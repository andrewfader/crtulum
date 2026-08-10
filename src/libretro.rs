// A minimal libretro host — enough to run an emulator core in-process and pull
// frames out of it.
//
// This is what makes scripted runs *frame-exact*. The alternative (drive a real
// emulator through a virtual gamepad, record its window) is at the mercy of real
// time and input latency; here we call `retro_run()` once per frame with the exact
// button mask the script says that frame has, so frame 1237 is frame 1237 every
// run. It's also much faster than real time and needs no window.
//
// Three rendering paths, and the core picks: a plain software framebuffer (every 2D
// system), OpenGL through a headless EGL context (glctx.rs), or Vulkan through an
// instance and device we stand up for it (vkctx.rs). Direct3D is declined, and those
// cores fall back on their own.
//
// libretro is a singleton C API — the callbacks are global function pointers with
// no user-data parameter, so a process hosts exactly one core at a time. `Core::load`
// enforces that.

use std::collections::HashMap;
use std::ffi::{c_char, c_uint, c_void, CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;

// ---------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------

#[repr(C)]
struct SystemInfo {
    library_name: *const c_char,
    library_version: *const c_char,
    valid_extensions: *const c_char,
    need_fullpath: bool,
    block_extract: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GameGeometry {
    base_width: c_uint,
    base_height: c_uint,
    max_width: c_uint,
    max_height: c_uint,
    aspect_ratio: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SystemTiming {
    fps: f64,
    sample_rate: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SystemAvInfo {
    geometry: GameGeometry,
    timing: SystemTiming,
}

#[repr(C)]
struct GameInfo {
    path: *const c_char,
    data: *const c_void,
    size: usize,
    meta: *const c_char,
}

/// `struct retro_hw_render_callback`. We fill in the two function pointers the core
/// calls into; it fills in everything else before handing it to us.
#[repr(C)]
#[derive(Clone, Copy)]
struct HwRenderCallback {
    context_type: c_uint,
    context_reset: Option<unsafe extern "C" fn()>,
    get_current_framebuffer: Option<unsafe extern "C" fn() -> usize>,
    get_proc_address: Option<crate::glctx::ProcAddressFn>,
    depth: bool,
    stencil: bool,
    bottom_left_origin: bool,
    version_major: c_uint,
    version_minor: c_uint,
    cache_context: bool,
    context_destroy: Option<unsafe extern "C" fn()>,
    debug_context: bool,
}

// retro_hw_context_type
const HW_CONTEXT_OPENGL: c_uint = 1;
const HW_CONTEXT_OPENGLES2: c_uint = 2;
const HW_CONTEXT_OPENGL_CORE: c_uint = 3;
const HW_CONTEXT_OPENGLES3: c_uint = 4;
const HW_CONTEXT_OPENGLES_VERSION: c_uint = 5;
const HW_CONTEXT_VULKAN: c_uint = 6;

// Both of these carry the EXPERIMENTAL bit (0x10000) in libretro.h.
const ENV_GET_HW_RENDER_INTERFACE: c_uint = 41 | 0x10000;
const ENV_SET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE: c_uint = 43 | 0x10000;

/// The sentinel `video_refresh` gets instead of a pixel pointer when the frame is
/// sitting in the GPU framebuffer.
const HW_FRAME_BUFFER_VALID: *const c_void = usize::MAX as *const c_void;

type EnvironmentFn = unsafe extern "C" fn(c_uint, *mut c_void) -> bool;
type VideoRefreshFn = unsafe extern "C" fn(*const c_void, c_uint, c_uint, usize);
type AudioSampleFn = unsafe extern "C" fn(i16, i16);
type AudioSampleBatchFn = unsafe extern "C" fn(*const i16, usize) -> usize;
type InputPollFn = unsafe extern "C" fn();
type InputStateFn = unsafe extern "C" fn(c_uint, c_uint, c_uint, c_uint) -> i16;

// The handful of environment commands worth answering. Everything else gets a
// "not supported", which every core copes with.
const ENV_GET_CAN_DUPE: c_uint = 3;
const ENV_SET_PERFORMANCE_LEVEL: c_uint = 8;
const ENV_GET_SYSTEM_DIRECTORY: c_uint = 9;
const ENV_SET_PIXEL_FORMAT: c_uint = 10;
const ENV_SET_HW_RENDER: c_uint = 14;
const ENV_GET_VARIABLE: c_uint = 15;
const ENV_SET_VARIABLES: c_uint = 16;
const ENV_GET_VARIABLE_UPDATE: c_uint = 17;
const ENV_SET_SUPPORT_NO_GAME: c_uint = 18;
const ENV_GET_SAVE_DIRECTORY: c_uint = 31;
const ENV_SET_GEOMETRY: c_uint = 37;
const ENV_GET_LANGUAGE: c_uint = 39;
const ENV_GET_CORE_OPTIONS_VERSION: c_uint = 52;
const ENV_GET_LOG_INTERFACE: c_uint = 27;
const ENV_GET_PERF_INTERFACE: c_uint = 28;

const DEVICE_JOYPAD: c_uint = 1;
const DEVICE_ANALOG: c_uint = 5;

/// libretro joypad button ids, which double as the bit positions in an input mask.
pub const BUTTONS: [(&str, u32); 20] = [
    ("b", 0),
    ("y", 1),
    ("select", 2),
    ("start", 3),
    ("up", 4),
    ("down", 5),
    ("left", 6),
    ("right", 7),
    ("a", 8),
    ("x", 9),
    ("l", 10),
    ("r", 11),
    ("l2", 12),
    ("r2", 13),
    ("l3", 14),
    ("r3", 15),
    // friendlier aliases
    ("shoulder-l", 10),
    ("shoulder-r", 11),
    ("trigger-l", 12),
    ("trigger-r", 13),
];

pub fn button_bit(name: &str) -> Option<u32> {
    let n = name.trim().to_ascii_lowercase();
    BUTTONS.iter().find(|(b, _)| *b == n).map(|(_, id)| 1u32 << id)
}

pub fn button_names() -> String {
    BUTTONS[..16].iter().map(|(n, _)| *n).collect::<Vec<_>>().join(" ")
}

#[derive(Clone, Copy, PartialEq)]
enum PixelFormat {
    Rgb1555,
    Xrgb8888,
    Rgb565,
}

// ---------------------------------------------------------------------------
// The logging and performance interfaces
// ---------------------------------------------------------------------------
//
// Not optional in practice. Cores ask for these and then use what they got — the N64
// cores here call straight through the perf callbacks, so declining the interface and
// leaving the struct full of uninitialised pointers is a segfault waiting to happen.

#[repr(C)]
struct LogCallback {
    log: unsafe extern "C" fn(c_uint, *const c_char, usize, usize, usize, usize),
}

/// `retro_log_printf_t` is variadic, which stable Rust cannot declare. What it *can*
/// do is take more fixed parameters than the caller's format string needs: on the
/// SysV x86-64 ABI the first six integer/pointer arguments arrive in registers, so
/// declaring four extra picks up the first four varargs. That's enough to render the
/// messages cores actually log, and without it every core error reads "[%s] %s".
///
/// Only `%s`/`%d`/`%u`/`%x`/`%c`/`%p` are expanded, and formatting stops at the first
/// specifier that would be passed in a vector register (`%f` and friends) or once the
/// four captured arguments run out — past that point the remaining format string is
/// emitted verbatim rather than guessed at.
unsafe extern "C" fn log_cb(
    level: c_uint,
    fmt: *const c_char,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
) {
    if std::env::var_os("CRTULUM_CORE_LOG").is_none() || fmt.is_null() {
        return;
    }
    let raw = CStr::from_ptr(fmt).to_string_lossy().into_owned();
    let args = [a1, a2, a3, a4];
    let mut out = String::with_capacity(raw.len() + 32);
    let mut next = 0usize;
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        // Skip flags/width/precision — we don't honour them, just the conversion.
        let mut spec = String::new();
        while let Some(&p) = chars.peek() {
            if p.is_ascii_alphabetic() || p == '%' {
                break;
            }
            spec.push(p);
            chars.next();
        }
        let conv = match chars.next() {
            Some(c) => c,
            None => break,
        };
        if conv == '%' {
            out.push('%');
            continue;
        }
        if next >= args.len() || matches!(conv, 'f' | 'g' | 'e' | 'F' | 'G' | 'E') {
            // Beyond here the register mapping is no longer knowable.
            out.push('%');
            out.push_str(&spec);
            out.push(conv);
            out.push_str("…");
            break;
        }
        let a = args[next];
        next += 1;
        match conv {
            's' => {
                if a == 0 {
                    out.push_str("(null)");
                } else {
                    out.push_str(&CStr::from_ptr(a as *const c_char).to_string_lossy());
                }
            }
            'd' | 'i' => out.push_str(&(a as i32).to_string()),
            'u' => out.push_str(&(a as u32).to_string()),
            'x' => out.push_str(&format!("{:x}", a as u32)),
            'X' => out.push_str(&format!("{:X}", a as u32)),
            'c' => out.push(char::from_u32(a as u32).unwrap_or('?')),
            'p' => out.push_str(&format!("{a:#x}")),
            'l' | 'z' => out.push_str(&a.to_string()),
            other => {
                out.push('%');
                out.push(other);
                next -= 1;
            }
        }
    }
    let tag = ["dbg", "info", "warn", "err"].get(level as usize).copied().unwrap_or("log");
    eprintln!("[core:{tag}] {}", out.trim_end());
}

#[repr(C)]
struct PerfCounter {
    ident: *const c_char,
    start: u64,
    total: u64,
    call_cnt: u64,
    registered: bool,
}

#[repr(C)]
struct PerfCallback {
    get_time_usec: unsafe extern "C" fn() -> i64,
    get_cpu_features: unsafe extern "C" fn() -> u64,
    get_perf_counter: unsafe extern "C" fn() -> u64,
    perf_register: unsafe extern "C" fn(*mut PerfCounter),
    perf_start: unsafe extern "C" fn(*mut PerfCounter),
    perf_stop: unsafe extern "C" fn(*mut PerfCounter),
    perf_log: unsafe extern "C" fn(),
}

fn now_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

unsafe extern "C" fn perf_get_time_usec() -> i64 {
    (now_nanos() / 1000) as i64
}

/// Zero means "assume nothing", which puts cores on their portable code paths — the
/// right answer for a renderer whose output has to be identical every run.
unsafe extern "C" fn perf_get_cpu_features() -> u64 {
    0
}

unsafe extern "C" fn perf_get_counter() -> u64 {
    now_nanos()
}

unsafe extern "C" fn perf_register(counter: *mut PerfCounter) {
    if !counter.is_null() {
        (*counter).registered = true;
    }
}

unsafe extern "C" fn perf_start(counter: *mut PerfCounter) {
    if !counter.is_null() {
        (*counter).start = now_nanos();
    }
}

unsafe extern "C" fn perf_stop(counter: *mut PerfCounter) {
    if !counter.is_null() {
        (*counter).total = (*counter).total.wrapping_add(now_nanos() - (*counter).start);
        (*counter).call_cnt = (*counter).call_cnt.wrapping_add(1);
    }
}

unsafe extern "C" fn perf_log() {}

// ---------------------------------------------------------------------------
// Host state shared with the C callbacks
// ---------------------------------------------------------------------------
//
// The callbacks are bare `extern "C" fn`s with nowhere to hang a context pointer,
// so the state lives in a global. It's only ever touched from the thread running
// `retro_run` (the core calls back into us synchronously from there), but a Mutex
// costs nothing at these rates and keeps it honest.

#[repr(C)]
struct Variable {
    key: *const c_char,
    value: *const c_char,
}

struct Host {
    pixel_format: PixelFormat,
    /// Core options the user asked for, as NUL-terminated strings the core can keep
    /// pointing at (this map is never cleared, so the pointers stay valid).
    options: HashMap<String, CString>,
    /// Option keys the core declared, for catching typos.
    declared: Vec<String>,
    /// Latest frame, already unpacked to RGBA8 and de-pitched.
    frame: Vec<u8>,
    frame_size: (u32, u32),
    got_frame: bool,
    audio: Vec<i16>,
    input: u32,
    analog: [i16; 2],
    dir: CString,
    /// Set when the core asked for (and got) hardware rendering.
    hw: Option<HwRenderCallback>,
    /// The FBO the core renders into. Read by the callback below, which has nowhere
    /// else to get it from.
    hw_fbo: u32,
    hw_proc: Option<crate::glctx::ProcAddressFn>,
    /// Vulkan: the core's context-negotiation interface, and the pointer we hand back
    /// for GET_HW_RENDER_INTERFACE. Kept as addresses because a raw pointer isn't
    /// `Send` and this lives in a global behind a mutex.
    vk_negotiation: usize,
    vk_interface: usize,
    /// True once a frame arrived as HW_FRAME_BUFFER_VALID, so `run_frame` knows to
    /// pull it out of the GPU instead of the CPU-side buffer.
    hw_frame_pending: bool,
}

// The two callbacks a hardware-rendering core calls into. Both are bare `extern "C"`
// with no user data, so they read the globals.
unsafe extern "C" fn hw_get_framebuffer() -> usize {
    host().lock().unwrap().hw_fbo as usize
}

unsafe extern "C" fn hw_get_proc_address(sym: *const c_char) -> *const c_void {
    let f = host().lock().unwrap().hw_proc;
    match f {
        Some(f) => f(sym),
        None => std::ptr::null(),
    }
}

static HOST: OnceLock<Mutex<Host>> = OnceLock::new();
static LOADED: AtomicBool = AtomicBool::new(false);

fn host() -> &'static Mutex<Host> {
    HOST.get_or_init(|| {
        Mutex::new(Host {
            pixel_format: PixelFormat::Rgb1555, // the libretro default
            options: HashMap::new(),
            declared: Vec::new(),
            frame: Vec::new(),
            frame_size: (0, 0),
            got_frame: false,
            audio: Vec::new(),
            input: 0,
            analog: [0, 0],
            dir: CString::new(".").unwrap(),
            hw: None,
            hw_fbo: 0,
            hw_proc: None,
            vk_negotiation: 0,
            vk_interface: 0,
            hw_frame_pending: false,
        })
    })
}

unsafe extern "C" fn env_cb(cmd: c_uint, data: *mut c_void) -> bool {
    if std::env::var_os("CRTULUM_TRACE_ENV").is_some() {
        eprintln!("  env {cmd}{}", if cmd == ENV_SET_HW_RENDER { "  <-- SET_HW_RENDER" } else { "" });
    }
    match cmd {
        // Yes, we can handle a NULL frame meaning "same as last time".
        ENV_GET_CAN_DUPE => {
            if !data.is_null() {
                *(data as *mut bool) = true;
            }
            true
        }
        ENV_GET_SYSTEM_DIRECTORY | ENV_GET_SAVE_DIRECTORY => {
            if data.is_null() {
                return false;
            }
            // The CString lives in the host for the process lifetime.
            let h = host().lock().unwrap();
            *(data as *mut *const c_char) = h.dir.as_ptr();
            true
        }
        ENV_SET_PIXEL_FORMAT => {
            if data.is_null() {
                return false;
            }
            let f = match *(data as *const c_uint) {
                0 => PixelFormat::Rgb1555,
                1 => PixelFormat::Xrgb8888,
                2 => PixelFormat::Rgb565,
                _ => return false,
            };
            host().lock().unwrap().pixel_format = f;
            true
        }
        // Hardware rendering: OpenGL through a headless EGL context (glctx.rs), or
        // Vulkan through an instance/device we create for the core (vkctx.rs). D3D
        // gets a no, and those cores fall back.
        ENV_SET_HW_RENDER => {
            if data.is_null() || std::env::var_os("CRTULUM_NO_HW").is_some() {
                return false;
            }
            let cb = data as *mut HwRenderCallback;
            match (*cb).context_type {
                HW_CONTEXT_OPENGL | HW_CONTEXT_OPENGLES2 | HW_CONTEXT_OPENGL_CORE
                | HW_CONTEXT_OPENGLES3 | HW_CONTEXT_OPENGLES_VERSION => {
                    // The GL entry points the core calls into. The context itself is
                    // built after load_game, once we know how big a frame can get.
                    (*cb).get_current_framebuffer = Some(hw_get_framebuffer);
                    (*cb).get_proc_address = Some(hw_get_proc_address);
                }
                // Vulkan cores don't use those two — they pull everything they need
                // out of the render interface below.
                HW_CONTEXT_VULKAN => {}
                _ => return false, // D3D9/10/11/12
            }
            host().lock().unwrap().hw = Some(*cb);
            true
        }
        // A Vulkan core can ask for a say in how the device is created — which
        // features and extensions its renderer needs. Stash it; it's consulted when
        // the Vulkan host is built, after load_game.
        ENV_SET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE => {
            if data.is_null() {
                return false;
            }
            let iface = data as *const crate::vkctx::NegotiationInterface;
            if (*iface).interface_type != crate::vkctx::NEGOTIATION_INTERFACE_VULKAN {
                return false; // not Vulkan negotiation
            }
            host().lock().unwrap().vk_negotiation = iface as usize;
            true
        }
        // The core is collecting the handles and callbacks it drives us through.
        ENV_GET_HW_RENDER_INTERFACE => {
            if data.is_null() {
                return false;
            }
            let ptr = host().lock().unwrap().vk_interface;
            if std::env::var_os("CRTULUM_TRACE_ENV").is_some() {
                eprintln!("  GET_HW_RENDER_INTERFACE -> {}", if ptr == 0 { "NOT READY (declined)" } else { "ok" });
            }
            if ptr == 0 {
                return false;
            }
            *(data as *mut *const c_void) = ptr as *const c_void;
            true
        }
        // The core is telling us which options it has. Record the keys so an
        // unrecognised `--option` can be reported instead of silently ignored.
        ENV_SET_VARIABLES => {
            if !data.is_null() {
                let mut p = data as *const Variable;
                let mut keys = Vec::new();
                // The array is terminated by a { NULL, NULL } entry.
                while !(*p).key.is_null() && keys.len() < 512 {
                    keys.push(CStr::from_ptr((*p).key).to_string_lossy().into_owned());
                    p = p.add(1);
                }
                host().lock().unwrap().declared = keys;
            }
            true
        }
        // Serve a value the user set; anything else falls back to the core's default.
        ENV_GET_VARIABLE => {
            if data.is_null() {
                return false;
            }
            let var = data as *mut Variable;
            if (*var).key.is_null() {
                return false;
            }
            let key = CStr::from_ptr((*var).key).to_string_lossy().into_owned();
            if std::env::var_os("CRTULUM_TRACE_ENV").is_some() {
                eprintln!("  GET_VARIABLE {key}");
            }
            let mut h = host().lock().unwrap();
            // Modern cores declare their options through SET_CORE_OPTIONS_V2, whose
            // layout we don't parse — but every option a core actually uses passes
            // through here, so this is the reliable place to learn the real key set.
            if !h.declared.iter().any(|d| *d == key) {
                h.declared.push(key.clone());
            }
            match h.options.get(&key) {
                Some(v) => {
                    (*var).value = v.as_ptr();
                    true
                }
                None => false,
            }
        }
        ENV_GET_VARIABLE_UPDATE => {
            if !data.is_null() {
                *(data as *mut bool) = false;
            }
            true
        }
        ENV_GET_LOG_INTERFACE => {
            if data.is_null() {
                return false;
            }
            (*(data as *mut LogCallback)).log = log_cb;
            true
        }
        ENV_GET_PERF_INTERFACE => {
            if data.is_null() {
                return false;
            }
            *(data as *mut PerfCallback) = PerfCallback {
                get_time_usec: perf_get_time_usec,
                get_cpu_features: perf_get_cpu_features,
                get_perf_counter: perf_get_counter,
                perf_register,
                perf_start,
                perf_stop,
                perf_log,
            };
            true
        }
        ENV_GET_CORE_OPTIONS_VERSION => {
            if !data.is_null() {
                *(data as *mut c_uint) = 0;
            }
            true
        }
        ENV_GET_LANGUAGE => {
            if !data.is_null() {
                *(data as *mut c_uint) = 0; // English
            }
            true
        }
        // Acknowledged, but nothing for us to do: frame geometry is read off the
        // video callback's own width/height, so a mid-run change just works.
        ENV_SET_PERFORMANCE_LEVEL | ENV_SET_SUPPORT_NO_GAME | ENV_SET_GEOMETRY => true,
        _ => false,
    }
}

unsafe extern "C" fn video_cb(data: *const c_void, width: c_uint, height: c_uint, pitch: usize) {
    let mut h = host().lock().unwrap();
    h.frame_size = (width, height);
    if data == HW_FRAME_BUFFER_VALID {
        // Nothing to copy here — the pixels are in the GPU framebuffer, and reading
        // them back needs the GL context, which lives on `Core`.
        h.hw_frame_pending = true;
        h.got_frame = true;
        return;
    }
    if data.is_null() {
        // Duped frame: keep whatever we had, and still count it as delivered.
        h.got_frame = true;
        return;
    }
    let (w, hgt) = (width as usize, height as usize);
    h.frame.resize(w * hgt * 4, 0);
    let fmt = h.pixel_format;
    for y in 0..hgt {
        let row = (data as *const u8).add(y * pitch);
        let out = y * w * 4;
        for x in 0..w {
            let (r, g, b) = match fmt {
                PixelFormat::Xrgb8888 => {
                    let p = *(row as *const u32).add(x);
                    (((p >> 16) & 0xFF) as u8, ((p >> 8) & 0xFF) as u8, (p & 0xFF) as u8)
                }
                PixelFormat::Rgb565 => {
                    let p = *(row as *const u16).add(x);
                    let (r5, g6, b5) = ((p >> 11) & 0x1F, (p >> 5) & 0x3F, p & 0x1F);
                    // Replicate the high bits into the low ones so full-scale stays full.
                    (
                        ((r5 << 3) | (r5 >> 2)) as u8,
                        ((g6 << 2) | (g6 >> 4)) as u8,
                        ((b5 << 3) | (b5 >> 2)) as u8,
                    )
                }
                PixelFormat::Rgb1555 => {
                    let p = *(row as *const u16).add(x);
                    let (r5, g5, b5) = ((p >> 10) & 0x1F, (p >> 5) & 0x1F, p & 0x1F);
                    (
                        ((r5 << 3) | (r5 >> 2)) as u8,
                        ((g5 << 3) | (g5 >> 2)) as u8,
                        ((b5 << 3) | (b5 >> 2)) as u8,
                    )
                }
            };
            let o = out + x * 4;
            h.frame[o] = r;
            h.frame[o + 1] = g;
            h.frame[o + 2] = b;
            h.frame[o + 3] = 255;
        }
    }
    h.got_frame = true;
}

unsafe extern "C" fn audio_sample_cb(left: i16, right: i16) {
    let mut h = host().lock().unwrap();
    h.audio.push(left);
    h.audio.push(right);
}

unsafe extern "C" fn audio_batch_cb(data: *const i16, frames: usize) -> usize {
    if !data.is_null() {
        let mut h = host().lock().unwrap();
        h.audio.extend_from_slice(std::slice::from_raw_parts(data, frames * 2));
    }
    frames
}

unsafe extern "C" fn input_poll_cb() {}

unsafe extern "C" fn input_state_cb(port: c_uint, device: c_uint, idx: c_uint, id: c_uint) -> i16 {
    if port != 0 {
        return 0;
    }
    let h = host().lock().unwrap();
    if device == DEVICE_JOYPAD && id <= 15 {
        ((h.input >> id) & 1) as i16
    } else if device == DEVICE_ANALOG && idx == 0 && id < 2 {
        h.analog[id as usize]
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// The core
// ---------------------------------------------------------------------------

pub struct Core {
    // Field order is drop order: the function pointers borrowed from `lib` must go
    // before it.
    run: unsafe extern "C" fn(),
    unload: unsafe extern "C" fn(),
    deinit: unsafe extern "C" fn(),
    /// Kept alive because `need_fullpath = false` cores may reference it for the
    /// whole session rather than copying.
    _rom: Vec<u8>,
    /// Present only for a hardware-rendering core. Declared before `_lib` so the GL
    /// objects are torn down before the core library is unloaded.
    gl: Option<crate::glctx::GlContext>,
    vk: Option<crate::vkctx::VkHost>,
    hw_bottom_left: bool,
    hw_reset: Option<unsafe extern "C" fn()>,
    hw_destroy: Option<unsafe extern "C" fn()>,
    /// The last frame the core actually drew, kept so a duped frame can repeat it.
    last_frame: Option<(Vec<u8>, u32, u32)>,
    /// Whether the core has ever drawn anything. Games take a while to get going —
    /// an N64 title runs hundreds of frames of boot before the first picture — so
    /// until then the tube shows black rather than an error.
    pub saw_frame: bool,
    _lib: Library,
    pub name: String,
    pub fps: f64,
    pub sample_rate: f64,
    pub geometry: (u32, u32),
}

macro_rules! sym {
    ($lib:expr, $t:ty, $name:literal) => {
        // The pointer is copied out of the Symbol, so it doesn't borrow the Library —
        // and the Library outlives it inside `Core`.
        *unsafe { $lib.get::<$t>(concat!($name, "\0").as_bytes()) }
            .with_context(|| format!("core is missing {}", $name))?
    };
}

impl Core {
    /// Load a core and (optionally) a ROM. One core per process.
    pub fn load(
        core_path: &Path,
        rom: Option<&Path>,
        system_dir: &Path,
        options: &[(String, String)],
    ) -> Result<Core> {
        if LOADED.swap(true, Ordering::SeqCst) {
            bail!("a libretro core is already loaded (the API is a process-wide singleton)");
        }
        let lib = unsafe { Library::new(core_path) }
            .with_context(|| format!("loading core {}", core_path.display()))?;

        let api: unsafe extern "C" fn() -> c_uint = sym!(lib, unsafe extern "C" fn() -> c_uint, "retro_api_version");
        let version = unsafe { api() };
        if version != 1 {
            bail!("core reports libretro API version {version}, expected 1");
        }

        let set_env: unsafe extern "C" fn(EnvironmentFn) = sym!(lib, unsafe extern "C" fn(EnvironmentFn), "retro_set_environment");
        let set_video: unsafe extern "C" fn(VideoRefreshFn) = sym!(lib, unsafe extern "C" fn(VideoRefreshFn), "retro_set_video_refresh");
        let set_audio: unsafe extern "C" fn(AudioSampleFn) = sym!(lib, unsafe extern "C" fn(AudioSampleFn), "retro_set_audio_sample");
        let set_audio_batch: unsafe extern "C" fn(AudioSampleBatchFn) = sym!(lib, unsafe extern "C" fn(AudioSampleBatchFn), "retro_set_audio_sample_batch");
        let set_poll: unsafe extern "C" fn(InputPollFn) = sym!(lib, unsafe extern "C" fn(InputPollFn), "retro_set_input_poll");
        let set_state: unsafe extern "C" fn(InputStateFn) = sym!(lib, unsafe extern "C" fn(InputStateFn), "retro_set_input_state");
        let init: unsafe extern "C" fn() = sym!(lib, unsafe extern "C" fn(), "retro_init");
        let get_info: unsafe extern "C" fn(*mut SystemInfo) = sym!(lib, unsafe extern "C" fn(*mut SystemInfo), "retro_get_system_info");
        let get_av: unsafe extern "C" fn(*mut SystemAvInfo) = sym!(lib, unsafe extern "C" fn(*mut SystemAvInfo), "retro_get_system_av_info");
        let load_game: unsafe extern "C" fn(*const GameInfo) -> bool = sym!(lib, unsafe extern "C" fn(*const GameInfo) -> bool, "retro_load_game");
        let run: unsafe extern "C" fn() = sym!(lib, unsafe extern "C" fn(), "retro_run");
        let unload: unsafe extern "C" fn() = sym!(lib, unsafe extern "C" fn(), "retro_unload_game");
        let deinit: unsafe extern "C" fn() = sym!(lib, unsafe extern "C" fn(), "retro_deinit");

        // Cores read the system directory during retro_init/load_game, so set it first.
        {
            let mut h = host().lock().unwrap();
            h.dir = CString::new(system_dir.to_string_lossy().as_bytes().to_vec())
                .unwrap_or_else(|_| CString::new(".").unwrap());
            h.pixel_format = PixelFormat::Rgb1555;
            h.audio.clear();
            h.input = 0;
            h.declared.clear();
            h.hw = None;
            h.hw_fbo = 0;
            h.vk_negotiation = 0;
            h.vk_interface = 0;
            h.hw_frame_pending = false;
            // Options have to be in place before retro_set_environment: cores read
            // them during init and load_game, not later.
            h.options = options
                .iter()
                .filter_map(|(k, v)| Some((k.clone(), CString::new(v.as_str()).ok()?)))
                .collect();
        }

        // retro_set_environment must come before retro_init.
        unsafe {
            set_env(env_cb);
            set_video(video_cb);
            set_audio(audio_sample_cb);
            set_audio_batch(audio_batch_cb);
            set_poll(input_poll_cb);
            set_state(input_state_cb);
            init();
        }

        let mut info = SystemInfo {
            library_name: std::ptr::null(),
            library_version: std::ptr::null(),
            valid_extensions: std::ptr::null(),
            need_fullpath: false,
            block_extract: false,
        };
        unsafe { get_info(&mut info) };
        let cstr = |p: *const c_char| -> String {
            if p.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            }
        };
        let name = format!("{} {}", cstr(info.library_name), cstr(info.library_version))
            .trim()
            .to_string();

        // Load the content. `need_fullpath` cores open the file themselves; the rest
        // want it in memory (and may keep referencing our buffer).
        let mut rom_bytes = Vec::new();
        let ok = match rom {
            Some(path) => {
                let c_path = CString::new(path.to_string_lossy().as_bytes().to_vec())?;
                if !info.need_fullpath {
                    rom_bytes = std::fs::read(path)
                        .with_context(|| format!("reading ROM {}", path.display()))?;
                }
                let gi = GameInfo {
                    path: c_path.as_ptr(),
                    data: if rom_bytes.is_empty() {
                        std::ptr::null()
                    } else {
                        rom_bytes.as_ptr() as *const c_void
                    },
                    size: rom_bytes.len(),
                    meta: std::ptr::null(),
                };
                unsafe { load_game(&gi) }
            }
            None => unsafe { load_game(std::ptr::null()) },
        };
        if !ok {
            unsafe { deinit() };
            LOADED.store(false, Ordering::SeqCst);
            bail!(
                "core `{}` refused to load {}",
                name,
                rom.map(|p| p.display().to_string()).unwrap_or_else(|| "(no content)".into())
            );
        }

        // A mistyped option would otherwise just sit there doing nothing.
        {
            let h = host().lock().unwrap();
            if !h.declared.is_empty() {
                for (k, _) in options {
                    if !h.declared.iter().any(|d| d == k) {
                        eprintln!(
                            "[emu] warning: `{k}` is not an option of this core — it declares {} \
                             (see the core's .info file for the full list)",
                            h.declared.len()
                        );
                    }
                }
            }
        }

        let mut av = SystemAvInfo::default();
        unsafe { get_av(&mut av) };
        // If the core asked for hardware rendering, now is the moment: av_info tells
        // us the largest frame it can produce, which is how big the render target has
        // to be. Then context_reset() lets the core build its GL state.
        let hw = host().lock().unwrap().hw;
        let (mut gl, mut hw_bottom_left, mut hw_reset, mut hw_destroy) = (None, false, None, None);
        let mut vk = None;
        if let Some(cb) = hw.filter(|cb| cb.context_type == HW_CONTEXT_VULKAN) {
            // Vulkan: build the instance/device (letting the core negotiate it if it
            // asked to), publish the interface, then let the core set itself up.
            let negotiation = {
                let ptr = host().lock().unwrap().vk_negotiation;
                if ptr == 0 {
                    None
                } else {
                    Some(unsafe { &*(ptr as *const crate::vkctx::NegotiationInterface) })
                }
            };
            let host_vk = crate::vkctx::VkHost::new(negotiation).with_context(|| {
                format!("`{name}` wants Vulkan {}.{} and it could not be set up", cb.version_major, cb.version_minor)
            })?;
            host().lock().unwrap().vk_interface = host_vk.interface_ptr() as usize;
            hw_reset = cb.context_reset;
            hw_destroy = cb.context_destroy;
            vk = Some(host_vk);
            if let Some(reset) = hw_reset {
                unsafe { reset() };
            }
        } else if let Some(cb) = hw {
            let gles = matches!(
                cb.context_type,
                HW_CONTEXT_OPENGLES2 | HW_CONTEXT_OPENGLES3 | HW_CONTEXT_OPENGLES_VERSION
            );
            let w = av.geometry.max_width.max(av.geometry.base_width).max(320);
            let h = av.geometry.max_height.max(av.geometry.base_height).max(240);
            let ctx = crate::glctx::GlContext::new(
                w,
                h,
                cb.depth,
                cb.stencil,
                cb.version_major,
                cb.version_minor,
                gles,
                cb.context_type == HW_CONTEXT_OPENGL_CORE,
            )
            .with_context(|| {
                format!(
                    "`{name}` needs a {} {}.{} context and one could not be created. \
                     Either the GPU/driver can't do headless GL, or the core has a \
                     software renderer worth using instead (e.g. \
                     --option parallel-n64-gfxplugin=angrylion)",
                    if gles { "GLES" } else { "OpenGL" },
                    cb.version_major,
                    cb.version_minor
                )
            })?;
            {
                let mut host = host().lock().unwrap();
                host.hw_fbo = ctx.framebuffer();
                host.hw_proc = Some(crate::glctx::raw_get_proc_address()?);
            }
            hw_bottom_left = cb.bottom_left_origin;
            hw_reset = cb.context_reset;
            hw_destroy = cb.context_destroy;
            gl = Some(ctx);
            // The core builds its shaders/FBOs in here, so the context must be current
            // (it is — GlContext::new leaves it so) and the FBO must already exist.
            if let Some(reset) = hw_reset {
                unsafe { reset() };
            }
        }

        let fps = if av.timing.fps > 1.0 { av.timing.fps } else { 60.0 };
        let sample_rate = if av.timing.sample_rate > 1.0 { av.timing.sample_rate } else { 48000.0 };

        Ok(Core {
            run,
            unload,
            deinit,
            _rom: rom_bytes,
            gl,
            vk,
            hw_bottom_left,
            hw_reset,
            hw_destroy,
            last_frame: None,
            saw_frame: false,
            _lib: lib,
            name,
            fps,
            sample_rate,
            geometry: (av.geometry.base_width.max(1), av.geometry.base_height.max(1)),
        })
    }

    /// Run exactly one frame with `input` held, and return the frame as RGBA8.
    ///
    /// The mask is the libretro joypad bit layout (see `BUTTONS`), so frame N of a
    /// script is frame N of the emulation — no timing slop.
    pub fn run_frame(&mut self, input: u32) -> Result<(Vec<u8>, u32, u32)> {
        self.run_frame_with_analog(input, [0, 0])
    }

    pub fn run_frame_with_analog(&mut self, input: u32, analog: [i16; 2]) -> Result<(Vec<u8>, u32, u32)> {
        {
            let mut h = host().lock().unwrap();
            h.input = input;
            h.analog = analog;
            h.got_frame = false;
            h.hw_frame_pending = false;
        }
        unsafe { (self.run)() };

        // A hardware frame is sitting in the GL framebuffer rather than in memory.
        let (hw_pending, size) = {
            let h = host().lock().unwrap();
            (h.hw_frame_pending, h.frame_size)
        };
        if hw_pending {
            let mut buf = Vec::new();
            let out = if let Some(vk) = self.vk.as_mut() {
                // Vulkan images are already top-down, so no flip here.
                vk.read_frame(size.0, size.1, &mut buf)?;
                (buf, size.0.max(1), size.1.max(1))
            } else {
                let gl = self
                    .gl
                    .as_mut()
                    .ok_or_else(|| anyhow!("core reported a GPU frame without a GL context"))?;
                // GL's origin is bottom-left; libretro frames are top-down.
                gl.read_rgba(size.0, size.1, self.hw_bottom_left, &mut buf);
                let (w, h) = (size.0.min(gl.size().0).max(1), size.1.min(gl.size().1).max(1));
                (buf, w, h)
            };
            self.saw_frame = true;
            self.last_frame = Some(out.clone());
            return Ok(out);
        }

        let produced = {
            let h = host().lock().unwrap();
            if h.got_frame && h.frame_size.0 > 0 && !h.frame.is_empty() {
                Some((h.frame.clone(), h.frame_size.0, h.frame_size.1))
            } else {
                None
            }
        };
        match produced {
            Some(f) => {
                self.saw_frame = true;
                self.last_frame = Some(f.clone());
                Ok(f)
            }
            // Nothing new this run. That's legal — we advertise GET_CAN_DUPE, and a
            // core drops frames while a game boots or whenever it can't keep up — so
            // the previous frame stands.
            None => match &self.last_frame {
                Some(f) => Ok(f.clone()),
                // Still booting: a black frame of the right shape.
                None => {
                    let (w, h) = self.geometry;
                    Ok((vec![0u8; (w * h * 4) as usize], w, h))
                }
            },
        }
    }

    /// Drain the audio the core has produced since the last call (interleaved S16 stereo).
    pub fn take_audio(&mut self) -> Vec<i16> {
        std::mem::take(&mut host().lock().unwrap().audio)
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        // Order matters, and it's the frontend's job to get right: the content goes
        // first, *then* the GPU context is taken away (a core may still be holding
        // GPU resources until unload returns), and retro_deinit last of all. Tearing
        // the context down first makes cores fall back to a software renderer
        // mid-shutdown, which is where SwanStation used to crash.
        let trace = std::env::var_os("CRTULUM_TRACE_TEARDOWN").is_some();
        macro_rules! step { ($m:literal) => { if trace { eprintln!("[teardown] {}", $m); } } }
        unsafe {
            // Order matters, and it's the frontend's job to get right. The GPU context
            // belongs to us, so the core is told it's going away while the core is
            // still fully alive — then the content unloads, then the core deinits.
            // Unloading first leaves cores tearing down GPU state they've already
            // released, which is a crash in SwanStation.
            if let Some(destroy) = self.hw_destroy {
                step!("context_destroy");
                destroy();
            }
            step!("retro_unload_game");
            (self.unload)();
            step!("retro_deinit");
            (self.deinit)();
            step!("core torn down; releasing the GPU context");
        }
        LOADED.store(false, Ordering::SeqCst);
    }
}

/// Where cores look for BIOS images and their own data files (mupen64plus keeps an
/// .ini there, PlayStation cores want a BIOS). RetroArch's system directory is the
/// one people actually have populated, so use it unless told otherwise.
pub fn system_dir(fallback: &Path) -> std::path::PathBuf {
    if let Some(d) = std::env::var_os("CRTULUM_SYSTEM_DIR") {
        return std::path::PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    for d in [
        format!("{home}/.config/retroarch/system"),
        format!("{home}/.var/app/org.libretro.RetroArch/config/retroarch/system"),
    ] {
        let p = std::path::PathBuf::from(d);
        if p.is_dir() {
            return p;
        }
    }
    fallback.to_path_buf()
}

/// Guess a libretro core for a ROM extension, then find its shared object.
pub fn find_core(rom: Option<&Path>, explicit: Option<&str>) -> Result<std::path::PathBuf> {
    let candidates: Vec<&str> = match explicit {
        Some(c) => vec![c],
        None => {
            let ext = rom
                .and_then(|r| r.extension())
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            match ext.as_str() {
                // 2D systems: every core here renders in software, which is all this
                // host supports — and all a CRT really wants anyway.
                "nes" | "fds" | "unf" | "unif" => vec!["nestopia", "fceumm", "quicknes", "mesen"],
                "sfc" | "smc" => vec!["snes9x", "bsnes_mercury_balanced", "bsnes", "mesen-s"],
                "gb" | "gbc" => vec!["gambatte", "sameboy", "mgba"],
                "gba" => vec!["mgba", "vbam", "gpsp"],
                "md" | "gen" | "smd" | "sgd" => vec!["genesis_plus_gx", "picodrive", "blastem"],
                "sms" | "gg" | "sg" => vec!["genesis_plus_gx", "gearsystem", "picodrive"],
                "32x" => vec!["picodrive"],
                "pce" | "sgx" => vec!["mednafen_pce", "mednafen_pce_fast", "mednafen_supergrafx"],
                "a26" => vec!["stella", "stella2014"],
                "col" => vec!["gearcoleco", "bluemsx"],
                "lnx" => vec!["mednafen_lynx"],
                "ngp" | "ngc" => vec!["mednafen_ngp"],
                "ws" | "wsc" => vec!["mednafen_wswan"],
                "vb" => vec!["mednafen_vb"],
                "int" => vec!["freeintv"],
                // PlayStation: Beetle PSX and PCSX-ReARMed rasterise on the CPU;
                // SwanStation/DuckStation default to a GPU renderer and need
                // `--option swanstation_GPU_Renderer=Software` (or the DuckStation
                // equivalent) to work here.
                "cue" | "chd" | "pbp" | "toc" | "ccd" | "mds" | "exe" | "psexe" => {
                    vec!["mednafen_psx", "pcsx_rearmed", "swanstation", "duckstation"]
                }
                // N64 is the hard one: these cores are built around GL/Vulkan. Only
                // ParaLLEl's angrylion path is a pure software rasteriser, and it has
                // to be asked for: `--option parallel-n64-gfxplugin=angrylion`.
                "n64" | "z64" | "v64" | "ndd" => vec!["parallel_n64", "mupen64plus_next"],
                // .bin and .iso mean half a dozen different machines — make the user say.
                _ => vec![],
            }
        }
    };
    if candidates.is_empty() {
        bail!(
            "cannot guess a libretro core for {} — that extension is used by several \
             systems, so name one with --core <name>",
            rom.map(|p| p.display().to_string()).unwrap_or_else(|| "(no rom)".into())
        );
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let dirs = [
        std::env::var("RETROARCH_CORE_DIR").unwrap_or_default(),
        format!("{home}/.config/retroarch/cores"),
        format!("{home}/.var/app/org.libretro.RetroArch/config/retroarch/cores"),
        "/usr/lib/libretro".into(),
        "/usr/local/lib/libretro".into(),
    ];
    for c in &candidates {
        let direct = Path::new(c);
        if direct.is_file() {
            return Ok(direct.to_path_buf());
        }
        for d in dirs.iter().filter(|d| !d.is_empty()) {
            for name in [format!("{c}_libretro.so"), format!("{c}.so")] {
                let p = Path::new(d).join(&name);
                if p.is_file() {
                    return Ok(p);
                }
            }
        }
    }
    Err(anyhow!(
        "no libretro core found (tried {}). Install one with RetroArch's core \
         downloader, or pass --core /path/to/<name>_libretro.so",
        candidates.join(", ")
    ))
}

/// libretro is a process-wide singleton, so every test that loads a core has to take
/// turns (cargo runs them on parallel threads).
#[cfg(test)]
static CORE_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;


    /// Probe: load every core named in CRTULUM_PROBE_CORES (no content) and report
    /// what it is and whether it insists on hardware rendering.
    #[test]
    #[ignore = "diagnostic"]
    fn probe_cores() {
        let _serialize = super::CORE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let name = std::env::var("CRTULUM_PROBE_CORE").unwrap();
        let path = find_core(None, Some(&name)).unwrap();
        let lib = unsafe { Library::new(&path) }.unwrap();
        let api: unsafe extern "C" fn() -> c_uint =
            *unsafe { lib.get(b"retro_api_version\0") }.unwrap();
        let set_env: unsafe extern "C" fn(EnvironmentFn) =
            *unsafe { lib.get(b"retro_set_environment\0") }.unwrap();
        let init: unsafe extern "C" fn() = *unsafe { lib.get(b"retro_init\0") }.unwrap();
        let get_info: unsafe extern "C" fn(*mut SystemInfo) =
            *unsafe { lib.get(b"retro_get_system_info\0") }.unwrap();
        let mut info = SystemInfo {
            library_name: std::ptr::null(),
            library_version: std::ptr::null(),
            valid_extensions: std::ptr::null(),
            need_fullpath: false,
            block_extract: false,
        };
        unsafe {
            set_env(env_cb);
            init();
            get_info(&mut info);
        }
        let cs = |p: *const c_char| {
            if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() }
        };
        eprintln!(
            "RESULT {name}: api={} lib=\"{} {}\" ext=[{}] need_fullpath={}",
            unsafe { api() },
            cs(info.library_name),
            cs(info.library_version),
            cs(info.valid_extensions),
            info.need_fullpath
        );
    }

    /// Diagnostic: run any core+ROM and print the centre pixel per button, so a new
    /// system can be checked the same way the NES one is.
    #[test]
    #[ignore = "diagnostic"]
    fn probe_rom() {
        let _serialize = super::CORE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let rom = std::env::var("CRTULUM_PROBE_ROM").unwrap();
        let rom = Path::new(&rom);
        let core = find_core(Some(rom), std::env::var("CRTULUM_PROBE_CORE").ok().as_deref()).unwrap();
        let options: Vec<(String, String)> = std::env::var("CRTULUM_PROBE_OPTIONS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|kv| kv.split_once('=').map(|(k, v)| (k.trim().into(), v.trim().into())))
            .collect();
        let mut core = Core::load(&core, Some(rom), Path::new("."), &options).unwrap();
        eprintln!("RESULT core={} {}x{} @{:.3}fps {:.0}Hz", core.name, core.geometry.0, core.geometry.1, core.fps, core.sample_rate);
        let centre = |f: &[u8], w: u32, h: u32| {
            let o = (((h / 2) * w + w / 2) * 4) as usize;
            [f[o], f[o + 1], f[o + 2]]
        };
        let mut idle = [0u8; 3];
        for _ in 0..90 {
            let (f, w, h) = core.run_frame(0).unwrap();
            idle = centre(&f, w, h);
        }
        eprintln!("RESULT idle {idle:?}");
        for name in ["up", "down", "left", "right", "b", "a", "start"] {
            let m = button_bit(name).unwrap();
            let mut c = [0u8; 3];
            let mut dim = (0, 0);
            for _ in 0..4 {
                let (f, w, h) = core.run_frame(m).unwrap();
                c = centre(&f, w, h);
                dim = (w, h);
            }
            eprintln!("RESULT {name:6} {c:?} {}  ({}x{})", if c == idle { "SAME as idle" } else { "distinct" }, dim.0, dim.1);
            for _ in 0..3 { core.run_frame(0).unwrap(); }
        }
    }

    /// The same end-to-end check on a second system and a second CPU architecture —
    /// a Mega Drive core with the 68k homebrew from examples/make_genesis_test_rom.py.
    /// (Start and Genesis-A live in the pad's other TH half, which that ROM doesn't
    /// read, so they're not part of this.)
    #[test]
    fn genesis_core_sees_scripted_input() {
        let _serialize = super::CORE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let rom = Path::new("examples/inputtest.md");
        if !rom.exists() {
            eprintln!("skipping: run examples/make_genesis_test_rom.py first");
            return;
        }
        let core = match find_core(Some(rom), None) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let mut core = Core::load(&core, Some(rom), Path::new("."), &[]).expect("load core");
        assert!(core.fps > 59.0 && core.fps < 60.5, "Mega Drive runs at ~59.92, got {}", core.fps);

        let centre = |f: &[u8], w: u32, h: u32| {
            let o = (((h / 2) * w + w / 2) * 4) as usize;
            [f[o], f[o + 1], f[o + 2]]
        };
        let mut idle = [0u8; 3];
        let mut dim = (0, 0);
        for _ in 0..90 {
            let (f, w, h) = core.run_frame(0).expect("frame");
            idle = centre(&f, w, h);
            dim = (w, h);
        }
        assert_eq!(dim, (320, 224), "H40 mode is a 320x224 frame");
        assert_eq!(idle, [0, 0, 0], "nothing held paints the screen black");

        let mut seen = vec![idle];
        for name in ["up", "down", "left", "right", "b", "a"] {
            let mask = button_bit(name).unwrap();
            let mut col = [0u8; 3];
            for _ in 0..4 {
                let (f, w, h) = core.run_frame(mask).expect("frame");
                col = centre(&f, w, h);
            }
            assert!(
                !seen.contains(&col),
                "`{name}` produced {col:?}, already seen — input is not reaching the 68k"
            );
            seen.push(col);
            for _ in 0..3 {
                core.run_frame(0).expect("frame");
            }
        }
    }

    /// The tightest thing a TAS can express is a one-frame press, so it has to
    /// survive a long run without drifting: tap for a single frame every 10 frames
    /// across 600 frames and require the emulation to see every single one.
    ///
    /// (This caught a real problem in the test ROM itself — polling from a spin loop
    /// instead of NMI, whose phase creeps against the frame boundary and dropped a
    /// tap every 90 frames.)
    #[test]
    #[ignore = "loads a real core; run with --ignored (also covered by the main test)"]
    fn one_frame_taps_never_drift() {
        let _serialize = super::CORE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let rom = Path::new("examples/inputtest.nes");
        let core = find_core(Some(rom), std::env::var("CRTULUM_TEST_CORE").ok().as_deref()).unwrap();
        let mut core = Core::load(&core, Some(rom), Path::new("."), &[]).unwrap();
        let centre = |f: &[u8], w: u32, h: u32| {
            let o = (((h / 2) * w + w / 2) * 4) as usize;
            [f[o], f[o + 1], f[o + 2]]
        };
        let b = button_bit("b").unwrap();
        // Tap for exactly one frame every 10 frames and see which taps the ROM sees.
        let mut idle = [0u8; 3];
        let mut hits = 0;
        let mut misses = Vec::new();
        for f in 0..600u32 {
            let tap = f >= 100 && f % 10 == 0;
            let (fr, w, h) = core.run_frame(if tap { b } else { 0 }).unwrap();
            let c = centre(&fr, w, h);
            if f == 50 { idle = c; }
            if tap {
                if c != idle { hits += 1; } else { misses.push(f); }
            }
        }
        assert!(
            misses.is_empty(),
            "{} of 50 one-frame taps were missed (at frames {misses:?}) — input is drifting \
             against the frame boundary",
            misses.len()
        );
        assert_eq!(hits, 50);
    }

    /// Load a real core with the homebrew test ROM (examples/inputtest.nes) and check
    /// that a scripted button mask actually reaches the emulation: the ROM paints the
    /// screen a colour derived from the buttons held that frame, so the centre pixel
    /// is a direct readout of what the core saw.
    #[test]
    fn core_runs_and_sees_scripted_input() {
        let _serialize = super::CORE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let rom = Path::new("examples/inputtest.nes");
        if !rom.exists() {
            eprintln!("skipping: run examples/make_test_rom.py first");
            return;
        }
        let forced = std::env::var("CRTULUM_TEST_CORE").ok();
        let core = match find_core(Some(rom), forced.as_deref()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let mut core = Core::load(&core, Some(rom), Path::new("."), &[]).expect("load core");
        assert!(core.fps > 59.0 && core.fps < 61.0, "NES should run at ~60 fps, got {}", core.fps);
        assert_eq!(core.geometry.1, 240, "NES is 240 lines");

        let centre = |f: &[u8], w: u32, h: u32| {
            let o = (((h / 2) * w + w / 2) * 4) as usize;
            [f[o], f[o + 1], f[o + 2]]
        };

        // Let the ROM finish its two-vblank warm-up, then sample with nothing held.
        let mut idle = [0u8; 3];
        for _ in 0..20 {
            let (f, w, h) = core.run_frame(0).expect("frame");
            idle = centre(&f, w, h);
        }

        // Every button should paint a distinct colour, and each must differ from idle.
        let mut seen = vec![idle];
        for name in ["a", "b", "start", "left", "right"] {
            let mask = button_bit(name).unwrap();
            let mut col = [0u8; 3];
            for _ in 0..3 {
                let (f, w, h) = core.run_frame(mask).expect("frame");
                col = centre(&f, w, h);
            }
            assert!(
                !seen.contains(&col),
                "`{name}` produced {col:?}, which the core had already shown — input is not reaching the emulation"
            );
            seen.push(col);
        }

        // Releasing everything must come back to the idle colour.
        let mut back = [0u8; 3];
        for _ in 0..3 {
            let (f, w, h) = core.run_frame(0).expect("frame");
            back = centre(&f, w, h);
        }
        assert_eq!(back, idle, "releasing every button should return to the idle colour");

        // A single-frame tap — the finest input a TAS can express — must show up in
        // the emulated output, and only briefly. (The ROM writes its palette after
        // polling, so the colour lands on the frame after the one that held it.)
        let b = button_bit("b").unwrap();
        let mut trace = Vec::new();
        for f in 0..8 {
            let (fr, w, hh) = core.run_frame(if f == 3 { b } else { 0 }).expect("frame");
            trace.push(centre(&fr, w, hh));
        }
        let flashes: Vec<usize> = trace
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != idle)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            flashes.len(),
            1,
            "a one-frame press should light exactly one emulated frame, got {flashes:?} from {trace:?}"
        );

        // The core should be producing audio too (silence is still samples).
        assert!(!core.take_audio().is_empty(), "core produced no audio samples");
    }
}

// ---------------------------------------------------------------------------
// Real games, real cores
// ---------------------------------------------------------------------------
//
// The homebrew test ROMs prove input timing exactly, but they only exercise two
// systems and they're deliberately trivial — no mappers, no BIOS, no 3D, no audio
// worth the name. These tests run actual games from a local library instead, one per
// system, and check the things that are true of any game regardless of which one it
// is: the core loads, it produces a picture, the picture has content and moves, and
// the run is reproducible.
//
// Nothing here asserts anything about a particular game's content — only structure —
// so the tests hold whichever titles happen to be in the library.
//
// The library isn't part of the repo. Point `CRTULUM_ROMS` at yours, or these skip.

#[cfg(test)]
mod real_roms {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    struct System {
        name: &'static str,
        dir: &'static str,
        exts: &'static [&'static str],
        core: Option<&'static str>,
        options: &'static [(&'static str, &'static str)],
        /// How long to wait for the game to come up. Games differ wildly — some are
        /// drawing within a second, some sit on a near-blank logo for ten — so this is
        /// a ceiling, not a target: the run stops as soon as the picture is alive.
        max_frames: u32,
    }

    const SYSTEMS: &[System] = &[
        System { name: "NES",         dir: "nes",       exts: &["nes"],        core: None, options: &[], max_frames: 1500 },
        System { name: "SNES",        dir: "snes",      exts: &["sfc", "smc"], core: None, options: &[], max_frames: 1500 },
        System { name: "Game Boy",    dir: "gb",        exts: &["gb", "gbc"],  core: None, options: &[], max_frames: 1500 },
        System { name: "Mega Drive",  dir: "megadrive", exts: &["md", "gen", "bin"], core: Some("genesis_plus_gx"), options: &[], max_frames: 1500 },
        // N64 on the software rasteriser: the GL core wants its own thread (see the
        // README), and angrylion is the deterministic path anyway.
        System { name: "N64",         dir: "n64",       exts: &["z64", "n64", "v64"], core: Some("parallel_n64"),
                 options: &[("parallel-n64-gfxplugin", "angrylion")], max_frames: 2000 },
        // PlayStation twice over — this is what keeps both GPU backends honest.
        System { name: "PSX/Vulkan",  dir: "psx",       exts: &["cue"], core: Some("swanstation"),
                 options: &[("swanstation_GPU_Renderer", "Vulkan")], max_frames: 1500 },
        System { name: "PSX/OpenGL",  dir: "psx",       exts: &["cue"], core: Some("swanstation"),
                 options: &[("swanstation_GPU_Renderer", "OpenGL")], max_frames: 1500 },
    ];

    fn library() -> Option<std::path::PathBuf> {
        let root = std::env::var("CRTULUM_ROMS").unwrap_or_else(|_| "/mnt/crucial/roms".into());
        let root = std::path::PathBuf::from(root);
        root.is_dir().then_some(root)
    }

    /// First game of a system, alphabetically, so a run picks the same one each time.
    fn pick_rom(root: &Path, sys: &System) -> Option<std::path::PathBuf> {
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(root.join(sys.dir))
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| sys.exts.contains(&e.to_ascii_lowercase().as_str()))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        files.into_iter().next()
    }

    fn hash(frame: &[u8]) -> u64 {
        let mut h = DefaultHasher::new();
        frame.hash(&mut h);
        h.finish()
    }

    /// A picture is "alive" once it has structure and has changed over time. Both
    /// bars have to clear hardware that predates colour: an original Game Boy has
    /// exactly four shades, so anything above a handful of "colours" would be asking
    /// it to do the impossible. Motion is the stronger signal anyway — a blank screen
    /// hashes to one frame forever, and a hung one stops changing.
    const MIN_COLOURS: usize = 3;
    const MIN_MOTION: usize = 8;

    /// Run a game until its picture comes alive, or give up at `max_frames`.
    /// Reports (frames taken, distinct frames, richest colour count, size, fps).
    fn run(sys: &System, rom: &Path) -> Result<(u32, usize, usize, (u32, u32), f64)> {
        let options: Vec<(String, String)> = sys
            .options
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let core_path = find_core(Some(rom), sys.core)?;
        let mut core = Core::load(&core_path, Some(rom), &system_dir(Path::new(".")), &options)?;
        let (fps, mut size) = (core.fps, core.geometry);

        let mut hashes = std::collections::HashSet::new();
        let mut richest = 0usize;
        let mut took = 0;
        let max = std::env::var("CRTULUM_TEST_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(sys.max_frames);
        for f in 0..max {
            took = f + 1;
            // Nudge start/A every so often so games that wait on a title screen move on.
            let mask = if f % 64 == 32 { button_bit("start").unwrap() | button_bit("a").unwrap() } else { 0 };
            let (frame, w, h) = core.run_frame(mask)?;
            size = (w, h);
            hashes.insert(hash(&frame));
            let colours = frame.chunks_exact(4).map(|p| u32::from_le_bytes([p[0], p[1], p[2], 0])).collect::<std::collections::HashSet<_>>();
            richest = richest.max(colours.len());
            if std::env::var_os("CRTULUM_TEST_TRACE").is_some() && f % 60 == 0 {
                eprintln!("    frame {f:4}: {}x{} {} colours", w, h, colours.len());
            }
            if richest >= MIN_COLOURS && hashes.len() > MIN_MOTION {
                break; // it's up and running; no need to sit through the rest
            }
        }
        Ok((took, hashes.len(), richest, size, fps))
    }

    #[test]
    fn real_games_run_on_every_supported_system() {
        let _serialize = super::CORE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(root) = library() else {
            eprintln!("skipping: no ROM library (set CRTULUM_ROMS)");
            return;
        };

        // `CRTULUM_TEST_SYSTEMS=SNES,N64` narrows the run when chasing one of them.
        let only = std::env::var("CRTULUM_TEST_SYSTEMS").unwrap_or_default();
        let mut checked = 0;
        for sys in SYSTEMS {
            if !only.is_empty() && !only.split(',').any(|w| sys.name.starts_with(w.trim())) {
                continue;
            }
            let Some(rom) = pick_rom(&root, sys) else {
                eprintln!("{:12} — no ROMs in {}/{}, skipped", sys.name, root.display(), sys.dir);
                continue;
            };
            let (took, distinct, colours, size, fps) =
                run(sys, &rom).unwrap_or_else(|e| panic!("{}: {e:#}", sys.name));

            assert!(
                colours >= MIN_COLOURS,
                "{}: the picture never got past {colours} colours in {took} frames — it's blank",
                sys.name
            );
            assert!(
                distinct > MIN_MOTION,
                "{}: only {distinct} distinct frames in {took} — the picture never moved",
                sys.name
            );
            assert!(size.0 >= 100 && size.1 >= 100, "{}: implausible frame {size:?}", sys.name);
            assert!(fps > 47.0 && fps < 61.0, "{}: implausible rate {fps}", sys.name);

            eprintln!(
                "{:12} ok — {}x{} @ {fps:.2} fps · alive after {took} frames · {distinct} distinct · {colours} colours · {}",
                sys.name,
                size.0,
                size.1,
                rom.file_name().unwrap_or_default().to_string_lossy()
            );
            checked += 1;
        }
        if only.is_empty() {
            assert!(checked >= 4, "only {checked} systems were exercised");
        }
    }

    /// The property the whole scripted-run feature rests on: same ROM, same inputs,
    /// same frames — across a full unload and reload of the core.
    #[test]
    fn a_real_game_renders_identically_twice() {
        let _serialize = super::CORE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(root) = library() else {
            eprintln!("skipping: no ROM library (set CRTULUM_ROMS)");
            return;
        };
        let sys = &SYSTEMS[0]; // NES
        let Some(rom) = pick_rom(&root, sys) else {
            eprintln!("skipping: no NES ROMs");
            return;
        };

        let take = |script: &[(u32, u32)]| -> Vec<u64> {
            let core_path = find_core(Some(&rom), sys.core).unwrap();
            let mut core =
                Core::load(&core_path, Some(&rom), &system_dir(Path::new(".")), &[]).unwrap();
            let mut out = Vec::new();
            for f in 0..180u32 {
                let mask = script.iter().find(|(at, _)| *at == f).map(|(_, m)| *m).unwrap_or(0);
                out.push(hash(&core.run_frame(mask).unwrap().0));
            }
            out
        };

        let script = [(30, button_bit("start").unwrap()), (90, button_bit("a").unwrap())];
        let first = take(&script);
        let second = take(&script);
        assert_eq!(first, second, "the same run produced different frames the second time");
        // Note: no "different input gives a different picture" check here — a real game
        // is free to ignore a button on its title screen, and that property is already
        // proven exactly by the homebrew ROM tests, where every button drives the screen.
    }
}
