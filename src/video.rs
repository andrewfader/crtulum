// crtulum — scripted video export.
//
// This is the offline sibling of the live window: instead of orbiting a tube with
// the mouse, you hand it a *source* and a *script*, and it renders the whole thing
// through the same CRT (same shader, same phosphor history planes, same presets)
// straight into ffmpeg, producing an .mp4/.mkv/.webm.
//
//   crtulum --render clip.mp4 out.mp4
//   crtulum --render 'https://youtu.be/…' out.mp4 --preset pvm
//   crtulum --render frames/ out.mkv --fps 30
//   crtulum --render --rom smb.nes --movie run.bsv out.mp4     (TAS via RetroArch)
//   crtulum --render clip.mp4 out.mp4 --script run.crts
//
// The script is a small timeline DSL — a Capybara-ish "do this, then that" list of
// camera moves, preset swaps, power cycles and degausses over the runtime of the
// video. See `parse_script` below for the grammar, and examples/ for a sample.
//
// Pipeline:
//
//   source ──ffmpeg decode──▶ raw RGBA frames ──▶ [ accum pass: signal → phosphor
//   plane w/ per-channel decay ] ──▶ [ tube pass @ SSAA ] ──▶ [ GPU box resolve ]
//   ──▶ readback ──ffmpeg encode──▶ out.mp4 (audio muxed back from the source)
//
// Two details that matter for fidelity:
//   * the phosphor planes persist across frames, so motion melts exactly like the
//     live path (this is why we can't just render stills and `ffmpeg -i %04d.png`);
//   * the tube is driven at 60 fields/sec regardless of the output frame rate, so a
//     30 fps export still scans each frame twice and 480i still twitters.

use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use wgpu::util::DeviceExt;

use crate::{
    accum_step, build_mesh, build_resources, create_depth, draw_tube, smoothstep01, write_uniforms,
    Orbit, Preset, Resources, ALL_PRESETS, COLLAPSE_DUR, DEGAUSS_DUR, DEGAUSS_TAU, TRINITRON,
    WARMUP_DUR,
};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Input {
    /// Anything ffmpeg can demux (also a directory of PNGs, via a glob pattern).
    Media(PathBuf),
    /// Anything yt-dlp can fetch (YouTube, Vimeo, a direct link, …).
    Url(String),
    /// A ROM run in-process by a libretro core, driven by the script's input
    /// timeline. Frame-exact, headless, and faster than real time.
    Rom {
        rom: PathBuf,
        core: Option<String>,
    },
    /// A ROM played back through RetroArch from a pre-authored replay/TAS file.
    /// RetroArch owns the emulation and the input; we consume its recording.
    Replay {
        rom: PathBuf,
        movie: PathBuf,
        core: Option<String>,
    },
}

pub struct Opts {
    pub input: Input,
    pub output: PathBuf,
    pub size: (u32, u32),
    pub fps: f64,                        // 0 → inherit the source rate
    pub ssaa: u32,                       // supersampling factor (1 = fast preview)
    pub source_size: Option<(u32, u32)>, // signal resolution fed to the tube
    pub start: f64,
    pub duration: Option<f64>,
    pub codec: String,
    pub crf: u32,
    pub audio: bool,
    pub script: Script,
    /// libretro core options, e.g. `parallel-n64-gfxplugin=angrylion`. Needed to put
    /// the 3D cores on a software renderer, since this host has no GL/Vulkan path.
    pub core_options: Vec<(String, String)>,
    /// Microsoft Agent character to put on the screen — a name or an asset directory.
    pub agent: Option<String>,
    pub dry_run: bool,
}

// ---------------------------------------------------------------------------
// The script DSL
// ---------------------------------------------------------------------------
//
// Blank lines and `#` comments are ignored. Two kinds of statement:
//
//   <setting> …            — applies before the first frame (a header directive)
//   at <time> <action> …   — fires at that point on the timeline
//
// Times are seconds (`12`, `12.5`) or clock (`1:03`, `1:02:30.5`).
//
//   # ---- header ----
//   source  clip.mp4          # so the script is self-contained (CLI arg wins)
//   size    1280x960
//   fps     60
//   ssaa    3
//   lines   240               # signal resolution fed to the tube (240p look)
//   preset  trinitron
//   camera  yaw=0.82 pitch=0.34 dist=3.7
//   exposure 1.0
//   interlace off
//   subpixel off
//
//   # ---- timeline ----
//   at 0:00  power on                              # cold start: raster blooms open
//   at 0:02  degauss
//   at 0:04  camera to yaw=-0.45 dist=3.1 over 5 ease
//   at 0:20  preset pvm                            # swap tubes mid-shot
//   at 0:30  exposure to 1.35 over 2
//   at 0:40  spin 1 over 8                         # one full orbit
//   at 1:10  power off                             # raster collapses to a dot
//
// With `rom` in the header, the same timeline also drives the *run* — this is the
// tool-assisted part. Inputs can be addressed in seconds like everything else, or
// by exact frame number, which is what a TAS actually cares about:
//
//   rom      smb.nes
//   core     nestopia          # optional; guessed from the extension
//   frames   3600              # how long to run (or `duration 60`)
//
//   at 1.0     press start                 # a momentary press (2 frames)
//   at 2.0     hold right                  # …stays down…
//   frame 210  press a for 18 frames       # a precisely-timed jump, mid-run
//   at 6.0     release right
//   at 6.5     tap b                       # exactly one frame
//
#[derive(Clone, Debug)]
pub enum Action {
    Preset(&'static str),
    /// Absolute camera move. `None` fields hold their current value.
    Camera {
        yaw: Option<f32>,
        pitch: Option<f32>,
        dist: Option<f32>,
        over: f32,
        ease: bool,
    },
    /// Relative yaw sweep, in turns.
    Spin {
        turns: f32,
        over: f32,
        ease: bool,
    },
    Exposure {
        to: f32,
        over: f32,
        ease: bool,
    },
    Power(bool),
    Degauss,
    Interlace(bool),
    Subpixel(bool),
    Bfi(bool),
    /// No-op; only extends the timeline's notion of "the end".
    Wait(f32),

    // --- emulator input (only meaningful with a `rom`) ---
    /// Press for a fixed span, then release. `dur: None` = a short default press.
    Press { buttons: u32, dur: Option<Dur> },
    /// Press and leave down until a matching `release`.
    Hold { buttons: u32 },
    Release { buttons: u32 },
    /// Set the left analog stick. Values use the conventional -1..1 range.
    Stick { x: f32, y: f32 },

    // --- the character on the screen (only meaningful with an `agent`) ---
    /// One instruction to the Microsoft Agent character. Kept out of [`Timeline`]
    /// because an animation isn't a function of the clock — it branches, so it has
    /// to be folded frame by frame. See [`crate::agent`].
    Agent(crate::agent::Cmd),
}

/// A span written either in seconds or in exact frames. Frames are what a TAS is
/// authored in; seconds are what a camera move is authored in. Both convert once
/// the core's frame rate is known.
#[derive(Clone, Copy, Debug)]
pub enum Dur {
    Seconds(f32),
    Frames(u64),
}

impl Dur {
    fn frames(self, fps: f64) -> u64 {
        match self {
            Dur::Seconds(s) => ((s as f64 * fps).round() as i64).max(1) as u64,
            Dur::Frames(f) => f.max(1),
        }
    }
}

/// When something happens: a wall-clock moment, or an exact emulated frame.
#[derive(Clone, Copy, Debug)]
pub enum When {
    Seconds(f32),
    Frame(u64),
}

impl When {
    fn seconds(self, fps: f64) -> f32 {
        match self {
            When::Seconds(s) => s,
            When::Frame(f) => (f as f64 / fps) as f32,
        }
    }
    fn frame(self, fps: f64) -> u64 {
        match self {
            When::Seconds(s) => ((s as f64 * fps).round() as i64).max(0) as u64,
            When::Frame(f) => f,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Script {
    pub source: Option<String>,
    pub rom: Option<String>,
    /// Character to put on the screen — a name to look up, or a path to an asset
    /// directory.
    pub agent: Option<String>,
    pub core: Option<String>,
    /// Explicit run length in emulated frames (a `duration` in seconds also works).
    pub frames: Option<u64>,
    pub options: Vec<(String, String)>,
    pub size: Option<(u32, u32)>,
    pub fps: Option<f64>,
    pub ssaa: Option<u32>,
    pub source_size: Option<(u32, u32)>,
    pub start: Option<f64>,
    pub duration: Option<f64>,
    pub preset: Option<&'static str>,
    pub yaw: Option<f32>,
    pub pitch: Option<f32>,
    pub dist: Option<f32>,
    pub exposure: Option<f32>,
    pub interlace: Option<bool>,
    pub subpixel: Option<bool>,
    pub bfi: Option<bool>,
    pub events: Vec<(When, Action)>,
}

fn preset_named(name: &str) -> Result<&'static Preset> {
    ALL_PRESETS
        .iter()
        .find(|p| p.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| {
            let names: Vec<&str> = ALL_PRESETS.iter().map(|p| p.name).collect();
            anyhow!("unknown preset `{name}` (have: {})", names.join(", "))
        })
}

fn parse_time(s: &str) -> Result<f32> {
    let s = s.trim_end_matches('s');
    let mut total = 0.0f32;
    for part in s.split(':') {
        let v: f32 = part
            .parse()
            .with_context(|| format!("bad time component `{part}` in `{s}`"))?;
        total = total * 60.0 + v;
    }
    Ok(total)
}

fn parse_size(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| anyhow!("bad size `{s}` (want WxH, e.g. 1280x960)"))?;
    Ok((w.trim().parse()?, h.trim().parse()?))
}

/// Split a script line into tokens, honouring quotes — `agent say "well hi there"`
/// is four tokens, and a `#` inside the quotes is text, not a comment.
fn tokenize(line: &str) -> Vec<String> {
    let (mut out, mut cur, mut quote, mut quoted) = (Vec::new(), String::new(), None, false);
    for c in line.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => cur.push(c),
            None => match c {
                '#' => break,
                '"' | '\'' => {
                    quote = Some(c);
                    quoted = true;
                }
                c if c.is_whitespace() => {
                    if quoted || !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                        quoted = false;
                    }
                }
                c => cur.push(c),
            },
        }
    }
    if quoted || !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// A point on the screen, in raster-normalised coordinates: `0.7,0.3` or `0.7 0.3`,
/// where (0,0) is the top-left of the picture and (1,1) the bottom-right.
fn parse_point(a: &mut Args) -> Result<[f32; 2]> {
    let first = a.next().ok_or_else(|| anyhow!("expected a point like `0.7,0.3`"))?;
    let (x, y) = match first.split_once(',') {
        Some((x, y)) if !y.is_empty() => (x.to_string(), y.to_string()),
        _ => (
            first.trim_end_matches(',').to_string(),
            a.next()
                .ok_or_else(|| anyhow!("`{first}` is only half a point — want `x,y`"))?
                .to_string(),
        ),
    };
    Ok([
        x.parse().with_context(|| format!("bad x in `{first}`"))?,
        y.parse().with_context(|| format!("bad y in `{first}`"))?,
    ])
}

fn parse_bool(s: &str) -> Result<bool> {
    match s.to_ascii_lowercase().as_str() {
        "on" | "true" | "yes" | "1" => Ok(true),
        "off" | "false" | "no" | "0" => Ok(false),
        _ => bail!("expected on/off, got `{s}`"),
    }
}

/// Pull `key=value` or `key value` out of a token stream, plus bare flags.
struct Args<'a> {
    toks: Vec<&'a str>,
    i: usize,
}

impl<'a> Args<'a> {
    fn new(toks: &[&'a str]) -> Self {
        Args { toks: toks.to_vec(), i: 0 }
    }
    fn next(&mut self) -> Option<&'a str> {
        let t = self.toks.get(self.i).copied();
        if t.is_some() {
            self.i += 1;
        }
        t
    }
    fn value_for(&mut self, key: &str, inline: Option<&'a str>) -> Result<f32> {
        match inline {
            Some(v) => v.parse().with_context(|| format!("bad value for `{key}`")),
            None => self
                .next()
                .ok_or_else(|| anyhow!("`{key}` needs a value"))?
                .parse()
                .with_context(|| format!("bad value for `{key}`")),
        }
    }
}

fn parse_action(toks: &[&str]) -> Result<Action> {
    let verb = toks[0].to_ascii_lowercase();
    let rest = &toks[1..];
    let mut a = Args::new(rest);
    match verb.as_str() {
        "preset" => {
            let name = rest.first().ok_or_else(|| anyhow!("`preset` needs a name"))?;
            Ok(Action::Preset(preset_named(name)?.name))
        }
        "camera" | "cam" | "move" => {
            let (mut yaw, mut pitch, mut dist) = (None, None, None);
            let (mut over, mut ease) = (0.0f32, true);
            while let Some(tok) = a.next() {
                let (key, inline) = match tok.split_once('=') {
                    Some((k, v)) => (k, Some(v)),
                    None => (tok, None),
                };
                match key.to_ascii_lowercase().as_str() {
                    "to" | "toward" => {}
                    "yaw" => yaw = Some(a.value_for("yaw", inline)?),
                    "pitch" => pitch = Some(a.value_for("pitch", inline)?),
                    "dist" | "distance" | "zoom" => dist = Some(a.value_for("dist", inline)?),
                    "over" | "in" => over = a.value_for("over", inline)?,
                    "ease" | "smooth" => ease = true,
                    "linear" => ease = false,
                    other => bail!("unknown camera option `{other}`"),
                }
            }
            Ok(Action::Camera { yaw, pitch, dist, over, ease })
        }
        "spin" | "orbit" => {
            let (mut turns, mut over, mut ease) = (1.0f32, 6.0f32, false);
            while let Some(tok) = a.next() {
                let (key, inline) = match tok.split_once('=') {
                    Some((k, v)) => (k, Some(v)),
                    None => (tok, None),
                };
                match key.to_ascii_lowercase().as_str() {
                    "over" | "in" => over = a.value_for("over", inline)?,
                    "turns" => turns = a.value_for("turns", inline)?,
                    "ease" | "smooth" => ease = true,
                    "linear" => ease = false,
                    v => turns = v.parse().with_context(|| format!("bad spin arg `{v}`"))?,
                }
            }
            Ok(Action::Spin { turns, over, ease })
        }
        "exposure" | "brightness" => {
            let (mut to, mut over, mut ease) = (None, 0.0f32, true);
            while let Some(tok) = a.next() {
                let (key, inline) = match tok.split_once('=') {
                    Some((k, v)) => (k, Some(v)),
                    None => (tok, None),
                };
                match key.to_ascii_lowercase().as_str() {
                    "to" => {}
                    "over" | "in" => over = a.value_for("over", inline)?,
                    "ease" | "smooth" => ease = true,
                    "linear" => ease = false,
                    v => to = Some(v.parse().with_context(|| format!("bad exposure `{v}`"))?),
                }
            }
            Ok(Action::Exposure {
                to: to.ok_or_else(|| anyhow!("`exposure` needs a value"))?,
                over,
                ease,
            })
        }
        "power" => Ok(Action::Power(parse_bool(
            rest.first().ok_or_else(|| anyhow!("`power` needs on/off"))?,
        )?)),
        "degauss" => Ok(Action::Degauss),
        "interlace" => Ok(Action::Interlace(parse_bool(rest.first().unwrap_or(&"on"))?)),
        "subpixel" => Ok(Action::Subpixel(parse_bool(rest.first().unwrap_or(&"on"))?)),
        "bfi" => Ok(Action::Bfi(parse_bool(rest.first().unwrap_or(&"on"))?)),
        "wait" => Ok(Action::Wait(parse_time(rest.first().unwrap_or(&"0"))?)),

        // --- the character (only meaningful with an `agent`) ---
        // agent show / hide
        // agent at 0.78,0.7            put him there
        // agent move to 0.25,0.6 over 1.5
        // agent point 0.3,0.55         turn and gesture at a spot on the picture
        // agent say "watch this"
        // agent play Congratulate
        // agent scale 1.4
        "agent" | "clippy" => {
            use crate::agent::Cmd;
            let verb = a
                .next()
                .ok_or_else(|| anyhow!("`agent` needs something to do (show/hide/at/move/point/say/play/scale)"))?
                .to_ascii_lowercase();
            let cmd = match verb.as_str() {
                "show" | "appear" => Cmd::Show,
                "hide" | "leave" => Cmd::Hide,
                "at" | "to" => Cmd::At(parse_point(&mut a)?),
                "move" | "walk" => {
                    // `move to 0.2,0.6 over 1.5` — `to` is optional sugar.
                    if a.toks.get(a.i).is_some_and(|t| t.eq_ignore_ascii_case("to")) {
                        a.next();
                    }
                    let to = parse_point(&mut a)?;
                    let mut over = 1.0;
                    if a.toks.get(a.i).is_some_and(|t| t.eq_ignore_ascii_case("over")) {
                        a.next();
                        over = a.value_for("over", None)?;
                    }
                    Cmd::MoveTo { to, over }
                }
                "point" | "gesture" => Cmd::Point(parse_point(&mut a)?),
                "say" | "speak" => Cmd::Say(
                    a.next()
                        .ok_or_else(|| anyhow!("`agent say` needs something to say, in quotes"))?
                        .to_string(),
                ),
                "play" | "animate" => Cmd::Play(
                    a.next()
                        .ok_or_else(|| anyhow!("`agent play` needs an animation name"))?
                        .to_string(),
                ),
                "scale" | "size" => Cmd::Scale(a.value_for("scale", None)?),
                other => bail!(
                    "unknown agent action `{other}` \
                     (show, hide, at, move, point, say, play, scale)"
                ),
            };
            Ok(Action::Agent(cmd))
        }

        // --- emulator input ---
        // press a right           momentary (PRESS_FRAMES)
        // press a for 18 frames   an exact span — how a TAS is written
        // press a for 0.4         …or in seconds
        // hold right / release right / tap b
        "stick" => {
            let p = parse_point(&mut a)?;
            Ok(Action::Stick { x: p[0].clamp(-1.0, 1.0), y: p[1].clamp(-1.0, 1.0) })
        }
        "center" if a.next().is_some_and(|v| v.eq_ignore_ascii_case("stick")) => {
            Ok(Action::Stick { x: 0.0, y: 0.0 })
        }
        "press" | "hold" | "release" | "tap" => {
            let mut buttons = 0u32;
            let mut dur = if verb == "tap" { Some(Dur::Frames(1)) } else { None };
            while let Some(tok) = a.next() {
                match tok.to_ascii_lowercase().as_str() {
                    "for" | "over" => {
                        let n: f32 = a
                            .next()
                            .ok_or_else(|| anyhow!("`for` needs a duration"))?
                            .parse()
                            .context("bad duration after `for`")?;
                        // A bare number is seconds; `frames`/`f` makes it exact.
                        let unit = a.toks.get(a.i).copied().unwrap_or("");
                        dur = Some(match unit.to_ascii_lowercase().as_str() {
                            "frames" | "frame" | "f" => {
                                a.next();
                                Dur::Frames(n.round().max(1.0) as u64)
                            }
                            "s" | "sec" | "secs" | "seconds" => {
                                a.next();
                                Dur::Seconds(n)
                            }
                            _ => Dur::Seconds(n),
                        });
                    }
                    "and" | "+" => {}
                    name => {
                        // `press a4` / `press 4f`-style typos land here as unknown buttons.
                        buttons |= crate::libretro::button_bit(name).ok_or_else(|| {
                            anyhow!("unknown button `{name}` (have: {})", crate::libretro::button_names())
                        })?;
                    }
                }
            }
            if buttons == 0 {
                bail!("`{verb}` needs at least one button (e.g. `{verb} a`)");
            }
            Ok(match verb.as_str() {
                "release" => Action::Release { buttons },
                "hold" => Action::Hold { buttons },
                _ => Action::Press { buttons, dur },
            })
        }
        other => bail!("unknown action `{other}`"),
    }
}

pub fn parse_script(text: &str) -> Result<Script> {
    let mut s = Script::default();
    for (lineno, raw) in text.lines().enumerate() {
        let owned = tokenize(raw);
        if owned.is_empty() {
            continue;
        }
        let toks: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        let ctx = || format!("script line {}: `{}`", lineno + 1, raw.trim());

        // `at <time> …` (wall clock) or `frame <n> …` (exact emulated frame).
        if toks[0].eq_ignore_ascii_case("at") || toks[0].eq_ignore_ascii_case("frame") {
            let by_frame = toks[0].eq_ignore_ascii_case("frame");
            if toks.len() < 3 {
                return Err(anyhow!("`{}` needs a time and an action", toks[0])).with_context(ctx);
            }
            let when = if by_frame {
                When::Frame(toks[1].parse().with_context(|| format!("bad frame number `{}`", toks[1])).with_context(ctx)?)
            } else {
                When::Seconds(parse_time(toks[1]).with_context(ctx)?)
            };
            let action = parse_action(&toks[2..]).with_context(ctx)?;
            s.events.push((when, action));
            continue;
        }

        // Header directives.
        let key = toks[0].to_ascii_lowercase();
        let arg = toks.get(1).copied().unwrap_or("");
        let r = (|| -> Result<()> {
            match key.as_str() {
                "source" | "input" => s.source = Some(arg.trim_matches(['"', '\'']).to_string()),
                "rom" | "game" => s.rom = Some(arg.trim_matches(['"', '\'']).to_string()),
                // `agent merlin` — a name to look up, or a path to an asset directory.
                "agent" | "character" => {
                    if arg.is_empty() {
                        bail!("`agent` needs a character (e.g. `agent merlin`)");
                    }
                    s.agent = Some(arg.to_string());
                }
                "core" => s.core = Some(arg.to_string()),
                "frames" => s.frames = Some(arg.parse()?),
                // `option key=value` (or `option key value`) — passed to the core.
                "option" | "core-option" => {
                    let rest = toks[1..].join(" ");
                    let (k, v) = rest
                        .split_once('=')
                        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                        .or_else(|| {
                            toks.get(2).map(|v| (arg.to_string(), v.to_string()))
                        })
                        .ok_or_else(|| anyhow!("`option` wants key=value"))?;
                    s.options.push((k, v));
                }
                "size" | "out-size" => s.size = Some(parse_size(arg)?),
                "fps" | "rate" => s.fps = Some(arg.parse()?),
                "ssaa" | "supersample" => s.ssaa = Some(arg.parse()?),
                "source-size" | "signal" => s.source_size = Some(parse_size(arg)?),
                // `lines N` = N-line signal, width derived from the source aspect.
                "lines" => s.source_size = Some((0, arg.parse()?)),
                "start" | "seek" => s.start = Some(parse_time(arg)? as f64),
                "duration" | "length" => s.duration = Some(parse_time(arg)? as f64),
                "preset" | "tube" => s.preset = Some(preset_named(arg)?.name),
                "camera" | "cam" => {
                    if let Action::Camera { yaw, pitch, dist, .. } = parse_action(&toks)? {
                        s.yaw = yaw.or(s.yaw);
                        s.pitch = pitch.or(s.pitch);
                        s.dist = dist.or(s.dist);
                    }
                }
                "exposure" | "brightness" => s.exposure = Some(arg.parse()?),
                "interlace" => s.interlace = Some(parse_bool(arg)?),
                "subpixel" => s.subpixel = Some(parse_bool(arg)?),
                "bfi" => s.bfi = Some(parse_bool(arg)?),
                other => bail!("unknown directive `{other}`"),
            }
            Ok(())
        })();
        r.with_context(ctx)?;
    }
    // Camera/tube events sort on wall clock. Frame-addressed events can't be placed
    // on that axis until the core's rate is known, so they sort again at compile time.
    s.events.sort_by(|a, b| a.0.seconds(60.0).total_cmp(&b.0.seconds(60.0)));
    Ok(s)
}

// ---------------------------------------------------------------------------
// Timeline — the script compiled into per-frame state
// ---------------------------------------------------------------------------

struct Seg {
    t0: f32,
    t1: f32,
    from: [f32; 3],
    to: [f32; 3],
    ease: bool,
}

impl Seg {
    fn eval(&self, t: f32) -> [f32; 3] {
        if t >= self.t1 {
            return self.to;
        }
        let mut k = ((t - self.t0) / (self.t1 - self.t0).max(1e-6)).clamp(0.0, 1.0);
        if self.ease {
            k = smoothstep01(k);
        }
        [
            self.from[0] + (self.to[0] - self.from[0]) * k,
            self.from[1] + (self.to[1] - self.from[1]) * k,
            self.from[2] + (self.to[2] - self.from[2]) * k,
        ]
    }
}

pub struct Timeline {
    cam: Vec<Seg>,
    cam0: [f32; 3],
    exp: Vec<Seg>,
    exp0: f32,
    presets: Vec<(f32, Preset)>,
    preset0: Preset,
    power: Vec<(f32, bool)>,
    degauss: Vec<f32>,
    interlace: Vec<(f32, bool)>,
    interlace0: bool,
    subpixel: Vec<(f32, bool)>,
    subpixel0: bool,
    bfi: Vec<(f32, bool)>,
    bfi0: bool,
    /// Last moment anything in the script happens (used only for reporting).
    pub end: f32,
}

/// Everything the renderer needs to draw one frame.
pub struct Shot {
    pub orbit: Orbit,
    pub preset: Preset,
    pub exposure: f32,
    pub pwr: [f32; 4],
    pub interlace: f32,
    pub subpixel: bool,
    pub bfi: bool,
}

fn step_at<T: Copy>(track: &[(f32, T)], t: f32, initial: T) -> T {
    track
        .iter()
        .rev()
        .find(|(et, _)| *et <= t)
        .map(|(_, v)| *v)
        .unwrap_or(initial)
}

impl Timeline {
    /// `fps` is the emulated/source frame rate, used to place `frame N` events on
    /// the wall clock the camera timeline runs on.
    pub fn compile(s: &Script, default_preset: Preset, fps: f64) -> Timeline {
        let cam0 = [
            s.yaw.unwrap_or(0.82),
            s.pitch.unwrap_or(0.34),
            s.dist.unwrap_or(3.7),
        ];
        let preset0 = s
            .preset
            .and_then(|n| preset_named(n).ok().copied())
            .unwrap_or(default_preset);
        let exp0 = s.exposure.unwrap_or(1.0);

        let mut tl = Timeline {
            cam: Vec::new(),
            cam0,
            exp: Vec::new(),
            exp0,
            presets: Vec::new(),
            preset0,
            power: Vec::new(),
            degauss: Vec::new(),
            interlace: Vec::new(),
            interlace0: s.interlace.unwrap_or(false),
            subpixel: Vec::new(),
            subpixel0: s.subpixel.unwrap_or(false),
            bfi: Vec::new(),
            bfi0: s.bfi.unwrap_or(false),
            end: 0.0,
        };

        // Camera and exposure keys are relative to whatever came before, so walk the
        // event list in order carrying a cursor.
        let mut cam = cam0;
        let mut exp = exp0;
        for (when, action) in &s.events {
            let t = when.seconds(fps);
            tl.end = tl.end.max(t);
            match action {
                Action::Preset(name) => {
                    if let Ok(p) = preset_named(name) {
                        tl.presets.push((t, *p));
                    }
                }
                Action::Camera { yaw, pitch, dist, over, ease } => {
                    let to = [
                        yaw.unwrap_or(cam[0]),
                        pitch.unwrap_or(cam[1]),
                        dist.unwrap_or(cam[2]),
                    ];
                    tl.cam.push(Seg { t0: t, t1: t + over, from: cam, to, ease: *ease });
                    cam = to;
                    tl.end = tl.end.max(t + over);
                }
                Action::Spin { turns, over, ease } => {
                    let to = [cam[0] + turns * std::f32::consts::TAU, cam[1], cam[2]];
                    tl.cam.push(Seg { t0: t, t1: t + over, from: cam, to, ease: *ease });
                    cam = to;
                    tl.end = tl.end.max(t + over);
                }
                Action::Exposure { to, over, ease } => {
                    tl.exp.push(Seg {
                        t0: t,
                        t1: t + over,
                        from: [exp, 0.0, 0.0],
                        to: [*to, 0.0, 0.0],
                        ease: *ease,
                    });
                    exp = *to;
                    tl.end = tl.end.max(t + over);
                }
                Action::Power(on) => {
                    tl.power.push((t, *on));
                    // The live path auto-degausses on power-on; match it.
                    if *on {
                        tl.degauss.push(t);
                    }
                    tl.end = tl.end.max(t + if *on { WARMUP_DUR } else { COLLAPSE_DUR });
                }
                Action::Degauss => {
                    tl.degauss.push(t);
                    tl.end = tl.end.max(t + DEGAUSS_DUR);
                }
                Action::Interlace(v) => tl.interlace.push((t, *v)),
                Action::Subpixel(v) => tl.subpixel.push((t, *v)),
                Action::Bfi(v) => tl.bfi.push((t, *v)),
                Action::Wait(d) => tl.end = tl.end.max(t + d),
                // Input actions drive the emulator, not the tube — see `InputTrack`.
                // They still count toward "when does this script end".
                Action::Press { dur, .. } => {
                    let d = dur.map(|d| d.frames(fps) as f32 / fps as f32).unwrap_or(0.0);
                    tl.end = tl.end.max(t + d);
                }
                Action::Hold { .. } | Action::Release { .. } | Action::Stick { .. } => {
                    tl.end = tl.end.max(t)
                }
                // The character is folded frame by frame, not sampled — see
                // `agent_events`. He still counts toward the end of the script.
                Action::Agent(_) => tl.end = tl.end.max(t),
            }
        }
        tl
    }

    pub fn eval(&self, t: f32) -> Shot {
        let cam = self
            .cam
            .iter()
            .rev()
            .find(|s| s.t0 <= t)
            .map(|s| s.eval(t))
            .unwrap_or(self.cam0);
        let exposure = self
            .exp
            .iter()
            .rev()
            .find(|s| s.t0 <= t)
            .map(|s| s.eval(t)[0])
            .unwrap_or(self.exp0);

        // Power: a `power on` at t0 ramps the raster open over WARMUP_DUR; `power off`
        // collapses it over COLLAPSE_DUR. No events at all = a warmed-up tube.
        let (warmup, collapse) = match self.power.iter().rev().find(|(et, _)| *et <= t) {
            Some((t0, true)) => (smoothstep01((t - t0) / WARMUP_DUR), 0.0),
            Some((t0, false)) => (1.0, smoothstep01((t - t0) / COLLAPSE_DUR)),
            None => (1.0, 0.0),
        };
        // Exponential AC burst, front-loaded — same envelope as the G key.
        let degauss = self
            .degauss
            .iter()
            .filter(|t0| t >= **t0 && t - **t0 < DEGAUSS_DUR)
            .map(|t0| (-(t - t0) / DEGAUSS_TAU).exp())
            .fold(0.0f32, f32::max);

        Shot {
            orbit: Orbit { yaw: cam[0], pitch: cam[1], distance: cam[2] },
            preset: step_at(&self.presets, t, self.preset0),
            exposure,
            pwr: [warmup, collapse, degauss, 0.0],
            interlace: if step_at(&self.interlace, t, self.interlace0) { 1.0 } else { 0.0 },
            subpixel: step_at(&self.subpixel, t, self.subpixel0),
            bfi: step_at(&self.bfi, t, self.bfi0),
        }
    }
}

/// The character's instructions, on the same wall clock the camera runs on. Pulled
/// out of the event list rather than compiled into [`Timeline`] because animation
/// state is a fold, not a sample: a frame's weighted branches mean where the
/// character *is* depends on where he's been.
pub fn agent_events(script: &Script, fps: f64) -> Vec<(f32, crate::agent::Cmd)> {
    script
        .events
        .iter()
        .filter_map(|(when, action)| match action {
            Action::Agent(cmd) => Some((when.seconds(fps), cmd.clone())),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Input track — the scripted run, compiled to a button mask per emulated frame
// ---------------------------------------------------------------------------
//
// This is the tool-assisted part. Every entry lands on an exact frame, so the run
// is deterministic: frame 1237 gets the same mask on every render, no matter how
// fast the machine is or how long the encoder stalls.

/// A press with no explicit span lasts this many frames — long enough for any game
/// polling once per frame to see it, short enough not to smear into the next input.
const DEFAULT_PRESS_FRAMES: u64 = 4;

#[derive(Clone, Copy, Debug)]
enum InputOp {
    Down(u32),
    Up(u32),
    /// Held for a fixed number of frames from this one.
    Timed(u32, u64),
    Stick([i16; 2]),
}

pub struct InputTrack {
    events: Vec<(u64, InputOp)>,
    cursor: usize,
    held: u32,
    analog: [i16; 2],
    timed: Vec<(u64, u32)>, // (end frame, exclusive) → mask
    /// Last frame any input happens — used to pick a run length when the script
    /// doesn't give one.
    pub last_frame: u64,
}

impl InputTrack {
    pub fn compile(script: &Script, fps: f64) -> InputTrack {
        let mut events: Vec<(u64, InputOp)> = Vec::new();
        let mut last_frame = 0;
        for (when, action) in &script.events {
            let f = when.frame(fps);
            let op = match action {
                Action::Press { buttons, dur } => {
                    let n = dur.map(|d| d.frames(fps)).unwrap_or(DEFAULT_PRESS_FRAMES);
                    last_frame = last_frame.max(f + n);
                    InputOp::Timed(*buttons, n)
                }
                Action::Hold { buttons } => {
                    last_frame = last_frame.max(f);
                    InputOp::Down(*buttons)
                }
                Action::Release { buttons } => {
                    last_frame = last_frame.max(f);
                    InputOp::Up(*buttons)
                }
                Action::Stick { x, y } => {
                    last_frame = last_frame.max(f);
                    InputOp::Stick([
                        (x * i16::MAX as f32) as i16,
                        (y * i16::MAX as f32) as i16,
                    ])
                }
                _ => continue,
            };
            events.push((f, op));
        }
        events.sort_by_key(|(f, _)| *f);
        InputTrack { events, cursor: 0, held: 0, analog: [0, 0], timed: Vec::new(), last_frame }
    }

    /// The button mask for `frame`. Must be called with non-decreasing frames — the
    /// render loop walks forward, so this stays O(1) amortized.
    pub fn advance(&mut self, frame: u64) -> u32 {
        self.advance_state(frame).0
    }

    pub fn advance_state(&mut self, frame: u64) -> (u32, [i16; 2]) {
        while self.cursor < self.events.len() && self.events[self.cursor].0 <= frame {
            let (at, op) = self.events[self.cursor];
            match op {
                InputOp::Down(m) => self.held |= m,
                InputOp::Up(m) => self.held &= !m,
                InputOp::Timed(m, n) => self.timed.push((at + n, m)),
                InputOp::Stick(v) => self.analog = v,
            }
            self.cursor += 1;
        }
        self.timed.retain(|(end, _)| *end > frame);
        (self.timed.iter().fold(self.held, |acc, (_, m)| acc | m), self.analog)
    }
}

// ---------------------------------------------------------------------------
// Source acquisition: URL → file, ROM+replay → file
// ---------------------------------------------------------------------------

fn tool(name: &str) -> Result<()> {
    which(name).map(|_| ())
}

fn which(name: &str) -> Result<PathBuf> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {name}"))
        .output()
        .with_context(|| format!("looking for {name}"))?;
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if !out.status.success() || path.is_empty() {
        bail!("`{name}` not found on PATH — it is required for this input/output");
    }
    Ok(PathBuf::from(path))
}

/// Staging area for downloads and emulator recordings — `.crtulum/` beside the
/// output file, overridable with `CRTULUM_TMP`.
///
/// Not `/tmp` and not `~/.cache`: these files are big (a fetched video, a TAS
/// recording) and worth keeping between runs, and the helpers that produce and
/// consume them are commonly sandboxed (firejail's stock ffmpeg/yt-dlp/retroarch
/// profiles use `private-tmp` and blacklist `~/.cache`, so a file staged there is
/// invisible to the next tool in the chain). Anywhere the user can write the
/// output, every tool in the pipeline can reach.
fn work_dir(output: &Path) -> Result<PathBuf> {
    let dir = match std::env::var_os("CRTULUM_TMP") {
        Some(d) => PathBuf::from(d),
        None => {
            let parent = output.parent().filter(|p| !p.as_os_str().is_empty());
            parent.unwrap_or(Path::new(".")).join(".crtulum")
        }
    };
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating the work directory {dir:?}"))?;
    Ok(dir)
}

fn hash_of(s: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Fetch a URL with yt-dlp into the work dir and return the downloaded file.
///
/// Keyed by a hash of the URL, so re-running the same export (tweaking the script,
/// say) reuses the download instead of hitting the network again.
fn fetch_url(url: &str, work: &Path, dry: bool) -> Result<PathBuf> {
    tool("yt-dlp")?;
    let dir = work.join(format!("dl-{}", hash_of(url)));
    std::fs::create_dir_all(&dir)?;
    if let Some(existing) = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.file_stem().and_then(|s| s.to_str()) == Some("source"))
    {
        eprintln!("[render] reusing cached download {}", existing.display());
        return Ok(existing);
    }
    let out = dir.join("source.%(ext)s");
    // Prefer an already-muxed mp4/mkv so ffmpeg gets a clean A/V file.
    let args = vec![
        "-f".to_string(),
        "bestvideo[height<=1080]+bestaudio/best".to_string(),
        "--merge-output-format".to_string(),
        "mkv".to_string(),
        "-o".to_string(),
        out.to_string_lossy().to_string(),
        url.to_string(),
    ];
    eprintln!("[render] yt-dlp {}", args.join(" "));
    if dry {
        return Ok(dir.join("source.mkv"));
    }
    let status = Command::new("yt-dlp").args(&args).status()?;
    if !status.success() {
        bail!("yt-dlp failed ({status})");
    }
    // yt-dlp picked the extension; find what actually landed.
    let f = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.file_stem().and_then(|s| s.to_str()) == Some("source"))
        .ok_or_else(|| anyhow!("yt-dlp reported success but no file appeared in {dir:?}"))?;
    Ok(f)
}

/// Guess a libretro core for a ROM extension, then find its .so.
fn core_for(rom: &Path, explicit: Option<&str>) -> Result<PathBuf> {
    let candidates: Vec<&str> = match explicit {
        Some(c) => vec![c],
        None => {
            let ext = rom
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            match ext.as_str() {
                "nes" | "fds" | "unf" => vec!["mesen", "nestopia", "fceumm", "quicknes"],
                "sfc" | "smc" => vec!["snes9x", "bsnes_mercury_balanced", "bsnes", "mesen-s"],
                "gb" | "gbc" => vec!["gambatte", "sameboy", "mgba"],
                "gba" => vec!["mgba", "vbam"],
                "n64" | "z64" | "v64" => vec!["mupen64plus_next", "parallel_n64"],
                "md" | "gen" | "smd" | "sms" | "gg" => vec!["genesis_plus_gx", "picodrive"],
                "pce" | "sgx" => vec!["mednafen_pce", "mednafen_pce_fast"],
                "cue" | "chd" | "pbp" => vec!["swanstation", "pcsx_rearmed", "beetle_psx"],
                "a26" | "bin" => vec!["stella"],
                "ws" | "wsc" => vec!["mednafen_wswan"],
                _ => vec![],
            }
        }
    };
    if candidates.is_empty() {
        bail!(
            "cannot guess a libretro core for {:?} — pass --core <name> (e.g. --core mesen)",
            rom
        );
    }
    let dirs = [
        std::env::var("RETROARCH_CORE_DIR").unwrap_or_default(),
        format!(
            "{}/.config/retroarch/cores",
            std::env::var("HOME").unwrap_or_default()
        ),
        "/usr/lib/libretro".into(),
        "/usr/local/lib/libretro".into(),
    ];
    for c in &candidates {
        // An explicit --core may already be a path.
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
    bail!(
        "no libretro core found for {:?} (tried {}). Install one with RetroArch's \
         core downloader, or pass --core /path/to/core_libretro.so",
        rom,
        candidates.join(", ")
    )
}

/// Run a ROM (optionally driving a TAS replay) through RetroArch, recording A/V.
///
/// RetroArch does the emulation and the input playback; we only consume its
/// recording. `--eof-exit` makes it quit when the replay ends, so a TAS run
/// produces a clip exactly as long as the run.
fn run_emulator(
    rom: &Path,
    movie: Option<&Path>,
    core: Option<&str>,
    work: &Path,
    dry: bool,
) -> Result<PathBuf> {
    tool("retroarch")?;
    let core = core_for(rom, core)?;
    // Recording a run costs real time (the emulator plays it at 1x), so cache it per
    // rom+replay pair; delete the file to force a fresh take.
    let key = hash_of(&format!("{rom:?}{movie:?}"));
    let out = work.join(format!("run-{key}.mkv"));
    if out.is_file() {
        eprintln!("[render] reusing cached recording {} (delete it to re-record)", out.display());
        return Ok(out);
    }
    let mut args: Vec<String> = vec![
        "-L".into(),
        core.to_string_lossy().into(),
        rom.to_string_lossy().into(),
        "-r".into(),
        out.to_string_lossy().into(),
    ];
    if let Some(m) = movie {
        args.push("-P".into());
        args.push(m.to_string_lossy().into());
        args.push("--eof-exit".into());
    }
    eprintln!("[render] retroarch {}", args.join(" "));
    if dry {
        return Ok(out);
    }
    eprintln!("[render] a RetroArch window will open and play the run in real time…");
    let status = Command::new("retroarch").args(&args).status()?;
    if !status.success() {
        bail!("retroarch exited with {status}");
    }
    if !out.is_file() {
        bail!("retroarch produced no recording at {out:?}");
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The emulator as a frame source
// ---------------------------------------------------------------------------

struct Emu {
    core: crate::libretro::Core,
    inputs: InputTrack,
    frame: u64,
    total: u64,
    /// Raw interleaved S16 stereo, muxed back in once the video is encoded.
    audio: Option<std::fs::File>,
    audio_frames: u64,
    last_mask: u32,
}

/// "a + right" for a button mask, or "—" for nothing held.
fn describe_mask(mask: u32) -> String {
    if mask == 0 {
        return "—".into();
    }
    crate::libretro::BUTTONS[..16]
        .iter()
        .filter(|(_, id)| mask & (1 << id) != 0)
        .map(|(n, _)| *n)
        .collect::<Vec<_>>()
        .join(" + ")
}

impl Emu {
    fn new(
        rom: &Path,
        core_name: Option<&str>,
        script: &Script,
        opts_duration: Option<f64>,
        work: &Path,
        want_audio: bool,
        options: &[(String, String)],
    ) -> Result<Emu> {
        let core_path = crate::libretro::find_core(Some(rom), core_name)?;
        eprintln!("[emu] core {}", core_path.display());
        for (k, v) in options {
            eprintln!("[emu] option {k} = {v}");
        }
        let system = crate::libretro::system_dir(work);
        eprintln!("[emu] system dir {}", system.display());
        let core = crate::libretro::Core::load(&core_path, Some(rom), &system, options)?;
        eprintln!(
            "[emu] {} · {}x{} @ {:.3} fps · {:.0} Hz audio",
            core.name, core.geometry.0, core.geometry.1, core.fps, core.sample_rate
        );

        let inputs = InputTrack::compile(script, core.fps);
        // How long to run: an explicit length wins, otherwise play out the scripted
        // run plus a couple of seconds so the last input is actually on screen.
        let total = match (script.frames, opts_duration.or(script.duration)) {
            (Some(f), _) => f,
            (None, Some(d)) => (d * core.fps).round() as u64,
            (None, None) => inputs.last_frame + (core.fps * 2.0) as u64,
        };
        let audio = if want_audio {
            Some(std::fs::File::create(work.join("emu-audio.raw")).context("creating the audio scratch file")?)
        } else {
            None
        };
        Ok(Emu { core, inputs, frame: 0, total, audio, audio_frames: 0, last_mask: 0 })
    }

    /// Run one emulated frame with the scripted input for it.
    fn next(&mut self) -> Result<Option<(Vec<u8>, u32, u32)>> {
        if self.frame >= self.total {
            if !self.core.saw_frame {
                bail!(
                    "the core never drew a single frame in {} frames. If it renders on the \
                     GPU it may want an API this host doesn't provide (OpenGL and Vulkan are \
                     supported, Direct3D is not); some cores also need a BIOS in the system \
                     directory. Its software renderer may work better, e.g.\n  \
                     --option parallel-n64-gfxplugin=angrylion   (N64)\n  \
                     --option swanstation_GPU_Renderer=Software  (PlayStation)",
                    self.total
                );
            }
            return Ok(None);
        }
        let (mask, analog) = self.inputs.advance_state(self.frame);
        let out = self.core.run_frame_with_analog(mask, analog)?;
        // `CRTULUM_DEBUG_INPUT=1` prints the run as it happens — every frame where the
        // held buttons change, which is what you want when a scripted run isn't doing
        // what you meant. (stderr, so it interleaves with the progress line.)
        if mask != self.last_mask && std::env::var_os("CRTULUM_DEBUG_INPUT").is_some() {
            eprintln!("\n[input] frame {:5}  {}", self.frame, describe_mask(mask));
        }
        self.last_mask = mask;
        if let Some(f) = &mut self.audio {
            let pcm = self.core.take_audio();
            if !pcm.is_empty() {
                self.audio_frames += (pcm.len() / 2) as u64;
                f.write_all(bytemuck::cast_slice(&pcm)).context("writing emulator audio")?;
            }
        }
        self.frame += 1;
        Ok(Some(out))
    }
}

// ---------------------------------------------------------------------------
// ffmpeg
// ---------------------------------------------------------------------------

struct Probe {
    width: u32,
    height: u32,
    fps: f64,
    duration: Option<f64>,
}

fn probe(path: &Path) -> Result<Probe> {
    tool("ffprobe")?;
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .context("running ffprobe")?;
    if !out.status.success() {
        bail!(
            "ffprobe failed on {path:?}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let get = |k: &str| -> Option<String> {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("{k}=")))
            .map(|v| v.trim().to_string())
    };
    let fps = get("r_frame_rate")
        .and_then(|r| {
            let (n, d) = r.split_once('/')?;
            let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
            (d > 0.0).then_some(n / d)
        })
        .unwrap_or(60.0);
    Ok(Probe {
        width: get("width").and_then(|v| v.parse().ok()).unwrap_or(0),
        height: get("height").and_then(|v| v.parse().ok()).unwrap_or(0),
        fps,
        duration: get("duration").and_then(|v| v.parse().ok()),
    })
}

/// A raw stereo PCM track waiting to be muxed onto the finished video.
struct PcmBed {
    path: PathBuf,
    rate: u32,
}

/// Does this file carry an audio stream? Only asked when a bed has to be mixed
/// *with* whatever the source brought, since naming a stream that isn't there is a
/// hard error inside `-filter_complex`.
fn has_audio_stream(path: &Path) -> bool {
    Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("audio"))
        .unwrap_or(false)
}

/// Mux one or more raw PCM beds onto `video`, mixing them with each other (and with
/// the video's own audio, if `embedded`). The video is stream-copied.
///
/// `normalize=0` matters: `amix` otherwise divides every input by the input count, so
/// adding a character who says one line halfway through would quietly halve the game's
/// volume for the whole run.
fn mux_beds(video: &Path, beds: &[PcmBed], embedded: bool, out: &Path) -> Result<()> {
    let mut args: Vec<String> = ["-hide_banner", "-v", "error", "-y", "-i"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    args.push(video.to_string_lossy().into());
    for bed in beds {
        args.extend([
            "-f".into(), "s16le".into(),
            "-ar".into(), bed.rate.to_string(),
            "-ac".into(), "2".into(),
            "-i".into(), bed.path.to_string_lossy().into(),
        ]);
    }

    let mut labels: Vec<String> = Vec::new();
    if embedded {
        labels.push("[0:a]".into());
    }
    labels.extend((1..=beds.len()).map(|i| format!("[{i}:a]")));

    args.extend(["-map".into(), "0:v:0".into()]);
    if labels.len() == 1 {
        args.extend(["-map".into(), labels[0].trim_matches(['[', ']']).to_string()]);
    } else {
        args.extend([
            "-filter_complex".into(),
            format!("{}amix=inputs={}:normalize=0[aout]", labels.concat(), labels.len()),
            "-map".into(),
            "[aout]".into(),
        ]);
    }
    args.extend([
        "-c:v".into(), "copy".into(),
        "-c:a".into(), "aac".into(),
        "-b:a".into(), "192k".into(),
        "-shortest".into(),
        out.to_string_lossy().into(),
    ]);

    let status = Command::new("ffmpeg").args(&args).status().context("muxing audio")?;
    if !status.success() {
        bail!("ffmpeg failed to mux the audio ({status})");
    }
    Ok(())
}

/// A directory of stills is fed to ffmpeg as a glob; probing it means probing the
/// first image, and the file count gives the progress bar something to aim at.
fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading frame directory {dir:?}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|s| s.to_str()).map(|s| s.to_ascii_lowercase()).as_deref(),
                Some("png" | "jpg" | "jpeg" | "bmp" | "tga")
            )
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no image frames (.png/.jpg/.bmp/.tga) found in {dir:?}");
    }
    Ok(files)
}

struct Decoder {
    child: Child,
    stdout: std::io::BufReader<std::process::ChildStdout>,
    frame_bytes: usize,
}

impl Decoder {
    /// One raw RGBA frame, or None at end of stream.
    fn next_frame(&mut self, buf: &mut [u8]) -> Result<bool> {
        debug_assert_eq!(buf.len(), self.frame_bytes);
        let mut read = 0;
        while read < buf.len() {
            match self.stdout.read(&mut buf[read..])? {
                0 => break,
                n => read += n,
            }
        }
        if read == 0 {
            return Ok(false);
        }
        if read < buf.len() {
            // A torn final frame means ffmpeg died mid-write; treat it as the end.
            eprintln!("[render] short read ({read}/{}) — ending", buf.len());
            return Ok(false);
        }
        Ok(true)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------------------
// GPU resolve pass (SSAA box downsample, in linear light, on the GPU)
// ---------------------------------------------------------------------------
//
// `--shot`/`--clip` do this on the CPU, which is fine for a handful of stills but
// costs ~100M ops per frame at 1280x960x3 — minutes per second of video. Same math,
// done in a fragment shader: sampling an `Rgba8UnormSrgb` texture decodes to linear,
// and writing to one re-encodes, so the average is a linear-light average.
struct Resolve {
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
}

impl Resolve {
    fn new(device: &wgpu::Device, src: &wgpu::TextureView, format: wgpu::TextureFormat, ss: u32) -> Resolve {
        let src_code = format!(
            r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {{
    let ss = {ss}u;
    let base = vec2<i32>(pos.xy) * i32(ss);
    var acc = vec4<f32>(0.0);
    for (var y: u32 = 0u; y < ss; y = y + 1u) {{
        for (var x: u32 = 0u; x < ss; x = x + 1u) {{
            acc = acc + textureLoad(src, base + vec2<i32>(i32(x), i32(y)), 0);
        }}
    }}
    return acc / f32(ss * ss);
}}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resolve"),
            source: wgpu::ShaderSource::Wgsl(src_code.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resolve-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resolve-bind"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resolve-pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("resolve-pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                targets: &[Some(format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        Resolve { pipeline, bind }
    }

    fn run(&self, enc: &mut wgpu::CommandEncoder, dst: &wgpu::TextureView) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("resolve-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

// ---------------------------------------------------------------------------
// The export itself
// ---------------------------------------------------------------------------

fn set_preset_res(device: &wgpu::Device, res: &mut Resources, preset: &Preset) {
    // Curvature is baked into the mesh, so a preset swap rebuilds the geometry —
    // same as `State::set_preset` on the live path.
    let (verts, indices) = build_mesh(preset.bulge, preset.curv_x, preset.curv_y, preset.cabinet);
    res.vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vbuf"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    res.ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ibuf"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    res.index_count = indices.len() as u32;
}

fn even(v: u32) -> u32 {
    v.max(2) & !1
}

pub fn render(opts: Opts) -> Result<()> {
    tool("ffmpeg")?;
    let work = work_dir(&opts.output)?;

    // --- 1. Resolve the source -------------------------------------------------
    // A scripted ROM run needs no media file at all: the core *is* the source, and
    // its frames go straight into the tube. Everything else resolves to a file (or
    // a directory of stills) that ffmpeg decodes for us.
    let want_audio_in = opts.audio;
    let mut emu: Option<Emu> = None;
    let (media, glob): (PathBuf, bool) = match &opts.input {
        Input::Media(p) => {
            if p.is_dir() {
                (p.clone(), true)
            } else {
                if !p.exists() {
                    bail!("input {p:?} does not exist");
                }
                (p.clone(), false)
            }
        }
        Input::Url(u) => (fetch_url(u, &work, opts.dry_run)?, false),
        Input::Replay { rom, movie, core } => (
            run_emulator(rom, Some(movie), core.as_deref(), &work, opts.dry_run)?,
            false,
        ),
        Input::Rom { rom, core } => {
            if !rom.exists() {
                bail!("ROM {rom:?} does not exist");
            }
            if !opts.dry_run {
                emu = Some(Emu::new(
                    rom,
                    core.as_deref(),
                    &opts.script,
                    opts.duration,
                    &work,
                    want_audio_in,
                    &opts.core_options,
                )?);
            }
            (rom.clone(), false)
        }
    };

    // --- 2. Probe it -----------------------------------------------------------
    let stills = if glob { Some(list_images(&media)?) } else { None };
    let probe_target = match &stills {
        Some(files) => files[0].clone(),
        None => media.clone(),
    };
    let src = match &emu {
        // The core already told us everything the probe would have.
        Some(e) => Probe {
            width: e.core.geometry.0,
            height: e.core.geometry.1,
            fps: e.core.fps,
            duration: Some(e.total as f64 / e.core.fps),
        },
        None if opts.dry_run && !probe_target.exists() => {
            Probe { width: 640, height: 480, fps: 60.0, duration: None }
        }
        None if matches!(opts.input, Input::Rom { .. }) => {
            Probe { width: 256, height: 240, fps: 60.0, duration: None } // dry run
        }
        None => probe(&probe_target)?,
    };
    // A still has no meaningful frame rate of its own, so a stills directory plays at
    // 30 fps unless told otherwise. An emulator's rate is not ours to choose — a core
    // runs at 60.0988 (NES) or 59.7275 (GB) and resampling that would judder.
    let fps = match (&emu, opts.fps > 0.0, glob) {
        (Some(e), asked, _) => {
            if asked && (opts.fps - e.core.fps).abs() > 0.01 {
                eprintln!(
                    "[render] ignoring --fps {}: the core runs at {:.4} fps and that's what gets encoded",
                    opts.fps, e.core.fps
                );
            }
            e.core.fps
        }
        (None, true, _) => opts.fps,
        (None, false, true) => 30.0,
        (None, false, false) => src.fps.max(1.0),
    };

    // The "signal" resolution fed to the tube. A CRT takes a 240p/480i signal, not a
    // 1080p one — feeding native 1080 makes the beam reconstruction sample ~1080 rows
    // and the scanline structure vanishes. So downscale tall sources to 480 lines by
    // default; `--source-size`/`lines` overrides. A core's native output is already
    // exactly the signal the tube wants, so it's passed through untouched.
    let (sw, sh) = match (&emu, opts.source_size) {
        (Some(e), req) => {
            if req.is_some() {
                eprintln!("[render] ignoring --lines/--source-size: the core's native output is the signal");
            }
            e.core.geometry
        }
        (None, Some((0, h))) => {
            let aspect = if src.height > 0 { src.width as f32 / src.height as f32 } else { 4.0 / 3.0 };
            (even((h as f32 * aspect).round() as u32), even(h))
        }
        (None, Some((w, h))) => (even(w), even(h)),
        (None, None) if src.height > 576 => {
            let aspect = src.width as f32 / src.height.max(1) as f32;
            (even((480.0 * aspect).round() as u32), 480)
        }
        (None, None) => (even(src.width.max(2)), even(src.height.max(2))),
    };

    let (ow, oh) = (even(opts.size.0), even(opts.size.1));
    let ss = opts.ssaa.clamp(1, 4);
    let (rw, rh) = (ow * ss, oh * ss);

    let timeline = Timeline::compile(
        &opts.script,
        opts.script
            .preset
            .and_then(|n| preset_named(n).ok().copied())
            .unwrap_or(TRINITRON),
        fps,
    );
    // The character, if the script (or `--agent`) asked for one. Loading him here —
    // after `fps` is settled but before the first frame — is also when his speech is
    // synthesised, since the balloon fills at the rate the voice reads it.
    let mut agent = match opts.agent.as_deref().or(opts.script.agent.as_deref()) {
        Some(name) if !opts.dry_run => {
            let path = crate::agent::resolve(name)?;
            let ch = crate::agent::Character::load_path(&path, opts.audio)?;
            eprintln!(
                "[agent] {} · {}x{} · {} animations",
                ch.name,
                ch.frame_w,
                ch.frame_h,
                ch.animation_names().len()
            );
            Some(crate::agent::Agent::new(
                ch,
                agent_events(&opts.script, fps),
                opts.audio,
            ))
        }
        _ => None,
    };

    // A run whose length was left to us stops when the last input has played out —
    // which would cut the character off mid-sentence. Give him room to finish.
    if let (Some(e), Some(a)) = (&mut emu, &agent) {
        if opts.script.frames.is_none() && opts.duration.or(opts.script.duration).is_none() {
            let needed = ((a.end() + 0.5) as f64 * fps).round() as u64;
            if needed > e.total {
                eprintln!("[render] extending the run to {needed} frames so the character finishes");
                e.total = needed;
            }
        }
    }

    let est_frames = match (&emu, &stills) {
        (Some(e), _) => Some(e.total),
        (None, Some(files)) => Some(files.len() as u64),
        (None, None) => opts
            .duration
            .or_else(|| src.duration.map(|d| (d - opts.start).max(0.0)))
            .map(|d| (d * fps).round() as u64),
    };

    // --- 3. Build the ffmpeg command lines -------------------------------------
    let mut dec_args: Vec<String> = vec!["-hide_banner".into(), "-v".into(), "error".into(), "-nostdin".into()];
    if opts.start > 0.0 && !glob {
        dec_args.extend(["-ss".into(), format!("{:.4}", opts.start)]);
    }
    if glob {
        dec_args.extend([
            "-framerate".into(),
            format!("{fps}"),
            "-pattern_type".into(),
            "glob".into(),
            "-i".into(),
            // Glob on the extension the frames actually use (f_0001.png, shot.jpg, …).
            format!(
                "{}/*.{}",
                media.display(),
                probe_target.extension().and_then(|e| e.to_str()).unwrap_or("png")
            ),
        ]);
    } else {
        dec_args.extend(["-i".into(), media.to_string_lossy().into()]);
    }
    if let Some(d) = opts.duration {
        dec_args.extend(["-t".into(), format!("{d:.4}")]);
    }
    dec_args.extend([
        "-vf".into(),
        format!("fps={fps},scale={sw}:{sh}:flags=bicubic"),
        "-f".into(),
        "rawvideo".into(),
        "-pix_fmt".into(),
        "rgba".into(),
        "pipe:1".into(),
    ]);

    let mut enc_args: Vec<String> = vec![
        "-hide_banner".into(),
        "-v".into(),
        "error".into(),
        "-y".into(),
        "-f".into(),
        "rawvideo".into(),
        "-pixel_format".into(),
        "rgba".into(),
        "-video_size".into(),
        format!("{ow}x{oh}"),
        "-framerate".into(),
        format!("{fps}"),
        "-i".into(),
        "pipe:0".into(),
    ];
    // The emulator hands us PCM alongside the frames rather than a file ffmpeg can
    // open, so its audio is muxed in a second (stream-copy) pass at the end.
    let emu_audio = emu.as_ref().map(|e| (e.core.sample_rate, e.audio.is_some())).filter(|(_, on)| *on);
    let want_audio = opts.audio && !glob && emu.is_none();
    // The character's voice and sound effects land on their own bed, mixed in at the
    // end alongside whatever the picture came with.
    let agent_audio = agent.is_some() && opts.audio;
    let video_target = if emu_audio.is_some() || agent_audio {
        work.join(format!(
            "video-only.{}",
            opts.output.extension().and_then(|e| e.to_str()).unwrap_or("mkv")
        ))
    } else {
        opts.output.clone()
    };
    if want_audio {
        if opts.start > 0.0 {
            enc_args.extend(["-ss".into(), format!("{:.4}", opts.start)]);
        }
        if let Some(d) = opts.duration {
            enc_args.extend(["-t".into(), format!("{d:.4}")]);
        }
        enc_args.extend([
            "-i".into(),
            media.to_string_lossy().into(),
            "-map".into(),
            "0:v:0".into(),
            "-map".into(),
            "1:a:0?".into(), // '?' = fine if the source has no audio
            "-c:a".into(),
            "aac".into(),
            "-b:a".into(),
            "192k".into(),
            "-shortest".into(),
        ]);
    } else {
        enc_args.push("-an".into());
    }
    let crf = opts.crf.to_string();
    match opts.codec.as_str() {
        "x265" | "hevc" => enc_args.extend([
            "-c:v".into(), "libx265".into(), "-crf".into(), crf,
            "-preset".into(), "medium".into(), "-pix_fmt".into(), "yuv420p".into(),
        ]),
        "vp9" => enc_args.extend([
            "-c:v".into(), "libvpx-vp9".into(), "-crf".into(), crf,
            "-b:v".into(), "0".into(), "-pix_fmt".into(), "yuv420p".into(),
        ]),
        "ffv1" => enc_args.extend([
            "-c:v".into(), "ffv1".into(), "-level".into(), "3".into(),
            "-pix_fmt".into(), "yuv444p".into(),
        ]),
        _ => enc_args.extend([
            "-c:v".into(), "libx264".into(), "-crf".into(), crf,
            "-preset".into(), "slow".into(), "-pix_fmt".into(), "yuv420p".into(),
        ]),
    }
    enc_args.extend([
        "-color_primaries".into(), "bt709".into(),
        "-color_trc".into(), "bt709".into(),
        "-colorspace".into(), "bt709".into(),
    ]);
    if opts.output.extension().and_then(|s| s.to_str()) == Some("mp4") {
        enc_args.extend(["-movflags".into(), "+faststart".into()]);
    }
    enc_args.push(video_target.to_string_lossy().into());

    match &emu {
        Some(e) => eprintln!(
            "[render] source  {} · {} frames ({:.1}s)",
            media.display(),
            e.total,
            e.total as f64 / fps
        ),
        None => eprintln!("[render] source  {} → signal {sw}x{sh}", probe_target.display()),
    }
    eprintln!("[render] output  {} @ {ow}x{oh} {fps} fps (ssaa {ss}x)", opts.output.display());
    eprintln!("[render] tube    {} · {} script event(s), {:.1}s of choreography",
        timeline.preset0.name, opts.script.events.len(), timeline.end);
    if opts.dry_run {
        if emu.is_none() {
            println!("ffmpeg {}", dec_args.join(" "));
        } else {
            println!("(no decoder: frames come straight from the libretro core)");
        }
        println!("ffmpeg {}", enc_args.join(" "));
        return Ok(());
    }

    // --- 4. GPU setup ----------------------------------------------------------
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| anyhow!("no GPU adapter"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("render-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut res = build_resources(&device, &queue, format, timeline.preset0);
    let mut cur_preset = timeline.preset0;

    let make_target = |w: u32, h: u32, label: &str, extra: wgpu::TextureUsages| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | extra,
            view_formats: &[],
        })
    };
    let hires = make_target(rw, rh, "render-hires", wgpu::TextureUsages::TEXTURE_BINDING);
    let hires_view = hires.create_view(&wgpu::TextureViewDescriptor::default());
    let out_tex = make_target(ow, oh, "render-out", wgpu::TextureUsages::COPY_SRC);
    let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = create_depth(&device, rw, rh);
    let resolve = Resolve::new(&device, &hires_view, format, ss);

    let padded = ((ow * 4 + 255) / 256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render-readback"),
        size: (padded * oh) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // --- 5. Spawn ffmpeg -------------------------------------------------------
    let mut decoder = if emu.is_some() {
        None
    } else {
        let mut dec_child = Command::new("ffmpeg")
            .args(&dec_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("spawning the ffmpeg decoder")?;
        Some(Decoder {
            stdout: std::io::BufReader::with_capacity(
                (sw * sh * 4) as usize,
                dec_child.stdout.take().expect("decoder stdout"),
            ),
            child: dec_child,
            frame_bytes: (sw * sh * 4) as usize,
        })
    };

    let mut enc_child = Command::new("ffmpeg")
        .args(&enc_args)
        .stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("spawning the ffmpeg encoder")?;
    let mut enc_stdin = BufWriter::with_capacity(
        (ow * oh * 4) as usize,
        enc_child.stdin.take().expect("encoder stdin"),
    );

    // --- 6. The frame loop -----------------------------------------------------
    // The tube runs at 60 fields/sec no matter the output rate, so each output frame
    // may consume several accumulation steps: a 30 fps export scans every frame twice,
    // which is what a real set does — and what makes 480i twitter at the right rate.
    let fields_per_frame = ((60.0 / fps).round() as u32).max(1);
    let field_dt = 1.0 / 60.0f32;

    let mut src_buf = vec![0u8; (sw * sh * 4) as usize];
    let mut row = vec![0u8; (ow * 4) as usize];
    let mut frame_idx: u64 = 0;
    let mut field_parity = 0.0f32;
    let mut src_dim;
    let mut last_dim = (0u32, 0u32);
    let wall = std::time::Instant::now();

    let write_err = loop {
        // One frame from the source: either ffmpeg's raw pipe, or one `retro_run()`
        // with the button mask this exact frame is scripted to hold.
        match (&mut decoder, &mut emu) {
            (Some(d), _) => {
                if !d.next_frame(&mut src_buf)? {
                    break None;
                }
                src_dim = (sw, sh);
            }
            (None, Some(e)) => match e.next()? {
                Some((buf, w, h)) => {
                    // A core's declared geometry is only its startup mode; the real
                    // signal size arrives with the frames, and some machines change it
                    // mid-run (a PS1 switching video modes, a Mega Drive H32↔H40).
                    if (w, h) != last_dim {
                        eprintln!("\n[render] signal {w}x{h}");
                        last_dim = (w, h);
                    }
                    src_buf = buf;
                    src_dim = (w, h);
                }
                None => break None,
            },
            (None, None) => break None,
        }
        let t = opts.start as f32 + frame_idx as f32 / fps as f32;
        let shot = timeline.eval(t);

        if shot.preset.name != cur_preset.name {
            set_preset_res(&device, &mut res, &shot.preset);
            cur_preset = shot.preset;
            eprintln!("[render] {:>8.2}s · preset → {}", t, cur_preset.name);
        }

        // The character goes into the *signal*, not over the finished picture — so he
        // is made of phosphor like everything else: scanlines across him, the beam
        // blooming his highlights, his motion trailing red as the green and blue decay
        // out from under it. Compositing him after the tube would have made him a
        // sticker on a photograph.
        if let Some(a) = &mut agent {
            a.step(t, 1.0 / fps as f32);
            a.draw(&mut src_buf, src_dim.0, src_dim.1);
        }

        res.set_source(&device, &queue, src_dim.0, src_dim.1, format, &src_buf);

        // Black-frame insertion blanks the emitted phosphor on alternate frames; only
        // meaningful for a high-rate export, so it's off unless the script asks.
        let bfi_mul = if shot.bfi && frame_idx % 2 == 1 { 0.0 } else { 1.0 };

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-enc"),
        });
        for f in 0..fields_per_frame {
            let ft = t + f as f32 * field_dt;
            write_uniforms(
                &queue, &res, &shot.orbit, ow as f32 / oh as f32, ft, &cur_preset, ss as f32,
                false, field_dt, shot.pwr, shot.interlace, field_parity, shot.exposure,
                shot.subpixel, bfi_mul,
            );
            accum_step(&mut enc, &mut res);
            field_parity = 1.0 - field_parity;
        }
        draw_tube(&mut enc, &res, &hires_view, &depth_view);
        resolve.run(&mut enc, &out_view);
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(oh),
                },
            },
            wgpu::Extent3d { width: ow, height: oh, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(enc.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        device.poll(wgpu::Maintain::Wait);
        let mut err = None;
        {
            let data = slice.get_mapped_range();
            for y in 0..oh {
                let o = (y * padded) as usize;
                row.copy_from_slice(&data[o..o + (ow * 4) as usize]);
                if let Err(e) = enc_stdin.write_all(&row) {
                    err = Some(e);
                    break;
                }
            }
        }
        readback.unmap();
        // A dead encoder (bad codec args, unwritable path) shows up as EPIPE here.
        if let Some(e) = err {
            break Some(e);
        }

        frame_idx += 1;
        if frame_idx % 30 == 0 || Some(frame_idx) == est_frames {
            let secs = wall.elapsed().as_secs_f64();
            let rate = frame_idx as f64 / secs.max(1e-6);
            match est_frames {
                Some(total) => {
                    let eta = (total.saturating_sub(frame_idx)) as f64 / rate.max(1e-6);
                    eprint!(
                        "\r[render] {frame_idx}/{total} frames · {rate:.1} fps · eta {:.0}m{:02.0}s   ",
                        eta / 60.0,
                        eta % 60.0
                    );
                }
                None => eprint!("\r[render] {frame_idx} frames · {rate:.1} fps   "),
            }
            let _ = std::io::stderr().flush();
        }
    };
    eprintln!();

    // --- 7. Shut the pipes down cleanly ---------------------------------------
    drop(decoder);
    let emu_audio_frames = emu.as_ref().map(|e| e.audio_frames).unwrap_or(0);
    drop(emu); // unloads the core
    enc_stdin.flush().ok();
    drop(enc_stdin); // EOF → ffmpeg finalises the container
    let status = enc_child.wait().context("waiting for the ffmpeg encoder")?;
    if let Some(e) = write_err {
        bail!("writing frames to ffmpeg failed: {e}");
    }
    if !status.success() {
        bail!("ffmpeg encoder exited with {status}");
    }
    if frame_idx == 0 {
        bail!("no frames were decoded from {media:?} — is it a video?");
    }

    // Anything that arrived as raw PCM rather than as a file ffmpeg could open — the
    // core's output, the character's voice — is muxed on after the fact against the
    // stream-copied video, which is why those paths encoded to a scratch file first.
    let mut beds: Vec<PcmBed> = Vec::new();
    if let Some((rate, _)) = emu_audio {
        if emu_audio_frames == 0 {
            eprintln!("[render] core produced no audio — leaving the game silent");
        } else {
            beds.push(PcmBed { path: work.join("emu-audio.raw"), rate: rate as u32 });
        }
    }
    if let Some(a) = &agent {
        if !a.audio.is_empty() {
            let path = work.join("agent-audio.raw");
            a.audio.write_raw(&path)?;
            beds.push(PcmBed { path, rate: crate::agent::AUDIO_RATE });
        }
    }
    if video_target != opts.output {
        if beds.is_empty() {
            std::fs::rename(&video_target, &opts.output).context("moving the finished video")?;
        } else {
            // `want_audio` put the source's own track into the scratch file already; it
            // joins the mix rather than being replaced by it.
            let embedded = want_audio && has_audio_stream(&video_target);
            mux_beds(&video_target, &beds, embedded, &opts.output)?;
            std::fs::remove_file(&video_target).ok();
        }
        for bed in &beds {
            std::fs::remove_file(&bed.path).ok();
        }
    }
    println!(
        "wrote {} — {frame_idx} frames, {:.1}s @ {fps} fps ({ow}x{oh}), in {:.0}s",
        opts.output.display(),
        frame_idx as f64 / fps,
        wall.elapsed().as_secs_f64()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

pub const USAGE: &str = "\
crtulum --render [INPUT] [OUTPUT] [options]

  INPUT     a video file · a URL (yt-dlp) · a directory of stills
            (or --rom to run a ROM through a libretro core, scripted)
  OUTPUT    .mp4 · .mkv · .webm  (default crtulum_out.mp4)

Options:
  --script FILE      timeline script — camera moves, preset swaps, power, degauss,
                     and (with a rom) the run itself: press/hold/release/tap, placed
                     on an exact frame or a wall-clock time
  --preset NAME      starting tube preset
  --size WxH         output size            (default 1280x960)
  --fps N            output frame rate      (default: the source's)
  --ssaa N           supersampling 1-4      (default 3; use 1 for a fast preview)
  --source-size WxH  signal resolution fed to the tube
  --lines N          …or just its line count, width from the source aspect
  --start S          seek into the source
  --duration S       how much to render
  --codec NAME       x264 (default) · x265 · vp9 · ffv1
  --crf N            quality, lower is better (default 18)
  --no-audio         drop the source audio
  --rom FILE         ROM to run, driven by the script's input timeline
                     (frame-exact, headless, faster than real time)
  --movie FILE       instead, play a pre-authored replay/TAS through RetroArch
  --core NAME        libretro core (guessed from the ROM extension otherwise)
  --option K=V       libretro core option, repeatable (e.g. for a software renderer)
  --agent NAME|ACS   put a Microsoft Agent character on screen (original .acs files
                     or Clippy, Merlin, … clippy.js asset directories)
                     and drive him from the script: show/say/point/play/move
  --dry-run          print the plan and the ffmpeg commands, then stop

  crtulum --fetch-agent NAME   download a character's assets first
";

/// Parse the `--render …` tail of the command line.
pub fn opts_from_args(args: &[String], default_preset: Preset) -> Result<Opts> {
    let start_at = args
        .iter()
        .position(|a| a == "--render")
        .ok_or_else(|| anyhow!("--render not found"))?;
    let tail = &args[start_at + 1..];

    let mut positionals: Vec<String> = Vec::new();
    let (mut script_path, mut rom, mut movie, mut core) = (None, None, None, None);
    let (mut size, mut fps, mut ssaa, mut source_size) = (None, None, None, None);
    let (mut start, mut duration, mut crf, mut codec) = (None, None, None, None);
    let (mut audio, mut dry_run, mut preset_arg) = (true, false, None);
    let mut agent = None;
    let mut core_options: Vec<(String, String)> = Vec::new();

    let mut i = 0;
    while i < tail.len() {
        let a = tail[i].as_str();
        let mut val = || -> Result<String> {
            i += 1;
            tail.get(i)
                .cloned()
                .ok_or_else(|| anyhow!("{} needs a value", tail[i - 1]))
        };
        match a {
            "--script" => script_path = Some(val()?),
            "--rom" => rom = Some(PathBuf::from(val()?)),
            "--movie" | "--replay" | "--tas" => movie = Some(PathBuf::from(val()?)),
            "--core" => core = Some(val()?),
            "--size" => size = Some(parse_size(&val()?)?),
            "--fps" => fps = Some(val()?.parse()?),
            "--ssaa" => ssaa = Some(val()?.parse()?),
            "--source-size" => source_size = Some(parse_size(&val()?)?),
            "--lines" => source_size = Some((0, val()?.parse()?)),
            "--start" | "--ss" => start = Some(parse_time(&val()?)? as f64),
            "--duration" | "-t" => duration = Some(parse_time(&val()?)? as f64),
            "--codec" => codec = Some(val()?),
            "--crf" => crf = Some(val()?.parse()?),
            "--option" => {
                let kv = val()?;
                let (k, v) = kv
                    .split_once('=')
                    .ok_or_else(|| anyhow!("--option wants key=value, got `{kv}`"))?;
                core_options.push((k.to_string(), v.to_string()));
            }
            "--agent" | "--character" => agent = Some(val()?),
            "--no-audio" | "-an" => audio = false,
            "--dry-run" => dry_run = true,
            "-o" | "--out" => positionals.insert(0, val()?),
            // `--preset` is consumed by main() but appears in the same tail.
            "--preset" => preset_arg = Some(val()?),
            other if other.starts_with('-') => bail!("unknown --render option `{other}`\n\n{USAGE}"),
            other => positionals.push(other.to_string()),
        }
        i += 1;
    }
    let _ = preset_arg;

    let script = match &script_path {
        Some(p) => {
            let text = std::fs::read_to_string(p).with_context(|| format!("reading script {p}"))?;
            parse_script(&text).with_context(|| format!("in script {p}"))?
        }
        None => Script::default(),
    };

    // A script can name its own ROM/source, so `--render out.mp4 --script run.crts`
    // is a complete command line.
    let rom = rom.or_else(|| script.rom.as_deref().map(PathBuf::from));
    let core = core.or_else(|| script.core.clone());

    // With a ROM (or a script that names its own source), the first positional is the
    // output; otherwise it's the input.
    let have_source = rom.is_some() || script.source.is_some();
    let (input_arg, output_arg) = if have_source {
        (None, positionals.first().cloned())
    } else {
        (positionals.first().cloned(), positionals.get(1).cloned())
    };

    let input = if let Some(rom) = rom {
        // A replay file means RetroArch owns the run; otherwise we drive the core
        // ourselves from the script's input timeline.
        match movie {
            Some(movie) => Input::Replay { rom, movie, core },
            None => Input::Rom { rom, core },
        }
    } else {
        let s = input_arg
            .or_else(|| script.source.clone())
            .ok_or_else(|| anyhow!("no input given\n\n{USAGE}"))?;
        if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("ytsearch") {
            Input::Url(s)
        } else {
            Input::Media(PathBuf::from(s))
        }
    };

    let output = PathBuf::from(output_arg.unwrap_or_else(|| "crtulum_out.mp4".into()));
    let default_codec = match output.extension().and_then(|s| s.to_str()) {
        Some("webm") => "vp9",
        _ => "x264",
    };

    Ok(Opts {
        input,
        size: size.or(script.size).unwrap_or((1280, 960)),
        fps: fps.or(script.fps).unwrap_or(0.0),
        ssaa: ssaa.or(script.ssaa).unwrap_or(3),
        source_size: source_size.or(script.source_size),
        start: start.or(script.start).unwrap_or(0.0),
        duration: duration.or(script.duration),
        codec: codec.unwrap_or_else(|| default_codec.to_string()),
        crf: crf.unwrap_or(18),
        agent: agent.or_else(|| script.agent.clone()),
        audio,
        // Script options first, so a --option on the command line overrides one.
        core_options: script.options.iter().cloned().chain(core_options).collect(),
        output,
        script,
        dry_run,
    }
    .with_default_preset(default_preset))
}

impl Opts {
    /// `--preset` on the command line seeds the timeline unless the script sets one.
    fn with_default_preset(mut self, preset: Preset) -> Opts {
        if self.script.preset.is_none() {
            self.script.preset = Some(preset.name);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_times() {
        assert_eq!(parse_time("12").unwrap(), 12.0);
        assert_eq!(parse_time("1:03").unwrap(), 63.0);
        assert_eq!(parse_time("1:02:30").unwrap(), 3750.0);
        assert!((parse_time("0:01.5").unwrap() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn the_example_agent_script_compiles() {
        use crate::agent::Cmd;
        let text = std::fs::read_to_string("examples/agent.crts").expect("examples/agent.crts");
        let s = parse_script(&text).expect("parse");
        assert_eq!(s.agent.as_deref(), Some("merlin"));
        // The run and the commentary share one timeline: inputs on exact frames,
        // the character on the wall clock.
        let mut inputs = InputTrack::compile(&s, 60.0);
        assert_eq!(inputs.advance(512), crate::libretro::button_bit("a").unwrap() | crate::libretro::button_bit("right").unwrap());
        let ev = agent_events(&s, 60.0);
        assert_eq!(ev.len(), 8);
        assert!(matches!(ev[0].1, Cmd::At(_)));
        assert!(matches!(ev.last().unwrap().1, Cmd::Hide));
        // Every animation the script names by hand has to exist on the character it
        // names, or the example teaches a script that prints a warning and does nothing.
        if let Ok(dir) = crate::agent::resolve("merlin") {
            let c = crate::agent::Character::load(&dir, false).expect("load Merlin");
            for (_, cmd) in &ev {
                if let Cmd::Play(name) = cmd {
                    assert!(c.animation(name).is_some(), "Merlin has no `{name}`");
                }
            }
        }
    }

    #[test]
    fn parses_the_agent_track() {
        let s = parse_script(
            r#"
            agent merlin
            at 0.2  agent at 0.75,0.62
            at 0.4  agent show
            at 1.2  agent say "Watch this # frame-perfect jump."
            at 4.0  agent point 0.2,0.55
            at 5.5  agent move to 0.25,0.35 over 1.5
            at 7.5  agent play Congratulate
            at 9.0  agent hide
            "#,
        )
        .unwrap();
        assert_eq!(s.agent.as_deref(), Some("merlin"));
        let ev = agent_events(&s, 60.0);
        use crate::agent::Cmd;
        assert_eq!(ev.len(), 7);
        assert_eq!(ev[0], (0.2, Cmd::At([0.75, 0.62])));
        assert_eq!(ev[1], (0.4, Cmd::Show));
        // A `#` inside quotes is text, not the start of a comment.
        assert_eq!(ev[2], (1.2, Cmd::Say("Watch this # frame-perfect jump.".into())));
        assert_eq!(ev[3], (4.0, Cmd::Point([0.2, 0.55])));
        assert_eq!(ev[4], (5.5, Cmd::MoveTo { to: [0.25, 0.35], over: 1.5 }));
        assert_eq!(ev[5], (7.5, Cmd::Play("Congratulate".into())));
        assert_eq!(ev[6], (9.0, Cmd::Hide));
    }

    #[test]
    fn parses_a_script() {
        let s = parse_script(
            r#"
            # a scripted export
            source clip.mp4
            size 1280x960
            lines 240
            preset pvm
            camera yaw=0.4 pitch=0.2 dist=3.2

            at 0:00 power on
            at 0:03 camera to yaw=-0.5 dist=2.9 over 4 ease
            at 0:10 preset rca
            at 0:12 exposure to 1.4 over 2
            at 0:20 spin 1 over 6 linear
            at 0:30 power off
            "#,
        )
        .unwrap();
        assert_eq!(s.source.as_deref(), Some("clip.mp4"));
        assert_eq!(s.size, Some((1280, 960)));
        assert_eq!(s.source_size, Some((0, 240)));
        assert_eq!(s.preset, Some("pvm"));
        assert_eq!(s.yaw, Some(0.4));
        assert_eq!(s.events.len(), 6);
    }

    #[test]
    fn rejects_typos() {
        assert!(parse_script("preset trinitrron").is_err());
        assert!(parse_script("at 0:01 kaboom").is_err());
        assert!(parse_script("wobble 3").is_err());
    }

    #[test]
    fn timeline_interpolates_and_steps() {
        let s = parse_script(
            "preset pvm\n\
             camera yaw=0.0 pitch=0.0 dist=4.0\n\
             at 0 power on\n\
             at 2 camera to yaw=1.0 over 2 linear\n\
             at 5 preset rca\n\
             at 6 power off\n",
        )
        .unwrap();
        let tl = Timeline::compile(&s, TRINITRON, 60.0);

        // Camera holds, ramps linearly, then holds again.
        assert!((tl.eval(1.0).orbit.yaw - 0.0).abs() < 1e-5);
        assert!((tl.eval(3.0).orbit.yaw - 0.5).abs() < 1e-5);
        assert!((tl.eval(9.0).orbit.yaw - 1.0).abs() < 1e-5);

        // Presets are a step function.
        assert_eq!(tl.eval(4.9).preset.name, "pvm");
        assert_eq!(tl.eval(5.1).preset.name, "rca");

        // Power on at t=0 ramps the raster open, then it collapses after t=6.
        assert!(tl.eval(0.05).pwr[0] < 0.1);
        assert!(tl.eval(1.99).pwr[0] > 0.9);
        assert!(tl.eval(6.0).pwr[1] < 0.05);
        assert!(tl.eval(6.0 + COLLAPSE_DUR).pwr[1] >= 1.0);

        // Power-on auto-degausses (matching the live G key), and it decays fast.
        assert!(tl.eval(0.0).pwr[2] > 0.9);
        assert!(tl.eval(1.0).pwr[2] < 0.02);
    }

    #[test]
    fn input_track_is_frame_exact() {
        use crate::libretro::button_bit;
        let s = parse_script(
            "rom x.nes\n\
             frame 10 press a for 4 frames\n\
             frame 30 hold right\n\
             frame 50 release right\n\
             frame 60 tap b\n\
             at 2.0 press start\n",
        )
        .unwrap();
        let mut t = InputTrack::compile(&s, 60.0);
        let (a, right, b, start) = (
            button_bit("a").unwrap(),
            button_bit("right").unwrap(),
            button_bit("b").unwrap(),
            button_bit("start").unwrap(),
        );
        let mut seen = Vec::new();
        for f in 0..130 {
            seen.push(t.advance(f));
        }
        // `press a for 4 frames` at 10 → frames 10..13 inclusive, and nothing either side.
        assert_eq!(seen[9], 0);
        for f in 10..14 {
            assert_eq!(seen[f], a, "frame {f} should hold A alone");
        }
        assert_eq!(seen[14], 0, "the press must end after exactly 4 frames");
        // hold/release spans 30..49.
        assert_eq!(seen[29], 0);
        for f in 30..50 {
            assert_eq!(seen[f], right, "frame {f} should still hold right");
        }
        assert_eq!(seen[50], 0, "release takes effect on its own frame");
        // A tap is exactly one frame — the finest thing a TAS can express.
        assert_eq!(seen[59], 0);
        assert_eq!(seen[60], b, "the tap frame");
        assert_eq!(seen[61], 0, "a tap must not bleed into the next frame");
        // `at 2.0` on a 60 fps core is frame 120.
        assert_eq!(seen[119], 0);
        assert_eq!(seen[120], start);
    }

    /// The shipped example script must actually do what its comments claim.
    #[test]
    fn example_tas_script_compiles_to_the_documented_run() {
        use crate::libretro::button_bit;
        let text = std::fs::read_to_string("examples/tas.crts").expect("examples/tas.crts");
        let s = parse_script(&text).expect("parse");
        assert_eq!(s.rom.as_deref(), Some("examples/inputtest.nes"));
        assert_eq!(s.frames, Some(480));
        let mut t = InputTrack::compile(&s, 60.0);
        let (a, b, right, start) = (
            button_bit("a").unwrap(),
            button_bit("b").unwrap(),
            button_bit("right").unwrap(),
            button_bit("start").unwrap(),
        );
        let masks: Vec<u32> = (0..480).map(|f| t.advance(f)).collect();
        assert_eq!(masks[150], start, "frame 150: press start");
        assert_eq!(masks[153], start, "…for 4 frames");
        assert_eq!(masks[154], 0, "…and no longer");
        assert_eq!(masks[200], right, "right is held from 180");
        assert_eq!(masks[245], right | a, "the jump happens mid-hold");
        assert_eq!(masks[259], right | a, "…for its full 20 frames");
        assert_eq!(masks[260], right, "…then just the hold again");
        assert_eq!(masks[300], 0, "right released at 300");
        assert_eq!(masks[330], b, "frame 330: one-frame tap");
        assert_eq!(masks[331], 0, "a tap must not bleed");
    }

    #[test]
    fn spin_is_relative() {
        let s = parse_script("camera yaw=0.5\nat 0 spin 1 over 4 linear\n").unwrap();
        let tl = Timeline::compile(&s, TRINITRON, 60.0);
        assert!((tl.eval(4.0).orbit.yaw - (0.5 + std::f32::consts::TAU)).abs() < 1e-4);
        assert!((tl.eval(2.0).orbit.yaw - (0.5 + std::f32::consts::PI)).abs() < 1e-4);
    }
}
