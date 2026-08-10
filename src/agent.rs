//! A Microsoft Agent character living inside the tube.
//!
//! Clippy, Merlin and friends were 8-bit indexed sprite sheets driven by a list of
//! frames — an image (plus overlays), a duration in hundredths of a second, and
//! optional weighted branches back into the same animation. That is the whole model,
//! and it is reproduced here.
//!
//! Original Microsoft Agent v2 `.acs` files are parsed and rendered directly. The
//! clippy.js `map.png` + `agent.js` export remains supported as a fallback.
//!
//! ## Where the character is drawn
//!
//! Into the **source raster**, before the tube ever sees it — so the agent is made of
//! phosphor like everything else on the screen. He picks up the scanlines, the aperture
//! grille, the red trailing edge of [`crate::main`]'s per-channel decay, and bends with
//! the glass at the corners. Compositing him over the finished 3D render instead would
//! have made him a sticker on a photograph.
//!
//! Native ACS characters use their `ACSOVERLAYINFO` mouth shapes (0x00–0x06), driven
//! by the synthesised speech envelope. Exported clippy.js assets use their pre-baked
//! speaking animation because the overlay data is no longer present.

use anyhow::{anyhow, bail, Context, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::font8x8;

/// Where `--fetch-agent` pulls from.
const ASSET_BASE: &str = "https://raw.githubusercontent.com/clippyjs/clippy.js/master/agents";

/// The characters clippy.js extracted. Any directory with an `agent.js` + `map.png`
/// works; this list is only for `--fetch-agent` and the error message.
pub const KNOWN: &[&str] = &[
    "Clippy", "Merlin", "Rover", "Links", "Genie", "Genius", "Peedy", "Bonzi", "F1", "Rocky",
];

/// Agent art was drawn for a 640x480 desktop. Scaling by the raster's height against
/// that keeps the character the same fraction of the screen he always was — on a
/// 240-line signal Merlin lands at 64 px, not 128.
const DESIGN_HEIGHT: f32 = 480.0;

/// Audio is mixed at CD rate and handed to ffmpeg as raw stereo.
pub const AUDIO_RATE: u32 = 44_100;

// ---------------------------------------------------------------------------
// The character: sprite sheet + frame table
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Frame {
    /// Milliseconds this frame is held (the `.acs` field is hundredths of a second).
    pub duration_ms: f32,
    /// Sheet coordinates, drawn back to front — index 0 is the body, the rest overlays.
    pub images: Vec<(u32, u32)>,
    pub sound: Option<String>,
    /// Where to jump when the animation has been asked to stop. This is how an idle
    /// loop unwinds to a rest pose instead of cutting.
    pub exit_branch: Option<usize>,
    /// Weighted jumps, as percentages. Empty means "advance to the next frame".
    pub branches: Vec<(usize, u32)>,
    /// Native ACS animation/frame indices. Exported clippy.js frames leave this unset.
    native: Option<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct Animation {
    pub name: String,
    pub frames: Vec<Frame>,
}

pub struct Character {
    pub name: String,
    pub frame_w: u32,
    pub frame_h: u32,
    sheet_w: u32,
    sheet_h: u32,
    /// Straight-alpha RGBA8. The sheets are paletted with one fully transparent
    /// entry, so alpha here is binary until we resample.
    sheet: Vec<u8>,
    anims: HashMap<String, Animation>,
    /// Lowercased name → the key in `anims`, so scripts needn't match case.
    index: HashMap<String, String>,
    sounds: HashMap<String, Vec<i16>>,
    native: Option<NativeCharacter>,
}

struct NativeCharacter {
    character: acs::Character,
    cache: RefCell<acs::ImageCache>,
}

impl Character {
    /// Load from a directory holding `agent.js` and `map.png` (and optionally
    /// `sounds-mp3.js`, which is only read when `with_sounds` is set — decoding it
    /// costs an ffmpeg spawn per clip).
    pub fn load(dir: &Path, with_sounds: bool) -> Result<Character> {
        let js = std::fs::read_to_string(dir.join("agent.js"))
            .with_context(|| format!("reading {:?}", dir.join("agent.js")))?;
        let (name, json) = unwrap_callback(&js).context("parsing agent.js")?;
        let v: serde_json::Value = serde_json::from_str(&json).context("agent.js is not JSON")?;

        let fs_ = v["framesize"]
            .as_array()
            .ok_or_else(|| anyhow!("agent.js has no `framesize`"))?;
        let frame_w = fs_[0].as_u64().unwrap_or(0) as u32;
        let frame_h = fs_.get(1).and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        if frame_w == 0 || frame_h == 0 {
            bail!("agent.js has a degenerate framesize {frame_w}x{frame_h}");
        }

        let png = std::fs::read(dir.join("map.png"))
            .with_context(|| format!("reading {:?}", dir.join("map.png")))?;
        let img = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
            .context("decoding map.png")?
            .to_rgba8();
        let (sheet_w, sheet_h) = (img.width(), img.height());

        let mut anims = HashMap::new();
        let mut index = HashMap::new();
        let obj = v["animations"]
            .as_object()
            .ok_or_else(|| anyhow!("agent.js has no `animations`"))?;
        for (aname, adef) in obj {
            let frames = adef["frames"]
                .as_array()
                .ok_or_else(|| anyhow!("animation `{aname}` has no frames"))?
                .iter()
                .map(parse_frame)
                .collect::<Result<Vec<_>>>()
                .with_context(|| format!("in animation `{aname}`"))?;
            index.insert(aname.to_ascii_lowercase(), aname.clone());
            anims.insert(
                aname.clone(),
                Animation {
                    name: aname.clone(),
                    frames,
                },
            );
        }
        if anims.is_empty() {
            bail!("{name} has no animations");
        }

        let sounds = if with_sounds {
            load_sounds(dir).unwrap_or_else(|e| {
                eprintln!("[agent] no character audio ({e:#}) — continuing silent");
                HashMap::new()
            })
        } else {
            HashMap::new()
        };

        Ok(Character {
            name,
            frame_w,
            frame_h,
            sheet_w,
            sheet_h,
            sheet: img.into_raw(),
            anims,
            index,
            sounds,
            native: None,
        })
    }

    /// Load either an original Microsoft Agent v2 `.acs` file or a clippy.js
    /// export directory. ACS images, overlays and embedded WAVE clips stay native.
    pub fn load_path(path: &Path, with_sounds: bool) -> Result<Character> {
        if path.is_file() {
            return Self::load_acs(path, with_sounds);
        }
        Self::load(path, with_sounds)
    }

    fn load_acs(path: &Path, with_sounds: bool) -> Result<Character> {
        let native = acs::load(path).with_context(|| format!("reading ACS character {path:?}"))?;
        let frame_w = native.info.width as u32;
        let frame_h = native.info.height as u32;
        let locale = std::env::var("LC_ALL")
            .ok()
            .or_else(|| std::env::var("LANG").ok());
        let primary = locale
            .as_deref()
            .and_then(acs::types::primary_language_from_locale);
        let name = native
            .info
            .name_for(primary)
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Microsoft Agent")
            })
            .to_string();

        let mut anims = HashMap::new();
        let mut index = HashMap::new();
        for (ai, animation) in native.animations.iter().enumerate() {
            let frames = animation
                .frames
                .iter()
                .enumerate()
                .map(|(fi, frame)| Frame {
                    duration_ms: frame.duration_ms() as f32,
                    images: if frame.images.is_empty() {
                        vec![]
                    } else {
                        vec![(0, 0)]
                    },
                    sound: frame.audio_index.map(|i| format!("acs:{i}")),
                    exit_branch: usize::try_from(frame.exit_frame).ok(),
                    branches: frame
                        .branches
                        .iter()
                        .map(|b| (b.frame_index as usize, b.probability as u32))
                        .collect(),
                    native: Some((ai, fi)),
                })
                .collect();
            index.insert(animation.name.to_ascii_lowercase(), animation.name.clone());
            anims.insert(
                animation.name.clone(),
                Animation {
                    name: animation.name.clone(),
                    frames,
                },
            );
        }
        if anims.is_empty() {
            bail!("{} has no animations", name);
        }

        let mut sounds = HashMap::new();
        if with_sounds {
            for i in 0..native.audio_count() {
                if let Some(wav) = native.audio(i) {
                    match wav_to_stereo(wav) {
                        Ok(pcm) => {
                            sounds.insert(format!("acs:{i}"), pcm);
                        }
                        Err(e) => eprintln!("[agent] ignoring ACS audio {i}: {e:#}"),
                    }
                }
            }
        }
        Ok(Character {
            name,
            frame_w,
            frame_h,
            sheet_w: 0,
            sheet_h: 0,
            sheet: vec![],
            anims,
            index,
            sounds,
            native: Some(NativeCharacter {
                character: native,
                cache: RefCell::new(acs::ImageCache::new()),
            }),
        })
    }

    pub fn animation(&self, name: &str) -> Option<&Animation> {
        self.index
            .get(&name.to_ascii_lowercase())
            .and_then(|k| self.anims.get(k))
    }

    /// The first of `names` this character actually has. Characters disagree on
    /// spelling — Clippy says `Greeting`, Merlin says `Greet`.
    fn first_of(&self, names: &[&str]) -> Option<String> {
        names
            .iter()
            .find_map(|n| self.animation(n).map(|a| a.name.clone()))
    }

    pub fn animation_names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.anims.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }

    /// Composite one frame's images into a `frame_w * frame_h` RGBA buffer.
    fn compose(&self, frame: &Frame, mouth: Option<acs::MouthShape>, out: &mut Vec<u8>) {
        if let (Some(native), Some((ai, fi))) = (&self.native, frame.native) {
            if let Some(frame) = native
                .character
                .animations
                .get(ai)
                .and_then(|a| a.frames.get(fi))
            {
                match native
                    .character
                    .render_frame(frame, mouth, &mut native.cache.borrow_mut())
                {
                    Ok(image) => {
                        *out = image.data;
                        return;
                    }
                    Err(e) => eprintln!("[agent] could not render ACS frame: {e}"),
                }
            }
        }
        out.clear();
        out.resize((self.frame_w * self.frame_h * 4) as usize, 0);
        for &(sx, sy) in &frame.images {
            for y in 0..self.frame_h {
                let src_y = sy + y;
                if src_y >= self.sheet_h {
                    break;
                }
                for x in 0..self.frame_w {
                    let src_x = sx + x;
                    if src_x >= self.sheet_w {
                        break;
                    }
                    let s = ((src_y * self.sheet_w + src_x) * 4) as usize;
                    if self.sheet[s + 3] == 0 {
                        continue; // the sheet's one transparent palette entry
                    }
                    let d = ((y * self.frame_w + x) * 4) as usize;
                    out[d..d + 4].copy_from_slice(&self.sheet[s..s + 4]);
                }
            }
        }
    }
}

fn parse_frame(f: &serde_json::Value) -> Result<Frame> {
    let images = f["images"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|p| {
                    (
                        p[0].as_u64().unwrap_or(0) as u32,
                        p[1].as_u64().unwrap_or(0) as u32,
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let branches = f["branching"]["branches"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|b| {
                    (
                        b["frameIndex"].as_u64().unwrap_or(0) as usize,
                        b["weight"].as_u64().unwrap_or(0) as u32,
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(Frame {
        duration_ms: f["duration"].as_f64().unwrap_or(100.0) as f32,
        images,
        sound: f["sound"].as_str().map(|s| s.to_string()),
        exit_branch: f["exitBranch"]
            .as_i64()
            .and_then(|v| usize::try_from(v).ok()),
        branches,
        native: None,
    })
}

/// clippy.js wraps its payload in a call: `clippy.ready('Merlin', { … });`. Pull the
/// name out of the first argument and the object out of the braces.
fn unwrap_callback(js: &str) -> Result<(String, String)> {
    let open = js
        .find('{')
        .ok_or_else(|| anyhow!("no JSON object in the asset"))?;
    let close = js
        .rfind('}')
        .ok_or_else(|| anyhow!("unterminated JSON object in the asset"))?;
    let name = js[..open]
        .split(['\'', '"'])
        .nth(1)
        .unwrap_or("agent")
        .to_string();
    Ok((name, js[open..=close].to_string()))
}

// ---------------------------------------------------------------------------
// Character audio — the boings and whooshes, base64 mp3 in `sounds-mp3.js`
// ---------------------------------------------------------------------------

fn load_sounds(dir: &Path) -> Result<HashMap<String, Vec<i16>>> {
    let path = dir.join("sounds-mp3.js");
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let js = std::fs::read_to_string(&path)?;
    // This file quotes with apostrophes, which JSON won't take. base64 never contains
    // one, so the swap is safe here in a way it wouldn't be for arbitrary text.
    let (_, json) = unwrap_callback(&js)?;
    let v: serde_json::Value = serde_json::from_str(&json.replace('\'', "\""))?;
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow!("sounds are not an object"))?;

    let mut out = HashMap::new();
    for (id, data) in obj {
        let Some(b64) = data.as_str().and_then(|s| s.split(",").nth(1)) else {
            continue;
        };
        let Ok(mp3) = base64_decode(b64) else {
            continue;
        };
        match decode_mp3(&mp3) {
            Ok(pcm) => {
                out.insert(id.clone(), pcm);
            }
            Err(e) => eprintln!("[agent] sound {id} would not decode ({e:#})"),
        }
    }
    Ok(out)
}

/// ffmpeg is already a hard dependency of `--render`, so it does the mp3 work.
fn decode_mp3(mp3: &[u8]) -> Result<Vec<i16>> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "mp3",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ar",
            &AUDIO_RATE.to_string(),
            "-ac",
            "2",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    child.stdin.take().unwrap().write_all(mp3)?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!("ffmpeg rejected the clip");
    }
    Ok(out
        .stdout
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect())
}

fn base64_decode(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => bail!("bad base64 byte {c:#x}"),
        } as u32;
        acc = acc << 6 | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Speech
// ---------------------------------------------------------------------------

/// A synthesised utterance: the audio, and how long the balloon should take to fill.
struct Utterance {
    pcm: Vec<i16>,
    dur: f32,
}

/// Microsoft Agent spoke through SAPI 4 and L&H TruVoice — a formant synthesiser.
/// `espeak-ng` is one too, which is why it's the default here rather than a neural
/// voice that would sound nothing like 1997. `CRTULUM_TTS` overrides the whole
/// command line; `{out}` is replaced with the WAV path and the text arrives on stdin.
fn synthesize(text: &str) -> Result<Utterance> {
    let dir = std::env::temp_dir();
    let wav = dir.join(format!("crtulum-tts-{}.wav", std::process::id()));
    // `--stdin` matters: a bare `-` is taken as a word to pronounce, not a filename.
    let template = std::env::var("CRTULUM_TTS")
        .unwrap_or_else(|_| "espeak-ng -v en-us -s 165 -p 45 --stdin -w {out}".to_string());
    let parts: Vec<String> = template
        .split_whitespace()
        .map(|p| p.replace("{out}", &wav.to_string_lossy()))
        .collect();
    let mut child = Command::new(&parts[0])
        .args(&parts[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("running `{}` for text-to-speech", parts[0]))?;
    child.stdin.take().unwrap().write_all(text.as_bytes())?;
    if !child.wait()?.success() {
        bail!("`{}` failed on {text:?}", parts[0]);
    }
    let bytes = std::fs::read(&wav).context("reading the synthesised wav")?;
    std::fs::remove_file(&wav).ok();
    let pcm = wav_to_stereo(&bytes)?;
    let dur = pcm.len() as f32 / 2.0 / AUDIO_RATE as f32;
    Ok(Utterance { pcm, dur })
}

/// Minimal RIFF reader: enough for what a speech synthesiser writes (16-bit PCM,
/// usually 22050 Hz mono), resampled up to our mix rate.
fn wav_to_stereo(b: &[u8]) -> Result<Vec<i16>> {
    if b.len() < 44 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }
    let (mut pos, mut channels, mut rate, mut bits) = (12usize, 1u16, AUDIO_RATE, 16u16);
    let mut data: &[u8] = &[];
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let len = u32::from_le_bytes(b[pos + 4..pos + 8].try_into()?) as usize;
        let body = &b[pos + 8..(pos + 8 + len).min(b.len())];
        match id {
            b"fmt " if body.len() >= 16 => {
                channels = u16::from_le_bytes(body[2..4].try_into()?).max(1);
                rate = u32::from_le_bytes(body[4..8].try_into()?).max(1);
                bits = u16::from_le_bytes(body[14..16].try_into()?);
            }
            b"data" => data = body,
            _ => {}
        }
        pos += 8 + len + (len & 1); // chunks are word-aligned
    }
    if bits != 16 {
        bail!("only 16-bit PCM speech is supported, got {bits}-bit");
    }
    // Mix to mono first, then linearly resample to the bed rate.
    let mono: Vec<i16> = data
        .chunks_exact(2 * channels as usize)
        .map(|f| {
            let sum: i32 = f
                .chunks_exact(2)
                .map(|s| i16::from_le_bytes([s[0], s[1]]) as i32)
                .sum();
            (sum / channels as i32) as i16
        })
        .collect();
    let n = (mono.len() as f64 * AUDIO_RATE as f64 / rate as f64) as usize;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let x = i as f64 * rate as f64 / AUDIO_RATE as f64;
        let (i0, f) = (x as usize, (x - x.floor()) as f32);
        let a = mono.get(i0).copied().unwrap_or(0) as f32;
        let b_ = mono.get(i0 + 1).copied().unwrap_or(a as i16) as f32;
        let s = (a + (b_ - a) * f) as i16;
        out.push(s);
        out.push(s);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The audio bed — every agent noise, mixed on one timeline
// ---------------------------------------------------------------------------

/// Stereo interleaved at [`AUDIO_RATE`], grown as sounds land on it. The emulator's
/// own audio is a separate track; ffmpeg mixes the two at the end.
pub struct AudioBed {
    pub data: Vec<i16>,
    used: bool,
}

impl AudioBed {
    fn new() -> AudioBed {
        AudioBed {
            data: Vec::new(),
            used: false,
        }
    }

    /// Add `pcm` starting at `t` seconds, saturating rather than wrapping.
    fn mix(&mut self, t: f32, pcm: &[i16], gain: f32) {
        if pcm.is_empty() {
            return;
        }
        let start = (t.max(0.0) * AUDIO_RATE as f32) as usize * 2;
        let end = start + pcm.len();
        if self.data.len() < end {
            self.data.resize(end, 0);
        }
        for (d, s) in self.data[start..end].iter_mut().zip(pcm) {
            *d = (*d as f32 + *s as f32 * gain).clamp(-32768.0, 32767.0) as i16;
        }
        self.used = true;
    }

    pub fn is_empty(&self) -> bool {
        !self.used
    }

    pub fn write_raw(&self, path: &Path) -> Result<()> {
        std::fs::write(path, bytemuck::cast_slice(&self.data))
            .with_context(|| format!("writing {path:?}"))
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Cmd {
    /// Play the character's entrance and become visible.
    Show,
    /// Play the exit and go away.
    Hide,
    /// Teleport, in raster-normalised coordinates (0,0 top-left → 1,1 bottom-right).
    At([f32; 2]),
    /// Walk there over `over` seconds, playing the matching Move animation.
    MoveTo { to: [f32; 2], over: f32 },
    /// Play a named animation once.
    Play(String),
    /// Say something: balloon, speaking animation, and speech if a synthesiser is
    /// available.
    Say(String),
    /// Turn and gesture at a point on the screen.
    Point([f32; 2]),
    /// Multiplier on the size the character would have had on a 640x480 desktop.
    Scale(f32),
}

// ---------------------------------------------------------------------------
// The animator
// ---------------------------------------------------------------------------

/// Deterministic by construction: the only randomness is branch selection and idle
/// choice, both drawn from this, and it is seeded from a constant and advanced only
/// by the frame loop. Render the same script twice and the same frames come out —
/// which is the whole point of driving a TAS with it.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }
}

struct Playing {
    anim: String,
    frame: usize,
    held_ms: f32,
    exiting: bool,
}

struct Balloon {
    text: String,
    start: f32,
    /// How long the text takes to fill in — the utterance length when there's speech.
    reveal: f32,
    /// When the balloon disappears.
    until: f32,
}

/// How long a balloon lingers after the last word.
const BALLOON_HANG: f32 = 0.7;
/// Reading rate used when there's no synthesiser to time against. espeak-ng's default
/// is 175 wpm; 165 is what we ask it for, and an average English word is ~5.5 chars.
const CHARS_PER_SEC: f32 = 165.0 * 5.5 / 60.0;
/// Idle animations kick in after this long with nothing to do, as they did on the desktop.
const IDLE_AFTER: f32 = 5.0;

pub struct Agent {
    pub character: Character,
    /// Scripted commands, sorted by time.
    events: Vec<(f32, Cmd)>,
    cursor: usize,
    /// Pre-synthesised speech, keyed by the event index that says it.
    speech: HashMap<usize, Utterance>,
    pub audio: AudioBed,

    visible: bool,
    playing: Option<Playing>,
    /// What he looks like when he isn't doing anything. Every character has a
    /// `RestPose`; without it he'd blink out of existence the moment an animation
    /// ran off its last frame.
    rest: Option<String>,
    /// Set when an animation ends, so idles only start after a real pause.
    idle_since: f32,
    rng: Rng,

    pos: [f32; 2],
    walk: Option<(f32, f32, [f32; 2], [f32; 2])>,
    scale: f32,
    balloon: Option<Balloon>,
    /// The clock as of the last `step`, so `draw` knows how much of the balloon has
    /// been spoken.
    now: f32,

    /// Scratch buffer for the composed frame, reused every draw.
    composed: Vec<u8>,
    /// Whether an animation frame's sound has been mixed already.
    sounded: bool,
}

impl Agent {
    /// Build the agent for a run. Speech is synthesised up front because the balloon's
    /// fill rate is timed against the utterance, and that has to be known before the
    /// first frame is drawn.
    pub fn new(character: Character, mut events: Vec<(f32, Cmd)>, want_audio: bool) -> Agent {
        events.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut speech = HashMap::new();
        if want_audio {
            for (i, (_, cmd)) in events.iter().enumerate() {
                if let Cmd::Say(text) = cmd {
                    match synthesize(text) {
                        Ok(u) => {
                            speech.insert(i, u);
                        }
                        Err(e) => {
                            eprintln!("[agent] no speech for {text:?} ({e:#}) — balloon only");
                            break; // one failure means the synthesiser is missing; stop trying
                        }
                    }
                }
            }
        }
        Agent {
            rest: character.first_of(&["RestPose", "Rest", "Alert", "Idle1_1"]),
            character,
            events,
            cursor: 0,
            speech,
            audio: AudioBed::new(),
            visible: false,
            playing: None,
            idle_since: 0.0,
            rng: Rng(0x9E37_79B9_7F4A_7C15),
            pos: [0.78, 0.70],
            walk: None,
            scale: 1.0,
            balloon: None,
            now: 0.0,
            composed: Vec::new(),
            sounded: false,
        }
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Advance to wall-clock `t`, having taken `dt` seconds since the last call.
    /// Called once per rendered frame, in order — the agent's state is a fold over
    /// the frame sequence, not a function of `t`, because animation branching is.
    pub fn step(&mut self, t: f32, dt: f32) {
        self.now = t;
        while self.cursor < self.events.len() && self.events[self.cursor].0 <= t {
            let (at, cmd) = self.events[self.cursor].clone();
            self.apply(&cmd, self.cursor, at.max(t - dt));
            self.cursor += 1;
        }

        if let Some((t0, t1, from, to)) = self.walk {
            let k = ((t - t0) / (t1 - t0).max(1e-6)).clamp(0.0, 1.0);
            // Smoothstep: the Move animations are hops, and a linear glide under a
            // hop reads as sliding.
            let k = k * k * (3.0 - 2.0 * k);
            self.pos = [
                from[0] + (to[0] - from[0]) * k,
                from[1] + (to[1] - from[1]) * k,
            ];
            if t >= t1 {
                self.walk = None;
            }
        }

        if let Some(b) = &self.balloon {
            if t >= b.until {
                self.balloon = None;
            }
        }

        self.advance_animation(t, dt);
    }

    fn apply(&mut self, cmd: &Cmd, idx: usize, at: f32) {
        match cmd {
            Cmd::Show => {
                self.visible = true;
                if let Some(a) = self
                    .character
                    .first_of(&["Show", "Greeting", "Greet", "Alert"])
                {
                    self.start(a, at);
                }
            }
            Cmd::Hide => {
                if let Some(a) = self.character.first_of(&["Hide", "GoodBye", "Goodbye"]) {
                    self.start(a, at);
                } else {
                    self.visible = false;
                }
                // The Hide animation plays out and *then* he's gone; `advance_animation`
                // clears `visible` when it ends.
            }
            Cmd::At(p) => {
                self.pos = *p;
                self.walk = None;
                self.visible = true;
            }
            Cmd::MoveTo { to, over } => {
                self.visible = true;
                let dir = [to[0] - self.pos[0], to[1] - self.pos[1]];
                if let Some(a) = self.directional("Move", dir) {
                    self.start(a, at);
                }
                self.walk = Some((at, at + over.max(0.01), self.pos, *to));
            }
            Cmd::Play(name) => {
                self.visible = true;
                match self.character.animation(name).map(|a| a.name.clone()) {
                    Some(a) => self.start(a, at),
                    None => eprintln!(
                        "[agent] {} has no animation `{name}` — try one of: {}",
                        self.character.name,
                        self.character.animation_names().join(", ")
                    ),
                }
            }
            Cmd::Point(target) => {
                self.visible = true;
                let dir = [target[0] - self.pos[0], target[1] - self.pos[1]];
                if let Some(a) = self.directional("Gesture", dir).or_else(|| {
                    self.character
                        .first_of(&["Alert", "GetAttention", "RestPose"])
                }) {
                    self.start(a, at);
                }
            }
            Cmd::Scale(s) => self.scale = s.max(0.01),
            Cmd::Say(text) => {
                self.visible = true;
                let dur = match self.speech.get(&idx) {
                    Some(u) => u.dur,
                    None => (text.chars().count() as f32 / CHARS_PER_SEC).max(1.0),
                };
                // Take the samples out rather than cloning: an utterance is only ever
                // spoken once, and `dur` (which `end()` still needs) stays behind.
                let pcm = self
                    .speech
                    .get_mut(&idx)
                    .map(|u| std::mem::take(&mut u.pcm));
                if let Some(pcm) = pcm {
                    self.audio.mix(at, &pcm, 0.9);
                }
                self.balloon = Some(Balloon {
                    text: text.clone(),
                    start: at,
                    reveal: dur,
                    until: at + dur + BALLOON_HANG,
                });
                if let Some(a) = self.character.first_of(&[
                    "Explain",
                    "Announce",
                    "Alert",
                    "GestureRight",
                    "RestPose",
                ]) {
                    self.start(a, at);
                }
            }
        }
    }

    /// Pick `GestureLeft`/`Right`/`Up`/`Down`-style names by dominant axis. The screen
    /// is wider than it is tall, so the horizontal component is weighted down to match
    /// what the direction *looks* like rather than what it measures.
    fn directional(&self, prefix: &str, dir: [f32; 2]) -> Option<String> {
        let (dx, dy) = (dir[0] * 0.75, dir[1]);
        let name = if dx.abs() >= dy.abs() {
            if dx < 0.0 {
                "Left"
            } else {
                "Right"
            }
        } else if dy < 0.0 {
            "Up"
        } else {
            "Down"
        };
        self.character
            .animation(&format!("{prefix}{name}"))
            .map(|a| a.name.clone())
    }

    fn start(&mut self, anim: String, t: f32) {
        self.playing = Some(Playing {
            anim,
            frame: 0,
            held_ms: 0.0,
            exiting: false,
        });
        self.sounded = false;
        self.idle_since = t;
    }

    fn advance_animation(&mut self, t: f32, dt: f32) {
        // Trigger the sound attached to the frame we're currently on, once.
        if !self.sounded {
            self.sounded = true;
            if let Some(id) = self.current_frame().and_then(|f| f.sound.clone()) {
                if let Some(pcm) = self.character.sounds.get(&id) {
                    let pcm = pcm.clone();
                    self.audio.mix(t, &pcm, 0.7);
                }
            }
        }

        let Some(p) = &mut self.playing else {
            // Nothing playing. After a pause, pick an idle, the way the desktop did.
            if self.visible && t - self.idle_since > IDLE_AFTER {
                let idles: Vec<String> = self
                    .character
                    .animation_names()
                    .iter()
                    .filter(|n| n.to_ascii_lowercase().starts_with("idle"))
                    .map(|n| n.to_string())
                    .collect();
                if !idles.is_empty() {
                    let pick = idles[self.rng.below(idles.len() as u32) as usize].clone();
                    self.start(pick, t);
                }
            }
            return;
        };

        p.held_ms += dt * 1000.0;
        let Some(anim) = self.character.anims.get(&p.anim) else {
            self.playing = None;
            return;
        };
        let Some(frame) = anim.frames.get(p.frame) else {
            self.playing = None;
            self.idle_since = t;
            return;
        };
        if p.held_ms < frame.duration_ms {
            return;
        }
        p.held_ms -= frame.duration_ms;

        // Where next: an exit branch if we're unwinding, else a weighted branch, else
        // simply onward. This is the `.acs` frame model verbatim.
        let next = if p.exiting {
            frame.exit_branch.unwrap_or(p.frame + 1)
        } else if !frame.branches.is_empty() {
            let total: u32 = frame.branches.iter().map(|(_, w)| w).sum();
            let mut roll = self.rng.below(total.max(1));
            let mut chosen = p.frame + 1;
            for (target, weight) in &frame.branches {
                if roll < *weight {
                    chosen = *target;
                    break;
                }
                roll -= *weight;
            }
            chosen
        } else {
            p.frame + 1
        };

        if next >= anim.frames.len() {
            let ended = p.anim.clone();
            self.playing = None;
            self.idle_since = t;
            if ended.eq_ignore_ascii_case("hide") || ended.eq_ignore_ascii_case("goodbye") {
                self.visible = false;
            }
        } else {
            p.frame = next;
            self.sounded = false;
        }
    }

    /// The frame to draw: whatever is playing, else the rest pose. An animation that
    /// has run out doesn't mean there's nobody there.
    fn current_frame(&self) -> Option<&Frame> {
        match &self.playing {
            Some(p) => self.character.anims.get(&p.anim)?.frames.get(p.frame),
            None => self
                .character
                .anims
                .get(self.rest.as_ref()?)?
                .frames
                .first(),
        }
    }

    /// Where the character's box sits on a `w`x`h` raster: (x, y, width, height) in
    /// pixels, top-left origin. `pos` is the centre of the box.
    fn box_on(&self, w: u32, h: u32) -> (i32, i32, u32, u32) {
        let k = (h as f32 / DESIGN_HEIGHT) * self.scale;
        let bw = (self.character.frame_w as f32 * k).round().max(1.0) as u32;
        let bh = (self.character.frame_h as f32 * k).round().max(1.0) as u32;
        let x = (self.pos[0] * w as f32 - bw as f32 / 2.0).round() as i32;
        let y = (self.pos[1] * h as f32 - bh as f32 / 2.0).round() as i32;
        (x, y, bw, bh)
    }

    /// Draw the character and his balloon into an RGBA raster.
    pub fn draw(&mut self, dst: &mut [u8], w: u32, h: u32) {
        if !self.visible || w == 0 || h == 0 {
            return;
        }
        let (x, y, bw, bh) = self.box_on(w, h);
        if let Some(frame) = self.current_frame() {
            if !frame.images.is_empty() {
                let frame = frame.clone();
                let mut buf = std::mem::take(&mut self.composed);
                let mouth = self.mouth_shape();
                self.character.compose(&frame, mouth, &mut buf);
                blit_scaled(
                    &buf,
                    self.character.frame_w,
                    self.character.frame_h,
                    dst,
                    w,
                    h,
                    x,
                    y,
                    bw,
                    bh,
                );
                self.composed = buf;
            }
        }
        if let Some(b) = &self.balloon {
            draw_balloon(dst, w, h, b, (x, y, bw, bh), self.now);
        }
    }

    /// Pick an ACS mouth overlay from the actual mixed speech amplitude near now.
    fn mouth_shape(&self) -> Option<acs::MouthShape> {
        let b = self.balloon.as_ref()?;
        if self.now < b.start || self.now >= b.start + b.reveal {
            return Some(acs::MouthShape::Closed);
        }
        let center = (self.now * AUDIO_RATE as f32) as usize * 2;
        let radius = (AUDIO_RATE / 100) as usize * 2;
        let lo = center.saturating_sub(radius);
        let hi = (center + radius).min(self.audio.data.len());
        let peak = self
            .audio
            .data
            .get(lo..hi)
            .unwrap_or(&[])
            .iter()
            .map(|s| s.unsigned_abs())
            .max()
            .unwrap_or(0);
        Some(match peak {
            0..=500 => acs::MouthShape::Closed,
            501..=1800 => acs::MouthShape::Narrow,
            1801..=4200 => acs::MouthShape::Medium,
            4201..=8000 => acs::MouthShape::WideOpen1,
            8001..=13000 => acs::MouthShape::WideOpen2,
            13001..=20000 => acs::MouthShape::WideOpen3,
            _ => acs::MouthShape::WideOpen4,
        })
    }

    /// Time of the last scripted event — how long the run has to be for the agent to
    /// finish saying his piece.
    pub fn end(&self) -> f32 {
        self.events
            .iter()
            .enumerate()
            .map(|(i, (t, cmd))| match cmd {
                Cmd::Say(text) => {
                    t + self
                        .speech
                        .get(&i)
                        .map(|u| u.dur)
                        .unwrap_or_else(|| text.chars().count() as f32 / CHARS_PER_SEC)
                        + BALLOON_HANG
                }
                Cmd::MoveTo { over, .. } => t + over,
                _ => *t,
            })
            .fold(0.0, f32::max)
    }
}

// ---------------------------------------------------------------------------
// Compositing
// ---------------------------------------------------------------------------

/// sRGB → linear, for the 256 possible byte values. Resampling has to happen in
/// light, not in code values, or the character's outline picks up a dark fringe as
/// he shrinks onto a 240-line raster.
fn srgb_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let c = i as f32 / 255.0;
            *v = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            };
        }
        t
    })
}

fn linear_to_srgb(v: f32) -> u8 {
    let c = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Box-filtered blit of a straight-alpha RGBA sprite into a straight-alpha RGBA
/// raster, premultiplied through the average so transparent pixels contribute no
/// colour. Downscaling is the normal case — the sprite is drawn for a desktop.
#[allow(clippy::too_many_arguments)]
fn blit_scaled(
    src: &[u8],
    sw: u32,
    sh: u32,
    dst: &mut [u8],
    dw: u32,
    dh: u32,
    dx: i32,
    dy: i32,
    bw: u32,
    bh: u32,
) {
    let lut = srgb_lut();
    for oy in 0..bh {
        let ty = dy + oy as i32;
        if ty < 0 || ty as u32 >= dh {
            continue;
        }
        // Source rows covered by this destination row.
        let y0 = (oy as f32 * sh as f32 / bh as f32) as u32;
        let y1 = (((oy + 1) as f32 * sh as f32 / bh as f32).ceil() as u32).clamp(y0 + 1, sh);
        for ox in 0..bw {
            let tx = dx + ox as i32;
            if tx < 0 || tx as u32 >= dw {
                continue;
            }
            let x0 = (ox as f32 * sw as f32 / bw as f32) as u32;
            let x1 = (((ox + 1) as f32 * sw as f32 / bw as f32).ceil() as u32).clamp(x0 + 1, sw);

            let (mut r, mut g, mut b, mut a, mut n) = (0.0f32, 0.0, 0.0, 0.0, 0.0f32);
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let s = ((sy * sw + sx) * 4) as usize;
                    let sa = src[s + 3] as f32 / 255.0;
                    r += lut[src[s] as usize] * sa;
                    g += lut[src[s + 1] as usize] * sa;
                    b += lut[src[s + 2] as usize] * sa;
                    a += sa;
                    n += 1.0;
                }
            }
            if n == 0.0 || a <= 0.0 {
                continue;
            }
            let (cov, r, g, b) = (a / n, r / n, g / n, b / n); // still premultiplied
            let d = ((ty as u32 * dw + tx as u32) * 4) as usize;
            let dr = lut[dst[d] as usize];
            let dg = lut[dst[d + 1] as usize];
            let db = lut[dst[d + 2] as usize];
            dst[d] = linear_to_srgb(r + dr * (1.0 - cov));
            dst[d + 1] = linear_to_srgb(g + dg * (1.0 - cov));
            dst[d + 2] = linear_to_srgb(b + db * (1.0 - cov));
            dst[d + 3] = 255;
        }
    }
}

fn put(dst: &mut [u8], w: u32, h: u32, x: i32, y: i32, c: [u8; 3]) {
    if x < 0 || y < 0 || x as u32 >= w || y as u32 >= h {
        return;
    }
    let d = ((y as u32 * w + x as u32) * 4) as usize;
    dst[d] = c[0];
    dst[d + 1] = c[1];
    dst[d + 2] = c[2];
    dst[d + 3] = 255;
}

// The Office Assistant's balloon: pale yellow fill, hairline black border, black text.
const BALLOON_FILL: [u8; 3] = [255, 255, 206];
const BALLOON_INK: [u8; 3] = [0, 0, 0];

/// Wrap `text` to `cols` characters, breaking on spaces where it can.
fn wrap(text: &str, cols: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if line.is_empty() {
            line = word.to_string();
        } else if line.chars().count() + 1 + word.chars().count() <= cols {
            line.push(' ');
            line.push_str(word);
        } else {
            lines.push(std::mem::take(&mut line));
            line = word.to_string();
        }
        // A single word longer than the line gets cut rather than overflowing.
        while line.chars().count() > cols {
            let head: String = line.chars().take(cols).collect();
            line = line.chars().skip(cols).collect();
            lines.push(head);
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn draw_balloon(dst: &mut [u8], w: u32, h: u32, b: &Balloon, ch: (i32, i32, u32, u32), t: f32) {
    // Text fills in over the length of the utterance, so the balloon keeps pace with
    // the voice instead of dumping the whole line at once. The last character lands a
    // little early — trailing silence in a synthesised clip shouldn't stall the line.
    let total = b.text.chars().count();
    let shown = if b.reveal <= 0.0 {
        total
    } else {
        let k = ((t - b.start) / b.reveal * 1.08).clamp(0.0, 1.0);
        (total as f32 * k).ceil() as usize
    };
    let cw = font8x8::CELL_W as u32;
    let cell_h = font8x8::CELL_H as u32;
    let pad = 4u32;

    // Keep the balloon to about three-fifths of the picture. A wider one covers the
    // game rather than commenting on it, and the real ones were never full-bleed.
    let max_cols = (((w as f32 * 0.62) as u32).saturating_sub(2 * pad) / cw).clamp(8, 32) as usize;
    let lines = wrap(&b.text, max_cols);
    let cols = lines
        .iter()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(1)
        .max(1) as u32;

    let bw = cols * cw + pad * 2;
    let bh = lines.len() as u32 * cell_h + pad * 2;
    let tail = 5i32;

    // Above the character by default, below if there's no room up there.
    let (cx, cy, cbw, _cbh) = ch;
    let mut bx = cx + cbw as i32 / 2 - bw as i32 / 2;
    let above = cy - tail - bh as i32 >= 0;
    let by = if above {
        cy - tail - bh as i32
    } else {
        cy + _cbh as i32 + tail
    };
    // Keep the balloon inside the safe area. A consumer set scans the raster larger
    // than the visible faceplate, and the tube models that (`focus.y`, ~3.5–5% a side
    // depending on the signal) — a balloon flush to the raster edge falls off the
    // glass. The BBC's old safe-area margin is the right number here too.
    let margin = (w as f32 * 0.06).round() as i32;
    bx = bx.clamp(margin, (w as i32 - bw as i32 - margin).max(margin));

    // Body, with the corner pixels knocked off — a 1-px round on an 8-px grid.
    for y in 0..bh as i32 {
        for x in 0..bw as i32 {
            let corner = (x == 0 || x == bw as i32 - 1) && (y == 0 || y == bh as i32 - 1);
            if corner {
                continue;
            }
            let edge = x == 0 || y == 0 || x == bw as i32 - 1 || y == bh as i32 - 1;
            put(
                dst,
                w,
                h,
                bx + x,
                by + y,
                if edge { BALLOON_INK } else { BALLOON_FILL },
            );
        }
    }

    // Tail: a small wedge from the balloon toward the character's head.
    let tip_x = (cx + cbw as i32 / 2).clamp(bx + 3, bx + bw as i32 - 4);
    for i in 0..tail {
        let (yy, half) = if above {
            (by + bh as i32 - 1 + i, tail - i)
        } else {
            (by - i, tail - i)
        };
        for x in (tip_x - half)..=(tip_x + half) {
            let edge = x == tip_x - half || x == tip_x + half || i == tail - 1;
            put(
                dst,
                w,
                h,
                x,
                yy,
                if edge { BALLOON_INK } else { BALLOON_FILL },
            );
        }
    }

    // Text, up to the reveal point.
    let mut left = shown;
    for (li, line) in lines.iter().enumerate() {
        for (ci, c) in line.chars().enumerate() {
            if left == 0 {
                return;
            }
            left -= 1;
            let Some(g) = font8x8::glyph(c) else { continue };
            let gx = bx + pad as i32 + ci as i32 * cw as i32;
            let gy = by + pad as i32 + li as i32 * cell_h as i32;
            for yy in 0..font8x8::CELL_H {
                for xx in 0..font8x8::CELL_W {
                    if font8x8::lit(g, xx, yy) {
                        put(dst, w, h, gx + xx as i32, gy + yy as i32, BALLOON_INK);
                    }
                }
            }
        }
        // The newline costs a character too, so wrapping doesn't run ahead of speech.
        left = left.saturating_sub(1);
    }
}

// ---------------------------------------------------------------------------
// Finding and fetching characters
// ---------------------------------------------------------------------------

/// Search order for `--agent <name>`: an explicit path first, then `$CRTULUM_AGENTS`,
/// then the per-user data directory `--fetch-agent` writes to, then `./agents`.
pub fn resolve(name_or_path: &str) -> Result<PathBuf> {
    let direct = Path::new(name_or_path);
    if direct.is_file() {
        return Ok(direct.to_path_buf());
    }
    if direct.join("agent.js").is_file() {
        return Ok(direct.to_path_buf());
    }
    let mut tried = vec![direct.to_path_buf()];
    for base in search_paths() {
        // Match case-insensitively: scripts say `merlin`, the directory says `Merlin`.
        if let Ok(entries) = std::fs::read_dir(&base) {
            for e in entries.flatten() {
                let path = e.path();
                let stem_matches = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().eq_ignore_ascii_case(name_or_path))
                    .unwrap_or(false);
                if (e
                    .file_name()
                    .to_string_lossy()
                    .eq_ignore_ascii_case(name_or_path)
                    && path.join("agent.js").is_file())
                    || (stem_matches && path.is_file())
                {
                    return Ok(path);
                }
            }
        }
        tried.push(base.join(name_or_path));
    }
    bail!(
        "no character `{name_or_path}` (looked in {}). `crtulum --fetch-agent {name_or_path}` \
         downloads one; known characters: {}",
        tried
            .iter()
            .map(|p| format!("{p:?}"))
            .collect::<Vec<_>>()
            .join(", "),
        KNOWN.join(", ")
    )
}

fn search_paths() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("CRTULUM_AGENTS") {
        v.push(PathBuf::from(p));
    }
    v.push(data_dir());
    v.push(PathBuf::from("agents"));
    v
}

/// Where downloaded characters live. Not in the repo: they are Microsoft's artwork,
/// and this project ships the reader for them, nothing more.
pub fn data_dir() -> PathBuf {
    std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("crtulum/agents")
}

/// Download one character's assets from the clippy.js repository.
pub fn fetch(name: &str) -> Result<PathBuf> {
    let canonical = KNOWN
        .iter()
        .find(|k| k.eq_ignore_ascii_case(name))
        .copied()
        .unwrap_or(name);
    let dir = data_dir().join(canonical);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {dir:?}"))?;
    for file in ["agent.js", "map.png", "sounds-mp3.js"] {
        let url = format!("{ASSET_BASE}/{canonical}/{file}");
        let out = dir.join(file);
        let status = Command::new("curl")
            .args(["-fsSL", "-o"])
            .arg(&out)
            .arg(&url)
            .status()
            .context("running curl — is it installed?")?;
        if !status.success() {
            std::fs::remove_file(&out).ok();
            // Only the sound bank is optional.
            if file != "sounds-mp3.js" {
                bail!(
                    "could not download {url} — is `{canonical}` a real character? ({})",
                    KNOWN.join(", ")
                );
            }
        }
    }
    println!("fetched {canonical} into {}", dir.display());
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sheet_char() -> Character {
        // A 2x1 sheet: one opaque red pixel, one transparent.
        let sheet = vec![255, 0, 0, 255, 0, 0, 0, 0];
        let mut anims = HashMap::new();
        let mut index = HashMap::new();
        anims.insert(
            "Wave".to_string(),
            Animation {
                name: "Wave".into(),
                frames: vec![
                    Frame {
                        duration_ms: 100.0,
                        images: vec![(0, 0)],
                        sound: None,
                        exit_branch: Some(2),
                        branches: vec![(0, 100)],
                        native: None,
                    },
                    Frame {
                        duration_ms: 100.0,
                        images: vec![(0, 0)],
                        sound: None,
                        exit_branch: None,
                        branches: vec![],
                        native: None,
                    },
                    Frame {
                        duration_ms: 100.0,
                        images: vec![(0, 0)],
                        sound: None,
                        exit_branch: None,
                        branches: vec![],
                        native: None,
                    },
                ],
            },
        );
        index.insert("wave".to_string(), "Wave".to_string());
        Character {
            name: "Test".into(),
            frame_w: 1,
            frame_h: 1,
            sheet_w: 2,
            sheet_h: 1,
            sheet,
            anims,
            index,
            sounds: HashMap::new(),
            native: None,
        }
    }

    /// The real characters, if they've been fetched. Everything here is skipped
    /// rather than failed when they haven't been — they're Microsoft's artwork and
    /// this repository doesn't carry it.
    fn real(name: &str) -> Option<Character> {
        let dir = resolve(name).ok()?;
        Character::load(&dir, false).ok()
    }

    #[test]
    fn a_real_character_loads_with_the_animations_a_script_can_name() {
        let Some(c) = real("Merlin") else {
            eprintln!("skipping: run `crtulum --fetch-agent Merlin`");
            return;
        };
        assert_eq!((c.frame_w, c.frame_h), (128, 128));
        // The four directional gestures are what `agent point` compiles down to, and
        // RestPose is what he falls back to between animations.
        for want in [
            "GestureLeft",
            "GestureRight",
            "GestureUp",
            "GestureDown",
            "RestPose",
            "Show",
            "Hide",
        ] {
            assert!(c.animation(want).is_some(), "Merlin should have {want}");
        }
        // Lookup is case-insensitive so scripts can say `agent play restpose`.
        assert!(c.animation("restpose").is_some());
        assert!(c.animation("NoSuchAnimation").is_none());
    }

    #[test]
    fn a_real_character_draws_visible_pixels_and_keeps_drawing_them() {
        let Some(c) = real("Merlin") else {
            eprintln!("skipping: run `crtulum --fetch-agent Merlin`");
            return;
        };
        let (w, h) = (256u32, 240u32);
        let mut a = Agent::new(c, vec![(0.0, Cmd::At([0.5, 0.5])), (0.0, Cmd::Show)], false);

        let count = |dst: &[u8]| dst.chunks_exact(4).filter(|p| p[3] > 0).count();
        // Step well past the end of the Show animation: the rest pose has to keep him
        // on the screen, which is the bug the first render caught.
        let mut seen_at = Vec::new();
        for i in 0..600 {
            let mut dst = vec![0u8; (w * h * 4) as usize];
            a.step(i as f32 / 60.0, 1.0 / 60.0);
            a.draw(&mut dst, w, h);
            // Skip frame 0 — the first frame of `Show` is the start of a materialise,
            // and he's legitimately barely there yet.
            if i > 0 && i % 120 == 0 {
                seen_at.push(count(&dst));
            }
        }
        assert!(
            seen_at.iter().all(|&n| n > 200),
            "the character should be on screen the whole time, drew {seen_at:?} pixels"
        );
    }

    #[test]
    fn he_is_scaled_to_the_raster_not_pasted_at_desktop_size() {
        let Some(c) = real("Merlin") else {
            eprintln!("skipping: run `crtulum --fetch-agent Merlin`");
            return;
        };
        let mut a = Agent::new(c, vec![(0.0, Cmd::At([0.5, 0.5]))], false);
        a.step(0.0, 1.0 / 60.0);
        // 128 px of art on a 240-line signal is the same fraction of the picture it
        // was on a 640x480 desktop: 128 * 240/480 = 64.
        let (x, y, bw, bh) = a.box_on(256, 240);
        assert_eq!((bw, bh), (64, 64));
        assert_eq!((x, y), (96, 88), "he should be centred on `pos`");
    }

    #[test]
    fn speaking_reveals_the_balloon_over_the_length_of_the_line() {
        let Some(c) = real("Merlin") else {
            eprintln!("skipping: run `crtulum --fetch-agent Merlin`");
            return;
        };
        let line = "Watch this frame perfect wall jump";
        let mut a = Agent::new(
            c,
            vec![(0.0, Cmd::At([0.5, 0.7])), (0.0, Cmd::Say(line.into()))],
            false, // no synthesis in tests: the reveal falls back to a reading rate
        );
        let ink = |dst: &[u8]| {
            dst.chunks_exact(4)
                .filter(|p| {
                    p[0] == BALLOON_FILL[0] && p[1] == BALLOON_FILL[1] && p[2] == BALLOON_FILL[2]
                })
                .count()
        };
        let mut text = Vec::new();
        for i in 0..90 {
            let mut dst = vec![0u8; (256 * 240 * 4) as usize];
            a.step(i as f32 / 60.0, 1.0 / 60.0);
            a.draw(&mut dst, 256, 240);
            if i == 1 || i == 45 || i == 89 {
                // Fill pixels shrink as glyphs (drawn in ink) eat into the balloon.
                text.push(ink(&dst));
            }
        }
        assert!(
            text[0] > 0,
            "the balloon should appear as soon as he speaks"
        );
        assert!(
            text[0] > text[1] && text[1] > text[2],
            "text should fill in progressively, saw {text:?} balloon-fill pixels"
        );
    }

    #[test]
    fn unwraps_the_clippy_js_callback() {
        let (name, json) = unwrap_callback("clippy.ready('Merlin', {\"a\": 1});").unwrap();
        assert_eq!(name, "Merlin");
        assert_eq!(json, "{\"a\": 1}");
    }

    #[test]
    fn parses_a_frame_the_way_the_acs_model_says() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"duration": 1200, "images": [[124, 0], [0, 93]], "sound": "7",
                "exitBranch": 21, "branching": {"branches": [{"frameIndex": 9, "weight": 60}]}}"#,
        )
        .unwrap();
        let f = parse_frame(&v).unwrap();
        assert_eq!(f.duration_ms, 1200.0);
        assert_eq!(f.images, vec![(124, 0), (0, 93)]);
        assert_eq!(f.sound.as_deref(), Some("7"));
        assert_eq!(f.exit_branch, Some(21));
        assert_eq!(f.branches, vec![(9, 60)]);
    }

    #[test]
    fn base64_round_trips_a_known_vector() {
        assert_eq!(base64_decode("TWFu").unwrap(), b"Man");
        assert_eq!(base64_decode("bGlnaHQgdw==").unwrap(), b"light w");
    }

    #[test]
    fn animation_timing_follows_frame_durations() {
        let mut a = Agent::new(sheet_char(), vec![(0.0, Cmd::Play("wave".into()))], false);
        a.step(0.0, 1.0 / 60.0);
        assert_eq!(a.playing.as_ref().unwrap().frame, 0);
        // A 100 ms frame at 60 fps is six frames; the seventh advances.
        for i in 1..=6 {
            a.step(i as f32 / 60.0, 1.0 / 60.0);
        }
        // The only branch has weight 100 back to frame 0, so it loops rather than ends.
        assert_eq!(a.playing.as_ref().unwrap().frame, 0);
    }

    #[test]
    fn the_same_script_animates_identically_twice() {
        let run = || {
            let mut a = Agent::new(sheet_char(), vec![(0.0, Cmd::Play("wave".into()))], false);
            let mut trace = Vec::new();
            for i in 0..300 {
                a.step(i as f32 / 60.0, 1.0 / 60.0);
                trace.push(a.playing.as_ref().map(|p| p.frame));
            }
            trace
        };
        assert_eq!(
            run(),
            run(),
            "branch selection must not depend on anything but the frame index"
        );
    }

    #[test]
    fn wraps_on_word_boundaries_and_cuts_long_words() {
        assert_eq!(
            wrap("the quick brown fox", 9),
            vec!["the quick", "brown fox"]
        );
        assert_eq!(
            wrap("supercalifragilistic", 8),
            vec!["supercal", "ifragili", "stic"]
        );
    }

    #[test]
    fn a_transparent_source_pixel_leaves_the_raster_alone() {
        let c = sheet_char();
        let mut frame_buf = Vec::new();
        c.compose(&c.anims["Wave"].frames[0], None, &mut frame_buf);
        assert_eq!(frame_buf, vec![255, 0, 0, 255]);

        let mut dst = vec![0u8, 0, 255, 255]; // one blue pixel
                                              // Blit the transparent half of the sheet: nothing should change.
        let clear = vec![0u8, 0, 0, 0];
        blit_scaled(&clear, 1, 1, &mut dst, 1, 1, 0, 0, 1, 1);
        assert_eq!(dst, vec![0, 0, 255, 255]);
        // Now the opaque half covers it.
        blit_scaled(&frame_buf, 1, 1, &mut dst, 1, 1, 0, 0, 1, 1);
        assert_eq!(dst, vec![255, 0, 0, 255]);
    }
}
