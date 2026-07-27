// Playing a game on the tube, live.
//
// The render path in `video.rs` drives a core as fast as the machine allows and writes
// the frames to a file. This is the same core host pointed the other way: one emulated
// frame per wall-clock frame, a real controller in your hands, and sound coming out —
// the emulator's picture arriving on the CRT's phosphor plane exactly as a screencast
// or a test pattern would.
//
// Three things have to be true for it to feel right:
//
//   * **pacing** — the core runs at its own rate (60.0988 Hz on a NES, 59.727 on a Game
//     Boy), which is not the monitor's. Emulation is advanced against the clock rather
//     than once per redraw, so the game runs at the speed it was written for however
//     fast the window refreshes.
//   * **input** — a gamepad if one is plugged in, the keyboard otherwise, read fresh
//     for every emulated frame rather than once per redraw. A frame that takes two
//     monitor refreshes still gets its own input poll.
//   * **sound** — the core hands us S16 stereo at a rate of its choosing, which is
//     resampled to whatever the audio device wanted and drained by the output stream.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::libretro::{button_bit, Core};

/// Keyboard fallback, chosen to stay clear of the CRT controls (P, G, I, M, B, Tab,
/// the digits and the brackets all already do something to the television).
const KEYMAP: &[(winit::keyboard::KeyCode, &str)] = {
    use winit::keyboard::KeyCode as K;
    &[
        (K::ArrowUp, "up"),
        (K::ArrowDown, "down"),
        (K::ArrowLeft, "left"),
        (K::ArrowRight, "right"),
        (K::KeyZ, "b"),
        (K::KeyX, "a"),
        (K::KeyA, "y"),
        (K::KeyS, "x"),
        (K::KeyQ, "l"),
        (K::KeyW, "r"),
        (K::Enter, "start"),
        (K::ShiftRight, "select"),
    ]
};

pub fn keymap_help() -> String {
    "arrows move · Z/X = B/A · A/S = Y/X · Q/W = L/R · Enter = Start · RShift = Select".into()
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

/// Interleaved stereo f32 at the device's rate, filled by the emulator thread and
/// drained by the audio callback.
type Samples = Arc<Mutex<VecDeque<f32>>>;

struct AudioOut {
    queue: Samples,
    device_rate: f64,
    channels: usize,
    /// Kept alive: dropping the stream stops playback.
    _stream: cpal::Stream,
    /// Fractional read position into the core's sample stream, for resampling.
    phase: f64,
    tail: [f32; 2],
    /// Roughly a tenth of a second of slack — enough to ride out a slow frame,
    /// little enough that the sound stays with the picture.
    target: usize,
}

impl AudioOut {
    fn open() -> Result<AudioOut> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no audio output device")?;
        let config = device.default_output_config().context("no output config")?;
        let device_rate = u32::from(config.sample_rate()) as f64;
        let channels = config.channels() as usize;

        let queue: Samples = Arc::new(Mutex::new(VecDeque::new()));
        let sink = queue.clone();
        let stream = device
            .build_output_stream(
                config.into(),
                move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut q = match sink.lock() {
                        Ok(q) => q,
                        Err(e) => e.into_inner(),
                    };
                    for s in out.iter_mut() {
                        *s = q.pop_front().unwrap_or(0.0); // silence on underrun
                    }
                },
                |e| eprintln!("[play] audio: {e}"),
                None,
            )
            .context("opening the audio stream")?;
        stream.play().context("starting audio")?;

        let target = (device_rate as usize / 10) * channels;
        eprintln!("[play] audio out: {device_rate} Hz, {channels} ch");
        Ok(AudioOut {
            queue,
            device_rate,
            channels,
            _stream: stream,
            phase: 0.0,
            tail: [0.0; 2],
            target,
        })
    }

    /// Push one frame's worth of the core's audio, resampling to the device rate.
    fn push(&mut self, pcm: &[i16], core_rate: f64) {
        if pcm.is_empty() || core_rate <= 0.0 {
            return;
        }
        let step = core_rate / self.device_rate; // core samples per device sample
        let frames = pcm.len() / 2;
        let mut q = match self.queue.lock() {
            Ok(q) => q,
            Err(e) => e.into_inner(),
        };

        // If we've drifted far ahead (a stall, or a core running fast), drop the
        // backlog rather than letting the delay grow without bound.
        if q.len() > self.target * 3 {
            let excess = q.len() - self.target;
            q.drain(..excess);
        }

        let at = |i: usize, ch: usize| -> f32 {
            if i == 0 {
                self.tail[ch]
            } else {
                pcm[(i - 1) * 2 + ch] as f32 / 32768.0
            }
        };
        while self.phase < frames as f64 {
            let i = self.phase.floor() as usize;
            let frac = (self.phase - i as f64) as f32;
            for ch in 0..2 {
                let a = at(i, ch);
                let b = if i < frames { pcm[i * 2 + ch] as f32 / 32768.0 } else { a };
                let v = a + (b - a) * frac;
                // Mono or surround devices just get the same signal in every channel.
                if self.channels >= 2 {
                    q.push_back(v);
                } else if ch == 0 {
                    q.push_back((a + b) * 0.5);
                }
            }
            for _ in 2..self.channels {
                q.push_back(0.0);
            }
            self.phase += step;
        }
        self.phase -= frames as f64;
        self.tail = [
            pcm[(frames - 1) * 2] as f32 / 32768.0,
            pcm[(frames - 1) * 2 + 1] as f32 / 32768.0,
        ];
    }
}

// ---------------------------------------------------------------------------
// The player
// ---------------------------------------------------------------------------

pub struct Player {
    core: Core,
    gilrs: Option<gilrs::Gilrs>,
    keys: u32,
    /// Seconds of emulation owed to the clock.
    owed: f64,
    last: std::time::Instant,
    audio: Option<AudioOut>,
    /// Newest frame, and whether it still needs uploading.
    pub frame: Vec<u8>,
    pub size: (u32, u32),
    pub fresh: bool,
    pub paused: bool,
    pub fps: f64,
    /// `CRTULUM_PLAY_STATS=1` reports the emulated rate and how much sound is
    /// buffered — the two numbers that tell you whether it's keeping up.
    stats: Option<(std::time::Instant, u32)>,
}

impl Player {
    pub fn new(rom: &Path, core_name: Option<&str>, options: &[(String, String)]) -> Result<Player> {
        let core_path = crate::libretro::find_core(Some(rom), core_name)?;
        eprintln!("[play] core {}", core_path.display());
        let system = crate::libretro::system_dir(Path::new("."));
        let core = Core::load(&core_path, Some(rom), &system, options)?;
        eprintln!(
            "[play] {} · {}x{} @ {:.3} fps",
            core.name, core.geometry.0, core.geometry.1, core.fps
        );

        let gilrs = match gilrs::Gilrs::new() {
            Ok(g) => {
                for (_id, pad) in g.gamepads() {
                    eprintln!("[play] controller: {}", pad.name());
                }
                Some(g)
            }
            Err(e) => {
                eprintln!("[play] no gamepad support ({e}) — keyboard only");
                None
            }
        };
        let audio = match AudioOut::open() {
            Ok(a) => Some(a),
            Err(e) => {
                eprintln!("[play] no audio ({e:#}) — running silent");
                None
            }
        };

        eprintln!("[play] {}", keymap_help());
        let fps = core.fps;
        Ok(Player {
            core,
            gilrs,
            keys: 0,
            owed: 0.0,
            last: std::time::Instant::now(),
            audio,
            frame: Vec::new(),
            size: (0, 0),
            fresh: false,
            paused: false,
            fps,
            stats: std::env::var_os("CRTULUM_PLAY_STATS")
                .map(|_| (std::time::Instant::now(), 0)),
        })
    }

    pub fn set_key(&mut self, code: winit::keyboard::KeyCode, down: bool) -> bool {
        let Some((_, name)) = KEYMAP.iter().find(|(k, _)| *k == code) else {
            return false;
        };
        let Some(bit) = button_bit(name) else { return false };
        if down {
            self.keys |= bit;
        } else {
            self.keys &= !bit;
        }
        true
    }

    /// Whatever is held right now, from the pad and the keyboard together.
    fn input(&mut self) -> u32 {
        let mut mask = self.keys;
        let Some(gilrs) = &mut self.gilrs else { return mask };
        // Drain the event queue so gilrs's own button states stay current.
        while gilrs.next_event().is_some() {}

        use gilrs::Button as B;
        const PAD: &[(B, &str)] = &[
            (B::DPadUp, "up"),
            (B::DPadDown, "down"),
            (B::DPadLeft, "left"),
            (B::DPadRight, "right"),
            (B::South, "b"),
            (B::East, "a"),
            (B::West, "y"),
            (B::North, "x"),
            (B::LeftTrigger, "l"),
            (B::RightTrigger, "r"),
            (B::LeftTrigger2, "l2"),
            (B::RightTrigger2, "r2"),
            (B::Start, "start"),
            (B::Select, "select"),
        ];
        for (_id, pad) in gilrs.gamepads() {
            for (button, name) in PAD {
                if pad.is_pressed(*button) {
                    if let Some(bit) = button_bit(name) {
                        mask |= bit;
                    }
                }
            }
            // Plenty of pads report the d-pad as an axis, and a stick should work as
            // one anyway.
            let (x, y) = (
                pad.value(gilrs::Axis::LeftStickX),
                pad.value(gilrs::Axis::LeftStickY),
            );
            const DEAD: f32 = 0.4;
            for (on, name) in [
                (x < -DEAD, "left"),
                (x > DEAD, "right"),
                (y > DEAD, "up"),
                (y < -DEAD, "down"),
            ] {
                if on {
                    if let Some(bit) = button_bit(name) {
                        mask |= bit;
                    }
                }
            }
        }
        mask
    }

    /// Advance emulation to catch up with the clock. Returns true if a new frame
    /// arrived that the tube hasn't seen yet.
    pub fn tick(&mut self) -> Result<bool> {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last).as_secs_f64();
        self.last = now;
        if self.paused {
            return Ok(false);
        }
        // A long stall (dragging the window, a hitch) must not turn into a burst of
        // fast-forward, so the debt is capped rather than repaid in full.
        self.owed = (self.owed + elapsed).min(0.25);

        let period = 1.0 / self.fps.max(1.0);
        let mut drew = false;
        while self.owed >= period {
            self.owed -= period;
            let mask = self.input();
            let (frame, w, h) = self.core.run_frame(mask)?;
            self.frame = frame;
            self.size = (w, h);
            drew = true;

            let pcm = self.core.take_audio();
            if let Some(audio) = &mut self.audio {
                audio.push(&pcm, self.core.sample_rate);
            }
        }
        if drew {
            self.fresh = true;
        }
        if let Some((since, count)) = &mut self.stats {
            *count += u32::from(drew);
            let elapsed = since.elapsed().as_secs_f64();
            if elapsed >= 1.0 {
                let backlog = self
                    .audio
                    .as_ref()
                    .map(|a| {
                        let n = a.queue.lock().map(|q| q.len()).unwrap_or(0);
                        (n as f64 / a.channels as f64 / a.device_rate * 1000.0) as u32
                    })
                    .unwrap_or(0);
                eprintln!(
                    "[play] {:.1} fps emulated (target {:.1}) · {backlog} ms of audio buffered",
                    *count as f64 / elapsed,
                    self.fps
                );
                *since = std::time::Instant::now();
                *count = 0;
            }
        }
        Ok(drew)
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        eprintln!("[play] {}", if self.paused { "paused" } else { "running" });
    }
}
