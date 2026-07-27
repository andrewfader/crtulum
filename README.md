# crtulum

A CRT you can hold in your hands, minus the 70 pounds of leaded glass and the risk
of the flyback transformer killing you in your garage.

It's a Wayland app that grabs another program's output — RetroArch, a terminal, a
browser, a video player, whatever — the same way OBS does window capture, and paints
it onto a 3D Trinitron
you can spin around with the mouse. Not a fullscreen filter. An actual tube, sitting
in your compositor, that you can orbit and zoom until the glare slides across the
glass the right way.

## Build & run

Current Rust toolchain (system rustup, stable). Then:

```sh
cargo run -- --capture
```

That pops the screencast picker. Point it at something. It lands on the tube.

No window? Take a picture instead — handy when your compositor won't do
wlr-screencopy and `grim` gives up:

```sh
cargo run -- --shot out.png 1000x800
```

## Exporting video

The other half of the app: hand it a source and it renders the whole thing through
the tube and out the far side as a video file. Same shader, same phosphor planes,
same presets — just offline, and pointed at ffmpeg instead of a window.

```sh
# any video file
cargo run --release -- --render clip.mp4 out.mp4

# a URL (anything yt-dlp handles)
cargo run --release -- --render 'https://youtu.be/…' out.mkv --preset rca

# a directory of stills
cargo run --release -- --render frames/ out.mp4 --fps 30

# a ROM, with the run itself scripted frame-by-frame (see below)
cargo run --release -- --render --rom smb.nes out.mp4 --script run.crts
```

Audio comes along from the source. The tube is driven at 60 fields/sec no matter
what frame rate you export at, so a 30 fps export still scans every frame twice and
480i still twitters. Signal resolution defaults to 480 lines (`--lines 240` for the
real thing — a CRT never saw a 1080p signal, and feeding it one dissolves the
scanline structure).

`--help` for the rest: `--size`, `--fps`, `--ssaa` (3 by default, `1` for a fast
preview), `--start`/`--duration`, `--codec x264|x265|vp9|ffv1`, `--crf`, `--no-audio`.
Roughly 500 fps at 640×480/ssaa 2 on an RX 9070 — most clips render faster than they
play.

### Scripts

A source alone gives you a static camera. A **script** gives you choreography — a
flat timeline of camera moves, tube swaps, power cycles and degausses:

```
size     1280x960
lines    240
preset   trinitron
camera   yaw=0.55 pitch=0.22 dist=3.4

at 0:00  power on                              # raster blooms open, auto-degauss
at 0:03  camera to yaw=-0.35 dist=3.0 over 6   # slow drift across the face
at 0:12  preset pvm                            # swap tubes mid-shot
at 0:14  exposure to 1.25 over 2
at 0:30  spin 1 over 10 linear                 # one full orbit
at 0:52  power off                             # collapse to a line, then a dot
```

```sh
cargo run --release -- --render clip.mp4 out.mp4 --script examples/demo.crts
```

Times are seconds or clock (`0:03`, `1:02:30.5`). Moves take `over <seconds>` and
ease by default (`linear` if you'd rather). Actions: `preset`, `camera`, `spin`,
`exposure`, `power on|off`, `degauss`, `interlace`, `subpixel`, `bfi`, `wait`. Set
`source` in the script and it's self-contained — `--render out.mp4 --script run.crts`.
Command-line flags override the script's setup lines, so one script works across
different sources and sizes. Typos are errors with a line number, not silent no-ops.

Downloads and emulator recordings are staged in `.crtulum/` next to the output, and
reused on the next run — so iterating on a script doesn't re-download or re-record.

### Scripting a run

Point the script at a ROM instead of a video and the same timeline drives the *run*
as well as the camera. crtulum loads a libretro core in-process and calls it one
frame at a time with the exact buttons that frame is scripted to hold — so it's
frame-exact, headless, deterministic, and faster than real time:

```
rom      smb.nes
core     nestopia          # optional; guessed from the extension
frames   3600              # how long to run (or `duration 60`)

preset   trinitron
camera   yaw=0.3 pitch=0.2 dist=3.2

at 0:00     power on
frame 150   press start              # momentary — 4 frames unless you say otherwise
frame 180   hold right               # …stays down…
frame 240   press a for 20 frames    # a precisely-placed jump, mid-hold
frame 300   release right
frame 330   tap b                    # exactly one frame
```

```sh
cargo run --release -- --render run.mp4 --script examples/tas.crts
```

`at <time>` is wall clock; `frame <n>` is exact — write the run in frames, write the
camera in seconds. Verbs: `press`, `hold`, `release`, `tap`, with `for <n> frames`
or `for <seconds>`. Buttons are the libretro names (`a b x y l r l2 r2 l3 r3 start
select up down left right`), several per line: `press a right`. Audio comes from the
core and is muxed in at the end. `CRTULUM_DEBUG_INPUT=1` prints the run as it plays,
one line per change, which is how you find out why a jump missed.

`examples/inputtest.nes` (built by `examples/make_test_rom.py`, ~70 bytes of 6502)
paints the screen a colour per button held, so a rendered run is a direct readout of
its own input timeline — that's how the frame-exactness above is tested rather than
asserted.

Pre-authored runs still work the other way: `--movie run.bsv` hands the whole thing
to RetroArch (`-P … --eof-exit -r`), which owns the emulation and input and records
a clip we then pipe through the tube. Use that for existing `.bsv`/replay files;
use `rom` + a script when you want to write the run here. The RetroArch pass runs in
real time in a window; the in-process path doesn't.

Cores are found in RetroArch's core directory (`--core` takes a name or a path). One
caveat: a libretro core is a shared library running in our process, so a core that
misbehaves takes the process with it — the mesen build on this machine segfaults when
hosted outside RetroArch, so the NES default order is nestopia, fceumm, quicknes,
then mesen.

### Which systems

Three rendering paths, and the core chooses: a software framebuffer, **OpenGL** via a
headless EGL context, or **Vulkan** via an instance and device crtulum stands up for
it — including the context-negotiation handshake, so the core builds the device with
the features its renderer needs. No window, no display server, still deterministic.

| | |
| --- | --- |
| **Verified here** | NES · SNES · Game Boy · Mega Drive · N64 · PlayStation on **both Vulkan and OpenGL** — each one boots a real game in `cargo test` (see below), plus frame-exact input against the homebrew ROMs in `examples/` |
| **Same class, should just work** | Game Boy Advance, Master System, Game Gear, PC Engine, 32X, Atari 2600, Lynx, Neo Geo Pocket, WonderSwan, ColecoVision |
| **Known not to work** | `mupen64plus_next` runs its emulator on its own thread and makes GL calls from there, where our context isn't current. Use `parallel_n64` instead |

The real-game tests need a library, which obviously isn't in the repo — point
`CRTULUM_ROMS` at yours (it looks for `nes/`, `snes/`, `gb/`, `megadrive/`, `n64/`,
`psx/` subdirectories) and `cargo test` boots one game per system, or skips if it
isn't there. They assert nothing about any particular game: only that the core loads,
the picture gains structure and moves, the rate and geometry are sane, and the core
tears down cleanly. A separate test runs the same ROM twice through a full unload and
reload and requires identical frames — the property scripted runs depend on.

A GPU core needs nothing special from you:

```sh
crtulum --render out.mp4 --rom game.cue --option swanstation_GPU_Renderer=Vulkan
```

Two things that made the difference, in case you hit them elsewhere: the frontend has
to provide the **log and performance interfaces** (cores call straight through those
pointers and crash if they're absent), and a Vulkan core negotiating its device looks
for a queue family that can *present* — with no surface that search fails and the core
records an out-of-range index it later trips over. `VK_EXT_headless_surface` gives it a
real surface with no window, and the whole class of problem goes away.

Core options are passed with `--option key=value` (repeatable), or `option key=value`
in a script — that's how you reach a core's renderer setting.
`CRTULUM_TRACE_ENV=1` lists every option key a core asks for, and `CRTULUM_CORE_LOG=1`
shows the core's own log, which is usually where the real answer is.

Ambiguous extensions are left to you: `.bin` and `.iso` belong to half a dozen
machines, so those need `--core`.

## Presets

Ten tubes, each one measured off real hardware — actual stripe pitch, actual TVL,
actual white point. `--preset <name>` (default `trinitron`), or keys **1–9,0** live,
**Tab** to cycle.

| Key | Name          | What it is                                           |
| --- | ------------- | ---------------------------------------------------- |
| 1   | `trinitron`   | the one everybody remembers — aperture grille, cylindrical |
| 2   | `panasonic`   | consumer shadow mask, spherical face                 |
| 3   | `slotmask`    | slot mask, the awkward middle child                  |
| 4   | `rca`         | warm, fuzzy console set your grandparents owned      |
| 5   | `pvm`         | the broadcast monitor you couldn't afford            |
| 6   | `arcade`      | coarse 15 kHz mask, scanlines you can count          |
| 7   | `vga`         | fine-pitch PC monitor, flatter, colder               |
| 8   | `diamondtron` | dead-flat aperture grille, blindingly bright         |
| 9   | `green`       | P1 green phosphor, long afterglow, terminal vibes    |
| 0   | `amber`       | P3 amber, same energy, warmer                        |

The Trinitron even has its damper wires — those two faint horizontal shadows across
the screen that drove people nuts and that nobody could explain.

## Controls

| Input        | Does                                    |
| ------------ | --------------------------------------- |
| left-drag    | orbit the tube                          |
| scroll       | zoom                                    |
| 1–9,0 / Tab  | pick / cycle preset                     |
| P            | power (warm-up, or collapse to a dot)   |
| G            | degauss                                 |
| I            | 480i / 240p                             |
| M            | subpixel mask (Megatron) / gaussian     |
| B            | black-frame insertion (needs 100 Hz+)   |
| `[` / `]`    | exposure trim (for HDR panels)          |
| Esc          | quit                                    |

## What's actually going on in there

Short version: it's not a texture with a scanline overlay. The light is simulated.

**Color is real.** Each tube runs its measured phosphor gamut (SMPTE-C, P22, sRGB)
and native white point through a CRT→sRGB matrix computed on the CPU. 9300K reads
blue the way a cheap TV did; D65 stays neutral. The greens desaturate exactly as
much as SMPTE-C says they should.

**The beam scans.** Two render passes: one integrates the picture into an HDR
phosphor plane with real per-channel decay (red lingers, blue snaps off in under a
millisecond — that's why fast motion trails warm), the other reconstructs the
electron beam from the source scanlines. Bright spots bloom and merge; saturated
colors stay thin with the gaps open. Leave a bright object moving and it drags a
fading tail, because the tube is genuinely remembering the last few fields.

**The glass is glass.** Snell refraction bends the view ray through the faceplate
to the phosphor behind it, traced separately per color channel, so you get real
chromatic fringing toward the corners. It's a mirror, too — dark screen catches a
daylight window and the room, and they slide across as you orbit. That last part
came straight off studying photos of real sets; a CRT head-on isn't black, it's a
4% mirror of whatever's lit in front of it. Bright content gets two separate glows:
a tight warm halation off the phosphor and a wider, softer diffusion haze scattering
through the thick glass — which is where CRT light gets its density.

**The consumer sets cheat, on purpose.** Composite and S-video tubes run scan
velocity modulation — the old Sony trick of goosing the beam speed at edges to fake
sharpness, complete with the bright overshoot halo videophiles complained about for
twenty years. The broadcast PVM, fed clean RGB, doesn't bother, so it stays honest
and razor-flat. Hit **M** for subpixel mask mapping, which lands each simulated
phosphor on a real panel subpixel for maximum density at native resolution, or **B**
for black-frame insertion, which strobes the tube dark between frames so motion snaps
like an actual CRT instead of smearing like an LCD (you'll want a 120 Hz panel).

**The signal path is period-correct.** RGB and component stay clean (PVM, arcade,
PC monitors). S-video keeps sharp luma but band-limits color. Composite gets the
full indignity — dot crawl, cross-color, bleed — tuned to real NTSC Y/I/Q
bandwidths. So the Panasonic smears its reds the way composite did and the PVM
doesn't.

**Plus the small stuff nobody asked for.** Deflection geometry errors (pincushion,
keystone, corner defocus that only the cheap tubes show), convergence drift toward
the edges, purity blotches a degauss actually clears, overscan eating the picture
edges, a rolling hum bar from beating against 59.94 Hz, analog grain, halation, and
a power switch that collapses the raster to a bright line, then a dot, then nothing
— and runs it backward with a degauss burst on the way up.

The cabinet's a real one too: a deep, near-cubic charcoal consumer set modeled on a
Sony KV-20TS20, chin grille and knobs and all, lit by a small HDR room so the plastic
and glass catch highlights instead of looking like a screensaver from 1999.

## HDR

If you've got the panel for it, it'll drive true HDR — BT.2020 linear, compositor
does the transfer, beam cores and speculars pushed past 1.0 so they actually glow.
This is the fussiest part on Linux and it took a vendored wgpu-hal patch to get the
colorspace mapping right. Use `[` / `]` to trim exposure to taste.

## Where things live

- `src/main.rs` — window, wgpu, tube + cabinet mesh, orbit camera, the two-pass
  render loop, all ten presets.
- `src/capture.rs` — the screencast portal handshake and PipeWire loop that feeds
  live frames onto the tube.
- `src/video.rs` — the `--render` export: the script DSL and its timeline, source
  acquisition (yt-dlp, RetroArch), the ffmpeg pipes, and the GPU SSAA resolve.
- `src/shader.wgsl` — the optics. Beam reconstruction, phosphor decay, refraction,
  masks, glass, PBR cabinet, the room it reflects. Tube curvature lives in
  `screen_z()` back in `main.rs`.
- `src/libretro.rs` — the in-process libretro host: loads a core, runs it a frame at
  a time with a scripted button mask, hands back RGBA frames and PCM.
- `src/glctx.rs` — the headless EGL/OpenGL context that hardware-rendering cores draw
  into, plus the readback.
- `src/vkctx.rs` — the Vulkan equivalent: instance, device, the negotiation handshake,
  the `retro_hw_render_interface_vulkan` callbacks, and the image copy back to RGBA.
- `examples/demo.crts` — a commented script showing every action.
- `examples/tas.crts` + `examples/make_test_rom.py` / `make_genesis_test_rom.py` — a
  scripted run, and the homebrew NES and Mega Drive ROMs it's verified against.
