# README implementation audit

The application has implementations for the major advertised workflows. This is
not a claim that every core, compositor, panel, character file, or video codec has
been tested. The table distinguishes code coverage from equipment-dependent checks.

| Advertised behavior | Implementation and verification |
| --- | --- |
| Wayland capture | `capture.rs`: portal selection, PipeWire frame copy; offset, stride and chunk-bound tests; an interactive portal session is still needed to verify the picker end to end. |
| 3D tube, cabinets, orbit, zoom, presets | `main.rs`: mesh generation, ten presets, window input, headless PNG rendering. |
| Live ROMs, keyboard/controller, pause, audio | `libretro.rs`, `play.rs`; core/input tests; mono/surround mapping is tested; integer and float device formats are supported. Desktop runs detected the connected controllers and initialized 48 kHz stereo output; keyboard controls were exercised. Physical controller presses and audible quality still require direct checks. |
| Software, EGL and Vulkan cores | `libretro.rs`, `glctx.rs`, `vkctx.rs`; EGL test and optional real-ROM integration tests. GPU core compatibility is core-specific. |
| Video, URL, stills, replay, ROM export | `video.rs`: ffmpeg pipelines, yt-dlp cache, RetroArch recording, direct libretro frame loop. All four encoders, audio padding, WebM/Opus, seek, mixed still formats, CLI overrides, ROM and Agent exports are exercised by `scripts/verify_exports.py`. |
| Timeline and exact button input | Parser, `Timeline`, `InputTrack`; tests cover examples, timing, holds/taps, interpolation, invalid syntax, actual-core-rate ordering, interrupted moves and degauss state. |
| Agent sprites, ACS, speech, balloons, mixing | `agent.rs` and `crates/acs`; character/animation/compositing tests. Voices require a configured synthesizer, and artwork is supplied separately. |
| Preset gamut and white point | CPU primary/white matrix; GPU gun response undoes sRGB exactly and applies gamma 2.4. No second arbitrary warm tint. |
| Blooming spot and horizontal held pixels | Normalized vertical generalized Gaussian; horizontal integration of each held pixel through a current-dependent Gaussian, separately per primary. |
| Interlace and field cadence | Independent 60000/1001 Hz field clock in live/export paths; tests at 24–144 fps plus GPU alternating-row readback. Export historical fields retain the preceding source instead of seeing a future frame. Uniform updates are submitted separately for each field. |
| Persistence | Published P22 response integrated with positive decay reservoirs; GPU readbacks match independent analytic impulse integrals. Legacy extended trails retain their regression checks. |
| Mask, damper wires, geometry, convergence | Faceplate-coordinate patterns and per-preset geometry/focus parameters in WGSL; mask mean/pitch and beam normalization tests, including GPU energy integration of a defocused spot. |
| Refraction and dispersion | One refracted ray per primary into an approximate phosphor plane; not a curved multilayer optical trace. |
| Glass, halation and diffusion | Surface reflections, diffuse ambient term, two scatter scales. Per-channel scatter fractions remove the same energy they add in a uniform-field kernel. |
| Composite and S-video | YIQ modulation/demodulation and finite Gaussian filters; fixed NTSC active-line timing, four-field carrier phase, phase alternation per scanned field line, native-pixel hold. These are approximate filters, not a circuit simulation. |
| SVM, sag, hum and grain | Luminance-driven auxiliary beam displacement, inverse trajectory and dwell-time deposition; GPU checks verify positive velocity, inversion and energy. APL supply loss, mains/field beat and voltage-domain noise remain. |
| Power and degauss | Warm-up/collapse animation, gun cutoff, fading history, degauss distortion and persistent clearing of residual purity. Power animation is stylized, not a simulated flyback/heater circuit. |
| Fullscreen, glare/reflection, exposure, BFI | Live controls and shot environment switches; BFI gates emission while leaving reflections visible. |
| HDR | User-confirmed working HDR. Linear sRGB or BT.2020 output matched to the configured swapchain, with vendored Vulkan colorspace mapping and numerical primary-conversion tests. SDR uses a presentation tone map, so exported pixel values are not calibrated luminance measurements. |

## Parameter provenance

Physical inputs remain in the renderer; “representative” does not mean that every
parameter is arbitrary. The distinction is between a measured/specification input
and applying that input to a modeled tube family.

| Input | Evidence and use |
| --- | --- |
| PVM grille and phosphors | [Sony's PVM-L5 brochure](https://consolemods.org/wiki/images/c/c1/Sony_PVM-20L5_PVM-14L5_Brochure.pdf) specifies 0.31 mm AG pitch, SMPTE-C phosphors and selectable D65/D93. The preset uses 0.31 mm, SMPTE-C and D65; its 388.4 mm effective width now follows the brochure directly. |
| Consumer mask pitches | [Samuel Goldwasser's direct scale measurements](https://www.repairfaq.org/sam/crtfaq.htm) report 0.75 mm for a 19-inch Samsung and 0.90 mm for a 25-inch RCA, both slot masks. The slot preset uses the latter; RCA borrows it for a dot-mask model. Panasonic borrows the former for its dot-mask model. The measured horizontal spacings do not establish the modeled dot geometry. |
| Other mask geometry | Trinitron 0.66 mm is estimated; arcade 0.63 mm is scaled from a reported larger-tube pitch; VGA 0.28 mm and Diamondtron 0.24 mm represent monitor classes. Assumed picture widths remain explicit in `Preset.screen_mm`. None of these is a new measurement. |
| Phosphor decay | Color uses the published P22 response described in [calibration sources](calibration-sources.md), including source scope and fit accuracy. Green/amber retain their 50/13 ms T10 targets. The 18/6/4 ms extended response remains selectable as legacy mode. |
| Signal noise | Assumed voltage SNR targets are 36 dB consumer composite, 42 dB S-video, 52 dB RGB/component and 64 dB PC/mono. Uniform peak-to-peak amplitude is sqrt(12) × 10^(-SNR/20), before clipping and gun transfer. These are link-quality scenarios, not measurements implied by connector type. |

The README's mask counts follow the current dimensions: Trinitron 606, PVM 1253,
Diamondtron 1525 horizontal periods. Geometry, focus and glass parameters remain
approximations; published TVL is not itself a measured beam profile.

## Model boundaries

- Presets are tube-family approximations. Mask dimensions have nominal hardware
  referents; focus, purity, supply regulation and room illumination have not been
  fitted to a complete set of lab measurements. P22 is a family of formulations,
  not a unique decay curve or gamut.
- The raster model times excitation across each active line, reserves blanking
  intervals, and integrates per-channel light analytically over a field shutter.
  Multiple processed fields contribute to each output exposure. Exact PC/HD
  modelines and eye-pursuit integration remain separate measurement/model work.
  Interlace alternates existing rows without silently resizing the input.
- Live capture/player polling exposes the latest available frame, not a timestamped
  frame queue. Catch-up fields use the previously held source before polling the
  next frame; intermediate source motion lost during a stall cannot be recovered.
  Latching after presentation adds up to one presentation interval of latency.
- Video frames are decoded at the requested export rate and held for field steps.
  That cannot recover motion samples absent from the input/export sampling. ROM
  emulation retains the core's own rate.
- The horizontal spot is Gaussian; the vertical profile is generalized Gaussian.
  Horizontal stripe filtering uses analytic Gaussian Fourier coefficients, pixel-box integration and a nonnegative spectral reconstruction window; vertical dot/slot and scanline filtering remain approximate.
  The phosphor planes are not supersampled at physical dot-pitch resolution.
- Scattering uses finite spatial taps, and some geometry/brightness terms are
  evaluated downstream. Kernel conservation does not imply a globally calibrated
  radiometric solution across screen boundaries and viewing angles.
- Subpixel mode assumes RGB stripes and native pixel mapping; OLED/BGR layouts and
  compositor scaling need different mappings. BFI needs sufficient refresh rate.
- In-process ROM exports cache native frames and PCM. Keys include ROM/core
  content, referenced CUE/M3U tracks, BIOS files, core options, compiled input
  events and run length. Camera, preset and output codec changes reuse the run.
  A complete recording is published atomically; interrupted recordings are not
  cache hits. External RetroArch replay configuration remains an independent input
  to its existing replay cache.


PNG sequence export (`--codec png`, also exposed as `--clip`) uses the same
export loop and GPU resolve as video. The export checks compare both spellings
byte for byte at 24 fps with monochrome persistence enabled.

## Running checks

`cargo test -- --test-threads=1` runs the main suite. GPU checks unconditionally
require a discrete or integrated Vulkan GPU using the application's automatic
selection policy. There is no requirement switch or software fallback.
`python3 scripts/verify_exports.py` checks a built binary under that same policy.
CI runs on GitHub-hosted `ubuntu-latest`. An explicit `ci-software-vulkan`
build feature permits headless numerical and export checks on Mesa lavapipe.
It does not enable software rendering in normal builds or windowed rendering,
and no test silently skips a missing Vulkan adapter. CI binaries use a separate
`target/ci-vulkan` directory. These checks do not establish physical GPU behavior.
`cargo test --manifest-path crates/acs/Cargo.toml` checks the native ACS reader.
Real-game tests require `CRTULUM_ROMS` and installed cores; a passing skipped test
is not evidence that the corresponding platform booted here.

## Earlier verification (2026-09-26)

- Main suite: 45 passed, 3 diagnostic tests ignored, with
  `CRTULUM_REQUIRE_GPU=1` and Mesa lavapipe selected explicitly.
- The separately invoked ignored one-frame input-timing test passed.
- Real games booted and produced moving pictures on NES, SNES, Game Boy,
  Mega Drive, N64, and PlayStation through both Vulkan and OpenGL. The NES
  unload/reload determinism check passed. These used the local ROM/core library.
- Native ACS reader: 1 test passed.
- `scripts/verify_exports.py`: x264, x265, VP9/WebM and FFV1 with audio; muted
  trimmed video; script/CLI overrides; mixed PNG/JPEG/BMP/TGA stills; a sought
  homebrew ROM run; Agent speech mixed into WebM; invalid codec rejection.
  ffprobe checked decoded frame counts, audio presence and output color metadata.
- Video output now explicitly converts RGB to the BT.709 YUV matrix and tags
  the retained sRGB transfer; it no longer labels an implicit BT.601 conversion
  as BT.709. Short audio beds are padded so they cannot truncate the picture.

Reproduce the media checks after `cargo build`:

```sh
python3 scripts/verify_exports.py
```

The media checks also run in CI. Optional Agent/ROM cases explicitly report
`SKIP` without local artwork/cores. Physical HDR output, interactive portal
selection, controller events and listening tests are not certified by these
headless checks. Desktop access outside the sandbox exposes an AMD Radeon RX 9070; the sandbox itself exposes no `/dev/dri`.

### Desktop retry

The first hardware run exposed a GPU reset during warm-up: inverse power-mapped
signal coordinates drove defocus, making the beam convolution loops enormous.
Defocus now uses physical faceplate position. A subsequent 90-second hardware
run sustained approximately 60 emulated fps (NES target 60.099). The full beam
convolution remains enabled. Desktop connection/window failures now exit with a
clear error instead of panicking.

A live uniform-gray screenshot also exposed colored mask moire. The stripe filter
now evaluates the Gaussian Fourier series with pixel integration and a nonnegative
Nyquist-limited reconstruction window. GPU readbacks check mean transmission at
three scales and constant per-pixel output for unresolved stripes.

Native Wayland and XWayland windows rendered on COSMIC. Targeted input confirmed
preset, interlace, mask, glare/reflection, fullscreen, exposure and pause/resume
controls. Compositor screenshots confirmed that holding the NES A button changes
the uniform-gray test picture to blue, power-off removes emission while retaining
reflections, and power-on restores the picture. Stereo audio initialization succeeded; this is not an audible-quality
test. That particular run offered SDR formats; it does not establish an HDR
failure. The user subsequently confirmed working HDR. The export
checks additionally render RCA warm-up at 0 and 0.25 and full collapse, with a
subprocess deadline to catch runaway workloads.


## Full-history review (2026-09-26)

[The commit ledger](history-review.md) traces all 35 commits and the corrections.
The current test suite additionally checks HDR primary conversion, linear cabinet
bounce and BFI, finite-kernel −3 dB cutoffs, Agent percentage branches and timing
across 10–120 fps, native odd/tall image rasters, clip/script precedence, and analog
axis conversion. The expanded export verifier renders all ten presets and compares
PNG output through both command spellings.

Full-history results before the hardware-only policy change (software Vulkan/llvmpipe):

- Full application suite: **54 passed, 3 ignored**. Concurrent and serial runs
  passed. Real games ran on NES, SNES, Game Boy, Mega Drive, N64, and PlayStation
  through both Vulkan and OpenGL; these were not skipped in this run.
- The normally ignored 600-frame single-frame input-tap check also passed when
  explicitly selected. The other two ignored cases are interactive diagnostics.
- ACS crate: its decompression test passed.
- Release build completed. The release-binary export verifier passed all ten presets, power
  transitions, x264/x265/VP9/FFV1 media and audio, trimming, script precedence,
  still sequences, identical clip/render PNGs, native/odd rasters, diagnostic
  sampling intervals, a real-core homebrew export, and an Agent export.

An earlier full-suite run failed the phosphor inactive-row assertion once. A
targeted rerun and subsequent concurrent and serial suites passed without relaxing
the assertion; the failure has not been explained. The assertion now reports the
field and all rows to aid diagnosis if it recurs. Passing reruns do not establish
that an intermittent defect is fixed.

This run did not validate physical HDR output, gamepad hardware, audible quality,
or real-time throughput on a desktop GPU. In particular, the corrected signal
quadrature increases sampling work; the earlier desktop timing above predates
that change. Export has no real-time pacing limit, but is not guaranteed to run
faster than real time at every resolution or quality setting.

## Physical Vulkan verification (2026-09-26)

The subsequent hardware-only policy removes the optional GPU requirement flag,
CPU/virtual adapter selection, and the export verifier's software-driver override.
Window, shot, export and test rendering all enumerate Vulkan adapters and select
physical hardware automatically; Vulkan libretro devices must also be physical.

On the host, default Vulkan discovery selected **AMD Radeon RX 9070 / RADV**.
This agent session inherited a stale NVIDIA ICD override; verification unset that
override rather than pinning an AMD driver. The sandbox itself has no GPU device,
so hardware checks ran on the host. The main suite passed **57 tests, 3 ignored**,
including all numerical GPU checks and the available real-core tests. The full
hardware export verifier passed on both debug and rebuilt release binaries, including source, ROM and Agent audio in videos
and PNG-sequence sidecars. A deliberately software-only invocation exited with
status 1 and the physical-GPU requirement message before writing a screenshot.
The live desktop window also selected the Radeon automatically and presented
through Vulkan; the compositor offered SDR formats only.

The host's `ffprobe` command is normally a Firejail wrapper that cannot see these
temporary test files. Export verification used `/usr/bin` binaries directly.
CI targets GitHub-hosted `ubuntu-latest` using the explicit headless test build
described above. No self-hosted runner is required. Physical HDR signaling and
audible/controller hardware behavior remain outside those CI checks.

The GitHub-hosted CI configuration was exercised locally with the explicit
`ci-software-vulkan` build and lavapipe: 60 tests passed, 3 were ignored, and
the complete export verifier passed. The two default-build adapter-policy tests
also passed. The normal build rejects CPU rendering, which the workflow checks
before building its CI-only renderer. These are local results; a remote GitHub
Actions run has not been triggered here. Coverage instrumentation was not rerun.

## Measured response and recording cache

See [calibration sources](calibration-sources.md) for published inputs, the
regenerable response dataset, numerical/GPU validation and the measurements still
needed to establish a complete calibration of every preset. Export checks now
exercise recorded-ROM reuse, byte-identical cached PNGs and input invalidation.
