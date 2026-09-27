# Commit history and feature preservation review

Reviewed 2026-09-26: all 35 reachable commits, from `22e279e` through
`8d98343`, plus the working-tree corrections. This review traces changes to
current implementations and tests; it is not a claim of physical calibration or
universal compatibility with every core, compositor, or character file.

## Commit ledger

| Commit | Contribution and current disposition |
| --- | --- |
| `22e279e` | Initial manipulable tube, ten presets, capture, shots, masks, color matrices, power, degauss, interlace, and HDR backend patch remain. Float-surface primaries needed the correction below. |
| `7a50baf` | Reduced exaggerated internal ghosting and dispersion; both optical effects remain. |
| `402fd4b` | Deep cabinet, shorter presentation queue, short capture lock, corner defocus and per-primary persistence remain. Frame-scale color trails were subsequently replaced by physical-scale decay; persistence itself was retained. |
| `172be4b` | Window reflection, exposure, scanline and refresh-band work remain. |
| `23bc8bc` | Signal filtering in cycles per active line remains; native capture width does not set the analog bandwidth. |
| `0e533ee` | Rust CI remains and now exercises GPU and encoded exports. |
| `00e528d` | SVM, wide diffusion, subpixel-mask control and BFI remain. |
| `304dbd4` | Motion/beam tests and PNG-sequence export remain. PNG export now shares the video loop; lost native image dimensions and script defaults were restored. |
| `6202da1` | Monitor refresh detection and re-detection on BFI toggle remain. |
| `1a52af2` | Per-preset cabinets, materials, speaker placements, bevels, controls and vents remain. |
| `28d64af` | Software/GL/Vulkan libretro hosting, live play/audio, timeline export, replay, downloads and homebrew input ROMs remain. Live analog forwarding was missing and is now connected. |
| `d2d5aed` | System-support and core troubleshooting documentation retained, with conditional-test limitations stated. |
| `c03c339` | Generalized beam profiles, conservation, glass transmission, asymmetric chroma filtering and mono persistence remain. Mono decay now distinguishes e-folding time from time to 10%. |
| `5d3c3e6` | Mask normalization, restrained convergence, supply sag, signal-dependent noise and slower hum remain. Additive highlight gain was replaced by existing conserving spot/scatter mechanisms. |
| `f03af55` | NTSC carrier/line timing, four-field phase, receiver bandwidth and separate luma trap remain. The bandwidth-to-Gaussian calculation needed the correction below. |
| `0e92f64` | Scripted characters, branching, idle/directional animations, balloons, speech and audio mixing remain. Percentage branching and short-frame timing needed corrections. |
| `671210f` | Physical mask pitches, footprint filtering, current-dependent profile shape, held source pixels, linear-light optics and physical scatter distances remain. New horizontal integration replaces the one-fetch approximation without deleting spot blur. |
| `972269a` | Project license and CI adjustments retained. |
| `cc6d4c5` | CI system dependencies retained. |
| `667e3d8` | Native ACS palettes/layers, sounds and mouth overlays, and scripted analog input retained. |
| `8bcd995` | Release build-script assertion override and tracked release-artifact convention retained. |
| `adc000a` | CI release-build prerequisites retained. |
| `3513526` | Vulkan coverage retained on GitHub-hosted CI through an explicit headless software-Vulkan test build; ordinary builds require physical hardware. |
| `7de0084` | Molded cabinet finish, independent glare/window controls, fullscreen and vendored binding fixes retained. |
| `64ef7da` | Cabinet lighting, contact/seam shading and unified output transform retained. Screen illumination no longer applies average brightness twice. |
| `6eea658` | README screenshot added; subsequent screenshots supersede it without removing rendering behavior. |
| `d816c6a` | ROM-plus-Agent screenshot example retained in the current gameplay illustration. |
| `9991f92` | Gameplay screenshot revision; no runtime feature changes. |
| `1969419` | Versioned screenshot asset; current README still references a tracked asset. |
| `ec4dab0` | Sonic illustration superseded by Donkey Kong Country; emulation/Agent functionality retained. |
| `7b440c6` | Failed-core-load cleanup, GPU test handling and build/dependency fixes retained. |
| `1ce8527` | Recoverable GPU initialization errors retained. |
| `9c89054` | Donkey Kong Country screenshot remains the README illustration. |
| `b04388e` | Release-binary refresh; source behavior checked against current builds. |
| `8d98343` | Screenshot update; no source behavior removed. |

## Corrections from this review

- **Automatic hardware selection:** windows, still shots, exports and GPU tests
  share a Vulkan-only selector that prefers discrete over integrated hardware.
  Ordinary builds reject software/virtual devices, and missing adapters fail
  rather than skipping tests. Vulkan core negotiation also enforces physical
  hardware. GitHub-hosted CI explicitly builds with `ci-software-vulkan` to run
  headless numerical/export checks on lavapipe; this is not a runtime fallback
  in the normal application. No self-hosted runner or GPU-enabling flag is needed.
- **HDR primaries:** the Vulkan backend can select either linear sRGB or BT.2020
  for the same float texture format. It now exposes the configured swapchain
  color space; the renderer applies the BT.2020 matrix only when appropriate,
  including after reconfiguration. Neither HDR color-space option was removed.
- **Cabinet illumination:** source statistics are already linear light. Multiplying
  them by APL again squared midtone intensity. Bounce now uses that light once,
  applies the tube's color matrix, and respects BFI. Room illumination/reflections
  remain present on a dark BFI frame.
- **Signal bandwidth:** the Gaussian conversion used `sqrt(2 ln 2)` for a −3 dB
  amplitude cutoff, giving −6 dB instead. It now uses `sqrt(ln 2)`, keeps the
  1.3/0.4 MHz encoder, 0.5 MHz receiver, 3 MHz composite luma and 4 MHz S-video
  luma targets, and treats Q=10 as a full trap bandwidth. Finer quadrature and
  longer support preserve these finite-filter responses. This costs more signal
  sampling; it does not replace the analog model with a cheaper effect.
- **PNG sequence parity:** `--clip` continues through the shared export renderer.
  Default flags no longer override script settings; native still pixels and
  per-image dimensions are preserved unless resizing is explicitly requested.
  PNG output preserves odd dimensions. Encoded YUV video retains codec alignment.
  `CRTULUM_DT` remains available as the clip sampling interval, subordinate to an
  explicit CLI/script fps; it no longer distorts decay independently of time.
- **Controller completeness:** live left-stick values now reach the existing
  libretro analog callback; L3/R3 are mapped. Digital controls and the existing
  stick-to-direction mapping remain. Scripted `stick`/`center stick` are documented.
- **Agent animation:** branch weights are percentages, with unused probability
  proceeding to the following frame, as described in
  [Microsoft's animation authoring documentation](https://learn.microsoft.com/en-us/windows/win32/lwef/creating-animations).
  Animation advancement now consumes every elapsed short/zero-duration frame,
  places commands at their timestamps, and mixes crossed-frame sounds at the
  animation boundary. Malformed zero-duration cycles terminate instead of hanging.
- **Script errors:** surplus stick/center/core-option tokens and empty option keys
  or values now report errors with the script line rather than silently succeeding.

## What was intentionally replaced

Removal of a parameter is not necessarily removal of its effect. The arbitrary
scanline-depth field was replaced by beam geometry and pixel-footprint filtering;
output-pixel mask pitch was replaced by physical dimensions; a second warm tint
was replaced by the existing native-white matrix; additive bloom was replaced by
energy redistribution. The generalized vertical spot, per-primary horizontal
spot, color matrices, mask dimensions, SVM, sag, internal reflection, refraction,
dispersion, halation, diffusion, room reflections and cabinet lighting remain.

The previous color persistence constants deliberately exaggerated real times to
make trails visible on an LCD. Restoring that exaggeration would regress accuracy.
The same applies to treating borrowed family parameters as measurements of ten
individual sets. The numerical hardware inputs and their provenance are retained
in [the accuracy audit](accuracy-audit.md#parameter-provenance).

The README previously reported roughly 500 export fps at 640×480 with 2×
supersampling on an RX 9070. That historical observation is retained here, but
its source/preset was not specified and it predates the corrected signal
quadrature. It cannot substantiate a current throughput guarantee.

## Verification boundaries

Headless tests verify numerical GPU output, signal-kernel frequency response,
script timing, core input, native image preservation, Agent timing/branching,
all ten presets, and actual encoded media. They cannot certify perceived motion
on a physical panel, compositor HDR signaling, audible quality, or actual gamepad
hardware events. Those checks remain explicit rather than being inferred from a
passing unit-test count.

The user confirms working HDR. The earlier absence of a color-management protocol
in a Wayland registry listing was not evidence of an HDR failure: the application
uses the negotiated Vulkan surface format and color space. Absolute luminance
calibration remains distinct from working HDR output.

The renderer still approximates within-field scanout, camera exposure, mask/scatter
filtering and power circuitry. The Agent host is not a complete implementation of
the Microsoft Agent request/return-animation queue. Native library/core tests are
conditional on installed assets. None of those boundaries was resolved merely by
changing README wording. See the current-run results in the accuracy audit.
