# crtulum

A manipulable 3D CRT tube in a window. Point of the project: a Wayland app that
renders another app's output (RetroArch, a terminal, …) onto a hyper-real,
movable CRT tube — think "OBS window capture" piped onto a 3D Trinitron.

## Build & run

Needs a current Rust toolchain (system rustup, stable). Then:

```sh
cargo run -- --capture
```

Headless render (writes a PNG instead of opening a window — useful when the
compositor lacks wlr-screencopy so `grim` can't grab the window):

```sh
cargo run -- --shot out.png 1000x800
```

## Presets

Ten web-grounded tube presets (`--preset <name>`, default `trinitron`; live keys
**1–9,0**, **Tab** cycles):

| Key | Name          | Character                                             |
| --- | ------------- | ---------------------------------------------------- |
| 1   | `trinitron`   | consumer aperture grille, cylindrical                |
| 2   | `panasonic`   | consumer shadow (dot) mask, spherical                |
| 3   | `slotmask`    | consumer slot mask                                   |
| 4   | `rca`         | fuzzy warm shadow-mask console TV (low TVL)          |
| 5   | `pvm`         | razor-sharp broadcast aperture-grille monitor        |
| 6   | `arcade`      | coarse 15 kHz shadow mask, big scanlines             |
| 7   | `vga`         | fine shadow-mask PC monitor, flatter, cool           |
| 8   | `diamondtron` | dead-flat aperture-grille superbright PC monitor     |
| 9   | `green`       | monochrome P1/P39 green terminal, long persistence   |
| 0   | `amber`       | monochrome P3 amber terminal                         |

Each preset carries real-derived geometry, mask pitch, beam focus (TVL), convergence,
geometry error, white point, and persistence; the mono terminals emit a single
phosphor colour driven by luminance (no colour mask/convergence).

## Controls

| Input       | Action                 |
| ----------- | ---------------------- |
| left-drag   | orbit the tube         |
| scroll      | zoom                   |
| 1–9,0 / Tab | pick / cycle preset    |
| P           | power (warmup ↔ off-collapse) |
| G           | degauss                |
| I           | toggle 480i / 240p     |
| Esc         | quit                   |

Colour is real: each preset maps its measured phosphor gamut (SMPTE-C / P22 / sRGB)
and native white point (9300K reads blue, D65 neutral) through a CPU-computed CRT→sRGB
matrix. Signal path per tube — RGB/component (clean: PVM, arcade, PC), S-video
(sharp luma, band-limited colour), or composite (dot crawl + cross-colour +
colour bleed) — with NTSC-grounded Y/I/Q bandwidths. Power-off collapses the raster
to a bright line then a fading dot; power-on runs it in reverse with an auto-degauss.

## Where things live

- `src/main.rs` — window, wgpu setup, tube+bezel mesh, orbit camera, render loop.
  The M2 hook is `make_test_pattern()`: swap its output for a captured frame.
- `src/shader.wgsl` — the CRT optics (aperture grille, scanlines, vignette,
  Fresnel) and bezel material. Tube geometry (bulge/curvature) is in
  `screen_z()` in `main.rs`.
