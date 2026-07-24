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
- `src/shader.wgsl` — the optics. Beam reconstruction, phosphor decay, refraction,
  masks, glass, PBR cabinet, the room it reflects. Tube curvature lives in
  `screen_z()` back in `main.rs`.
