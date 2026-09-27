# Calibration inputs and reproducible checks

The renderer now uses the following published inputs directly. These sources
provide measured material responses and hardware specifications; they do not
constitute an independent laboratory calibration of ten individual CRT specimens.

| Component | Primary source | Applied data |
| --- | --- | --- |
| Color phosphor time response | [Markus Kuhn, IEEE S&P 2002, equations 8–10](https://www.cl.cam.ac.uk/~mgk25/ieee02-optical.pdf) | Measured P22 response of a Dell D1025HE: exponential red components and exponential/power-law green and blue components. |
| Consumer P22 color | [Phosphor Technology Ltd material specifications](https://www.phosphor-technology.com/crt-phosphors/) | QKL63/N-C1 red (0.647, 0.343), GL29A/N-C1 green (0.310, 0.594), GL47/N-C2 blue (0.148, 0.062), CIE 1931 xy. |
| PVM geometry and color | [Sony PVM-20L5 brochure, specifications](https://www.adcom.it/public/images/pdf/pvm-20l5.pdf) | 388.4 mm effective picture width, 0.31 mm aperture grille, SMPTE-C, D65/D93. |
| Diamondtron geometry | [Mitsubishi Diamond Pro 930SB service manual, page 16](https://www.manualslib.com/manual/875365/Mitsubishi-Electric-Dpro930sb-Bk.html?page=16) | 0.24 mm grille, 366 mm full-scan width; the 356 mm factory setting is a different raster-size adjustment. |
| Scan velocity modulation | [US5600381A, circuit description](https://patents.google.com/patent/US5600381A/en) | Luminance differentiation drives auxiliary deflection. The shader inverts the resulting position map and derives deposited light from inverse velocity. |
| SD scan timing | [ITU-R BT.470-5](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S%21%21PDF-E.pdf) | 525-line/59.94-field NTSC timing and blanking. The raster model uses 262.5 lines per field and the README's 52.6 µs active-line convention. |

## Numerical verification

Run `python3 scripts/generate_phosphor.py` to regenerate the positive exponential
reservoirs and their machine-readable coefficients. The fit's impulse-response
error is below 0.14% over 1 microsecond to 1 second. The longest tail is represented
by a residual-energy reservoir; this is a numerical continuation, not an additional
measured decay curve. The measured response is shared by the color presets until
set-specific response measurements are available.

`cargo test -- --test-threads=1` checks the compiled shader coefficients against
the dataset, compares integrated response against independent closed-form
integrals, and reads GPU output to check scan timing and per-channel light.
Additional GPU checks exercise positive beam velocity, trajectory inversion,
transport energy, and neutral-field brightness. The old response retains its
existing persistence tests under the explicit legacy mode.

A newly opened warm tube initializes its reservoirs at the periodic steady state
of its initial signal. Starting with a blank signal instead starts the reservoirs
dark. Field exposures are integrated analytically; `--shutter` selects the open
fraction of each field. Exposures from multiple processed fields are combined for
an exported frame. Raster size changes recreate the reservoirs.

## Remaining measurement work

Online sources found here do not provide complete, set-specific measurements of
beam profiles, glass scatter, convergence, geometry, ambient contrast, and decay
for all ten presets. Existing sourced inputs and effects remain active. P1/P3
terminal persistence targets remain 50/13 ms to 10%; the newly found manufacturer
P1 color specification does not establish those decay times. PC/HD inputs retain
their native row counts with a normalized blanking budget; exact source modelines
are not yet supplied to the raster scheduler.

HDR output is confirmed working by the user, and HDR primary conversion is
numerically tested. Absolute luminance calibration is a separate measurement
task, not an HDR functionality gap. Panel subpixel layout, perceived BFI motion,
controller hardware, and audible quality require the corresponding equipment.
Web research cannot turn a missing measurement into a measured parameter.

## Connected display evidence (2026-09-26)

Read-only EDID inspection found HDR-capable XG27UCG and LG displays advertising
ST2084; the LG also advertises HLG. The user confirms that HDR works. The earlier
inference that a missing color-management protocol in `wayland-info` prevented
HDR was incorrect: that listing does not determine the application's Vulkan WSI
capabilities. The renderer selects an advertised floating-point HDR surface and
the backend negotiates linear sRGB or BT.2020, with shader primaries matched to
the configured swapchain. The application's surface-format and output-primaries
logs describe that rendering path. No compositor/display settings were changed.
