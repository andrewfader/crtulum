// crtulum CRT optics. One pipeline, two materials (screen = 0.0, bezel = 1.0).
//
// Screen path models: parallax (phosphor recessed behind thick glass), a selectable
// phosphor mask (aperture grille / shadow / slot) with gaussian subpixels, a
// brightness-dependent beam scanline, reddish halation, glass Fresnel reflection,
// and tube vignette.

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    cam_pos: vec4<f32>, // xyz = camera world position
    params: vec4<f32>,  // x=src_w, y=src_h, z=time, w=render_scale (SS factor)
    optics: vec4<f32>,  // x=mask_type(0 grille,1 shadow,2 slot), y=mask_strength, z=reserved, w=halation
    glass: vec4<f32>,   // x=faceplate thickness, y=reflection, z=vignette, w=mask triads across the face
    tone: vec4<f32>,    // x=hdr(0 tonemap→SDR, 1 scRGB passthrough), y=peak(white pt), z=beam_drive, w=ntsc_strength
    scan: vec4<f32>,    // beam math: x=beam_min(width, dark), y=beam_max(width, bright), z=beam_shape, w=beam_range
    env: vec4<f32>,     // xyz=avg source color, w=avg picture level (screen area-light)
    look: vec4<f32>,    // x=convergence, y=corner_radius, z=grain, w=ghost
    phys: vec4<f32>,    // x=crt_gamma, y=reserved, z=glow_bounce, w=HV sag
    temporal: vec4<f32>,// x=dt(sec), y=persist_mult, z=interlace, w=field_parity
    raster: vec4<f32>, // measured response, shutter fraction, exposure group, reserved
    ptau: vec4<f32>,    // per-phosphor decay tau: xyz = R,G,B (sec); w = power-law tail exponent
    geom: vec4<f32>,    // raster geometry: x=pincushion, y=trapezoid, z=corner_pin, w=purity
    mono: vec4<f32>,    // monochrome phosphor tint (rgb) + flag (w>0.5 = single-gun)
    cmat0: vec4<f32>,   // CRT-phosphor → sRGB colour matrix rows (real gamut + white pt)
    cmat1: vec4<f32>,
    cmat2: vec4<f32>,
    pwr: vec4<f32>,     // power: x=warmup, y=collapse, z=degauss, w=specular glare enabled
    focus: vec4<f32>,   // x=edge defocus (deflection spot growth), y=overscan(per side), z=roll rate, w=roll amp
    fx: vec4<f32>,      // x=svm (scan-velocity crispening), y=diffusion (wide glass glow), z=subpixel-mask flag, w=bfi screen multiplier
    camera: vec4<f32>, // enabled, tan(horizontal FOV/2), aspect, reserved
    camera_ambient: vec4<f32>, // mean linear webcam radiance
    beam2: vec4<f32>,   // x=spot profile exponent p at low beam current,
                        // y=window reflection enabled, z=ambient diffuse wash, w=scatter redistribution
};

@group(0) @binding(0) var<uniform> u: Uniforms;
// Tube pass: t_screen = the persisted phosphor plane. Accum pass: t_screen = the
// raw source frame and t_prev = the previous phosphor plane (fed back for decay).
@group(0) @binding(1) var t_screen: texture_2d<f32>;
@group(0) @binding(2) var s_screen: sampler;
@group(0) @binding(7) var t_camera: texture_2d<f32>;
@group(0) @binding(3) var t_prev: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> phos_bank0: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> phos_bank1: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> phos_bank2: array<vec4<f32>>;

fn history_load(index: u32) -> vec4<f32> {
    let term = index % 33u;
    // Store each reservoir as a contiguous image. Neighboring fragment lanes
    // access the same term, so this avoids a 176-byte stride between pixels.
    let offset = (term % 11u) * (u32(u.params.x) * u32(u.params.y)) + index / 33u;
    if (term < 11u) { return phos_bank0[offset]; }
    if (term < 22u) { return phos_bank1[offset]; }
    return phos_bank2[offset];
}
fn history_store(index: u32, value: vec4<f32>) {
    let term = index % 33u;
    let offset = (term % 11u) * (u32(u.params.x) * u32(u.params.y)) + index / 33u;
    if (term < 11u) { phos_bank0[offset] = value; }
    else if (term < 22u) { phos_bank1[offset] = value; }
    else { phos_bank2[offset] = value; }
}

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) material: f32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>, // fragment stage: framebuffer pixel coords
    @location(0) uv: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) material: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world = u.model * vec4<f32>(in.pos, 1.0);
    out.world_pos = world.xyz;
    out.clip = u.view_proj * world;
    out.uv = in.uv;
    out.world_normal = (u.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.material = in.material;
    return out;
}

fn gauss(t: f32, c: f32, w: f32) -> f32 {
    let d = t - c;
    return exp(-(d * d) / (2.0 * w * w));
}

// Cheap hash → [0,1) for animated analog grain (the noise floor of a real signal).
fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 345.45));
    q = q + dot(q, q + 34.345);
    return fract(q.x * q.y);
}

// Smooth, object-space value noise for molded materials. Unlike the old per-cell hash,
// this has no square pixel boundaries and does not swim with the camera. Triplanar
// projection keeps the pebble scale continuous across front, side and chamfer faces.
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

fn molded_noise(p: vec3<f32>, n: vec3<f32>, scale: f32) -> f32 {
    let an = pow(abs(n), vec3<f32>(4.0));
    let w = an / max(an.x + an.y + an.z, 1e-4);
    return value_noise(p.yz * scale + vec2<f32>(17.0, 41.0)) * w.x
         + value_noise(p.zx * scale + vec2<f32>(53.0, 11.0)) * w.y
         + value_noise(p.xy * scale + vec2<f32>(29.0, 67.0)) * w.z;
}

// Periodic Gaussian phosphor stripes, evaluated in the frequency domain.
// The coefficients are analytic: mean * exp(-2*pi^2*sigma^2*n^2).
// Integrate each harmonic over the pixel box (sinc), then reject frequencies
// the output grid cannot resolve. Merely adding footprint variance to sigma
// left substantial energy above Nyquist and produced colored moire on gray.
// Eight harmonics leave < 5e-8 absolute truncation error for sigma=0.105.
fn phosphor3(t: f32, fw: f32) -> vec3<f32> {
    let w = 0.105;
    let mean = w * sqrt(TAU);
    var result = vec3<f32>(mean);
    let phase = vec3<f32>(t) - vec3<f32>(1.0 / 6.0, 0.5, 5.0 / 6.0);
    for (var k = 1; k <= 8; k = k + 1) {
        let frequency = f32(k);
        let pixel_frequency = frequency * fw;
        // A triangular spectral window is a nonnegative reconstruction kernel:
        // unlike a sharp/tapered cutoff it cannot ring into negative light.
        let cutoff = max(1.0 - 2.0 * pixel_frequency, 0.0);
        let a = PI * pixel_frequency;
        var aperture = 1.0;
        if (abs(a) > 1e-5) { aperture = sin(a) / a; }
        let amplitude = 2.0 * mean * exp(-2.0 * PI * PI * w * w * frequency * frequency);
        result = result + amplitude * aperture * cutoff * cos(TAU * frequency * phase);
    }
    return result;
}

// Mean transmission of `mask()` over one full period, per channel — the DC term of the
// mask pattern, needed to normalise it to unit energy (see the call site in fs_main).
// Closed form rather than a numeric sum, because every factor is analytic:
//   * one stripe is a gaussian of sigma w summed over its periodic images, so over a unit
//     period its mean is exactly w·sqrt(2*pi) = 0.105 × 2.5066 = 0.2632;
//   * the shadow mask multiplies that by mix(0.35, 1, gauss(ty, 0.5, 0.30)), whose mean is
//     0.35 + 0.65 · sigma·sqrt(2*pi)·erf(0.5/(sigma·sqrt2)) = 0.35 + 0.65 × 0.6803 = 0.792;
//   * the slot mask multiplies by mix(0.45, 1, slot), and a smoothstep ramp of width a
//     integrates to a/2, so the duty is 1 − 0.12 = 0.88 → 0.45 + 0.55 × 0.88 = 0.934.
// The resulting transmissions — grille 0.263, slot 0.246, shadow 0.208 — land right on the
// published open-area figures for real masks (aperture grille ~22-25%, slot ~20%, shadow
// mask ~15-18%) — the grille a touch above the top of its range, the other two inside
// theirs, which is a good sign the stripe geometry above is honest.
fn mask_mean(kind: f32) -> f32 {
    let stripe = 0.26317;                      // 0.105 · sqrt(2·pi)
    if (kind < 0.5) {
        return stripe;                         // aperture grille
    } else if (kind < 1.5) {
        return stripe * 0.79220;               // shadow mask (dot triads)
    }
    return stripe * 0.93400;                   // slot mask
}

// Phosphor mask weights at triad coordinate `tc` (position on the faceplate measured in
// mask triads — see the call site), with `fw` the screen-space pixel footprint in the
// same units. The vertical structure of a dot/slot mask band-limits the same way the
// stripes do, but toward its own exact mean (the factors in mask_mean), so the DC the
// normaliser divides out stays correct at every scale.
fn mask(tc: vec2<f32>, fw: vec2<f32>, kind: f32) -> vec3<f32> {
    if (kind < 0.5) {
        // aperture grille (Trinitron): continuous vertical RGB stripes
        return phosphor3(fract(tc.x), fw.x);
    } else if (kind < 1.5) {
        // shadow mask: RGB dot triads, every other row staggered by half a triad
        let row = floor(tc.y);
        let stagger = select(0.0, 0.5, (i32(row) - (i32(row) / 2) * 2) != 0);
        let stripes = phosphor3(fract(tc.x + stagger), fw.x);
        let ty = fract(tc.y);
        let dot = mix(0.35, 1.0, gauss(ty, 0.5, 0.30));
        return stripes * mix(dot, 0.79220, smoothstep(0.15, 0.60, fw.y));
    } else {
        // slot mask (many consumer sets): vertical slots, columns staggered
        let stripes = phosphor3(fract(tc.x), fw.x);
        let seg = floor(tc.x);
        let stagger = select(0.0, 0.5, (i32(seg) - (i32(seg) / 2) * 2) == 0);
        let ty = fract(tc.y * 0.5 + stagger);
        let slot = mix(0.45, 1.0, smoothstep(0.0, 0.12, ty) * smoothstep(1.0, 0.88, ty));
        return stripes * mix(slot, 0.93400, smoothstep(0.15, 0.60, fw.y * 0.5));
    }
}

// Subpixel-accurate mask (the Sony-Megatron trick): instead of a resolution-
// independent phosphor pattern, light each REAL panel subpixel directly. On an
// RGB-stripe LCD every output pixel is one R, G or B subpixel, so a triad of the
// tube's phosphors maps onto three physical subpixels — the mask resolves as true
// per-subpixel light density at native resolution instead of averaging to a tint.
// Only meaningful at render_scale 1 on an RGB-stripe panel (a shot SSAAs it away).
fn mask_subpixel(px: vec2<f32>) -> vec3<f32> {
    let sp = i32(floor(px.x)) - (i32(floor(px.x)) / 3) * 3; // 0,1,2 across the LCD triad
    var w = vec3<f32>(0.06);        // tiny floor so black subpixels aren't dead
    if (sp == 0) { w.r = 1.0; } else if (sp == 1) { w.g = 1.0; } else { w.b = 1.0; }
    return w;
}

// Extended Reinhard tonemap: identity-ish below 1.0, rolls HDR highlights up to
// `peak` (the white point) back into the displayable [0,1] range for SDR output.
fn tonemap(c: vec3<f32>, peak: f32) -> vec3<f32> {
    let w2 = max(peak * peak, 1.0);
    return (c * (1.0 + c / w2)) / (1.0 + c);
}

// ACES filmic tonemap (Narkowicz 2015 fit): a filmic S-curve with a graceful highlight
// shoulder and a slight toe — far more photographic HDR→SDR rolloff than Reinhard, and
// it keeps saturated bright colours from clipping to flat white. Input linear HDR.
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Unified HDR tonemapping and colorimetry pipeline for both chassis and screen.
fn output_color(col: vec3<f32>) -> vec4<f32> {
    // Output. col is HDR (linear light, BT.709/sRGB primaries, highlights >1.0).
    if (u.tone.x > 0.5) {
        // Float surfaces can be scRGB (1) or BT.2020 linear (2). Convert only
        // for the color space actually configured by the compositor/backend.
        if (u.tone.x < 1.5) { return vec4<f32>(col * u.tone.y, 1.0); }
        let bt2020 = mat3x3<f32>(
            0.6274, 0.0691, 0.0164,
            0.3293, 0.9195, 0.0880,
            0.0433, 0.0114, 0.8956,
        ) * col;
        // tone.y = HDR exposure (scales SDR-white → the compositor's reference
        // white; bump if the picture looks dim, drop if it's blinding).
        if (u.tone.x > 2.5) {
            // ST 2084: scRGB reference white is 80 cd/m², PQ is absolute 0–10000.
            // Encode once, after linear-light shading. The UNORM swapchain must
            // not apply an sRGB transfer on top of PQ.
            let n = pow(clamp(bt2020 * u.tone.y * 0.008, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2610.0 / 16384.0));
            let pq = pow((vec3<f32>(3424.0 / 4096.0) + (2413.0 / 128.0) * n)
                / (vec3<f32>(1.0) + (2392.0 / 128.0) * n), vec3<f32>(2523.0 / 32.0));
            return vec4<f32>(pq, 1.0);
        }
        return vec4<f32>(bt2020 * u.tone.y, 1.0);
    }
    // SDR display: filmic-tonemap HDR highlights back into range (ACES). Target is
    // sRGB, so return linear — the swapchain encodes the transfer function. The small
    // exposure lift keeps midtones from darkening under the ACES toe.
    let toned = aces(col * u.tone.y);
    // ACES desaturates bright colours as it rolls them off — that is the RRT's film
    // "path to white", a print-stock emulation. A CRT has no such behaviour: the guns clip
    // per-channel at maximum beam current, so a saturated primary stays saturated right up
    // to clipping and never bleaches toward white. Undoing that desaturation is therefore a
    // correction, not a look. But it has to be applied WHERE ACES actually does it: the
    // shoulder. Gate it on luminance so only the rolled-off highlights get their chroma back.
    let l = dot(toned, vec3<f32>(0.2126, 0.7152, 0.0722));
    let shoulder = smoothstep(0.45, 1.0, l);
    return vec4<f32>(clamp(toned + (toned - vec3<f32>(l)) * 0.22 * shoulder, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

// Synthetic HDR room reflected in the glass and the plastic. Values run well above
// 1.0 so the light sources bloom in the reflections — a dark room with a soft
// ceiling area-light, a warm lamp to the right, and a faint cool fill to the left.
fn room(r: vec3<f32>) -> vec3<f32> {
    if (u.camera.x > 0.5) {
        // A camera mounted at the display faces +Z into the room. Its image
        // right is world -X: this produces a mirror, not a pasted selfie.
        // Keep this basis fixed in world space as the viewer orbits the tube.
        let plane = vec2<f32>(-r.x, -r.y * u.camera.z) / (max(r.z, 0.0001) * u.camera.y);
        let uv = plane * 0.5 + 0.5;
        let coverage = (1.0 - smoothstep(0.94, 1.0, max(abs(plane.x), abs(plane.y))))
            * select(0.0, 1.0, r.z > 0.0);
        let captured = textureSampleLevel(t_camera, s_screen, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
        // A single camera cannot observe behind itself. Blend to its mean
        // radiance rather than inventing another window or stretching the edges.
        return mix(u.camera_ambient.rgb, captured, coverage);
    }
    let up = clamp(r.y * 0.5 + 0.5, 0.0, 1.0);
    // A NORMALLY-LIT interior — not a black void. Every photo of a real CRT shows the
    // dark glass mirroring a whole room (walls, ceiling, a window), so the environment
    // has to read as a lit room: warm mid walls, brighter cool ceiling, darker floor.
    var c = mix(vec3<f32>(0.085, 0.080, 0.072), vec3<f32>(0.30, 0.32, 0.37), up);
    // Daylight window with mullion bars — the single most recognisable reflection in a
    // CRT photo. Projected to the left of the room; a bright bluish rectangle crossed by
    // a dark 2×2 mullion grid. Moves across the glass as the camera orbits.
    let wx = r.x * 1.7 + 0.55;
    let wy = r.y * 1.9 - 0.10;
    let inwin = smoothstep(0.52, 0.42, abs(wx)) * smoothstep(0.52, 0.42, abs(wy));
    let barx = smoothstep(0.05, 0.11, abs(fract(wx * 1.6) - 0.5));
    let bary = smoothstep(0.05, 0.11, abs(fract(wy * 1.6) - 0.5));
    c = c + vec3<f32>(1.6, 1.78, 2.15) * inwin * mix(0.18, 1.0, min(barx, bary)) * 0.9 * u.beam2.y;
    // Soft rectangular ceiling softbox (broad area highlight on the gloss).
    let win = smoothstep(0.45, 0.97, r.y) * smoothstep(0.66, 0.06, abs(r.x - 0.35));
    c = c + vec3<f32>(1.2, 1.25, 1.42) * win * 1.7;
    // Warm practical lamp off to the right.
    let lamp = pow(max(dot(r, normalize(vec3<f32>(0.78, 0.02, 0.55))), 0.0), 28.0);
    c = c + vec3<f32>(1.0, 0.68, 0.40) * lamp * 4.0;
    // Floor: darker, warm — downward rays pick it up (fills shadowed undersides).
    let down = clamp(-r.y, 0.0, 1.0);
    c = c + vec3<f32>(0.11, 0.095, 0.078) * down;
    return c;
}

fn f_schlick(cos_t: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

// Cook-Torrance GGX specular for one light (HDR: highlights can exceed 1.0).
fn ggx_spec(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let a = max(rough * rough, 1e-3);
    let a2 = a * a;
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 1e-3);
    let ndl = max(dot(n, l), 0.0);
    let denom = ndh * ndh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * denom * denom);
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    let gv = ndv / (ndv * (1.0 - k) + k);
    let gl = ndl / (ndl * (1.0 - k) + k);
    let f = f_schlick(max(dot(h, v), 0.0), f0);
    return d * (gv * gl) * f * ndl;
}

// Physically-based shade for the tube body / bezel: three matching HDR room lights,
// hemispheric ambient, energy-conserving Cook-Torrance specular, and roughness-blurred
// HDR environment reflection with Fresnel.
fn shade_body(base: vec3<f32>, rough: f32, metal: f32, n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    if (u.camera.x > 0.5) {
        let f0 = mix(vec3<f32>(0.04), base, metal);
        let fres = f_schlick(max(dot(n, v), 0.0), f0);
        let refl = reflect(-v, n);
        let axis = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(refl.y) > 0.9);
        let tangent = normalize(cross(axis, refl)) * rough * rough * 0.5;
        let bitangent = cross(refl, tangent);
        let filtered = (room(refl) * 2.0 + room(normalize(refl + tangent))
            + room(normalize(refl - tangent)) + room(normalize(refl + bitangent))
            + room(normalize(refl - bitangent))) / 6.0;
        let specular = mix(filtered, u.camera_ambient.rgb, rough * rough);
        return base * (vec3<f32>(1.0) - fres) * (1.0 - metal) * u.camera_ambient.rgb
            + specular * fres;
    }
    // 1. Matching room lights:
    // l0: Ceiling softbox (key light from upper-front)
    let l0 = normalize(vec3<f32>(0.35, 0.85, 0.40));
    let rad0 = vec3<f32>(1.30, 1.32, 1.38) * 1.35;

    // l1: Daylight window from upper-left (gated by u.beam2.y / R key)
    let l1 = normalize(vec3<f32>(-0.75, 0.25, 0.50));
    let rad1 = vec3<f32>(1.25, 1.40, 1.65) * 0.70 * u.beam2.y;

    // l2: Warm practical lamp from lower-right
    let l2 = normalize(vec3<f32>(0.78, 0.05, 0.55));
    let rad2 = vec3<f32>(1.15, 0.80, 0.48) * 0.55;

    let f0 = mix(vec3<f32>(0.04), base, metal);
    let fres_v = f_schlick(max(dot(n, v), 0.0), f0);
    // Strict energy conservation: diffuse fraction is reduced by Fresnel reflection
    let kd = base * (vec3<f32>(1.0) - fres_v) * (1.0 - metal);

    // Hemispheric ambient from the lit room interior
    let amb_sky = vec3<f32>(0.024, 0.026, 0.033);
    let amb_ground = vec3<f32>(0.008, 0.007, 0.006);
    let amb = mix(amb_ground, amb_sky, n.y * 0.5 + 0.5);

    // Direct lighting accumulation (diffuse + GGX specular)
    var col = kd * amb;

    let ndl0 = max(dot(n, l0), 0.0);
    col = col + kd * rad0 * ndl0 + ggx_spec(n, v, l0, rough, f0) * rad0;

    let ndl1 = max(dot(n, l1), 0.0);
    col = col + kd * rad1 * ndl1 + ggx_spec(n, v, l1, rough, f0) * rad1;

    let ndl2 = max(dot(n, l2), 0.0);
    col = col + kd * rad2 * ndl2 + ggx_spec(n, v, l2, rough, f0) * rad2;

    // Roughness-filtered HDR environment reflection
    let refl = reflect(-v, n);
    let env_spec = mix(room(refl), amb * 2.5, rough * rough);
    col = col + env_spec * fres_v * (1.0 - rough * 0.65) * 0.35;

    return col;
}

const HALF_W: f32 = 0.667;
const HALF_H: f32 = 0.5;

// Trace the view ray refracting through the curved faceplate to the phosphor plane
// behind it, returning the uv it lands on. The rasterizer already hands us the
// outer-glass point (world_pos) and normal, so this is one analytic Snell bounce,
// not a march. `eta` = air/glass IOR ratio; a per-channel eta gives dispersion.
fn refract_uv(base_uv: vec2<f32>, n: vec3<f32>, v: vec3<f32>, thick: f32, eta: f32) -> vec2<f32> {
    let r = refract(-v, n, eta);     // ray bent into the glass (heads toward -z)
    let t = thick / max(-r.z, 1e-3); // distance along it to the phosphor plane
    let off = r.xy * t;              // local-space lateral shift over the glass depth
    return base_uv + vec2<f32>(off.x / HALF_W, -off.y / HALF_H) * 0.5;
}

// Raster deflection geometry: the yoke never paints a perfect rectangle. Warps the
// image-sampling coordinate (NOT the physical tube face) with pincushion/barrel
// (radial), corner pincushion (4th-order, corners only), and trapezoid/keystone
// (horizontal width varies with height). Sampling past the edge clamps → mild overscan.
fn geometry_warp(uv: vec2<f32>) -> vec2<f32> {
    var p = uv - vec2<f32>(0.5);
    let r2 = dot(p, p);
    p = p * (1.0 + u.geom.x * r2);       // pincushion / barrel
    p = p * (1.0 + u.geom.z * r2 * r2);  // corner pincushion (4th order)
    p.x = p.x * (1.0 + u.geom.y * p.y);  // trapezoid / keystone
    return p + vec2<f32>(0.5);
}

const PI: f32 = 3.14159265;
const TAU: f32 = 6.28318530;
// Colour-subcarrier cycles per *content* pixel (on a virtual ~320-wide line — see the
// `step` remap in ntsc()/svideo(), not per captured texel). This is a MEASURED ratio, not
// a convenience: 320 active pixels across NTSC's 52.6 µs active line is a 6.0837 MHz
// content-pixel rate, so f_sc = 3.579545 / 6.0837 = 0.58839 cycles per content pixel —
// a 1.70-px subcarrier period. That number decides which picture detail turns into false
// colour, so it cannot be chosen for arithmetic convenience: at 0.25 (4 samples/cycle,
// what this used to be) the subcarrier sits at 1.52 MHz, cross-colour fires on 3.2–5.3-px
// detail instead of 1.25–2.7-px, and ordinary 4-px-period pixel art — 2-px text stems,
// wide dither — demodulates to full-strength false chroma that a real set leaves grey.
const NTSC_FSC: f32 = 0.58839;
// Four integration samples per virtual pixel resolve both the subcarrier and
// the narrow luma Gaussian. The trap needs ±18 pixels for four sigma of support.
// Source pixels are held at their native centers; carrier phase stays continuous.
const NTSC_STEP: f32 = 0.25;
const NTSC_TAPS: i32 = 72;

// --- NTSC bandwidths, all in content pixels as gaussian σ ---
// σ = sqrt(ln2)·6.0837 / (2π·f_MHz): Gaussian amplitude is 1/sqrt(2)
// at the -3 dB cutoff. sqrt(2 ln2) incorrectly places that cutoff at -6 dB.
//
// Chroma is the CASCADE of two real filters, which is the part that is easy to get wrong.
// The NTSC *encoder* transmits I at ~1.3 MHz and Q at only ~0.4 MHz, so green–magenta
// detail leaves the studio mushier than orange–cyan. But a consumer RCA/Panasonic-class
// receiver does not do wideband-I demodulation — that was high-end-set territory, because
// it needs the asymmetric-sideband correction. It demodulates equiband at about 0.5 MHz on
// both axes. Cascading the two (σ² adds) is what a consumer set actually delivers: the
// encoder's 3.25:1 asymmetry survives, but compressed to 1.49:1, because the receiver's own
// 0.5 MHz limit dominates the I axis and barely touches the already-narrower Q axis.
const NTSC_SIG_I: f32 = 1.727367; // enc 1.3 MHz ⊗ rx 0.5 MHz → 0.467 MHz
const NTSC_SIG_Q: f32 = 2.580828; // enc 0.4 MHz ⊗ rx 0.5 MHz → 0.312 MHz
// Luma: the set's video amplifier, ~3.0 MHz for a mid-range consumer set (cheap RF-fed
// sets run nearer 2.5). It is NOT narrowed to reject the subcarrier — that is the trap's
// job, below. Making one gaussian do both jobs is what forced the old 0.88 MHz luma path,
// i.e. most of the composite softness was a side effect of the f_sc error, not a tube.
const NTSC_SIG_Y: f32 = 0.268705;
// Approximate a Q=10 trap by estimating quadrature carrier amplitude through
// Gaussian baseband filters. Its 0.358 MHz full width means a 0.179 MHz cutoff
// on EACH side of the carrier, not a 0.358 MHz one-sided low-pass cutoff.
const NTSC_SIG_TRAP: f32 = 4.504009;
const SVIDEO_SIG_Y: f32 = 0.201529; // 4 MHz luma at -3 dB
const NTSC_TRAP: f32 = 0.90;

// NTSC is defined on GAMMA-CORRECTED video. The camera (or the console's DAC) applies
// roughly a 1/2.2 opto-electric curve and the tube's ~2.4 EOTF undoes it at the other end,
// so Y'IQ, the 0.299/0.587/0.114 luma weights, the encoder bandwidths and every artifact
// that falls out of them all live in that ENCODED space — never in light. The source
// texture here is sRGB, which the hardware has already linearised for us, so the encode has
// to be put back before modulating and taken off again after decoding. Running the decode
// in linear light instead (as this did) is not a small error: composite artifacts are
// differences between neighbouring samples, and the encoding curve is what decides how big
// a difference a given step in the picture makes. In linear, a dark edge barely modulates
// the subcarrier, so dot crawl and cross-colour all but vanish from shadows and pile up in
// highlights — the opposite of what a real set does, where the worst rainbows in any
// composite capture are in the mid-dark detail.
// Exact sRGB inverse of the GPU texture decode; the toe is not gamma 2.2.
fn oetf(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055,
                  12.92 * x, x <= vec3<f32>(0.0031308));
}
fn eotf(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(pow((x + 0.055) / 1.055, vec3<f32>(2.4)),
                  x / 12.92, x <= vec3<f32>(0.04045));
}

fn rgb2yiq(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.299, 0.587, 0.114)),
        dot(c, vec3<f32>(0.596, -0.274, -0.322)),
        dot(c, vec3<f32>(0.211, -0.523, 0.312)),
    );
}
fn yiq2rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        c.x + 0.956 * c.y + 0.619 * c.z,
        c.x - 0.272 * c.y - 0.647 * c.z,
        c.x - 1.106 * c.y + 1.703 * c.z,
    );
}
// Subcarrier phase at source column `px` on `line`. Both offsets are NTSC line timing,
// not free parameters. f_sc is 227.5 cycles per line, so the half cycle flips the phase
// 180° every scanline (+PI*line). Over a field that is 227.5 × 262.5 = 59718.75 cycles —
// a 0.75-cycle remainder, so the phase steps +270° per field and closes on itself after
// four, which is the NTSC four-field colour sequence. Quantising to the 59.94 Hz field
// index is what makes the residual a *crawl*: the pattern alternates at field rate and the
// phosphor integrator averages most of it away, leaving the low-contrast creep a real set
// shows. Drifting the phase on wall-clock time instead (this was `t * 6.0`, a 0.955 Hz
// rotation) sweeps the hue of every cross-colour artifact through the wheel once a second,
// which no set does — that reads as colour morphing rather than as an artifact.
fn subcarrier(px: f32, line: f32, t: f32) -> f32 {
    // mod 4 is free: four fields is 3.0 whole cycles, so it wraps exactly. It also keeps the
    // argument small — an unwrapped field index passes 10⁵ within half an hour, where f32
    // cos/sin has lost the fractional radians the phase is made of.
    let field = u.temporal.w % 4.0;
    return TAU * NTSC_FSC * px + PI * line + TAU * 0.75 * field;
}

// Encode RGB→composite along the scanline, then decode. Band-limited luma low-pass +
// quadrature chroma demod: imperfect luma/subcarrier separation → dot crawl; luma
// energy near the subcarrier leaking into chroma → cross-colour rainbow; the narrow
// chroma passband → horizontal colour bleed. This is the analog-signal look.
//
// `step` remaps the whole decode onto a virtual ~320-wide content line, so the bandwidths
// and subcarrier are fixed in cycles-per-*line* — they track the source pixel grid no
// matter how upscaled the captured frame is. Without the remap, an integer-scaled capture
// (e.g. 1280-wide RetroArch) makes each content pixel several texels, the fixed-texel
// filters can't reach across a checkerboard, and everything below is mistuned by the
// capture scale. At native capture step→1.
//
// That remap also preserves the physical frequency of console dither: some detail
// survives the luma low-pass and some demodulates into false color near the carrier.
// The luma filter and the separate carrier trap serve different roles; neither is
// narrowed just to make checkerboards disappear. See the cutoff regression test.
fn ntsc(uv: vec2<f32>, res: vec2<f32>, t: f32) -> vec3<f32> {
    let step = res.x / 320.0; // texels per virtual-320 content pixel (1 at native capture)
    let cx = uv.x * (res.x / step);     // column on the virtual content line
    // Adjacent scanned lines in an interlaced field are two texture rows apart.
    // Phase alternates per scanned line, not per full-frame row.
    let row = floor(uv.y * res.y);
    let line = select(row, floor(row * 0.5), u.temporal.z > 0.5);
    var y_acc = 0.0;  var yw = 0.0;     // luma low-pass
    var i_acc = 0.0;  var iw_s = 0.0;   // I demod
    var q_acc = 0.0;  var qw_s = 0.0;   // Q demod
    var it_acc = 0.0; var qt_acc = 0.0; var tw_s = 0.0; // trap's narrow-band subcarrier estimate
    // The luma kernel's own projection onto the subcarrier. Σcos(ph)·lw / Σlw is the luma
    // filter's complex gain at f_sc rotated to the centre tap's phase, so subtracting
    // trap·(Î·lc + Q̂·ls) removes the subcarrier residual with the right amplitude AND
    // phase without a second pass over the taps — the alternative, notching each tap before
    // the low-pass, needs Î/Q̂ before the loop can start.
    var lc_acc = 0.0; var ls_acc = 0.0;
    for (var k = -NTSC_TAPS; k <= NTSC_TAPS; k = k + 1) {
        let d = f32(k) * NTSC_STEP;     // offset in content pixels
        let scx = cx + d;
        // Zero-order hold: the DAC holds each content pixel for its whole dwell, so the
        // baseband being modulated is a staircase. Snapping the fetch to the content-pixel
        // centre IS that staircase, at no cost — which resolves the simplification noted at
        // NTSC_STEP the right way round. (Trusting the linear sampler instead reconstructed
        // the source with a triangle kernel, which carries only sinc²=28% of the input at
        // f_sc where a real hold carries sinc=52%, so stair-step detail cross-coloured at
        // half strength.) The subcarrier phase stays on the CONTINUOUS position: a real
        // encoder modulates a held baseband onto a free-running 3.58 MHz carrier.
        let sxc = (floor(scx * step) + 0.5) / res.x;
        let src = textureSampleLevel(t_screen, s_screen, vec2<f32>(sxc, uv.y), 0.0).rgb;
        let yiq = rgb2yiq(oetf(src));
        let ph = subcarrier(scx, line, t);
        let c = cos(ph);
        let s = sin(ph);
        let comp = yiq.x + yiq.y * c + yiq.z * s; // composite sample
        let dd = d * d;
        let lw = exp(-dd / (2.0 * NTSC_SIG_Y * NTSC_SIG_Y));
        let iw = exp(-dd / (2.0 * NTSC_SIG_I * NTSC_SIG_I));
        let qw = exp(-dd / (2.0 * NTSC_SIG_Q * NTSC_SIG_Q));
        let tw = exp(-dd / (2.0 * NTSC_SIG_TRAP * NTSC_SIG_TRAP));
        y_acc = y_acc + comp * lw;      yw = yw + lw;
        i_acc = i_acc + comp * c * iw;  iw_s = iw_s + iw;
        q_acc = q_acc + comp * s * qw;  qw_s = qw_s + qw;
        it_acc = it_acc + comp * c * tw;
        qt_acc = qt_acc + comp * s * tw;
        tw_s = tw_s + tw;
        lc_acc = lc_acc + c * lw;
        ls_acc = ls_acc + s * lw;
    }
    // ×2 recovers the demodulation loss: comp·cos = I/2 + (terms at 2·f_sc, rejected above).
    let i_hat = 2.0 * i_acc / iw_s;
    let q_hat = 2.0 * q_acc / qw_s;
    let i_trap = 2.0 * it_acc / tw_s;
    let q_trap = 2.0 * qt_acc / tw_s;
    let y = y_acc / yw - NTSC_TRAP * (i_trap * lc_acc / yw + q_trap * ls_acc / yw);
    return eotf(max(yiq2rgb(vec3<f32>(y, i_hat, q_hat)), vec3<f32>(0.0)));
}

// S-video: luma and chroma travel on separate wires, so there's perfect Y/C
// separation — no dot crawl, no cross-colour rainbow — but chroma is still
// band-limited (the horizontal colour bleed remains). Sharp luma, soft colour.
fn svideo(uv: vec2<f32>, res: vec2<f32>) -> vec3<f32> {
    let step = res.x / 320.0; // same content-line remap as ntsc(): bleed tracks the source grid
    let cx = uv.x * (res.x / step);
    var y = 0.0;
    var yw = 0.0;
    var i = 0.0;
    var q = 0.0;
    var cw = 0.0;
    var qw = 0.0;
    // Integrate at the same quarter-pixel spacing: integer-only taps undersample
    // the 4 MHz luma kernel and incorrectly turn it into an almost perfect wire.
    for (var k = -36; k <= 36; k = k + 1) {
        let d = f32(k) * NTSC_STEP;
        let scx = cx + d;
        // Same zero-order hold and same gamma-encoded working space as ntsc(): S-video
        // splits the wires, it does not change what is on them.
        let sxc = (floor(scx * step) + 0.5) / res.x;
        let yiq = rgb2yiq(oetf(textureSampleLevel(t_screen, s_screen, vec2<f32>(sxc, uv.y), 0.0).rgb));
        let kk = d * d;
        // Luma rides its own wire, so no 3.58 trap has to cut into it — the limit is just
        // the set's video amp at ~4.0 MHz. That is wider than the 3.04 MHz content Nyquist,
        // so single content pixels remain distinguishable, with finite bandwidth
        // rolloff rather than an unfiltered passthrough. Composite's softness is the trap's shoulder,
        // not the tube's, which is why the two paths differ here at all.
        let lw = exp(-kk / (2.0 * SVIDEO_SIG_Y * SVIDEO_SIG_Y));
        // Chroma is unchanged from composite: it is still a demodulated subcarrier, so it
        // still cascades the encoder's 1.3/0.4 MHz asymmetry with the consumer receiver's
        // 0.5 MHz equiband demodulator. Separate wires buy perfect Y/C separation — no dot
        // crawl, no cross-colour — but they do not widen the chroma passband, so the old
        // "no receiver-side narrowing of I either" was wrong: that narrowing is the
        // demodulator, which S-video does not bypass.
        let iw = exp(-kk / (2.0 * NTSC_SIG_I * NTSC_SIG_I));
        let qw_k = exp(-kk / (2.0 * NTSC_SIG_Q * NTSC_SIG_Q));
        y = y + yiq.x * lw;
        yw = yw + lw;
        i = i + yiq.y * iw;
        q = q + yiq.z * qw_k;
        cw = cw + iw;
        qw = qw + qw_k;
    }
    return eotf(max(yiq2rgb(vec3<f32>(y / yw, i / cw, q / qw)), vec3<f32>(0.0)));
}

// Per-channel electron-beam width from that channel's drive (the guest-advanced /
// Sony-Megatron "beam math"). A bright channel draws more beam current, so its spot
// blooms wider vertically and its scanlines merge; a dim channel stays a tight,
// separated line. beam_min/max are half-widths in source-texel rows; beam_shape
// curves how fast width grows with signal.
fn beam_drive(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(u.scan.z));
}
fn beam_width(c: vec3<f32>) -> vec3<f32> {
    return mix(vec3<f32>(u.scan.x), vec3<f32>(u.scan.y), beam_drive(c));
}

// Γ(1 + x) for x = 1/p over p ∈ [1.2, 5] — a cubic fit, max error 3.3e-4. The spot
// profile's area is 2·w·Γ(1+1/p), so this is what normalises each row's beam to unit
// energy. It has to be evaluated per pixel now that p varies with drive (see the note on
// scan_reconstruct); it used to be one CPU-side Lanczos Γ per frame.
fn gamma1p(x: f32) -> f32 {
    return ((-0.10654 * x + 0.58755) * x - 0.47554) * x + 0.99029;
}

// Reconstruct the beam-scanned color at `uv` from the phosphor plane (already
// NTSC-decoded and time-integrated by the accum pass). Each nearby source row emits
// a per-channel beam spot; summing the overlapping profiles gives bright cores
// that bloom and dark gaps that stay open — resolution-correct, in linear light.
// (Explicit-LOD sampling so it stays callable once per primary for dispersion.)
//
// Two pieces of physics decide the shape of the sum:
//
// 1. ENERGY. A phosphor emits photons in linear proportion to the beam current landing
//    on it — the ~2.4 gamma of a CRT comes from the gun's grid transfer characteristic
//    (drive volts → beam current), not from any phosphor nonlinearity. So when a bright
//    row's spot blooms wider it must SPREAD that row's light, not create more of it.
//    Each row is therefore normalised to unit area (`norm / w` — the profile's area
//    is 2·w·Γ(1+1/p)), which makes a flat field reconstruct to exactly its own value for
//    any tube. Scanline structure then emerges purely as redistribution: a dark row keeps
//    a tight core with a black gap either side, a bright row fattens until the gaps close.
//    That is the real bloom, and it no longer smuggles in an extra gamma — the previous
//    unnormalised sum scaled total energy by c·w(c), so a 2:1 signal ratio rendered as
//    2.4:1 and every tube's beam setting silently re-exposed the whole picture.
//
// 2. PROFILE. The spot is the gun's imaged cathode crossover blurred by the optics'
//    aberrations, so it is not a bell unless the tube is soft: a well-focused Trinitron or
//    shadow-mask CRT has a plateau at the scanline centre with a steep wall down to the
//    gap. exp(-|d/w|^p) covers both — p=2 is the plain gaussian of a defocused consumer
//    set, p≈4 the flat-topped core of a broadcast monitor. The flat top also gives
//    halation a concentrated source to spread from rather than a soft peak.
//
//    But p is not a constant of the tube, it is a constant of the tube AT LOW CURRENT.
//    A spot blooms because space charge and the lens's spherical aberration take over as
//    the beam gets fatter, and both of those add tails — the crisp plateau is exactly the
//    thing that goes first. Holding p fixed while only w grew produced a result no tube
//    shows: a flat-topped profile wider than half the line pitch has a NEGATIVE aperture
//    response at the line frequency, so the sum of overlapping rows peaked BETWEEN the
//    scanlines instead of on them. Measured on the presets it inverted by 11% on a white
//    Trinitron field and, worse, flipped sign across a gradient on the arcade tube — the
//    scanline phase visibly jumping half a line partway down a ramp. Relaxing the exponent
//    toward a plain gaussian on the same drive term that widens the spot fixes it at the
//    cause: modulation now falls monotonically from a deep dark-field gap to a merged
//    highlight, and a sharp tube (PVM ~9%, Diamondtron ~10%) still holds visible scanlines
//    at peak white where a fuzzy RCA goes to zero.
//
// 3. THE OTHER AXIS. Vertically a raster really is a stack of discrete lines, so summing
//    point emitters is the right model. Horizontally it is not: the video signal is
//    continuous, the DAC holds each source pixel for its whole dwell time, and the beam
//    paints that staircase blurred by the spot. Leaving the horizontal axis to the
//    hardware's linear filter — as this did — models neither: a triangle kernel a full
//    source pixel wide turns every pixel into a ramp between its neighbours' centres, so
//    nothing ever reaches a flat top and the tube's focus has no say on this axis at all.
//    A razor PVM and a fuzzy RCA came out identically soft horizontally. `horizontal_signal` below
//    restores the held signal with an integrated horizontal spot.
// Zero-mean supply ripple: changing its phase redistributes brightness without
// changing a uniform field's mean. Keep this separate from phosphor persistence.
fn hum_modulation(y: f32, time: f32, beat_hz: f32, amplitude: f32) -> f32 {
    return 1.0 + amplitude * sin(TAU * (fract(time * beat_hz) - 2.0 * y));
}

// Integral of exp(-x²), Abramowitz/Stegun 7.1.26 (absolute error < 1.5e-7).
fn spot_erf(x: vec3<f32>) -> vec3<f32> {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                  - 0.284496736) * t + 0.254829592) * t;
    return sign(x) * (1.0 - poly * exp(-x * x));
}

// Convolve the DAC's held pixel intervals with a horizontal Gaussian spot.
// Integrating each interval, instead of remapping one bilinear fetch, includes
// multi-pixel tails and gives each gun its own current-dependent spot width.
fn horizontal_signal(uvx: f32, row: f32, res: vec2<f32>, wscale: f32, footprint: f32) -> vec3<f32> {
    let x = uvx * res.x;
    let scale = wscale * (res.x / res.y) * (HALF_H / HALF_W);
    let widest = max(u.scan.y * scale, 0.5 * footprint);
    let radius = i32(ceil(3.0 * widest)) + 1;
    let center = floor(x);
    var result = vec3<f32>(0.0);
    for (var k = -radius; k <= radius; k = k + 1) {
        let column = center + f32(k);
        let c = textureSampleLevel(t_screen, s_screen,
            vec2<f32>((column + 0.5) / res.x, (row + 0.5) / res.y), 0.0).rgb;
        let w = max(beam_width(c) * scale, vec3<f32>(max(0.5 * footprint, 1e-4)));
        let weight = 0.5 * (spot_erf(vec3<f32>(column + 1.0 - x) / w)
                          - spot_erf(vec3<f32>(column - x) / w));
        result = result + c * weight;
    }
    return result;
}

fn scan_reconstruct(uv: vec2<f32>, res: vec2<f32>, wscale: f32, src_px: vec2<f32>) -> vec3<f32> {
    let fy = uv.y * res.y - 0.5;
    let row0 = floor(fy);
    var beam = vec3<f32>(0.0); // energy-normalised beam sum (blooms where lines overlap)
    var flat = vec3<f32>(0.0); // profile-weighted reference (the settled picture)
    var wsum = vec3<f32>(0.0);
    // Defocused bright spots can extend beyond the preset's minimum support.
    // Three widths cover >99.99% of a Gaussian; a fixed ±1 row lost corner energy.
    let range = max(i32(u.scan.w), i32(ceil(3.0 * u.scan.y * wscale)));
    let p = u.beam2.x;
    for (var k = -range; k <= range + 1; k = k + 1) {
        let row = row0 + f32(k);
        let c = horizontal_signal(uv.x, row, res, wscale, src_px.x);
        // wscale > 1 near the edges: deflection defocus widens the vertical spot.
        let s = beam_drive(c);
        let w = mix(vec3<f32>(u.scan.x), vec3<f32>(u.scan.y), s) * wscale;
        // Same drive term relaxes the flat top toward a plain gaussian as the spot blooms.
        let pe = mix(vec3<f32>(p), vec3<f32>(2.0), s);
        let d = abs(fy - row);
        // Generalized gaussian, per channel, normalised to unit area over the row.
        let g = exp(-pow(vec3<f32>(d) / w, pe));
        let norm = vec3<f32>(1.0 / (2.0 * gamma1p(1.0 / pe.r)),
                             1.0 / (2.0 * gamma1p(1.0 / pe.g)),
                             1.0 / (2.0 * gamma1p(1.0 / pe.b)));
        beam = beam + c * g * (norm / w);
        flat = flat + c * g;
        wsum = wsum + g;
    }
    flat = flat / max(wsum, vec3<f32>(1e-4));
    // Band-limit against the display, the same way the mask does. Scanline depth is NOT a
    // per-tube taste setting: it falls out of the spot width against the line pitch, and both
    // of those are already here. Every preset used to carry a hand-set `scanline` from 0.30
    // to 0.62 that damped the structure the beam math had just computed — a second opinion
    // about a quantity the physics had already answered, and one that pulled the sharpest
    // tubes (Diamondtron 0.30) furthest from what they really do with a 240p signal. What
    // that knob was actually standing in for is RESOLVABILITY: once one output pixel covers
    // a whole source row the raster structure cannot be drawn, and must integrate into the
    // smooth picture rather than alias into it. That is a property of the camera and the
    // signal, not of the tube, so it is measured here — `src_px.y` is source rows per output
    // pixel — and it correctly leaves a 240p console showing hard scanlines on the same tube
    // where a 1080p desktop capture shows none. Both terms carry the same total energy, so
    // this trades structure for smoothness at constant brightness.
    // Nyquist: the line pattern needs two output pixels per scanline to be drawn at all, so
    // it is intact at 0.35 rows/pixel (~3 px per line) and gone by 0.8 (~1.25 px per line).
    let resolvable = 1.0 - smoothstep(0.35, 0.8, src_px.y);
    return mix(flat, beam, resolvable) * u.tone.z;
}

// ---------------------------------------------------------------------------
// Pass A — phosphor persistence. A fullscreen pass that decodes the source signal
// (NTSC) and integrates it into the phosphor plane over time: the phosphor charges
// to the fresh excitation, then decays exponentially toward the previous field, so
// moving content leaves a real fading trail (and interlaced fields can flicker).
// ---------------------------------------------------------------------------

struct FullOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) vid: u32) -> FullOut {
    var out: FullOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.uv = vec2<f32>(x, y);              // uv (0,0) = top-left, matching the source
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

// Auxiliary-coil displacement is driven by differentiated luminance (US5600381A).
// Integrate the held input through a Gaussian differentiator. Its bounded second
// derivative keeps velocity positive even on a black/white transition.
// Returns displacement in source columns and dx_screen/dx_time.
fn svm_trajectory(x: f32, row: f32, res: vec2<f32>) -> vec2<f32> {
    if (u.fx.x <= 0.0) { return vec2<f32>(0.0, 1.0); }
    let sigma = max(0.65, 0.004375 * res.x);
    let radius = i32(ceil(4.0 * sigma)) + 1;
    let center = floor(x);
    var gradient = 0.0;
    var curvature = 0.0;
    for (var k = -radius; k <= radius; k = k + 1) {
        let column = center + f32(k);
        let c = textureSampleLevel(t_screen, s_screen,
            vec2<f32>((column + 0.5) / res.x, (row + 0.5) / res.y), 0.0).rgb;
        let y = dot(clamp(oetf(c), vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.299, 0.587, 0.114));
        let left = x - column;
        let right = left - 1.0;
        let gl = exp(-0.5 * left * left / (sigma * sigma)) / (sqrt(TAU) * sigma);
        let gr = exp(-0.5 * right * right / (sigma * sigma)) / (sqrt(TAU) * sigma);
        gradient = gradient + y * (gl - gr);
        curvature = curvature + y * (right * gr - left * gl) / (sigma * sigma);
    }
    let gain = u.fx.x * sigma * sigma;
    return vec2<f32>(gain * gradient, 1.0 + gain * curvature);
}

fn svm_source(uv: vec2<f32>, res: vec2<f32>) -> vec3<f32> {
    let target_x = uv.x * res.x;
    var x = target_x;
    var trajectory = vec2<f32>(0.0, 1.0);
    // Newton inversion maps a phosphor position back to the instant the beam
    // crossed it. Dividing current by velocity preserves deposited energy.
    for (var i = 0; i < 4; i = i + 1) {
        trajectory = svm_trajectory(x, floor(uv.y * res.y), res);
        x = x - (x + trajectory.x - target_x) / max(trajectory.y, 0.1);
    }
    trajectory = svm_trajectory(x, floor(uv.y * res.y), res);
    return vec3<f32>(x / res.x, uv.y, 1.0 / max(trajectory.y, 0.1));
}

// Time of the beam crossing within the field. The horizontal retrace and
// vertical blanking intervals contain no excitation. SD uses NTSC's 262.5-line
// field / 52.6 us active line; denser PC rasters retain their rows and use the
// same normalized blanking budget until an explicit source timing is supplied.
fn raster_arrival(uv: vec2<f32>, res: vec2<f32>, interlace: f32) -> f32 {
    let rows = res.y / (1.0 + interlace);
    let line = floor(floor(uv.y * res.y) / (1.0 + interlace));
    let total = rows * (262.5 / 240.0);
    let vertical_blank = total - rows;
    let horizontal = (63.5555556 - 52.6) / 63.5555556 + uv.x * (52.6 / 63.5555556);
    return (vertical_blank + line + horizontal) / total * (1001.0 / 60000.0);
}

// Stable 1-exp(-x) for slow reservoirs, avoiding loss of a small time step.
fn released(x: vec3<f32>) -> vec3<f32> {
    let small = x * (1.0 - 0.5 * x + x * x / 6.0);
    return select(1.0 - exp(-x), small, x < vec3<f32>(0.001));
}

fn measured_phosphor(uv: vec2<f32>, scan_uv: vec2<f32>, sig: vec3<f32>, lit: f32) -> vec3<f32> {
    let dt = u.temporal.x;
    if (dt <= 0.0) { return sig * lit; }
    let period = 1001.0 / 60000.0;
    let arrival = raster_arrival(scan_uv, u.params.xy, u.temporal.z);
    let remaining = max(dt - arrival, 0.0);
    let shutter = dt * clamp(u.raster.y, 0.000001, 1.0);
    let begin = dt - shutter;
    let index = (u32(uv.y * u.params.y) * u32(u.params.x) + u32(uv.x * u.params.x)) * 33u;
    let deposited = sig * lit * period * (1.0 + u.temporal.z) * select(0.0, 1.0, arrival <= dt);
    var integral = vec3<f32>(0.0);
    let mono = u.mono.w > 0.5;
    let material = select(0.0, u.ptau.x, mono);
    for (var k = 0u; k < 31u; k = k + 1u) {
        var rate = PHOS_RATE[k];
        var weight = PHOS_ENERGY[k];
        if (mono) {
            rate = vec3<f32>(1.0) / u.ptau.rgb;
            weight = select(vec3<f32>(0.0), vec3<f32>(1.0), k == 0u);
        }
        var previous = history_load(index + k).rgb;
        // A newly opened, already-warm tube starts at the periodic steady state
        // of its initial signal. Power-up from a blank signal still starts dark.
        if (history_load(index + 32u).w == 0.0 || history_load(index + 32u).y != material) {
            let cycle = period * (1.0 + u.temporal.z);
            previous = deposited * weight * rate * exp(-rate * max(cycle - arrival, 0.0))
                / max(released(rate * cycle), vec3<f32>(1e-12));
        }
        integral = integral + previous * exp(-rate * begin) * released(rate * shutter) / rate;
        // Integrate the new impulse only over the open shutter interval.
        let age_begin = max(begin - arrival, 0.0);
        integral = integral + deposited * weight * exp(-rate * age_begin)
            * released(rate * max(remaining - age_begin, 0.0));
        let next = previous * exp(-rate * dt) + deposited * weight * rate * exp(-rate * remaining);
        history_store(index + k, vec4<f32>(next, 0.0));
        if (mono) { break; }
    }
    // Accumulate all field exposures belonging to one output frame. A 30 fps
    // frame must include both fields' light, not just the final field snapshot.
    var exposure = history_load(index + 31u);
    if (history_load(index + 32u).x != u.raster.z) { exposure = vec4<f32>(0.0); }
    exposure = exposure + vec4<f32>(integral, shutter);
    history_store(index + 31u, exposure);
    history_store(index + 32u, vec4<f32>(u.raster.z, material, 0.0, 1.0));
    return exposure.rgb / max(exposure.w, 1e-9);
}

@fragment
fn fs_phosphor(in: FullOut) -> @location(0) vec4<f32> {
    let res = u.params.xy;
    let beam = svm_source(in.uv, res);
    let uv = beam.xy;
    // Input signal path (tone.w): 0 = RGB/component (clean — PVM, arcade board, PC),
    // 1 = S-video (sharp luma, band-limited colour, no dot crawl), 2 = composite
    // (dot crawl + cross-colour rainbow + colour bleed), 3 = RF tuner approximation.
    var sig: vec3<f32>;
    if (u.tone.w >= 1.5) {
        sig = ntsc(uv, res, u.params.z);
        if (u.tone.w >= 2.5) {
            // Approximate the extra bandwidth loss of a console RF modulator/tuner.
            // Voltage-space filtering precedes the gun; this is not an RF circuit simulation.
            let dx = vec2<f32>(0.5 / 320.0, 0.0);
            sig = eotf(0.5 * oetf(sig)
                + 0.25 * oetf(ntsc(uv - dx, res, u.params.z))
                + 0.25 * oetf(ntsc(uv + dx, res, u.params.z)));
        }
    } else if (u.tone.w >= 0.5) {
        sig = svideo(uv, res);
    } else {
        sig = textureSampleLevel(t_screen, s_screen, uv, 0.0).rgb; // clean RGB
    }

    // Signal noise belongs before the nonlinear gun and phosphor history.
    // Analog noise correlation follows the fixed signal bandwidth, not capture width.
    let noise_res = vec2<f32>(select(res.x, 320.0, u.tone.w > 0.5), res.y);
    let grain = hash21(floor(uv * noise_res) + vec2<f32>(u.params.z * 61.0, u.params.z * 37.0)) - 0.5;
    var voltage = max(oetf(sig) + vec3<f32>(grain * u.look.z), vec3<f32>(0.0));
    if (u.mono.w > 0.5) {
        voltage = vec3<f32>(dot(voltage, vec3<f32>(0.299, 0.587, 0.114)));
    }
    sig = pow(voltage, vec3<f32>(u.phys.x)) * beam.z;
    // Cut excitation when fully off; stored light must decay rather than secretly
    // tracking the source while the power switch hides the output.
    sig = sig * select(1.0, 0.0, u.pwr.y >= 1.0 || u.pwr.x <= 0.0);

    let prev = textureSampleLevel(t_prev, s_screen, in.uv, 0.0).rgb;    // last phosphor

    let dt = max(u.temporal.x, 0.0);
    // Per-phosphor decay: each primary keeps its own fraction of last field's charge.
    // Red lingers a whole persistence class longer than green and blue → moving highlights
    // trail warm (the real P22 look; see the ptau comment on the CPU side).
    let tau = max(u.ptau.rgb * max(u.temporal.y, 1e-4), vec3<f32>(1e-8));
    // Sulfide phosphors do not decay as a single exponential. The measured curve is a fast
    // near-exponential drop followed by a slow power-law tail (I = a/(t+t₀)^b), which is
    // where the visible afterglow lives — a pure exponential drops it entirely. Physically
    // that tail is bimolecular: the recombination rate scales with the remaining trapped
    // charge, so dim charge decays SLOWER than bright charge. Model it as a level-dependent
    // time constant rather than a true power law, which would decay as a fraction per frame
    // forever and leave a faint permanent smudge; this keeps the correct fast-then-slow
    // shape while still clearing exponentially. ptau.w sets how much the tail stretches.
    let tail = 1.0 + u.ptau.w * (1.0 - clamp(prev, vec3<f32>(0.0), vec3<f32>(1.0)));
    let decay = exp(-vec3<f32>(dt) / (tau * tail));

    // Interlace: on an interlaced field only alternate lines are re-excited this
    // frame; the others coast on their decayed charge, giving line twitter.
    let line = floor(uv.y * res.y);
    let parity = f32(i32(u.temporal.w) & 1);
    let odd = f32(i32(line) - (i32(line) / 2) * 2);
    let lit = 1.0 - u.temporal.z * abs(odd - parity);
    let excite = sig * lit;
    if (u.raster.x > 0.5) { return vec4<f32>(measured_phosphor(in.uv, beam.xy, sig, lit), 1.0); }

    // Phosphor charges instantly to the beam excitation, then decays. max() keeps a
    // freshly-lit pixel bright while unlit pixels fall off toward the previous field.
    let out = max(excite, prev * decay);
    return vec4<f32>(out, 1.0);
}

// Source statistics already contain average linear light. Do not multiply by
// APL again: that squares brightness and loses the midtone screen illumination.
fn screen_bounce_color() -> vec3<f32> {
    var color = vec3<f32>(dot(u.cmat0.xyz, u.env.rgb), dot(u.cmat1.xyz, u.env.rgb), dot(u.cmat2.xyz, u.env.rgb));
    if (u.mono.w > 0.5) { color = dot(u.env.rgb, vec3<f32>(0.299, 0.587, 0.114)) * u.mono.rgb; }
    return max(color, vec3<f32>(0.0)) * u.fx.w;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(u.cam_pos.xyz - in.world_pos);

    // Screen-space footprint of the faceplate coordinate — how much of the tube's face one
    // output pixel covers. This is what decides whether the mask and the scanlines are
    // resolvable at the current zoom and viewing angle, i.e. how much of the tube's fine
    // structure the display can actually show; both band-limit against it below. Taken here,
    // at the top, because a derivative has to be evaluated in uniform control flow and the
    // material branch just below returns early for every non-screen fragment.
    let uv_fw = fwidth(in.uv);

    // ---- Body: 1=leaded glass, 2=yoke, 3=cabinet plastic, 4=speaker cloth ----
    if (in.material > 0.5) {
        // Two-sided: cull is off so the viewer can see interior faces; flip the normal
        // toward the camera so lighting is correct regardless of triangle winding.
        var nn = n;
        if (dot(nn, v) < 0.0) { nn = -nn; }

        var base: vec3<f32>;
        var rough: f32;
        var metal: f32;
        if (in.material < 1.5) {
            base = vec3<f32>(0.016, 0.017, 0.021); // near-black leaded glass, glossy
            rough = 0.20;
            metal = 0.0;
        } else if (in.material < 2.5) {
            base = vec3<f32>(0.42, 0.26, 0.15);    // deflection yoke: dull copper
            rough = 0.55;
            metal = 1.0;
        } else if (in.material > 3.5 && in.material < 4.5) {
            // Speaker grille (material 4): near-black woven cloth — matte, light-drinking,
            // with interlaced warp/weft catching different amounts of light. The weave is
            // low contrast so it resolves nearby and integrates to cloth from a distance.
            let warp = 0.5 + 0.5 * sin(in.world_pos.x * 760.0);
            let weft = 0.5 + 0.5 * sin(in.world_pos.y * 690.0 + warp * 0.7);
            let weave = mix(warp, weft, 0.48);
            base = vec3<f32>(0.010, 0.010, 0.012) * (0.82 + 0.20 * weave);
            rough = 0.96 - weave * 0.025;
            metal = 0.0;
        } else {
            // Molded cabinet plastic. The base tone + finish are per-brand (the vertex
            // material id selects the family): 3 = Sony charcoal, 5 = RCA warm walnut-
            // brown, 6 = Panasonic cool silver-grey, 7 = beige computer-terminal case.
            metal = 0.0;
            var pbase: vec3<f32>;
            var prough_lo = 0.46;
            var prough_hi = 0.82;
            if (in.material < 3.5) {
                pbase = vec3<f32>(0.030, 0.027, 0.024); // charcoal (Trinitron / PVM / arcade)
            } else if (in.material < 5.5) {
                pbase = vec3<f32>(0.055, 0.032, 0.020); // warm dark walnut-brown (RCA console)
            } else if (in.material < 6.5) {
                pbase = vec3<f32>(0.145, 0.150, 0.160); // cool silver-grey (Panasonic / PC)
                prough_lo = 0.34; prough_hi = 0.66;     // a touch glossier than TV charcoal
                // Silver-painted ABS remains a dielectric; treating it as partially metal
                // tinted every reflection grey and made the cabinet look like cast alloy.
                metal = 0.0;
            } else {
                pbase = vec3<f32>(0.235, 0.210, 0.165); // warm beige (terminal case)
                prough_lo = 0.55; prough_hi = 0.88;     // chalky matte
            }
            base = pbase;
            // Real injection-molded ABS is nearly uniform in colour. Its visible texture
            // is mostly a sub-millimetre roughness/normal variation, with only faint,
            // broad pigment clouds. The previous 40% albedo cells read as camouflage-like
            // CG noise and exposed every projection seam.
            let pigment = molded_noise(in.world_pos, nn, 34.0);
            let pebble = molded_noise(in.world_pos, nn, 310.0);
            let micro = molded_noise(in.world_pos + vec3<f32>(0.37, 0.19, 0.71), nn, 620.0);
            base = base * (0.985 + 0.030 * pigment);
            rough = clamp(0.61 + (pebble - 0.5) * 0.11 + (micro - 0.5) * 0.035,
                          prough_lo, prough_hi);

            // Project a tiny isotropic perturbation into the geometric tangent plane.
            // It breaks a highlight without changing the cabinet silhouette or turning
            // the material into hammered metal. Raised knobs are subtly smoother where
            // years of handling polish the molded texture.
            let rvec = vec3<f32>(
                molded_noise(in.world_pos + vec3<f32>(0.13, 0.31, 0.07), nn, 430.0) - 0.5,
                molded_noise(in.world_pos + vec3<f32>(0.47, 0.17, 0.59), nn, 430.0) - 0.5,
                molded_noise(in.world_pos + vec3<f32>(0.73, 0.61, 0.23), nn, 430.0) - 0.5,
            );
            let tangent_jit = rvec - nn * dot(rvec, nn);
            nn = normalize(nn + tangent_jit * 0.022);
            if (fract(in.material) > 0.10) {
                rough = max(prough_lo, rough - 0.10);
                base = base * 1.015;
            }
            // Ventilation slots: real sets vent heat through fine louvres across the TOP
            // toward the rear. Thin dark grooves (darker albedo + a normal tilt so they
            // self-shade) where the face points up and we're behind the front box.
            if (nn.y > 0.6 && in.world_pos.z < -0.35) {
                let g = abs(fract(in.world_pos.z * 6.5) - 0.5) * 2.0; // 0 at each groove
                let groove = smoothstep(0.10, 0.32, g);
                base = base * mix(0.35, 1.0, groove);
                nn = normalize(nn + vec3<f32>(0.0, 0.0, (fract(in.world_pos.z * 6.5) - 0.5) * 0.7));
            }
            // Cabinet seam: the moulded front bezel meets the rear cabinet along a fine
            // parting line. A thin dark groove around the side/top/bottom faces at z≈-0.5.
            if (abs(nn.z) < 0.6) {
                let s = abs(in.world_pos.z + 0.5);
                base = base * mix(0.5, 1.0, smoothstep(0.0, 0.014, s));
            }
        }

        // Micro-ambient occlusion: contact shadowing for louvres, parting seam, and floor contact
        var ao = 1.0;
        let ground_ao = clamp(in.world_pos.y * 1.1 + 1.15, 0.35, 1.0);
        ao = ao * ground_ao;

        if (abs(nn.z) < 0.6) {
            let s = abs(in.world_pos.z + 0.5);
            ao = ao * mix(0.55, 1.0, smoothstep(0.0, 0.016, s));
        }
        if (nn.y > 0.6 && in.world_pos.z < -0.35) {
            let g = abs(fract(in.world_pos.z * 6.5) - 0.5) * 2.0;
            ao = ao * mix(0.40, 1.0, smoothstep(0.08, 0.30, g));
        }

        var col = shade_body(base, rough, metal, nn, v) * ao;

        // HDR Screen Area Light Bounce onto Front Bezel
        // The active phosphor screen faceplate emits light forward (centered at z ~ 0, within [-HALF_W, HALF_W] x [-HALF_H, HALF_H]).
        // Front fragments catch this emission with directional inverse-square falloff and inner-bevel specular glints.
        let fz = smoothstep(-0.75, -0.02, in.world_pos.z);
        let front = fz * fz;
        let glow_col = screen_bounce_color();
        let son = min(u.pwr.x, 1.0 - u.pwr.y);

        let screen_pt = vec3<f32>(clamp(in.world_pos.x, -HALF_W, HALF_W), clamp(in.world_pos.y, -HALF_H, HALF_H), 0.0);
        let to_screen = screen_pt - in.world_pos;
        let screen_dist = max(length(to_screen), 1e-4);
        let screen_dir = to_screen / screen_dist;
        let screen_ndl = max(dot(nn, screen_dir), 0.0);
        let f0 = mix(vec3<f32>(0.04), base, metal);
        let screen_spec = ggx_spec(nn, v, screen_dir, rough, f0);

        let screen_falloff = 1.0 / (1.0 + screen_dist * screen_dist * 2.8);
        let screen_spill = glow_col * u.phys.z * front * (screen_ndl * 0.90 + 0.18 * max(dot(nn, v), 0.0) + screen_spec * 0.45) * screen_falloff * son;
        col = col + screen_spill;

        return output_color(col);
    }

    // ---- Screen ----
    let res = u.params.xy;

    // Refraction through the thick curved faceplate. The phosphor sits behind the
    // glass, so the view ray bends (Snell) on the way in and lands off-axis — the
    // image shifts and magnifies as you move around the tube. Tracing each primary
    // with its own IOR (blue bends most) adds the chromatic dispersion fringing that
    // real leaded CRT glass shows toward the corners.
    let thick = u.glass.x;

    // --- Power theatre: warmup expand / power-off collapse + degauss wobble ---
    // `open` = 1 is a full raster; as the tube powers off it shrinks to a bright
    // horizontal line (vertical deflection dies), then to a fading phosphor dot
    // (horizontal dies). Warmup runs the same in reverse.
    let open = min(u.pwr.x, 1.0 - u.pwr.y);
    // Overscan: a consumer set scans the raster larger than the visible faceplate, so
    // the picture's outer edges fall off the tube. Sample the centre (1 - 2*os) of the
    // image across the full screen; PC monitors / mono terminals run os≈0 (full raster).
    var base_uv = vec2<f32>(0.5) + (in.uv - vec2<f32>(0.5)) * (1.0 - 2.0 * u.focus.y);
    if (u.pwr.z > 0.001) {
        // Degauss: a decaying AC wobble as the coil demagnetises the shadow mask.
        let tt = u.params.z;
        base_uv = base_uv + vec2<f32>(sin(base_uv.y * 34.0 + tt * 62.0),
                                      cos(base_uv.x * 26.0 + tt * 55.0)) * u.pwr.z * 0.006;
    }
    let vy = max(clamp((open - 0.5) * 2.0, 0.0, 1.0), 0.006); // raster height fraction
    let hx = max(clamp(open * 2.0, 0.0, 1.0), 0.004);         // raster width fraction
    base_uv = vec2<f32>((base_uv.x - 0.5) / hx + 0.5, (base_uv.y - 0.5) / vy + 0.5);
    let in_raster = step(0.0, base_uv.x) * step(base_uv.x, 1.0)
                  * step(0.0, base_uv.y) * step(base_uv.y, 1.0);
    let concentrate = clamp(1.0 / sqrt(vy * hx), 1.0, 6.0); // beam energy concentration
    let hot = smoothstep(0.5, 0.0, open);                   // white-hot near collapse

    // Raster geometry: warp the (power-mapped) sampling coordinate for the yoke's
    // deflection errors (pincushion/keystone/etc.). Physical tube-face effects
    // (corner rounding, vignette, damper wires, glare) keep using the true in.uv.
    let ruv = geometry_warp(base_uv);
    // Convergence error: the three electron guns never register perfectly, and the
    // misalignment grows radially toward the corners (a well-set PVM is tight, a
    // tired consumer set fringes red/blue at the edges). Push red out, blue in.
    let cvec = ruv - vec2<f32>(0.5);
    let conv = cvec * dot(cvec, cvec) * u.look.x * 0.9;
    // Dispersion. CRT faceplates are a barium/strontium silicate, n_d ≈ 1.523 with an Abbe
    // number around 55, so n_F − n_C = 0.523/55 = 0.0095 between the blue and red Fraunhofer
    // lines — the three primaries land near 1.5185 / 1.5230 / 1.5280. The old spread was
    // 1.518-1.522, i.e. 0.004, which quietly understated a real panel's fringing by 2.4×.
    let uv_r = refract_uv(ruv, n, v, thick, 1.0 / 1.5185) + conv;
    let uv_g = refract_uv(ruv, n, v, thick, 1.0 / 1.5230);
    let uv_b = refract_uv(ruv, n, v, thick, 1.0 / 1.5280) - conv;
    let uv = uv_g; // base uv for halation / vignette

    // Deflection defocus: off-axis the electron beam travels farther and the deflection
    // field grows, so the spot widens (astigmatic — elongates horizontally at the sides,
    // worst in the corners) and the picture softens toward the edges. r2 grows to the
    // corners; a 4th-order term makes the corners bloom hardest. u.focus.x = the tube's
    // edge-focus quality (a PVM ~0, a fuzzy RCA/arcade blooms). Deflection is
    // a position on the physical faceplate, not the inverse-mapped source signal.
    // During power collapse that signal coordinate can be far outside [0,1];
    // using it here invents enormous spots and unbounded convolution workloads.
    let dfv = in.uv - vec2<f32>(0.5);
    let r2 = dot(dfv, dfv);
    let vscale = 1.0 + u.focus.x * (2.0 * r2 + 3.5 * r2 * r2);
    let src_px = uv_fw * res; // source columns/scanlines covered by one output pixel
    var col = vec3<f32>(
        scan_reconstruct(uv_r, res, vscale, src_px).r,
        scan_reconstruct(uv_g, res, vscale, src_px).g,
        scan_reconstruct(uv_b, res, vscale, src_px).b,
    );
    // Horizontal astigmatism: the spot elongates most horizontally along the side
    // edges (|dfv.x|), so blur the sampled colour laterally there. Two taps, ~0 in the
    // centre, so it only softens the edges/corners like a real over-deflected beam.
    if (u.focus.x > 0.0) {
        let hamt = clamp(u.focus.x * (0.7 * abs(dfv.x) + 1.6 * r2), 0.0, 0.5);
        // Spot elongation, so again a distance on the glass: 0.16% to 1.7% of the picture
        // width (0.6 mm to 7 mm), which is the range a badly over-deflected corner spot
        // actually smears over.
        let hoff = vec2<f32>(0.0015625 + 0.00625 * hamt, 0.0);
        // These are raw phosphor-plane samples in signal units, but `col` has already been
        // through the tube drive, so scale them to match before mixing — otherwise the
        // "blur" halves the brightness of whatever it softens and the astigmatism reads as
        // an extra corner vignette. (The CRT transfer curve is applied further down, to
        // both together, so it must NOT be pre-applied here.)
        let hb = 0.5 * (textureSampleLevel(t_screen, s_screen, uv + hoff, 0.0).rgb
                      + textureSampleLevel(t_screen, s_screen, uv - hoff, 0.0).rgb) * u.tone.z;
        col = mix(col, hb, hamt);
    }

    // (The CRT transfer curve and the phosphor white point are applied at the signal→light
    // boundary in fs_phosphor, so `col` and every phosphor-plane tap below are already in
    // the same light units — see the note there.)

    // High-voltage sag, driven by average picture level (APL). The flyback supplies the
    // final anode through a finite source impedance, so total beam current pulls the anode
    // voltage down and the whole picture dims — the "breathing" you see when a scene cuts
    // to full white. It is why a CRT's full-field white is well below its small-window
    // white, the same measurement an LCD calls ANSI vs peak contrast: a pro monitor with a
    // regulated supply gives up only a few percent, a cheap consumer chassis 10-15%.
    // phys.w carries the per-tube coefficient (see write_uniforms).
    //
    // The additive highlight bloom that used to sit here is gone — it double-counted the
    // beam growth already in the reconstruction and added energy on top. See phys.w's note.
    let apl = u.env.w;
    col = col * (1.0 - apl * u.phys.w);

    // Supply ripple sampled by successive fields. The CPU supplies the signed
    // difference between full-wave mains and twice the field rate. Two spatial
    // periods retain the rectified supply's two cycles per field; the slowly
    // changing phase is the beat, not a second phosphor-decay animation.
    col = col * hum_modulation(in.uv.y, u.params.z, u.focus.z, u.focus.w);

    // Monochrome tube: a single electron gun paints ONE phosphor colour scaled by the
    // signal's luminance — no colour triads, no convergence (a green/amber terminal).
    // The colour mask is already skipped via mask_strength=0; damper wires, halation
    // and glow tint are gated on this flag below.
    if (u.mono.w > 0.5) {
        col = dot(col, vec3<f32>(0.299, 0.587, 0.114)) * u.mono.rgb;
    }

    // (Phosphor persistence + the raster field sweep are now integrated over real
    // frame history in the accum pass, so there's no per-fragment temporal fake here.)

    // The scatter taps below read the phosphor plane directly. It now holds LIGHT (the
    // transfer curve and white point are applied at the top of the accum pass), so the only
    // thing between a tap and `col` is the tube drive — one multiply, no second copy of the
    // exponent. Getting this wrong is what used to break conservation: the fraction taken
    // off `col` was worth about twice the fraction added back from the taps, so every tube
    // lost a few percent of brightness in proportion to how much it scattered.
    let emit = u.tone.z * (1.0 - apl * u.phys.w);

    // Halation: light scattering laterally inside the glass, biased warm/red
    // because the red phosphor persists longest. Sampled around the parallax uv.
    // Every radius below is a distance ON THE GLASS, so it is expressed as a fraction of the
    // picture width (× 4/3 on the short axis to stay circular on a 4:3 face) — NOT, as they
    // all were, as a count of source texels. That distinction is invisible while the source
    // is a 320-wide console frame, which is exactly what these numbers were calibrated
    // against, and wrong for everything else this app is normally pointed at: capture a
    // 1920-wide desktop window and a "2.5 texel" halation radius shrinks from 3.1 mm on the
    // faceplate to 0.5 mm, so the glass stops scattering the moment the source gets sharper.
    // Light in a panel does not know the resolution of the signal.
    let halo = u.optics.w;
    if (halo > 0.0) {
        let hr = 0.0078; // ≈3.1 mm on a 400 mm face — the phosphor→front-surface→back bounce
        let px = vec2<f32>(hr, hr * HALF_W / HALF_H);
        var glow = vec3<f32>(0.0);
        glow = glow + textureSample(t_screen, s_screen, uv + vec2<f32>(px.x, 0.0)).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv - vec2<f32>(px.x, 0.0)).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv + vec2<f32>(0.0, px.y)).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv - vec2<f32>(0.0, px.y)).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv + px).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv - px).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv + vec2<f32>(px.x, -px.y)).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv + vec2<f32>(-px.x, px.y)).rgb;
        glow = max(glow / 8.0, vec3<f32>(0.0)) * emit;
        // Scattering REDISTRIBUTES light: whatever bounces off the phosphor and comes back
        // out somewhere else left the spot it started from. So the scattered fraction comes
        // OFF the direct term before the blurred copy is added back, and the two use the
        // same fraction — that is what makes this a glow instead of a brightness offset.
        // Added on its own, as it was before, a flat white field simply got uniformly
        // brighter, which is a gain; only a pixel's difference from its surround is glow.
        // Conserving it means a flat field passes through untouched while an isolated
        // highlight dims a shade as it blooms, which is what one does on real glass.
        let hshare = halo * u.beam2.w;
        // Wavelength-dependent scattering redistributes each primary separately.
        // Tinting only the added term creates red energy even on a flat field.
        let share = vec3<f32>(hshare) * select(vec3<f32>(1.0, 0.6, 0.45), vec3<f32>(1.0), u.mono.w > 0.5);
        let gl = select(glow, vec3<f32>(dot(glow, vec3<f32>(0.299, 0.587, 0.114))) * u.mono.rgb, u.mono.w > 0.5);
        col = col * (vec3<f32>(1.0) - share) + gl * share;
    }

    // Diffusion — a SECOND, wider bloom scale, physically distinct from halation.
    // Halation (above) is light reflecting off the phosphor back through the glass:
    // tight and warm/red. Diffusion is light SCATTERING inside the thick imperfect
    // faceplate — a broad, soft, near-neutral haze that lifts the whole lit region and
    // gives bright CRT content its dense, "wet" glow. Two rings so the falloff is smooth
    // rather than a single hard radius, and the ring weights matter as much as the radii:
    // scattering in glass has a long-tailed PSF — most of the light stays close to where it
    // was emitted and a thin tail reaches far. Light launched into a ~10 mm panel past the
    // critical angle walks ~2·t·tan(41°) ≈ 17 mm before it escapes, which is a good 7% of a
    // 240 mm face, so the outer ring goes wider than it used to; but that far excursion is
    // the tail, not the bulk, so it carries proportionally much less weight. Giving the wide
    // ring a third of the energy (as an even split did) is what turns a haze into a wash:
    // it drags neighbouring bright content across narrow dark features and desaturates.
    let diff_amt = u.fx.y;
    if (diff_amt > 0.0) {
        // 1.6% and 4.4% of the picture width — ~6 mm and ~18 mm on a 400 mm face, the second
        // matching the ~2·t·tan(41°) an over-critical-angle ray walks inside a 10 mm panel.
        let r1 = vec2<f32>(0.015625, 0.015625 * HALF_W / HALF_H);
        let r2 = vec2<f32>(0.04375, 0.04375 * HALF_W / HALF_H);
        var d = textureSample(t_screen, s_screen, uv + vec2<f32>(r1.x, 0.0)).rgb
              + textureSample(t_screen, s_screen, uv - vec2<f32>(r1.x, 0.0)).rgb
              + textureSample(t_screen, s_screen, uv + vec2<f32>(0.0, r1.y)).rgb
              + textureSample(t_screen, s_screen, uv - vec2<f32>(0.0, r1.y)).rgb
              + textureSample(t_screen, s_screen, uv + r1).rgb
              + textureSample(t_screen, s_screen, uv - r1).rgb
              + textureSample(t_screen, s_screen, uv + vec2<f32>(r1.x, -r1.y)).rgb
              + textureSample(t_screen, s_screen, uv + vec2<f32>(-r1.x, r1.y)).rgb;
        var d2 = textureSample(t_screen, s_screen, uv + vec2<f32>(r2.x, 0.0)).rgb
               + textureSample(t_screen, s_screen, uv - vec2<f32>(r2.x, 0.0)).rgb
               + textureSample(t_screen, s_screen, uv + vec2<f32>(0.0, r2.y)).rgb
               + textureSample(t_screen, s_screen, uv - vec2<f32>(0.0, r2.y)).rgb
               + textureSample(t_screen, s_screen, uv + r2).rgb
               + textureSample(t_screen, s_screen, uv - r2).rgb
               + textureSample(t_screen, s_screen, uv + vec2<f32>(r2.x, -r2.y)).rgb
               + textureSample(t_screen, s_screen, uv + vec2<f32>(-r2.x, r2.y)).rgb;
        // Long-tailed PSF: tight core, faint wide skirt — and through the same transfer as
        // the halation taps, so the fraction added back matches the fraction taken off.
        let diff = max((d + d2 * 0.30) / 10.4, vec3<f32>(0.0)) * emit;
        // Same conservation as halation: light scattered sideways inside the panel is light
        // that did not come straight out, so the same fraction comes off the direct term as
        // goes back on. A single-phosphor tube keeps the same hue in its scattered light.
        let dshare = diff_amt * u.beam2.w;
        let share = vec3<f32>(dshare) * select(vec3<f32>(1.0, 0.95, 0.9), vec3<f32>(1.0), u.mono.w > 0.5);
        let dl = select(diff, vec3<f32>(dot(diff, vec3<f32>(0.299, 0.587, 0.114))) * u.mono.rgb, u.mono.w > 0.5);
        col = col * (vec3<f32>(1.0) - share) + dl * share;
    }

    // Phosphor mask. The mask is a physical object GLUED TO THE TUBE — a grille of
    // 583 stripe triads across a 20" Trinitron's 385 mm face, 1235 across a PVM-20L5's
    // 0.31 mm grille, 1467 across a 0.24 mm Diamondtron. So its coordinate is the
    // faceplate itself (in.uv), not the framebuffer: it curves with the glass, foreshortens
    // as the tube turns, and magnifies when the camera moves in. Keying it off in.clip.xy —
    // a fixed pitch in output pixels, as this did — pinned the grille to the screen instead,
    // so the picture slid across a stationary screen-door as you orbited, the tube's real
    // pitch was replaced by a hand-set 2.2-4.5 px (which had the PVM's grille COARSER than a
    // consumer Trinitron's, the reverse of the truth), and at any zoom the pattern beat
    // against the pixel grid into coloured moire.
    //
    // glass.w carries the triad count across the picture width, computed from the tube's
    // measured mask pitch and visible width. The vertical count is scaled by the 3:4 face
    // aspect so a triad is square on the glass, as it is on a real mask.
    // fwidth gives the pixel footprint, which is what band-limits the pattern (see
    // phosphor3): far away it integrates to a flat field, up close it resolves.
    let tri_x = max(u.glass.w, 1.0);
    let face = vec2<f32>(1.0, HALF_H / HALF_W); // uv → square units on the 4:3 face
    let tc = in.uv * face * tri_x;
    let tc_fw = uv_fw * face * tri_x;
    // The subpixel path is a deliberate exception: it is the one pattern that belongs to the
    // DISPLAY rather than to the tube, so it stays keyed to output pixels — and it is
    // meaningless once the frame is supersampled, because the resolve averages the very
    // subpixels it is trying to drive. Fall through to the physical mask in that case rather
    // than drawing a pattern that cannot survive to the screen.
    if (u.fx.z > 0.5 && u.mono.w < 0.5 && u.params.w < 1.5) {
        // Subpixel-accurate mask: each real LCD subpixel driven by the matching phosphor.
        // Same unit-mean normalisation as the resolution-independent path below — the DC of
        // mask_subpixel is (1.0 + 0.06 + 0.06)/3 = 0.37333 per channel across the LCD triad.
        // The old `× (1 + strength·2.0)` over-compensated by ~22% at strength 0.9 (2.8 where
        // the mean calls for 2.29), so turning the subpixel mask on brightened the picture.
        let m = mask_subpixel(in.clip.xy);
        let mm = mix(1.0, 0.37333, u.optics.y);
        col = col * mix(vec3<f32>(1.0), m, u.optics.y) / max(mm, 1e-3);
    } else {
        // The mask is a spatial modulation, not a dimmer. A real tube's rated white is
        // measured THROUGH its own mask: the set runs whatever beam current it needs to hit
        // that white, which is exactly why a shadow-mask tube (~16% open area) drives harder
        // than an aperture grille (~24%) and still reaches the same peak. So the mask must
        // be normalised to unit mean — then mask_strength controls how deep the stripes cut
        // (structure) and nothing else, and the tube drive alone sets brightness.
        //
        // It used to be `× (1 + strength·0.7)`, a hand-fitted number that under-compensated:
        // at the Trinitron's strength 0.9 the mask path passed mix(1, 0.263, 0.9) = 0.337 of
        // the light and handed back 1.63×, netting 0.549 — a 45% loss the drive constant had
        // silently absorbed. Worse, the loss varied with mask_strength AND mask type, so the
        // ten tubes were sitting at brightnesses that differed for no physical reason. The
        // exact normaliser is 1/mean(mix(1, m, strength)), which is one reciprocal.
        let m = mask(tc, tc_fw, u.optics.x);
        let mm = mix(1.0, mask_mean(u.optics.x), u.optics.y); // DC of the modulation
        col = col * mix(vec3<f32>(1.0), m, u.optics.y) / max(mm, 1e-3);
    }

    // Damper wires: the signature of an aperture-grille (Trinitron) tube. The
    // vertical phosphor strips are steadied by 1-2 fine horizontal tension wires
    // that cast a soft thin shadow across the whole picture. Aperture grille only
    // (not on a monochrome tube, which has no grille).
    if (u.optics.x < 0.5 && u.mono.w < 0.5) {
        let wy = in.uv.y;
        // two wires (large-set layout) at ~1/3 and ~2/3 height; ~1.5px soft shadow.
        // Depth: the wire is a tungsten filament ~0.015 mm across on a 20" tube (Sony ran
        // roughly 0.012-0.020 mm), strung a few mm in front of the grille. It occludes its
        // own width of beam, but it sits far enough forward that the shadow is penumbral —
        // the beam converges through a finite crossover, so the wire never fully blocks any
        // one point, and what lands on the phosphor is a soft dip a few tenths of a mm wide.
        // Against a ~0.4 mm scanline pitch that is a shallow attenuation, which is why the
        // wires are famously invisible on picture content and only show against a flat
        // bright field. 0.45 was drawing them as hard black rules across the screen; the
        // measured depth is more like a fifth of the local brightness at the very centre.
        let w = 0.0016;
        let s1 = exp(-(wy - 0.333) * (wy - 0.333) / (2.0 * w * w));
        let s2 = exp(-(wy - 0.667) * (wy - 0.667) / (2.0 * w * w));
        col = col * (1.0 - 0.20 * (s1 + s2));
    }

    // Secondary internal reflection ("ghost"): the double exposure on a thick, glossy CRT
    // faceplate. Light leaves the phosphor, part of it reflects back off the glass-air front
    // surface, bounces off the aluminised backing behind the phosphor, and escapes on a
    // second pass — so it has crossed the panel three times where the direct image crossed
    // it once. That is the geometry, and it means the offset is the SAME refraction the
    // direct image gets, traced to three times the depth: zero when you look straight on,
    // sliding out and growing as you move off-axis, always in the direction the picture
    // itself is displaced. It used to be a fixed (0.011, -0.008) in uv — a hard-coded
    // up-and-right double image that sat there even head-on and never moved as the tube
    // turned, which is the one thing a reflection cannot do.
    var gcol = max(textureSampleLevel(t_screen, s_screen,
                                      refract_uv(ruv, n, v, thick * 3.0, 1.0 / 1.5230), 0.0).rgb,
                   vec3<f32>(0.0)) * emit;
    if (u.mono.w > 0.5) { gcol = dot(gcol, vec3<f32>(0.299, 0.587, 0.114)) * u.mono.rgb; }
    // Strength is the product of the two reflectances on that path: ~4% at the glass-air
    // front surface (Schlick F0 for n=1.52) times the aluminium backing's ~75%, so ~3% —
    // and it is light that did NOT escape on the first pass, so it comes off the direct term
    // rather than being added on top of it.
    col = col * (1.0 - u.look.w) + gcol * u.look.w;

    // Rounded phosphor rectangle: the usable screen area is a rounded rect, not a
    // sharp box, so the extreme corners fade to black. Aspect-correct x so the
    // radius is geometrically round on the 4:3 face.
    let ar = HALF_W / HALF_H;
    let acc = (in.uv - vec2<f32>(0.5)) * 2.0 * vec2<f32>(ar, 1.0);
    let ext = vec2<f32>(ar, 1.0);
    let cr = max(u.look.y, 0.001);
    let cd = length(max(abs(acc) - (ext - cr), vec2<f32>(0.0))) - cr;
    col = col * (1.0 - smoothstep(0.0, 0.06, cd));

    // Purity: residual magnetization mislands the beam onto the wrong phosphors,
    // tinting broad patches of the picture (the discoloration a degauss clears). Two
    // soft off-axis blotches shift the colour balance — one warm, one cool.
    if (u.geom.w > 0.0) {
        let d1 = in.uv - vec2<f32>(0.20, 0.24);
        let d2 = in.uv - vec2<f32>(0.84, 0.78);
        let b1 = exp(-dot(d1, d1) * 5.0);
        let b2 = exp(-dot(d2, d2) * 6.0);
        let tint = vec3<f32>(1.0)
            + vec3<f32>(0.10, -0.06, -0.05) * b1
            + vec3<f32>(-0.05, 0.03, 0.09) * b2;
        col = col * mix(vec3<f32>(1.0), tint, u.geom.w);
    }

    // Tube vignette.
    let vd = distance(in.uv, vec2<f32>(0.5, 0.5));
    col = col * mix(1.0, 1.0 - u.glass.z, smoothstep(0.30, 0.92, vd));

    // Real phosphor colorimetry: map the tube's drive RGB through its measured gamut
    // and native white point into sRGB (SMPTE-C green is less saturated, its red is
    // oranger; a 9300K set reads blue). Mono tubes pass through (identity matrix).
    // Applied to the phosphor light only — the glass reflections below stay neutral.
    col = max(vec3<f32>(dot(u.cmat0.xyz, col), dot(u.cmat1.xyz, col), dot(u.cmat2.xyz, col)),
              vec3<f32>(0.0));

    // Power collapse/warmup: mask the black surround, concentrate the beam into the
    // shrinking line/dot, whiten it hot, then fade the final phosphor dot to black.
    col = col * in_raster * concentrate;
    col = mix(col, vec3<f32>(max(max(col.r, col.g), col.b)), hot * 0.7);
    col = col * (1.0 - smoothstep(0.82, 1.0, u.pwr.y)); // dot fades out at the end
    // Degauss rainbow purity: moving colour bands that ripple across and fade.
    if (u.pwr.z > 0.001) {
        let p = base_uv.y * 16.0 + base_uv.x * 6.0 + u.params.z * 22.0;
        let rainbow = vec3<f32>(sin(p), sin(p + 2.094), sin(p + 4.188));
        col = col * (vec3<f32>(1.0) + rainbow * u.pwr.z * 0.55);
    }

    // Black-frame insertion (motion): a real CRT flashes each pixel for well under a
    // millisecond and is dark the rest of the field, so motion is impulse-sharp — where
    // an LCD holds every frame for its whole refresh and smears. On a high-refresh panel
    // we strobe the emitted phosphor light dark on alternate refreshes to imitate that
    // impulse. Only the EMISSION strobes (fx.w→0 on a dark frame); the glass still
    // mirrors the lit room below, exactly as a switched-off-but-lit tube does.
    col = col * u.fx.w;

    // Faceplate glass = a dark, slightly-reflective mirror. This is THE defining CRT
    // cue (see any photo of a real set): even head-on the glass bounces ~4% of the room
    // (Schlick F0≈0.043 for glass↔air), rising to a full mirror at grazing — so a dark
    // screen clearly reflects the lit room, and the reflection warps over the curved
    // faceplate and slides as you orbit. Additive, so a bright picture washes it out
    // (just like a real tube) while dark content mirrors the room.
    let ndotv = max(dot(n, v), 0.0);
    let fres = 0.043 + 0.957 * pow(1.0 - ndotv, 5.0);
    let refl = reflect(-v, n);
    col = col + room(refl) * fres * (0.35 + 1.1 * u.glass.y);
    // Ambient wash: the other half of "a CRT in a lit room isn't black". The specular term
    // above is the front surface acting as a 4% mirror — sharp, and it slides as you orbit.
    // This is the DIFFUSE half: room light that gets through the panel, scatters off the
    // phosphor powder and the mask behind it, and comes back out. It crosses the tinted
    // faceplate twice against the picture's once, which is the entire reason tubes were
    // built with 40–60% transmission glass in the first place — halving transmission costs
    // you half your brightness but quarters the ambient wash, so contrast doubles. Being
    // diffuse it lifts the black floor evenly instead of mirroring anything, so blacks go
    // to a dark grey and stay there whatever angle you view from: the thing that makes a
    // real dark screen read as glass over a grey powder rather than as a hole.
    // Diffuse, so it samples the room broadly (around the normal) rather than in the mirror
    // direction, and it falls off at grazing by (1-F)² — the same light has to get in
    // through the front surface and back out through it, and both get harder as F rises.
    let amb_room = select((room(n) * 2.0 + room(refl)) / 3.0, u.camera_ambient.rgb, u.camera.x > 0.5);
    col = col + amb_room * u.beam2.z * (1.0 - fres) * (1.0 - fres);
    // Tight specular glare from the ceiling softbox — a hot spot sliding across the
    // curved glass as you move; the single most CRT-reading highlight.
    //
    // Magnitude is a Fresnel problem, not a taste problem. Bare glass reflects R0 ≈ 4% at
    // normal incidence (n ≈ 1.52); a bonded/AR-coated pro panel is nearer 0.5-1%, which is
    // what `glass.y` (the preset's `reflection`) is scaling between. What the eye sees is
    // R0 × the source's luminance relative to the tube's white, and a ceiling fixture runs
    // maybe 5-15× a CRT's ~100 nit white — so the hot spot lands around 0.04 × 10 ≈ 0.4 of
    // white, bright enough to bloom through the tonemapper but not a blown highlight. The
    // old × 2.0 put it at ~1.6 — four times over, a mirror-finish sheen no faceplate has.
    let light_dir = normalize(vec3<f32>(-0.35, 0.55, 0.95));
    let glare = pow(max(dot(refl, light_dir), 0.0), 130.0);
    col = col + vec3<f32>(1.0, 0.98, 0.92) * glare * (0.3 + u.glass.y) * 0.5 * u.pwr.w * select(1.0, 0.0, u.camera.x > 0.5);

    return output_color(col);
}
