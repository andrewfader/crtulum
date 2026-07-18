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
    optics: vec4<f32>,  // x=mask_type(0 grille,1 shadow,2 slot), y=mask_strength, z=scanline, w=halation
    glass: vec4<f32>,   // x=parallax_depth, y=reflection, z=vignette, w=mask_pitch(px)
    tone: vec4<f32>,    // x=hdr(0 tonemap→SDR, 1 scRGB passthrough), y=peak(white pt), z=beam_drive, w=ntsc_strength
    scan: vec4<f32>,    // beam math: x=beam_min(width, dark), y=beam_max(width, bright), z=beam_shape, w=beam_range
    env: vec4<f32>,     // xyz=avg source color, w=avg picture level (screen area-light)
    look: vec4<f32>,    // x=convergence, y=corner_radius, z=grain, w=ghost
    phys: vec4<f32>,    // x=crt_gamma, y=warmth, z=glow_bounce, w=bloom
    temporal: vec4<f32>,// x=dt(sec), y=persist_mult, z=interlace, w=field_parity
    ptau: vec4<f32>,    // per-phosphor decay tau: xyz = R,G,B (sec); w unused
    geom: vec4<f32>,    // raster geometry: x=pincushion, y=trapezoid, z=corner_pin, w=purity
    mono: vec4<f32>,    // monochrome phosphor tint (rgb) + flag (w>0.5 = single-gun)
    cmat0: vec4<f32>,   // CRT-phosphor → sRGB colour matrix rows (real gamut + white pt)
    cmat1: vec4<f32>,
    cmat2: vec4<f32>,
    pwr: vec4<f32>,     // power: x=warmup(0..1), y=collapse(0..1), z=degauss(0..1), w unused
};

@group(0) @binding(0) var<uniform> u: Uniforms;
// Tube pass: t_screen = the persisted phosphor plane. Accum pass: t_screen = the
// raw source frame and t_prev = the previous phosphor plane (fed back for decay).
@group(0) @binding(1) var t_screen: texture_2d<f32>;
@group(0) @binding(2) var s_screen: sampler;
@group(0) @binding(3) var t_prev: texture_2d<f32>;

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

// Three phosphor stripes (R,G,B) across a triad, evaluated periodically so the
// pattern wraps cleanly. `t` in [0,1) is the position within one triad.
fn phosphor3(t: f32) -> vec3<f32> {
    let w = 0.13;
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    // include neighbor copies (t-1, t+1) so gaussians wrap at triad seams
    for (var k = -1; k <= 1; k = k + 1) {
        let tk = t + f32(k);
        r = r + gauss(tk, 1.0 / 6.0, w);
        g = g + gauss(tk, 3.0 / 6.0, w);
        b = b + gauss(tk, 5.0 / 6.0, w);
    }
    return vec3<f32>(r, g, b);
}

// Phosphor mask weights at framebuffer pixel `px`.
fn mask(px: vec2<f32>, kind: f32, pitch: f32) -> vec3<f32> {
    if (kind < 0.5) {
        // aperture grille (Trinitron): continuous vertical RGB stripes
        return phosphor3(fract(px.x / pitch));
    } else if (kind < 1.5) {
        // shadow mask: RGB dot triads, every other row staggered by half a triad
        let row = floor(px.y / pitch);
        let stagger = select(0.0, 0.5, (i32(row) - (i32(row) / 2) * 2) != 0);
        let stripes = phosphor3(fract(px.x / pitch + stagger));
        let ty = fract(px.y / pitch);
        let dot = gauss(ty, 0.5, 0.30);
        return stripes * mix(0.35, 1.0, dot);
    } else {
        // slot mask (many consumer sets): vertical slots, columns staggered
        let stripes = phosphor3(fract(px.x / pitch));
        let seg = floor(px.x / pitch);
        let stagger = select(0.0, 0.5, (i32(seg) - (i32(seg) / 2) * 2) == 0);
        let ty = fract(px.y / (pitch * 2.0) + stagger);
        let slot = smoothstep(0.0, 0.12, ty) * smoothstep(1.0, 0.88, ty);
        return stripes * mix(0.45, 1.0, slot);
    }
}

// Extended Reinhard tonemap: identity-ish below 1.0, rolls HDR highlights up to
// `peak` (the white point) back into the displayable [0,1] range for SDR output.
fn tonemap(c: vec3<f32>, peak: f32) -> vec3<f32> {
    let w2 = max(peak * peak, 1.0);
    return (c * (1.0 + c / w2)) / (1.0 + c);
}

// Synthetic HDR room reflected in the glass and the plastic. Values run well above
// 1.0 so the light sources bloom in the reflections — a dark room with a soft
// ceiling area-light, a warm lamp to the right, and a faint cool fill to the left.
fn room(r: vec3<f32>) -> vec3<f32> {
    let up = clamp(r.y * 0.5 + 0.5, 0.0, 1.0);
    var c = mix(vec3<f32>(0.014, 0.013, 0.016), vec3<f32>(0.055, 0.065, 0.085), up);
    // soft rectangular ceiling light (area window)
    let win = smoothstep(0.45, 0.98, r.y) * smoothstep(0.6, 0.05, abs(r.x + 0.1));
    c = c + vec3<f32>(1.3, 1.35, 1.5) * win * 3.0;
    // warm lamp off to the right
    let lamp = pow(max(dot(r, normalize(vec3<f32>(0.75, 0.05, 0.55))), 0.0), 24.0);
    c = c + vec3<f32>(1.0, 0.72, 0.45) * lamp * 4.0;
    // faint cool fill from the left
    let fill = pow(max(dot(r, normalize(vec3<f32>(-0.7, 0.2, 0.4))), 0.0), 6.0);
    c = c + vec3<f32>(0.30, 0.35, 0.45) * fill * 0.6;
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

// Physically-based shade for the tube body / bezel: two lights, hemispheric ambient,
// roughness-blurred HDR environment reflection with Fresnel.
fn shade_body(base: vec3<f32>, rough: f32, metal: f32, n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let l0 = normalize(vec3<f32>(0.30, 0.75, 0.55)); // key (ceiling light direction)
    let l1 = normalize(vec3<f32>(-0.60, 0.10, 0.50)); // warm fill
    let f0 = mix(vec3<f32>(0.04), base, metal);
    let kd = base * (1.0 - metal);
    let amb = mix(vec3<f32>(0.03, 0.032, 0.04), vec3<f32>(0.11, 0.12, 0.15), n.y * 0.5 + 0.5);
    var col = kd * (amb
        + vec3<f32>(1.0, 1.0, 0.98) * max(dot(n, l0), 0.0) * 0.8
        + vec3<f32>(1.0, 0.80, 0.60) * max(dot(n, l1), 0.0) * 0.35);
    col = col + ggx_spec(n, v, l0, rough, f0) * 3.0 + ggx_spec(n, v, l1, rough, f0) * 1.5;
    let refl = reflect(-v, n);
    let env = mix(room(refl), amb * 2.0, rough); // rougher → duller reflection
    let fres = f_schlick(max(dot(n, v), 0.0), f0);
    col = col + env * fres * (1.0 - rough * 0.7);
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
const NTSC_FSC: f32 = 0.25; // colour-subcarrier cycles per source texel (~4 texels/cycle)

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
// Subcarrier phase at source column `px` on `line`. The +PI*line term flips the
// subcarrier 180° every scanline (real NTSC line timing) and the time term makes the
// residual crawl, so the dot pattern shimmers up the screen like a real set.
fn subcarrier(px: f32, line: f32, t: f32) -> f32 {
    return TAU * NTSC_FSC * px + PI * line + t * 6.0;
}

// Encode RGB→composite along the scanline, then decode: band-limited luma low-pass +
// quadrature chroma demod. Imperfect luma/subcarrier separation → dot crawl; luma
// energy near the subcarrier leaking into chroma → cross-colour rainbow; the narrow
// chroma passband → horizontal colour bleed. This is the analog-signal look.
fn ntsc(uv: vec2<f32>, res: vec2<f32>, t: f32) -> vec3<f32> {
    let px = uv.x * res.x;
    let line = floor(uv.y * res.y);
    var y_acc = 0.0;
    var i_acc = 0.0;
    var q_acc = 0.0;
    var yw = 0.0;
    var cw = 0.0;
    for (var k = -8; k <= 8; k = k + 1) {
        let sx = px + f32(k);
        let src = textureSampleLevel(t_screen, s_screen, vec2<f32>(sx / res.x, uv.y), 0.0).rgb;
        let yiq = rgb2yiq(src);
        let ph = subcarrier(sx, line, t);
        let comp = yiq.x + yiq.y * cos(ph) + yiq.z * sin(ph); // composite sample
        let kk = f32(k * k);
        // Grounded NTSC bandwidths: luma ~4.2 MHz (fairly sharp), chroma I/Q ~1.3/0.4
        // MHz (≈1/3 of luma → ~3x wider kernel → the horizontal colour bleed).
        let lw = exp(-kk / (2.0 * 1.3 * 1.3)); // luma low-pass (leaves some subcarrier)
        let bw = exp(-kk / (2.0 * 3.4 * 3.4)); // chroma band (narrow → bleed)
        y_acc = y_acc + comp * lw;
        yw = yw + lw;
        i_acc = i_acc + comp * cos(ph) * bw;
        q_acc = q_acc + comp * sin(ph) * bw;
        cw = cw + bw;
    }
    let yiq = vec3<f32>(y_acc / yw, 2.0 * i_acc / cw, 2.0 * q_acc / cw);
    return max(yiq2rgb(yiq), vec3<f32>(0.0));
}

// S-video: luma and chroma travel on separate wires, so there's perfect Y/C
// separation — no dot crawl, no cross-colour rainbow — but chroma is still
// band-limited (the horizontal colour bleed remains). Sharp luma, soft colour.
fn svideo(uv: vec2<f32>, res: vec2<f32>) -> vec3<f32> {
    let px = uv.x * res.x;
    var y = 0.0;
    var yw = 0.0;
    var i = 0.0;
    var q = 0.0;
    var cw = 0.0;
    for (var k = -6; k <= 6; k = k + 1) {
        let sx = px + f32(k);
        let yiq = rgb2yiq(textureSampleLevel(t_screen, s_screen, vec2<f32>(sx / res.x, uv.y), 0.0).rgb);
        let kk = f32(k * k);
        let lw = exp(-kk / (2.0 * 0.9 * 0.9)); // sharp luma (no subcarrier to reject)
        let bw = exp(-kk / (2.0 * 3.4 * 3.4)); // chroma bandwidth → colour bleed
        y = y + yiq.x * lw;
        yw = yw + lw;
        i = i + yiq.y * bw;
        q = q + yiq.z * bw;
        cw = cw + bw;
    }
    return max(yiq2rgb(vec3<f32>(y / yw, i / cw, q / cw)), vec3<f32>(0.0));
}

// Per-channel electron-beam width from that channel's drive (the guest-advanced /
// Sony-Megatron "beam math"). A bright channel draws more beam current, so its spot
// blooms wider vertically and its scanlines merge; a dim channel stays a tight,
// separated line. beam_min/max are half-widths in source-texel rows; beam_shape
// curves how fast width grows with signal.
fn beam_width(c: vec3<f32>) -> vec3<f32> {
    let s = pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(u.scan.z));
    return mix(vec3<f32>(u.scan.x), vec3<f32>(u.scan.y), s);
}

// Reconstruct the beam-scanned color at `uv` from the phosphor plane (already
// NTSC-decoded and time-integrated by the accum pass). Each nearby source row emits
// a per-channel gaussian beam; summing the overlapping profiles gives bright cores
// that bloom and dark gaps that stay open — resolution-correct, in linear light.
// (Explicit-LOD sampling so it stays callable once per primary for dispersion.)
fn scan_reconstruct(uv: vec2<f32>, res: vec2<f32>) -> vec3<f32> {
    let fy = uv.y * res.y - 0.5;
    let row0 = floor(fy);
    var beam = vec3<f32>(0.0); // energy-weighted beam sum (blooms where lines overlap)
    var flat = vec3<f32>(0.0); // profile-normalised reference (the settled picture)
    var wsum = vec3<f32>(0.0);
    let range = i32(u.scan.w);
    for (var k = -range; k <= range + 1; k = k + 1) {
        let row = row0 + f32(k);
        let ly = (row + 0.5) / res.y;
        let c = textureSampleLevel(t_screen, s_screen, vec2<f32>(uv.x, ly), 0.0).rgb;
        let w = beam_width(c);
        let d = fy - row;
        let g = exp(-(d * d) / (w * w)); // per-channel gaussian beam profile
        beam = beam + c * g;
        flat = flat + c * g;
        wsum = wsum + g;
    }
    flat = flat / max(wsum, vec3<f32>(1e-4));
    let col = mix(flat, beam * u.tone.z, u.optics.z);
    return col * (1.0 + u.optics.z * 0.5);
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

@fragment
fn fs_phosphor(in: FullOut) -> @location(0) vec4<f32> {
    let res = u.params.xy;
    let uv = in.uv;
    // Input signal path (tone.w): 0 = RGB/component (clean — PVM, arcade board, PC),
    // 1 = S-video (sharp luma, band-limited colour, no dot crawl), 2 = composite
    // (dot crawl + cross-colour rainbow + colour bleed — RF/antenna consumer TV).
    var sig: vec3<f32>;
    if (u.tone.w >= 1.5) {
        sig = ntsc(uv, res, u.params.z);
    } else if (u.tone.w >= 0.5) {
        sig = svideo(uv, res);
    } else {
        sig = textureSampleLevel(t_screen, s_screen, uv, 0.0).rgb; // clean RGB
    }
    let prev = textureSampleLevel(t_prev, s_screen, uv, 0.0).rgb;    // last phosphor

    let dt = max(u.temporal.x, 0.0);
    // Per-phosphor decay: each primary keeps its own fraction of last field's charge.
    // Red lingers, blue snaps off → moving highlights trail warm (the real P22 look).
    let tau = max(u.ptau.rgb * max(u.temporal.y, 1e-4), vec3<f32>(1e-4));
    let decay = exp(-vec3<f32>(dt) / tau);

    // Interlace: on an interlaced field only alternate lines are re-excited this
    // frame; the others coast on their decayed charge, giving line twitter.
    let line = floor(uv.y * res.y);
    let parity = f32(i32(u.temporal.w) & 1);
    let odd = f32(i32(line) - (i32(line) / 2) * 2);
    let lit = 1.0 - u.temporal.z * abs(odd - parity);
    let excite = sig * lit;

    // Phosphor charges instantly to the beam excitation, then decays. max() keeps a
    // freshly-lit pixel bright while unlit pixels fall off toward the previous field.
    let out = max(excite, prev * decay);
    return vec4<f32>(out, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(u.cam_pos.xyz - in.world_pos);

    // ---- Tube body: 1 = leaded funnel/neck glass, 2 = deflection yoke, 3 = plastic ----
    if (in.material > 0.5) {
        // Two-sided: cull is off so the viewer can see interior faces; flip the normal
        // toward the camera so lighting is correct regardless of triangle winding.
        var nn = n;
        if (dot(nn, v) < 0.0) { nn = -nn; }

        var base: vec3<f32>;
        var rough: f32;
        var metal: f32;
        if (in.material < 1.5) {
            base = vec3<f32>(0.018, 0.019, 0.024); // near-black leaded glass, glossy
            rough = 0.22;
            metal = 0.0;
        } else if (in.material < 2.5) {
            base = vec3<f32>(0.42, 0.26, 0.15);    // deflection yoke: dull copper
            rough = 0.55;
            metal = 1.0;
        } else {
            base = vec3<f32>(0.060, 0.054, 0.045); // aged warm-gray molded plastic
            // Injection-molded pebble finish: mottle the albedo, perturb roughness and
            // the normal so the specular breaks into a matte sparkle, not a mirror.
            let tx = hash21(floor(in.world_pos.xy * 140.0)) + hash21(floor(in.world_pos.yz * 130.0));
            rough = clamp(0.50 + (tx - 1.0) * 0.14, 0.30, 0.76);
            base = base * (0.82 + 0.34 * hash21(floor(in.world_pos.xy * 70.0)));
            let jit = vec3<f32>(hash21(in.world_pos.xy * 95.0) - 0.5,
                                hash21(in.world_pos.yz * 95.0) - 0.5,
                                (hash21(in.world_pos.zx * 95.0) - 0.5) * 0.4) * 0.07;
            nn = normalize(nn + jit);
            metal = 0.0;
        }

        var col = shade_body(base, rough, metal, nn, v);
        // HDR bounce: the phosphor screen is an area light. Fragments near the front
        // (world z ~ 0 — the bezel and faceplate block) catch the picture's average
        // colour and brightness; it falls off down the funnel. This is what makes a
        // real set — and its bezel — glow with the on-screen colour in a dark room.
        let front = smoothstep(-1.6, -0.05, in.world_pos.z);
        var glow_col = u.env.rgb;
        if (u.mono.w > 0.5) { glow_col = dot(u.env.rgb, vec3<f32>(0.299, 0.587, 0.114)) * u.mono.rgb; }
        // Screen-off / warming tubes don't light the bezel: gate the bounce by power.
        let son = min(u.pwr.x, 1.0 - u.pwr.y);
        col = col + glow_col * u.env.w * u.phys.z * front * (0.30 + 0.70 * max(dot(nn, v), 0.0)) * son;
        return vec4<f32>(col, 1.0);
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
    var base_uv = in.uv;
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
    let uv_r = refract_uv(ruv, n, v, thick, 1.0 / 1.518) + conv;
    let uv_g = refract_uv(ruv, n, v, thick, 1.0 / 1.520);
    let uv_b = refract_uv(ruv, n, v, thick, 1.0 / 1.522) - conv;
    let uv = uv_g; // base uv for halation / vignette
    var col = vec3<f32>(
        scan_reconstruct(uv_r, res).r,
        scan_reconstruct(uv_g, res).g,
        scan_reconstruct(uv_b, res).b,
    );

    // CRT transfer + phosphor colour. A real tube's response deepens the blacks
    // (extra gamma) and its P22 phosphors give a characteristically warm-white
    // point rather than a neutral D65.
    col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(u.phys.x));
    col = col * mix(vec3<f32>(1.0), vec3<f32>(1.06, 1.015, 0.93), u.phys.y);

    // Beam bloom + high-voltage sag, driven by average picture level (APL). On a
    // bright full-screen scene the power supply sags — the whole image dims a hair
    // and the beam widens, so highlights bleed. This "breathing" is a real-set tell.
    let apl = u.env.w;
    col = col * (1.0 - apl * 0.06);
    let bright = max(col - vec3<f32>(0.72), vec3<f32>(0.0));
    col = col + bright * u.phys.w * (0.6 + apl);

    // Monochrome tube: a single electron gun paints ONE phosphor colour scaled by the
    // signal's luminance — no colour triads, no convergence (a green/amber terminal).
    // The colour mask is already skipped via mask_strength=0; damper wires, halation
    // and glow tint are gated on this flag below.
    if (u.mono.w > 0.5) {
        col = dot(col, vec3<f32>(0.299, 0.587, 0.114)) * u.mono.rgb;
    }

    // (Phosphor persistence + the raster field sweep are now integrated over real
    // frame history in the accum pass, so there's no per-fragment temporal fake here.)

    // Halation: light scattering laterally inside the glass, biased warm/red
    // because the red phosphor persists longest. Sampled around the parallax uv.
    let halo = u.optics.w;
    if (halo > 0.0) {
        let px = vec2<f32>(2.5, 2.5) / res;
        var glow = vec3<f32>(0.0);
        glow = glow + textureSample(t_screen, s_screen, uv + vec2<f32>(px.x, 0.0)).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv - vec2<f32>(px.x, 0.0)).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv + vec2<f32>(0.0, px.y)).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv - vec2<f32>(0.0, px.y)).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv + px).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv - px).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv + vec2<f32>(px.x, -px.y)).rgb;
        glow = glow + textureSample(t_screen, s_screen, uv + vec2<f32>(-px.x, px.y)).rgb;
        glow = glow / 8.0;
        if (u.mono.w > 0.5) {
            // mono: the glow is the phosphor's own colour, not the warm-red halation.
            col = col + dot(glow, vec3<f32>(0.299, 0.587, 0.114)) * u.mono.rgb * halo;
        } else {
            col = col + glow * vec3<f32>(1.0, 0.6, 0.45) * halo;
        }
    }

    // Phosphor mask. Pitch is in *final* output pixels, scaled up by the render
    // scale so supersampling anti-aliases the mask instead of erasing it.
    let mask_pitch = max(u.glass.w, 1.0) * max(u.params.w, 1.0);
    let m = mask(in.clip.xy, u.optics.x, mask_pitch);
    col = col * mix(vec3<f32>(1.0), m, u.optics.y);
    col = col * (1.0 + u.optics.y * 0.7); // compensate mask darkening

    // Damper wires: the signature of an aperture-grille (Trinitron) tube. The
    // vertical phosphor strips are steadied by 1-2 fine horizontal tension wires
    // that cast a soft thin shadow across the whole picture. Aperture grille only
    // (not on a monochrome tube, which has no grille).
    if (u.optics.x < 0.5 && u.mono.w < 0.5) {
        let wy = in.uv.y;
        // two wires (large-set layout) at ~1/3 and ~2/3 height; ~1.5px soft shadow
        let w = 0.0016;
        let s1 = exp(-(wy - 0.333) * (wy - 0.333) / (2.0 * w * w));
        let s2 = exp(-(wy - 0.667) * (wy - 0.667) / (2.0 * w * w));
        col = col * (1.0 - 0.45 * (s1 + s2));
    }

    // Secondary internal reflection ("ghost"): a faint, offset second image from
    // the light that bounces off the inner glass surface before reaching the eye —
    // the double exposure you catch on a thick, glossy CRT faceplate.
    var gcol = textureSampleLevel(t_screen, s_screen, uv + vec2<f32>(0.011, -0.008), 0.0).rgb;
    if (u.mono.w > 0.5) { gcol = dot(gcol, vec3<f32>(0.299, 0.587, 0.114)) * u.mono.rgb; }
    col = col + gcol * u.look.w;

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

    // Analog noise floor: a little animated grain, strongest in the shadows where
    // a real signal's snow is visible.
    let lum = dot(col, vec3<f32>(0.299, 0.587, 0.114));
    let grain = (hash21(in.uv * res + vec2<f32>(u.params.z * 61.0, u.params.z * 37.0)) - 0.5);
    col = col + grain * u.look.z * (1.0 - smoothstep(0.0, 0.5, lum));

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

    // Glass Fresnel reflection of the synthetic room, plus a tight specular glare
    // from a room light. The glare is a hot spot that slides across the curved
    // faceplate as you orbit — the single most CRT-reading reflection cue.
    let fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    let refl = reflect(-v, n);
    col = col + room(refl) * fres * u.glass.y;
    let light_dir = normalize(vec3<f32>(-0.35, 0.55, 0.95));
    let glare = pow(max(dot(refl, light_dir), 0.0), 130.0);
    col = col + vec3<f32>(1.0, 0.98, 0.92) * glare * u.glass.y * 1.6;

    // Output. col is HDR (linear light, BT.709/sRGB primaries, highlights >1.0).
    if (u.tone.x > 0.5) {
        // HDR swapchain: emit linear light where 1.0 = SDR white and values above
        // 1.0 drive the panel's extra nits. The surface is BT.2020 linear, so
        // rotate our BT.709 primaries into BT.2020 (else colors read oversaturated).
        let bt2020 = mat3x3<f32>(
            0.6274, 0.0691, 0.0164,
            0.3293, 0.9195, 0.0880,
            0.0433, 0.0114, 0.8956,
        ) * col;
        // tone.y = HDR exposure (scales SDR-white → the compositor's reference
        // white; bump if the picture looks dim, drop if it's blinding).
        return vec4<f32>(bt2020 * u.tone.y, 1.0);
    }
    // SDR display: tonemap HDR highlights back into range. Target is sRGB, so
    // return linear — the swapchain encodes the transfer function.
    return vec4<f32>(tonemap(col, u.tone.y), 1.0);
}
