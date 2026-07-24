// crtulum — a manipulable 3D CRT tube in a window.
//
//   cargo run              : test-pattern source
//   cargo run -- --capture : live source via the ScreenCast portal + PipeWire (M2)
//   cargo run -- --shot out.png 1000x800 : headless PNG render
//
// Controls: left-drag orbit · scroll zoom · Esc quit

mod capture;

use std::sync::Arc;

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
    normal: [f32; 3],
    material: f32, // 0.0 = screen, 1.0 = bezel
}

impl Vertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3, // pos
            1 => Float32x2, // uv
            2 => Float32x3, // normal
            3 => Float32,   // material
        ],
    };
}

const HALF_W: f32 = 0.667; // 4:3 half-extents
const HALF_H: f32 = 0.5;

// Front-glass bulge: center proud by `bulge`, falling off with cx (horizontal) and
// cy (vertical) curvature. Trinitron ≈ cylindrical (cy≈0); consumer ≈ spherical.
fn screen_z(x: f32, y: f32, bulge: f32, cx: f32, cy: f32) -> f32 {
    let nx = x / HALF_W;
    let ny = y / HALF_H;
    bulge * (1.0 - cx * nx * nx - cy * ny * ny)
}

// Bare picture-tube dimensions (units of the screen half-width). A real 13"
// Trinitron is roughly as deep as it is wide, tapering through a glass bell to a
// thin electron-gun neck — see the KV-13M service-manual tube diagrams.
const GLASS_T: f32 = 0.13; // faceplate glass thickness (front face → block back)
const FUNNEL_DEPTH: f32 = 1.42; // block back → neck
const NECK_R: f32 = 0.095; // electron-gun neck radius
const NECK_LEN: f32 = 0.48;

// Push a flat quad p0→p1→p2→p3 with an auto-computed outward face normal.
fn push_quad(verts: &mut Vec<Vertex>, indices: &mut Vec<u32>, p: [[f32; 3]; 4], mat: f32) {
    let a = Vec3::from_array(p[0]);
    let b = Vec3::from_array(p[1]);
    let c = Vec3::from_array(p[2]);
    let n = (b - a).cross(c - a).normalize_or_zero();
    let base = verts.len() as u32;
    for pk in p {
        verts.push(Vertex { pos: pk, uv: [0.0, 0.0], normal: [n.x, n.y, n.z], material: mat });
    }
    indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
}

// One cross-section of the funnel. t=0 is the (near-rectangular) faceplate rim,
// t=1 the round neck: the superellipse exponent morphs rectangle→circle while the
// section shrinks and recedes, giving the bulged CRT bell.
fn funnel_ring(t: f32, m: usize) -> Vec<[f32; 3]> {
    let es = t * t; // section scale: stays wide, then pulls in toward the neck
    let ez = t.powf(0.8); // depth easing
    let ax = HALF_W * (1.0 - es) + NECK_R * es;
    let ay = HALF_H * (1.0 - es) + NECK_R * es;
    let n_exp = 8.0 * (1.0 - t) + 2.0 * t; // rounded-rect → circle
    let zz = -GLASS_T - FUNNEL_DEPTH * ez;
    let mut pts = Vec::with_capacity(m);
    for k in 0..m {
        let th = std::f32::consts::TAU * k as f32 / m as f32;
        let (c, s) = (th.cos(), th.sin());
        let e = 2.0 / n_exp;
        let x = ax * c.signum() * c.abs().powf(e);
        let y = ay * s.signum() * s.abs().powf(e);
        pts.push([x, y, zz]);
    }
    pts
}

// A rounded-rectangle ring of points at depth `z`, wound CCW. Every ring uses the
// same per-corner/per-edge point budget so consecutive rings loft into clean quads.
fn rrect(hw: f32, hh: f32, r: f32, z: f32) -> Vec<[f32; 3]> {
    use std::f32::consts::{FRAC_PI_2, PI};
    let r = r.min(hw).min(hh);
    let (ix, iy) = (hw - r, hh - r);
    const CP: usize = 6; // points per rounded corner
    const EP: usize = 8; // points per straight edge
    let corners = [
        ([ix, -iy], -FRAC_PI_2), // bottom-right, arc -90°..0°
        ([ix, iy], 0.0),         // top-right, 0°..90°
        ([-ix, iy], FRAC_PI_2),  // top-left, 90°..180°
        ([-ix, -iy], PI),        // bottom-left, 180°..270°
    ];
    let mut pts: Vec<[f32; 3]> = Vec::with_capacity(4 * (CP + EP));
    for ci in 0..4 {
        let (c, a0) = corners[ci];
        let mut last = [0.0f32, 0.0];
        for j in 0..CP {
            let a = a0 + FRAC_PI_2 * (j as f32 / (CP as f32 - 1.0));
            last = [c[0] + r * a.cos(), c[1] + r * a.sin()];
            pts.push([last[0], last[1], z]);
        }
        let (nc, na0) = corners[(ci + 1) % 4];
        let nfirst = [nc[0] + r * na0.cos(), nc[1] + r * na0.sin()];
        for j in 1..=EP {
            let t = j as f32 / (EP as f32 + 1.0);
            pts.push([last[0] + (nfirst[0] - last[0]) * t, last[1] + (nfirst[1] - last[1]) * t, z]);
        }
    }
    pts
}

// Loft a quad strip between two equal-length rings, material `mat`. Winding is
// irrelevant — the body shader is two-sided (normals face the viewer).
fn ring_strip(verts: &mut Vec<Vertex>, indices: &mut Vec<u32>, a: &[[f32; 3]], b: &[[f32; 3]], mat: f32) {
    let m = a.len();
    for k in 0..m {
        let k2 = (k + 1) % m;
        push_quad(verts, indices, [a[k], a[k2], b[k2], b[k]], mat);
    }
}

fn build_mesh(bulge: f32, cx: f32, cy: f32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    let z = |x: f32, y: f32| screen_z(x, y, bulge, cx, cy);

    // --- 1. Curved faceplate (phosphor screen): an N x N displaced grid, mat 0 ---
    const N: usize = 128;
    let e = 0.001_f32;
    for j in 0..=N {
        for i in 0..=N {
            let fx = i as f32 / N as f32; // 0..1
            let fy = j as f32 / N as f32;
            let x = (fx * 2.0 - 1.0) * HALF_W;
            let y = (fy * 2.0 - 1.0) * HALF_H;
            let zz = z(x, y);

            // analytic-ish normal via finite differences of the bulge
            let dzdx = (z(x + e, y) - z(x - e, y)) / (2.0 * e);
            let dzdy = (z(x, y + e) - z(x, y - e)) / (2.0 * e);
            let normal = Vec3::new(-dzdx, -dzdy, 1.0).normalize();

            verts.push(Vertex {
                pos: [x, y, zz],
                uv: [fx, 1.0 - fy], // texture v=0 at top
                normal: [normal.x, normal.y, normal.z],
                material: 0.0,
            });
        }
    }
    let stride = (N + 1) as u32;
    for j in 0..N as u32 {
        for i in 0..N as u32 {
            let a = j * stride + i;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    // --- 2. Faceplate glass sides: extrude the curved screen edge straight back
    // to the block-back plane (z = -GLASS_T) for the thick curved-glass look. ---
    let zb = -GLASS_T;
    let es = 64;
    for s in 0..es {
        let t0 = 2.0 * s as f32 / es as f32 - 1.0;
        let t1 = 2.0 * (s + 1) as f32 / es as f32 - 1.0;
        let (x0, x1) = (HALF_W * t0, HALF_W * t1);
        let (y0, y1) = (HALF_H * t0, HALF_H * t1);
        // top / bottom edges
        push_quad(&mut verts, &mut indices,
            [[x0, HALF_H, z(x0, HALF_H)], [x1, HALF_H, z(x1, HALF_H)], [x1, HALF_H, zb], [x0, HALF_H, zb]], 1.0);
        push_quad(&mut verts, &mut indices,
            [[x0, -HALF_H, zb], [x1, -HALF_H, zb], [x1, -HALF_H, z(x1, -HALF_H)], [x0, -HALF_H, z(x0, -HALF_H)]], 1.0);
        // right / left edges
        push_quad(&mut verts, &mut indices,
            [[HALF_W, y0, z(HALF_W, y0)], [HALF_W, y1, z(HALF_W, y1)], [HALF_W, y1, zb], [HALF_W, y0, zb]], 1.0);
        push_quad(&mut verts, &mut indices,
            [[-HALF_W, y0, zb], [-HALF_W, y1, zb], [-HALF_W, y1, z(-HALF_W, y1)], [-HALF_W, y0, z(-HALF_W, y0)]], 1.0);
    }

    // --- 3. Funnel (glass bell): faceplate block-back → neck, lofted rings. ---
    let m = 56usize;
    let rings = 24usize;
    let mut prev = funnel_ring(0.0, m);
    for r in 1..=rings {
        let cur = funnel_ring(r as f32 / rings as f32, m);
        for k in 0..m {
            let k2 = (k + 1) % m;
            push_quad(&mut verts, &mut indices, [prev[k], prev[k2], cur[k2], cur[k]], 1.0);
        }
        prev = cur;
    }

    // --- 4. Neck (electron-gun tube) + back cap, material 1 ---
    let z_neck0 = -GLASS_T - FUNNEL_DEPTH;
    let z_neck1 = z_neck0 - NECK_LEN;
    let ring_at = |rad: f32, zz: f32, k: usize| -> [f32; 3] {
        let th = std::f32::consts::TAU * k as f32 / m as f32;
        [rad * th.cos(), rad * th.sin(), zz]
    };
    for k in 0..m {
        let k2 = (k + 1) % m;
        push_quad(&mut verts, &mut indices,
            [ring_at(NECK_R, z_neck0, k), ring_at(NECK_R, z_neck0, k2), ring_at(NECK_R, z_neck1, k2), ring_at(NECK_R, z_neck1, k)], 1.0);
    }
    // neck end cap (triangle fan as degenerate quads to the center)
    for k in 0..m {
        let k2 = (k + 1) % m;
        push_quad(&mut verts, &mut indices,
            [ring_at(NECK_R, z_neck1, k), [0.0, 0.0, z_neck1], [0.0, 0.0, z_neck1], ring_at(NECK_R, z_neck1, k2)], 1.0);
    }

    // --- 5. Deflection yoke: a collar at the funnel/neck junction, material 2 ---
    let yr = NECK_R * 2.1;
    let (zy0, zy1) = (z_neck0 + 0.04, z_neck0 - 0.22);
    for k in 0..m {
        let k2 = (k + 1) % m;
        // outer wall
        push_quad(&mut verts, &mut indices,
            [ring_at(yr, zy0, k), ring_at(yr, zy0, k2), ring_at(yr, zy1, k2), ring_at(yr, zy1, k)], 2.0);
        // front face ring (yoke shoulder), yr → neck
        push_quad(&mut verts, &mut indices,
            [ring_at(NECK_R, zy0, k), ring_at(NECK_R, zy0, k2), ring_at(yr, zy0, k2), ring_at(yr, zy0, k)], 2.0);
    }

    // --- 6. TV cabinet (material 3 = charcoal plastic, material 4 = speaker cloth) ---
    // A deep, near-cubic consumer set grounded in the Sony Trinitron KV-20TS
    // proportions (513 × 487 × 481 mm — a real CRT TV is almost as DEEP as it is wide,
    // NOT a thin picture frame): the 4:3 screen recessed in the upper-centre, a speaker
    // grille + control cluster across the tall chin below, a side/top bezel, then a
    // tapered rear hump that encloses the glass funnel and electron-gun neck.
    let sb = 0.150; // side bezel
    let tb = 0.130; // top bezel
    let bc = 0.450; // bottom chin (speakers + controls) — the tall part below the tube
    let hw_cab = HALF_W + sb; // outer half-width  ≈ 0.817
    let cab_t = HALF_H + tb; //  top edge          ≈ 0.630
    let cab_b = -(HALF_H + bc); // bottom edge     ≈ -0.950  (cabinet ratio ≈ 1.04:1, real 1.05:1)
    let (cab_l, cab_r) = (-hw_cab, hw_cab);
    let (ox, oy) = (HALF_W * 1.018, HALF_H * 1.024); // screen opening (a touch over the glass)
    let z_front = bulge * 0.6 + 0.05; // front face plane, just behind the glass apex
    let z_rear = -1.46; //  main box back (depth ≈ width → the real near-cube)
    let z_rear2 = -1.94; // rear hump back (encloses the funnel/neck)
    // Edge chamfer: real injection-molded cabinets are never razor-edged — every outer
    // edge has a few-mm bevel that catches a bright highlight line. The front face is
    // inset by `cf`, then a 45° bevel (+ mitred corner facets) runs out to the full
    // cabinet extent where the side walls begin (at z = zc).
    let cf = 0.05;
    let (fl, fr, fb, ft) = (cab_l + cf, cab_r - cf, cab_b + cf, cab_t - cf);
    let zc = z_front - cf;
    let quad = |v: &mut Vec<Vertex>, ix: &mut Vec<u32>, p: [[f32; 3]; 4], m: f32| push_quad(v, ix, p, m);

    // 6a. Front bezel around the screen opening: top strip, two side strips (to the
    // chamfered front-face extents fl/fr/ft, not the full cabinet edge).
    quad(&mut verts, &mut indices,
        [[fl, oy, z_front], [fr, oy, z_front], [fr, ft, z_front], [fl, ft, z_front]], 3.0); // top
    quad(&mut verts, &mut indices,
        [[fl, -oy, z_front], [-ox, -oy, z_front], [-ox, oy, z_front], [fl, oy, z_front]], 3.0); // left
    quad(&mut verts, &mut indices,
        [[ox, -oy, z_front], [fr, -oy, z_front], [fr, oy, z_front], [ox, oy, z_front]], 3.0); // right
    // inner lip: recess the opening edge back to the glass block so the tube sits inset.
    let zl = -GLASS_T - 0.02;
    quad(&mut verts, &mut indices, [[-ox, oy, z_front], [ox, oy, z_front], [ox, oy, zl], [-ox, oy, zl]], 3.0);
    quad(&mut verts, &mut indices, [[-ox, -oy, z_front], [ox, -oy, z_front], [ox, -oy, zl], [-ox, -oy, zl]], 3.0);
    quad(&mut verts, &mut indices, [[-ox, -oy, z_front], [-ox, oy, z_front], [-ox, oy, zl], [-ox, -oy, zl]], 3.0);
    quad(&mut verts, &mut indices, [[ox, -oy, z_front], [ox, oy, z_front], [ox, oy, zl], [ox, -oy, zl]], 3.0);

    // 6a-bevel. 45° chamfer ring: the front face plane (z_front) out to the full cabinet
    // rectangle (zc), with four edge bevels + four mitred corner facets.
    quad(&mut verts, &mut indices, [[fl, ft, z_front], [fr, ft, z_front], [fr, cab_t, zc], [fl, cab_t, zc]], 3.0); // top
    quad(&mut verts, &mut indices, [[fl, fb, z_front], [fr, fb, z_front], [fr, cab_b, zc], [fl, cab_b, zc]], 3.0); // bottom
    quad(&mut verts, &mut indices, [[fl, fb, z_front], [fl, ft, z_front], [cab_l, ft, zc], [cab_l, fb, zc]], 3.0); // left
    quad(&mut verts, &mut indices, [[fr, fb, z_front], [fr, ft, z_front], [cab_r, ft, zc], [cab_r, fb, zc]], 3.0); // right
    quad(&mut verts, &mut indices, [[fr, ft, z_front], [fr, cab_t, zc], [cab_r, cab_t, zc], [cab_r, ft, zc]], 3.0); // TR corner
    quad(&mut verts, &mut indices, [[fl, ft, z_front], [fl, cab_t, zc], [cab_l, cab_t, zc], [cab_l, ft, zc]], 3.0); // TL corner
    quad(&mut verts, &mut indices, [[fr, fb, z_front], [fr, cab_b, zc], [cab_r, cab_b, zc], [cab_r, fb, zc]], 3.0); // BR corner
    quad(&mut verts, &mut indices, [[fl, fb, z_front], [fl, cab_b, zc], [cab_l, cab_b, zc], [cab_l, fb, zc]], 3.0); // BL corner

    // 6b. Chin: a plastic frame around a recessed panel that carries the speaker
    // grille (left ~66%) and the control cluster (right ~34%).
    let chin_top = -oy;
    let (rx0, rx1) = (fl + 0.05, fr - 0.05); // recess extents
    let (ry0, ry1) = (fb + 0.075, chin_top - 0.06);
    let z_rec = z_front - 0.05; // recess depth
    // chin frame strips (plastic, at z_front)
    quad(&mut verts, &mut indices, [[fl, fb, z_front], [fr, fb, z_front], [fr, ry0, z_front], [fl, ry0, z_front]], 3.0); // below recess
    quad(&mut verts, &mut indices, [[fl, ry1, z_front], [fr, ry1, z_front], [fr, chin_top, z_front], [fl, chin_top, z_front]], 3.0); // above recess
    quad(&mut verts, &mut indices, [[fl, ry0, z_front], [rx0, ry0, z_front], [rx0, ry1, z_front], [fl, ry1, z_front]], 3.0); // left of recess
    quad(&mut verts, &mut indices, [[rx1, ry0, z_front], [fr, ry0, z_front], [fr, ry1, z_front], [rx1, ry1, z_front]], 3.0); // right of recess
    // recess side walls (front → back)
    quad(&mut verts, &mut indices, [[rx0, ry0, z_front], [rx1, ry0, z_front], [rx1, ry0, z_rec], [rx0, ry0, z_rec]], 3.0);
    quad(&mut verts, &mut indices, [[rx0, ry1, z_front], [rx1, ry1, z_front], [rx1, ry1, z_rec], [rx0, ry1, z_rec]], 3.0);
    quad(&mut verts, &mut indices, [[rx0, ry0, z_front], [rx0, ry1, z_front], [rx0, ry1, z_rec], [rx0, ry0, z_rec]], 3.0);
    quad(&mut verts, &mut indices, [[rx1, ry0, z_front], [rx1, ry1, z_front], [rx1, ry1, z_rec], [rx1, ry0, z_rec]], 3.0);
    // recessed panels: grille (mat 4) on the left, control plate (mat 3) on the right,
    // with a thin plastic divider between them.
    let gx1 = rx0 + (rx1 - rx0) * 0.64; // grille / controls split
    let div = 0.02;
    quad(&mut verts, &mut indices, [[rx0, ry0, z_rec], [gx1, ry0, z_rec], [gx1, ry1, z_rec], [rx0, ry1, z_rec]], 4.0); // speaker grille
    quad(&mut verts, &mut indices, [[gx1, ry0, z_rec], [gx1 + div, ry0, z_rec], [gx1 + div, ry1, z_rec], [gx1, ry1, z_rec], ], 3.0); // divider
    quad(&mut verts, &mut indices, [[gx1 + div, ry0, z_rec], [rx1, ry0, z_rec], [rx1, ry1, z_rec], [gx1 + div, ry1, z_rec]], 3.0); // control plate
    // two control knobs (short cylinders) on the control plate
    let kc = 0.06;
    for (i, &kx) in [gx1 + (rx1 - gx1) * 0.36, gx1 + (rx1 - gx1) * 0.70].iter().enumerate() {
        let ky = ry0 + (ry1 - ry0) * 0.5;
        let kn = 16usize;
        let zk = z_rec + 0.035;
        for k in 0..kn {
            let k2 = (k + 1) % kn;
            let a0 = std::f32::consts::TAU * k as f32 / kn as f32;
            let a1 = std::f32::consts::TAU * k2 as f32 / kn as f32;
            let p0 = [kx + kc * a0.cos(), ky + kc * a0.sin(), z_rec];
            let p1 = [kx + kc * a1.cos(), ky + kc * a1.sin(), z_rec];
            let q0 = [kx + kc * a0.cos(), ky + kc * a0.sin(), zk];
            let q1 = [kx + kc * a1.cos(), ky + kc * a1.sin(), zk];
            quad(&mut verts, &mut indices, [p0, p1, q1, q0], 3.0); // knob wall
            quad(&mut verts, &mut indices, [q0, q1, [kx, ky, zk], [kx, ky, zk]], 3.0); // knob top
        }
        let _ = i;
    }

    // 6c. Side walls: from the chamfer edge (zc) straight back to the main box.
    quad(&mut verts, &mut indices, [[cab_l, cab_t, zc], [cab_r, cab_t, zc], [cab_r, cab_t, z_rear], [cab_l, cab_t, z_rear]], 3.0); // top
    quad(&mut verts, &mut indices, [[cab_l, cab_b, zc], [cab_r, cab_b, zc], [cab_r, cab_b, z_rear], [cab_l, cab_b, z_rear]], 3.0); // bottom
    quad(&mut verts, &mut indices, [[cab_l, cab_b, zc], [cab_l, cab_t, zc], [cab_l, cab_t, z_rear], [cab_l, cab_b, z_rear]], 3.0); // left
    quad(&mut verts, &mut indices, [[cab_r, cab_b, zc], [cab_r, cab_t, zc], [cab_r, cab_t, z_rear], [cab_r, cab_b, z_rear]], 3.0); // right

    // 6d. Rear hump: taper the box in toward the tube axis and cap it (with a neck
    // hole). This is the classic bulging back of a CRT set enclosing the deflection bell.
    let (rhw, rht, rhb) = (hw_cab * 0.60, cab_t * 0.60, cab_b * 0.60);
    quad(&mut verts, &mut indices, [[cab_l, cab_t, z_rear], [cab_r, cab_t, z_rear], [rhw, rht, z_rear2], [-rhw, rht, z_rear2]], 3.0); // top taper
    quad(&mut verts, &mut indices, [[cab_l, cab_b, z_rear], [cab_r, cab_b, z_rear], [rhw, rhb, z_rear2], [-rhw, rhb, z_rear2]], 3.0); // bottom taper
    quad(&mut verts, &mut indices, [[cab_l, cab_b, z_rear], [cab_l, cab_t, z_rear], [-rhw, rht, z_rear2], [-rhw, rhb, z_rear2]], 3.0); // left taper
    quad(&mut verts, &mut indices, [[cab_r, cab_b, z_rear], [cab_r, cab_t, z_rear], [rhw, rht, z_rear2], [rhw, rhb, z_rear2]], 3.0); // right taper
    // rear face with a neck hole (ring of 4 strips around the hole)
    let nh = NECK_R * 1.6;
    quad(&mut verts, &mut indices, [[-rhw, rhb, z_rear2], [rhw, rhb, z_rear2], [nh, -nh, z_rear2], [-nh, -nh, z_rear2]], 3.0);
    quad(&mut verts, &mut indices, [[-rhw, rht, z_rear2], [rhw, rht, z_rear2], [nh, nh, z_rear2], [-nh, nh, z_rear2]], 3.0);
    quad(&mut verts, &mut indices, [[-rhw, rhb, z_rear2], [-rhw, rht, z_rear2], [-nh, nh, z_rear2], [-nh, -nh, z_rear2]], 3.0);
    quad(&mut verts, &mut indices, [[rhw, rhb, z_rear2], [rhw, rht, z_rear2], [nh, nh, z_rear2], [nh, -nh, z_rear2]], 3.0);

    (verts, indices)
}

// ---------------------------------------------------------------------------
// Test-pattern source texture (SMPTE-style color bars)
// ---------------------------------------------------------------------------

fn make_test_pattern() -> (u32, u32, Vec<u8>) {
    let w = 320u32;
    let h = 240u32;
    let mut data = vec![0u8; (w * h * 4) as usize];

    // 100% color bars: gray, yellow, cyan, green, magenta, red, blue
    let bars: [[u8; 3]; 7] = [
        [191, 191, 191],
        [191, 191, 0],
        [0, 191, 191],
        [0, 191, 0],
        [191, 0, 191],
        [191, 0, 0],
        [0, 0, 191],
    ];

    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            let rgb = if y < h * 3 / 4 {
                // color bars in the top three-quarters
                bars[(x * 7 / w) as usize]
            } else {
                // bottom quarter: black/white castellation to show scanline response
                let block = (x / (w / 12)) % 2;
                if block == 0 { [235, 235, 235] } else { [8, 8, 8] }
            };
            data[idx] = rgb[0];
            data[idx + 1] = rgb[1];
            data[idx + 2] = rgb[2];
            data[idx + 3] = 255;
        }
    }
    (w, h, data)
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

struct Orbit {
    yaw: f32,
    pitch: f32,
    distance: f32,
}

impl Orbit {
    fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        Vec3::new(
            self.distance * self.yaw.sin() * cp,
            self.distance * self.pitch.sin(),
            self.distance * self.yaw.cos() * cp,
        )
    }

    fn view_proj(&self, aspect: f32) -> (Mat4, Vec3) {
        let eye = self.eye();
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, 0.1, 100.0);
        (proj * view, eye)
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [f32; 16],
    model: [f32; 16],
    cam_pos: [f32; 4],
    params: [f32; 4], // src_w, src_h, time, render_scale
    optics: [f32; 4], // mask_type, mask_strength, scanline, halation
    glass: [f32; 4],  // parallax, reflection, vignette, mask_pitch
    tone: [f32; 4],   // hdr flag, peak/white-point, beam_drive, ntsc_strength
    scan: [f32; 4],   // beam math: beam_min, beam_max, beam_shape, beam_range
    env: [f32; 4],    // avg_r, avg_g, avg_b, apl  (screen-as-area-light bounce)
    look: [f32; 4],   // convergence, corner_radius, grain, ghost
    phys: [f32; 4],   // crt_gamma, warmth, glow_bounce, bloom
    temporal: [f32; 4], // dt(sec), persist_mult, interlace, field_parity
    ptau: [f32; 4],   // per-phosphor decay tau: R, G, B (sec), _ (P22: red lingers, blue snaps off)
    geom: [f32; 4],   // raster geometry errors: pincushion, trapezoid, corner_pin, purity
    mono: [f32; 4],   // monochrome phosphor tint (rgb) + flag (w>0.5 = single-gun tube)
    cmat0: [f32; 4],  // CRT-phosphor → sRGB colour matrix, row 0 (real gamut + white pt)
    cmat1: [f32; 4],  // row 1
    cmat2: [f32; 4],  // row 2
    pwr: [f32; 4],    // power state: warmup(0..1), collapse(0..1), degauss(0..1), _
    focus: [f32; 4],  // x=edge defocus (deflection spot growth), y=overscan (per side), z/w _
}

// ---------------------------------------------------------------------------
// CRT presets — observed geometry + phosphor of real monitor families.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Preset {
    name: &'static str,
    // geometry
    bulge: f32,
    curv_x: f32,
    curv_y: f32,
    // optics
    mask_type: f32, // 0 aperture grille, 1 shadow (dot), 2 slot
    mask_strength: f32,
    scanline: f32,
    halation: f32,
    // glass
    parallax: f32, // faceplate glass thickness driving refraction/dispersion
    reflection: f32,
    vignette: f32,
    mask_pitch: f32,
    // beam/geometry imperfections
    convergence: f32,   // RGB misregistration magnitude at the corners
    corner_radius: f32, // rounding of the active phosphor rectangle
    // raster deflection geometry errors (a pro monitor is near-perfect; consumer
    // sets bow and drift): [pincushion, trapezoid, corner_pincushion, purity]
    geom: [f32; 4],
    // guest/Megatron beam focus [beam_min, beam_max, beam_shape, beam_range] — this
    // is the tube's sharpness / TVL: a tight beam = a sharp PVM, a wide one = fuzzy.
    beam: [f32; 4],
    // phosphor white point warmth (0 = cool/bright PC monitor, 1 = warm/aged TV).
    warmth: f32,
    // global persistence multiplier (1.0 = colour TV; long-persistence mono ~3+).
    persist: f32,
    // input signal path: 0=RGB/component (clean), 1=S-video (Y/C split), 2=composite.
    signal: u8,
    // phosphor set: 0=SMPTE-C, 1=P22, 2=sRGB/709, 3=mono (identity — mono tints itself).
    phos: u8,
    // native CRT white point (CIE xy) — 9300K reads cool/blue, D65 neutral, warm=aged.
    white_xy: [f32; 2],
    // monochrome phosphor: tint rgb + flag in .w (0 = colour CRT, 1 = single-gun mono).
    mono: [f32; 4],
}

// Real phosphor primaries (CIE 1931 xy). SMPTE-C is the standardized NTSC CRT set
// (a tightened P22); P22 is the looser consumer set; sRGB/709 for PC monitors.
const PHOS_SMPTE_C: [[f32; 2]; 3] = [[0.630, 0.340], [0.310, 0.595], [0.155, 0.070]];
const PHOS_P22: [[f32; 2]; 3] = [[0.625, 0.340], [0.280, 0.605], [0.155, 0.070]];
const PHOS_SRGB: [[f32; 2]; 3] = [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]];

// Build the 3x3 that maps CRT-phosphor drive RGB (linear) → linear sRGB (D65 display),
// baking in the tube's real gamut AND white point (so a 9300K set reads blue). Rows
// are returned for per-channel dot products in the shader. phos==3 (mono) → identity.
fn preset_color_matrix(preset: &Preset) -> [[f32; 4]; 3] {
    if preset.phos == 3 {
        return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    }
    let prim = match preset.phos {
        1 => PHOS_P22,
        2 => PHOS_SRGB,
        _ => PHOS_SMPTE_C,
    };
    let xyz = |x: f32, y: f32| Vec3::new(x / y, 1.0, (1.0 - x - y) / y);
    let (r, g, b) = (
        xyz(prim[0][0], prim[0][1]),
        xyz(prim[1][0], prim[1][1]),
        xyz(prim[2][0], prim[2][1]),
    );
    let w = xyz(preset.white_xy[0], preset.white_xy[1]);
    let m = glam::Mat3::from_cols(r, g, b);
    let s = m.inverse() * w; // per-primary scale so the primaries sum to the white point
    let rgb2xyz = glam::Mat3::from_cols(r * s.x, g * s.y, b * s.z);
    // XYZ → linear sRGB (D65), column-major for glam.
    let xyz2srgb = glam::Mat3::from_cols_array(&[
        3.2406, -0.9689, 0.0557, -1.5372, 1.8758, -0.2040, -0.4986, 0.0415, 1.0570,
    ]);
    let c = (xyz2srgb * rgb2xyz).to_cols_array(); // column-major: [c0.xyz, c1.xyz, c2.xyz]
    // row i = [c0[i], c1[i], c2[i]]
    [
        [c[0], c[3], c[6], 0.0],
        [c[1], c[4], c[7], 0.0],
        [c[2], c[5], c[8], 0.0],
    ]
}

// Sony Trinitron / PVM: aperture grille, near-flat vertically (cylindrical).
const TRINITRON: Preset = Preset {
    name: "trinitron",
    bulge: 0.10,
    curv_x: 0.34,
    curv_y: 0.16,
    mask_type: 0.0,
    mask_strength: 0.90,
    scanline: 0.55,
    halation: 0.35,
    parallax: 0.10,
    reflection: 0.50,
    vignette: 0.30,
    mask_pitch: 3.0,
    // a pro Trinitron/PVM is tightly converged with squarer corners.
    convergence: 0.010,
    corner_radius: 0.05,
    // studio-grade geometry: nearly straight, minimal impurity.
    geom: [0.008, 0.0, 0.010, 0.03],
    beam: [0.34, 0.74, 0.75, 1.0],
    warmth: 0.5,
    persist: 1.0,
    phos: 0,
    white_xy: [0.2831, 0.2971],
    signal: 1,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// Panasonic-style consumer set: shadow (dot) mask, spherical bulge.
const PANASONIC: Preset = Preset {
    name: "panasonic",
    bulge: 0.13,
    curv_x: 0.50,
    curv_y: 0.50,
    mask_type: 1.0,
    mask_strength: 0.85,
    scanline: 0.55,
    halation: 0.42,
    parallax: 0.13,
    reflection: 0.45,
    vignette: 0.48,
    mask_pitch: 3.0,
    // consumer set: looser convergence, rounder tube corners.
    convergence: 0.038,
    corner_radius: 0.12,
    // consumer geometry: visible pincushion + a little keystone and purity drift.
    geom: [0.055, 0.022, 0.060, 0.10],
    beam: [0.36, 0.78, 0.75, 1.0],
    warmth: 0.5,
    persist: 1.0,
    phos: 0,
    white_xy: [0.2831, 0.2971],
    signal: 2,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// Slot-mask consumer set (e.g. many 90s TVs).
const SLOTMASK: Preset = Preset {
    name: "slotmask",
    bulge: 0.10,
    curv_x: 0.42,
    curv_y: 0.38,
    mask_type: 2.0,
    mask_strength: 0.85,
    scanline: 0.52,
    halation: 0.40,
    parallax: 0.11,
    reflection: 0.45,
    vignette: 0.42,
    mask_pitch: 3.0,
    convergence: 0.028,
    corner_radius: 0.10,
    geom: [0.040, -0.015, 0.040, 0.08],
    beam: [0.35, 0.76, 0.75, 1.0],
    warmth: 0.5,
    persist: 1.0,
    phos: 0,
    white_xy: [0.2831, 0.2971],
    signal: 2,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// RCA ColorTrak-style console TV: shadow mask, soft/fuzzy, warm, very curved. Old
// consumer sets were low-TVL (~300s) with a wide, unfocused beam and lots of bloom.
const RCA: Preset = Preset {
    name: "rca",
    bulge: 0.13,
    curv_x: 0.52,
    curv_y: 0.48, // deeply curved old spherical tube
    mask_type: 1.0, // shadow-mask dot triads
    mask_strength: 0.62, // soft, low-contrast mask
    scanline: 0.45,
    halation: 0.62, // glowy, blooms warm
    parallax: 0.13,
    reflection: 0.55,
    vignette: 0.50,
    mask_pitch: 3.7, // coarse
    convergence: 0.055, // loose → colour fringing
    corner_radius: 0.13,
    geom: [0.060, 0.025, 0.070, 0.14], // consumer bow + purity drift
    beam: [0.48, 0.98, 0.65, 1.0], // WIDE, unfocused beam = fuzzy / low TVL
    warmth: 0.72, // warm, aged/yellowed white point
    persist: 1.0,
    phos: 0,
    white_xy: [0.305, 0.322],
    signal: 2,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// Sony PVM/BVM broadcast monitor: aperture grille, razor-sharp (600–800 TVL), fine
// stripe pitch, near-flat cylindrical face, studio-grade geometry, AR-coated glass.
const PVM: Preset = Preset {
    name: "pvm",
    bulge: 0.07,
    curv_x: 0.26,
    curv_y: 0.09, // near-flat, cylindrical
    mask_type: 0.0, // aperture grille
    mask_strength: 0.95,
    scanline: 0.56, // crisp, clean 240p scanlines
    halation: 0.22, // low bloom
    parallax: 0.09,
    reflection: 0.34,
    vignette: 0.26,
    mask_pitch: 2.5, // fine ~0.25 mm stripe
    convergence: 0.008, // tight
    corner_radius: 0.04, // squarish pro face
    geom: [0.006, 0.0, 0.008, 0.02], // near-perfect
    beam: [0.26, 0.56, 0.85, 1.0], // TIGHT beam = sharp / high TVL
    warmth: 0.34, // calibrated, slightly warm of D65
    persist: 1.0,
    phos: 0,
    white_xy: [0.3127, 0.329],
    signal: 0,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// 15 kHz arcade monitor (Wells Gardner / Hantarex chassis on a consumer-grade tube):
// shadow-mask triads, coarse pitch, big visible 240p scanlines, often misconverged.
const ARCADE: Preset = Preset {
    name: "arcade",
    bulge: 0.12,
    curv_x: 0.42,
    curv_y: 0.40,
    mask_type: 1.0, // shadow-mask triads
    mask_strength: 0.80,
    scanline: 0.62, // big, proud scanlines
    halation: 0.44,
    parallax: 0.13,
    reflection: 0.55, // bare (uncoated) glass
    vignette: 0.46,
    mask_pitch: 4.5, // coarse big-tube pitch
    convergence: 0.045, // frequently misadjusted
    corner_radius: 0.11,
    geom: [0.050, 0.020, 0.050, 0.10],
    beam: [0.40, 0.90, 0.70, 1.0], // wide, strong scanline gaps
    warmth: 0.50,
    persist: 1.0,
    phos: 0,
    white_xy: [0.2831, 0.2971],
    signal: 0,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// NEC MultiSync-style VGA PC monitor: fine (~0.28 mm) shadow mask, flatter late-CRT
// face, high line count (subtle scanlines), good geometry, cool/bright white.
const VGA: Preset = Preset {
    name: "vga",
    bulge: 0.05,
    curv_x: 0.22,
    curv_y: 0.20, // flatter
    mask_type: 1.0, // fine shadow mask
    mask_strength: 0.85,
    scanline: 0.34, // 480+ lines → subtle scanlines
    halation: 0.28,
    parallax: 0.09,
    reflection: 0.42,
    vignette: 0.30,
    mask_pitch: 2.6, // fine ~0.28 mm
    convergence: 0.018,
    corner_radius: 0.07,
    geom: [0.018, 0.005, 0.020, 0.04], // good geometry
    beam: [0.28, 0.60, 0.85, 1.0], // sharp
    warmth: 0.15, // cool / bright
    persist: 1.0,
    phos: 2,
    white_xy: [0.2831, 0.2971],
    signal: 0,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// NEC Diamondtron / FE-series "totally flat" aperture-grille PC monitor: very fine
// (~0.24 mm) stripe, dead-flat face, superbright, minimal scanlines, cool white.
const DIAMONDTRON: Preset = Preset {
    name: "diamondtron",
    bulge: 0.02,
    curv_x: 0.05,
    curv_y: 0.05, // dead flat
    mask_type: 0.0, // aperture grille (has damper wires)
    mask_strength: 0.92,
    scanline: 0.30, // high-res, minimal scanlines
    halation: 0.20,
    parallax: 0.07, // thin flat faceplate
    reflection: 0.30, // AR coated
    vignette: 0.22,
    mask_pitch: 2.2, // very fine ~0.24 mm
    convergence: 0.010,
    corner_radius: 0.03,
    geom: [0.010, 0.0, 0.010, 0.02], // flat, well-corrected
    beam: [0.26, 0.55, 0.88, 1.0], // very sharp / bright
    warmth: 0.10, // cool superbright
    persist: 1.0,
    phos: 2,
    white_xy: [0.2831, 0.2971],
    signal: 0,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// Monochrome green terminal (P1/P39 green phosphor): a single electron gun, no
// colour mask, and long persistence — the lingering green afterglow of a VT-style
// text terminal / IBM 5151. Crisp text beam, warm glow, gently curved small tube.
const GREEN: Preset = Preset {
    name: "green",
    bulge: 0.09,
    curv_x: 0.34,
    curv_y: 0.30,
    mask_type: 0.0,     // unused (mono skips the RGB triad mask)
    mask_strength: 0.0, // no colour mask on a single-phosphor tube
    scanline: 0.42,
    halation: 0.55, // strong phosphor glow/bleed
    parallax: 0.10,
    reflection: 0.42,
    vignette: 0.36,
    mask_pitch: 3.0,
    convergence: 0.0, // one gun → no RGB misconvergence
    corner_radius: 0.08,
    geom: [0.028, 0.0, 0.030, 0.0], // no purity error on a mono tube
    beam: [0.30, 0.62, 0.85, 1.0],  // fairly tight for readable text
    warmth: 0.0,                    // colour comes from `mono`, not the warm tint
    persist: 3.6,                   // long P39-style green afterglow
    phos: 3,
    white_xy: [0.3127, 0.329],
    signal: 0,
    mono: [0.10, 1.0, 0.14, 1.0], // P1 green (CIE ~0.218,0.712) → sRGB, normalized
};

// Monochrome amber terminal (P3 amber phosphor): the easier-on-the-eyes amber of a
// Wyse/late-80s terminal. Same tube, warmer phosphor, a touch less persistence.
const AMBER: Preset = Preset {
    name: "amber",
    bulge: 0.09,
    curv_x: 0.34,
    curv_y: 0.30,
    mask_type: 0.0,
    mask_strength: 0.0,
    scanline: 0.42,
    halation: 0.52,
    parallax: 0.10,
    reflection: 0.42,
    vignette: 0.36,
    mask_pitch: 3.0,
    convergence: 0.0,
    corner_radius: 0.08,
    geom: [0.028, 0.0, 0.030, 0.0],
    beam: [0.30, 0.62, 0.85, 1.0],
    warmth: 0.0,
    persist: 2.6,                  // amber P3: medium-long
    phos: 3,
    white_xy: [0.3127, 0.329],
    signal: 0,
    mono: [1.0, 0.44, 0.06, 1.0], // P3 amber (CIE ~0.523,0.469) → sRGB, normalized
};

fn preset_by_name(name: &str) -> Preset {
    match name {
        "panasonic" => PANASONIC,
        "slotmask" => SLOTMASK,
        "rca" => RCA,
        "pvm" => PVM,
        "arcade" => ARCADE,
        "vga" => VGA,
        "diamondtron" => DIAMONDTRON,
        "green" => GREEN,
        "amber" => AMBER,
        _ => TRINITRON,
    }
}

// Cycle order for the Tab key + digit selection (1..9, 0).
const ALL_PRESETS: [Preset; 10] =
    [TRINITRON, PANASONIC, SLOTMASK, RCA, PVM, ARCADE, VGA, DIAMONDTRON, GREEN, AMBER];

// ---------------------------------------------------------------------------
// Shared GPU resources (surface-independent, so the live window and the
// headless `--shot` path build them the same way)
// ---------------------------------------------------------------------------

struct Resources {
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    ubuf: wgpu::Buffer,
    // The source texture is swappable: capture frames change size/format at runtime,
    // so the texture + bind groups are rebuilt on demand. Everything else is fixed.
    layout: wgpu::BindGroupLayout, // tube pass (samples the phosphor plane)
    sampler: wgpu::Sampler,
    source_size: (u32, u32),
    source_format: wgpu::TextureFormat,
    retained_texture: Option<wgpu::Texture>,
    source_view: wgpu::TextureView,
    // Average source color + average picture level (APL), refreshed on every
    // frame upload. Drives the screen-as-area-light bounce and the beam bloom/sag.
    avg: [f32; 4],

    // --- Phosphor persistence (pass A) ---
    // The signal is decoded and integrated over time into a floating-point phosphor
    // plane with exponential decay, so moving content leaves real fading trails. Two
    // textures ping-pong: pass A reads phosphor[cur] + source, writes phosphor[1-cur];
    // the tube pass then samples phosphor[1-cur] as its screen. accum_bind[i] reads
    // phosphor[i] (previous field); screen_bind[i] binds phosphor[i] for the tube.
    accum_pipeline: wgpu::RenderPipeline,
    accum_layout: wgpu::BindGroupLayout,
    phosphor: [wgpu::Texture; 2],
    phosphor_view: [wgpu::TextureView; 2],
    accum_bind: [wgpu::BindGroup; 2],
    screen_bind: [wgpu::BindGroup; 2],
    phos_cur: usize, // index holding the most recently written phosphor plane
}

const PHOSPHOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

// An HDR phosphor plane (render target + sampleable) at the source's resolution.
fn make_phosphor(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("phosphor"),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: PHOSPHOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

// Cheap average color + luma of a source frame (sampled, not every pixel), used to
// treat the screen as an area light and to modulate beam bloom by picture level.
fn source_stats(data: &[u8], w: u32, h: u32, bgra: bool) -> [f32; 4] {
    let px = (w as usize) * (h as usize);
    if px == 0 || data.len() < 4 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let step = (px / 4096).max(1); // cap at ~4k samples regardless of resolution
    let (mut ar, mut ag, mut ab, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut i = 0;
    while i < px {
        let o = i * 4;
        if o + 2 >= data.len() {
            break;
        }
        let (r, g, b) = if bgra {
            (data[o + 2], data[o + 1], data[o])
        } else {
            (data[o], data[o + 1], data[o + 2])
        };
        ar += r as f64;
        ag += g as f64;
        ab += b as f64;
        n += 1.0;
        i += step;
    }
    if n == 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let r = (ar / n / 255.0) as f32;
    let g = (ag / n / 255.0) as f32;
    let b = (ab / n / 255.0) as f32;
    [r, g, b, 0.299 * r + 0.587 * g + 0.114 * b]
}

impl Resources {
    // Rebuilds every bind group from the current source view + phosphor views. Call
    // after (re)creating the source texture or the phosphor planes.
    fn rebuild_binds(&mut self, device: &wgpu::Device) {
        for i in 0..2 {
            // tube pass: samples phosphor[i] as the "screen".
            self.screen_bind[i] = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("screen_bind"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.phosphor_view[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            // accum pass: reads source + phosphor[i] (previous field).
            self.accum_bind[i] = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("accum_bind"),
                layout: &self.accum_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.phosphor_view[i]),
                    },
                ],
            });
        }
    }

    // Uploads a new source frame, recreating the source (and matching phosphor
    // planes) if the size or format changed, then rebuilds all bind groups.
    fn set_source(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        rgba: &[u8],
    ) {
        if (width, height) != self.source_size || format != self.source_format {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("source"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.source_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            write_source(queue, &texture, width, height, rgba);
            self.retained_texture = Some(texture);
            // Phosphor planes track the source resolution, so rebuild them on a resize.
            if (width, height) != self.source_size {
                let (t0, v0) = make_phosphor(device, width, height);
                let (t1, v1) = make_phosphor(device, width, height);
                self.phosphor = [t0, t1];
                self.phosphor_view = [v0, v1];
                self.phos_cur = 0;
            }
            self.source_size = (width, height);
            self.source_format = format;
            self.rebuild_binds(device);
        } else if let Some(texture) = &self.retained_texture {
            write_source(queue, texture, width, height, rgba);
        }
        self.avg = source_stats(rgba, width, height, format == wgpu::TextureFormat::Bgra8UnormSrgb);
    }
}

fn write_source(queue: &wgpu::Queue, texture: &wgpu::Texture, width: u32, height: u32, rgba: &[u8]) {
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

// The source-texture + uniforms + mask are always driven in linear/sRGB; the only
// thing that varies between the window and a headless shot is the color target format.
fn build_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_format: wgpu::TextureFormat,
    preset: Preset,
) -> Resources {
    // --- source texture (initial: test pattern; swappable at runtime for capture) ---
    let (tw, th, texels) = make_test_pattern();
    let source_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let tex_size = wgpu::Extent3d {
            width: tw,
            height: th,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("source"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: source_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &texels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * tw),
                rows_per_image: Some(th),
            },
            tex_size,
        );
        let source_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // Phosphor persistence planes (ping-pong), sized to the source.
        let (p0t, p0v) = make_phosphor(device, tw, th);
        let (p1t, p1v) = make_phosphor(device, tw, th);
        let phosphor = [p0t, p1t];
        let phosphor_view = [p0v, p1v];

        // --- uniforms ---
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // accum (pass A) layout: uniforms + source tex + sampler + prev phosphor.
        let accum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("accum_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // --- pipeline ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // see the tube from any angle
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // --- accum (phosphor persistence) pipeline: fullscreen, no depth ---
        let accum_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("accum_pipeline_layout"),
            bind_group_layouts: &[&accum_layout],
            push_constant_ranges: &[],
        });
        let accum_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("accum_pipeline"),
            layout: Some(&accum_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_full",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_phosphor",
                targets: &[Some(wgpu::ColorTargetState {
                    format: PHOSPHOR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Initial bind groups (rebuilt on any source resize via rebuild_binds).
        let mk_screen = |pv: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("screen_bind"),
                layout: &bind_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(pv) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            })
        };
        let mk_accum = |pv: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("accum_bind"),
                layout: &accum_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&source_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(pv) },
                ],
            })
        };
        let screen_bind = [mk_screen(&phosphor_view[0]), mk_screen(&phosphor_view[1])];
        let accum_bind = [mk_accum(&phosphor_view[0]), mk_accum(&phosphor_view[1])];

    // --- geometry buffers ---
    let (verts, indices) = build_mesh(preset.bulge, preset.curv_x, preset.curv_y);
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vbuf"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ibuf"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    Resources {
        pipeline,
        vbuf,
        ibuf,
        index_count: indices.len() as u32,
        ubuf,
        layout: bind_layout,
        sampler,
        source_size: (tw, th),
        source_format,
        retained_texture: Some(texture),
        source_view,
        avg: source_stats(&texels, tw, th, false),
        accum_pipeline,
        accum_layout,
        phosphor,
        phosphor_view,
        accum_bind,
        screen_bind,
        phos_cur: 0,
    }
}

fn create_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn write_uniforms(
    queue: &wgpu::Queue,
    res: &Resources,
    orbit: &Orbit,
    aspect: f32,
    time: f32,
    preset: &Preset,
    render_scale: f32,
    hdr: bool,
    dt: f32,
    pwr: [f32; 4],
    interlace: f32,
    field: f32,
    exposure: f32,
) {
    let (view_proj, eye) = orbit.view_proj(aspect);
    let cmat = preset_color_matrix(preset);
    let uniforms = Uniforms {
        view_proj: view_proj.to_cols_array(),
        model: Mat4::IDENTITY.to_cols_array(),
        cam_pos: [eye.x, eye.y, eye.z, 1.0],
        params: [
            res.source_size.0 as f32,
            res.source_size.1 as f32,
            time,
            render_scale,
        ],
        optics: [
            preset.mask_type,
            preset.mask_strength,
            preset.scanline,
            preset.halation,
        ],
        glass: [
            preset.parallax,
            preset.reflection,
            preset.vignette,
            preset.mask_pitch,
        ],
        // HDR path: on a scRGB swapchain, emit linear light with highlights >1.0
        // (peak/drive push the beam above white). On SDR, tonemap to `peak` white
        // point. `beam_drive` is the extra gain applied to scanline-beam cores.
        // tone.w = input signal path (0=RGB/component clean, 1=S-video, 2=composite).
        // tone.y carries the exposure trim on the HDR path (scales SDR-white → panel
        // reference white) and the tonemap white-point × exposure on the SDR path, so
        // the [ and ] keys tune brightness identically in both.
        tone: if hdr {
            [1.0, exposure, 1.9, preset.signal as f32]
        } else {
            [0.0, 1.08 * exposure, 1.7, preset.signal as f32] // ACES exposure (was Reinhard white pt)
        },
        // Guest/Megatron beam math (per-tube focus/TVL): per-channel beam half-width
        // runs from beam_min (dark → tight) to beam_max (bright → wide); beam_shape
        // curves the growth; beam_range = ± source rows summed. Tight = sharp PVM,
        // wide = fuzzy RCA.
        scan: preset.beam,
        // The screen radiates its average color/brightness onto the tube body.
        env: res.avg,
        // convergence + corner rounding come from the preset; grain (analog noise
        // floor) and ghost (secondary internal glass reflection) are global.
        look: [preset.convergence, preset.corner_radius, 0.015, 0.012],
        // CRT gamma (deepens blacks), per-tube warm/cool phosphor white point,
        // screen→tube glow bounce strength, and highlight bloom gain.
        phys: [1.12, preset.warmth, 0.42, 0.5],
        // Phosphor persistence + interlace: dt drives per-frame decay; temporal.y is
        // the per-tube persistence multiplier; temporal.z = interlace amount, .w =
        // field parity (alternate fields excite alternate lines → 480i twitter).
        temporal: [dt.max(0.0), preset.persist, interlace, field],
        // Per-phosphor decay constants (seconds). Grounded in measured P22 decay-to-10%
        // times (ePanorama/labguysworld phosphor data): red Y2O2S:Eu lingers a few
        // hundred µs to ~1 ms, while green ZnS:Cu and blue ZnS:Ag both snap off in
        // <100 µs (green a hair slower than blue) — so red ≈ 5× the blue/green tail.
        // The absolute scale is exaggerated ~50× so the trail is visible at 60 Hz on an
        // LCD, but the R:G:B ratio (≈5 : 1.3 : 1) now matches the real phosphors: bright
        // motion trails warm/reddish, blue/green edges stay crisp together.
        ptau: [0.055, 0.014, 0.011, 0.0],
        // Raster deflection geometry errors, per tube (see Preset.geom).
        geom: preset.geom,
        // Monochrome phosphor tint + flag (single-gun green/amber terminals).
        mono: preset.mono,
        // Real phosphor-gamut + white-point colour matrix (computed on the CPU).
        cmat0: cmat[0],
        cmat1: cmat[1],
        cmat2: cmat[2],
        // Power-on warmup / power-off collapse / degauss animation state.
        pwr,
        // Deflection defocus + overscan, derived from the tube's character (below).
        focus: {
            // Edge/corner defocus (physics: off-axis the beam path lengthens and the
            // deflection field grows, so the spot widens astigmatically toward the
            // edges — worst in the corners; US6329746/US6525459). Scale it off the
            // tube's beam-focus quality (preset.beam[1] = bright-beam half-width): a
            // razor PVM/Diamondtron (~0.55) barely blooms, a fuzzy RCA/arcade (~0.9–1.0)
            // softens hard at the edges. Applies to every tube, mono included.
            let defocus = ((preset.beam[1] - 0.55) * 1.15).clamp(0.0, 0.7);
            // Overscan (per side): consumer sets deliberately draw the raster larger
            // than the visible faceplate so the picture edges fall off (BBC-safe-area
            // convention ~3.5–5%); PC monitors and mono terminals run essentially full
            // raster. Composite RF consumer ~4.5%, S-video ~3.5%, component/RGB
            // broadcast ~2%; phos 2 (PC sRGB) / 3 (mono terminal) → 0.
            let overscan = if preset.phos >= 2 {
                0.0
            } else {
                match preset.signal {
                    2 => 0.045,
                    1 => 0.035,
                    _ => 0.02,
                }
            };
            // Rolling refresh band (focus.z = roll rate Hz, focus.w = amplitude): the
            // vertical "hum bar" you see on a CRT is the BEAT between the viewing/capture
            // rate and the tube's 59.94 Hz field sweep — a soft freshly-scanned bright
            // band rolling down the picture. Dead-on by eye it's invisible; here (a
            // "captured" CRT on an LCD) a gentle ~0.45 Hz beat reads as a living tube.
            // 480i doubles the beat feel via field twitter (handled in the accum pass).
            let roll_rate = if preset.mono[3] > 0.5 { 0.30 } else { 0.45 };
            [defocus, overscan, roll_rate, 0.05]
        },
    };
    queue.write_buffer(&res.ubuf, 0, bytemuck::bytes_of(&uniforms));
}

// Pass A: advance the phosphor plane one field. Reads phosphor[cur] (previous) +
// source, writes phosphor[1-cur], then flips cur. This is where the signal is
// decoded and integrated over time with exponential decay (real persistence).
fn accum_step(encoder: &mut wgpu::CommandEncoder, res: &mut Resources) {
    let src = res.phos_cur;
    let dst = 1 - res.phos_cur;
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("accum_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &res.phosphor_view[dst],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&res.accum_pipeline);
        pass.set_bind_group(0, &res.accum_bind[src], &[]);
        pass.draw(0..3, 0..1); // fullscreen triangle
    }
    res.phos_cur = dst;
}

// Pass B: draw the 3D tube, sampling the current phosphor plane as its screen.
fn draw_tube(
    encoder: &mut wgpu::CommandEncoder,
    res: &Resources,
    color: &wgpu::TextureView,
    depth: &wgpu::TextureView,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("tube_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.01,
                    g: 0.01,
                    b: 0.015,
                    a: 1.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(&res.pipeline);
    pass.set_bind_group(0, &res.screen_bind[res.phos_cur], &[]);
    pass.set_vertex_buffer(0, res.vbuf.slice(..));
    pass.set_index_buffer(res.ibuf.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..res.index_count, 0, 0..1);
}

// ---------------------------------------------------------------------------
// Power theatre — warmup / power-off collapse / degauss, grounded in real timing:
// power-off collapses the raster vertically to a bright line, then horizontally to a
// fading phosphor dot (~1.1s total); warmup runs that in reverse (~2s); degauss runs
// a decaying AC wobble + rainbow purity for ~1.8s (auto-fires on power-on).
// ---------------------------------------------------------------------------

const WARMUP_DUR: f32 = 2.0;
const COLLAPSE_DUR: f32 = 1.1;
const DEGAUSS_DUR: f32 = 0.9; // cutoff; the visible burst is front-loaded (see envelope)
const DEGAUSS_TAU: f32 = 0.22; // exponential decay of the AC burst — quick, snappy

#[derive(Clone, Copy)]
enum PowerState {
    Warmup(std::time::Instant),
    On,
    Collapse(std::time::Instant),
    Off,
}

fn smoothstep01(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    res: Resources,
    depth_view: wgpu::TextureView,
    orbit: Orbit,
    start: std::time::Instant,
    last_frame: std::time::Instant, // for per-frame dt (phosphor decay)
    dragging: bool,
    last_cursor: (f64, f64),
    window: Arc<Window>,
    capture: Option<capture::SharedFrame>,
    last_seq: u64,
    preset: Preset,
    hdr: bool, // true = scRGB HDR swapchain, false = SDR (tonemap on output)
    power: PowerState,
    degauss_start: Option<std::time::Instant>,
    frame: u64,       // field counter for 480i interlace
    interlace: bool,  // 480i (alternating fields) vs 240p progressive
    exposure: f32,    // live HDR/SDR exposure trim ([ and ] keys) for tuning on the panel
}

impl State {
    async fn new(
        window: Arc<Window>,
        capture: Option<capture::SharedFrame>,
        preset: Preset,
    ) -> State {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY, // Vulkan on Linux
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("failed to create device");

        let caps = surface.get_capabilities(&adapter);
        eprintln!("[surface] offered formats: {:?}", caps.formats);
        // Prefer a true HDR swapchain: Rgba16Float is scRGB (linear, 1.0 = SDR
        // white, values >1.0 = extra nits). Fall back to sRGB 8-bit otherwise.
        // NOTE: a compositor must actually advertise the float format for HDR to
        // engage; many Wayland compositors (mutter, most X11) only expose sRGB,
        // in which case we render HDR internally and tonemap to SDR for display.
        let hdr_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Rgba16Float);
        let format = hdr_format
            .or_else(|| caps.formats.iter().copied().find(|f| f.is_srgb()))
            .unwrap_or(caps.formats[0]);
        let hdr = format == wgpu::TextureFormat::Rgba16Float;
        eprintln!(
            "[surface] using {:?} — HDR output {}",
            format,
            if hdr { "ENABLED (Rgba16Float, BT.2020 linear)" } else { "unavailable → SDR tonemap" }
        );
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            // 1 (not 2): the compositor holds one fewer in-flight frame, cutting ~16 ms
            // of input→display latency so orbiting the tube tracks the cursor tighter.
            // We're nowhere near GPU-bound, so the shorter queue doesn't cost throughput.
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);

        let res = build_resources(&device, &queue, format, preset);
        let depth_view = create_depth(&device, config.width, config.height);

        State {
            surface,
            device,
            queue,
            config,
            size,
            res,
            depth_view,
            orbit: Orbit {
                yaw: 0.0,
                pitch: 0.15,
                distance: 2.6,
            },
            start: std::time::Instant::now(),
            last_frame: std::time::Instant::now(),
            // Power up with a warmup + auto-degauss, like a real set switching on.
            power: PowerState::Warmup(std::time::Instant::now()),
            degauss_start: Some(std::time::Instant::now()),
            frame: 0,
            interlace: false,
            exposure: 1.0,
            dragging: false,
            last_cursor: (0.0, 0.0),
            window,
            capture,
            last_seq: 0,
            preset,
            hdr,
        }
    }

    // Upload the latest captured frame, if any, before drawing.
    fn poll_capture(&mut self) {
        let Some(shared) = &self.capture else { return };
        // Move the latest frame OUT of the shared slot under a brief lock, then drop the
        // lock before the (comparatively slow) GPU upload + stats. The PipeWire capture
        // thread only ever blocks on this lock for the duration of an Option::take, so
        // our per-frame GPU work can't stall capture and cause frame-drop / micro-stutter.
        let frame = {
            let Ok(mut guard) = shared.lock() else { return };
            match guard.as_ref() {
                Some(f) if f.seq != self.last_seq => guard.take().unwrap(),
                _ => return,
            }
        };
        let format = if frame.is_bgra {
            wgpu::TextureFormat::Bgra8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8UnormSrgb
        };
        self.res.set_source(
            &self.device,
            &self.queue,
            frame.width,
            frame.height,
            format,
            &frame.data,
        );
        self.last_seq = frame.seq;
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth(&self.device, self.config.width, self.config.height);
    }

    // Switch tube/mask preset live. Optics come from `self.preset` each frame, but
    // the curvature is baked into the mesh, so the geometry buffers are rebuilt.
    fn set_preset(&mut self, preset: Preset) {
        let (verts, indices) = build_mesh(preset.bulge, preset.curv_x, preset.curv_y);
        self.res.vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vbuf"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.res.ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ibuf"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.res.index_count = indices.len() as u32;
        self.preset = preset;
        eprintln!("[preset] {}", preset.name);
    }

    // Advance the power state and return [warmup, collapse, degauss, 0] for the shader.
    fn power_params(&mut self) -> [f32; 4] {
        let now = std::time::Instant::now();
        let (warmup, collapse) = match self.power {
            PowerState::Warmup(t0) => {
                let e = (now - t0).as_secs_f32();
                if e >= WARMUP_DUR {
                    self.power = PowerState::On;
                    (1.0, 0.0)
                } else {
                    (smoothstep01(e / WARMUP_DUR), 0.0)
                }
            }
            PowerState::On => (1.0, 0.0),
            PowerState::Collapse(t0) => {
                let e = (now - t0).as_secs_f32();
                if e >= COLLAPSE_DUR {
                    self.power = PowerState::Off;
                    (1.0, 1.0)
                } else {
                    (1.0, smoothstep01(e / COLLAPSE_DUR))
                }
            }
            PowerState::Off => (1.0, 1.0),
        };
        let degauss = match self.degauss_start {
            Some(t0) => {
                let e = (now - t0).as_secs_f32();
                if e >= DEGAUSS_DUR {
                    self.degauss_start = None;
                    0.0
                } else {
                    (-e / DEGAUSS_TAU).exp() // exponential AC burst — snaps then fades fast
                }
            }
            None => 0.0,
        };
        [warmup, collapse, degauss, 0.0]
    }

    // 'P' toggles power (with the collapse/warmup animation); auto-degauss on power-on.
    fn toggle_power(&mut self) {
        let now = std::time::Instant::now();
        self.power = match self.power {
            PowerState::On | PowerState::Warmup(_) => PowerState::Collapse(now),
            PowerState::Off | PowerState::Collapse(_) => {
                self.degauss_start = Some(now);
                PowerState::Warmup(now)
            }
        };
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.poll_capture();
        let dt = self.last_frame.elapsed().as_secs_f32().clamp(0.0, 0.1);
        self.last_frame = std::time::Instant::now();
        let pwr = self.power_params();
        self.frame = self.frame.wrapping_add(1);
        let (interlace, field) = if self.interlace {
            (0.7, (self.frame & 1) as f32)
        } else {
            (0.0, 0.0)
        };
        let aspect = self.config.width as f32 / self.config.height as f32;
        write_uniforms(
            &self.queue,
            &self.res,
            &self.orbit,
            aspect,
            self.start.elapsed().as_secs_f32(),
            &self.preset,
            1.0, // live window renders at surface resolution (no supersampling)
            self.hdr,
            dt,
            pwr,
            interlace,
            field,
            self.exposure,
        );

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        // Advance the phosphor plane one field, then draw the tube sampling it.
        accum_step(&mut encoder, &mut self.res);
        draw_tube(&mut encoder, &self.res, &view, &self.depth_view);
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Headless capture: render one frame to a PNG (`--shot out.png`)
// ---------------------------------------------------------------------------

fn save_shot(path: &str, width: u32, height: u32, preset: Preset) {
    // Supersample: the CRT's fine mask + scanline structure sits near the output
    // Nyquist limit, so render at SSxSS and box-downsample (in linear light) to
    // the requested size. Without this the fine detail aliases into flat color.
    const SS: u32 = 3;
    let rw = width * SS;
    let rh = height * SS;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no GPU adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("headless-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("device");

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut res = build_resources(&device, &queue, format, preset);

    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot-color"),
        size: wgpu::Extent3d {
            width: rw,
            height: rh,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = create_depth(&device, rw, rh);

    // Camera can be overridden for verification/tuning (defaults show the funnel).
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let orbit = Orbit {
        yaw: envf("CRTULUM_YAW", 0.82),
        pitch: envf("CRTULUM_PITCH", 0.34),
        distance: envf("CRTULUM_DIST", 3.7),
    };
    // The shot path writes an 8-bit sRGB PNG, so always tonemap to SDR.
    // CRTULUM_TIME lets a still capture pick a moment in the beam-scan cycle.
    let shot_t = std::env::var("CRTULUM_TIME").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let dt = 1.0 / 60.0;
    // Power state for stills: default fully on; override to capture a warmup/collapse/
    // degauss phase (CRTULUM_WARMUP / _COLLAPSE / _DEGAUSS in 0..1).
    let pwr = [
        envf("CRTULUM_WARMUP", 1.0),
        envf("CRTULUM_COLLAPSE", 0.0),
        envf("CRTULUM_DEGAUSS", 0.0),
        0.0,
    ];
    let interlace = envf("CRTULUM_INTERLACE", 0.0);
    let field = envf("CRTULUM_FIELD", 0.0);
    let exposure = std::env::var("CRTULUM_EXPOSURE").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    write_uniforms(&queue, &res, &orbit, width as f32 / height as f32, shot_t, &preset, SS as f32, false, dt, pwr, interlace, field, exposure);

    // Warm up the phosphor plane. A single headless frame has no history, so run
    // the accumulation a few fields to reach steady state. CRTULUM_MOTION=1 instead
    // sweeps a bright bar across a dark source so the persistence trail is visible
    // in the still (headless proof that the history buffer actually integrates).
    let motion = std::env::var("CRTULUM_MOTION").ok().as_deref() == Some("1");
    let steps: u32 = if motion { 18 } else { 4 };
    for s in 0..steps {
        if motion {
            let (mw, mh) = (320u32, 240u32);
            let barx = (mw as f32 * (0.12 + 0.76 * s as f32 / (steps - 1) as f32)) as i32;
            let mut buf = vec![0u8; (mw * mh * 4) as usize];
            for y in 0..mh {
                for x in 0..mw {
                    let idx = ((y * mw + x) * 4) as usize;
                    let on = (x as i32 - barx).abs() < 6 && y > mh / 6 && y < mh * 5 / 6;
                    let v = if on { 240 } else { 6 };
                    buf[idx] = v;
                    buf[idx + 1] = v;
                    buf[idx + 2] = v;
                    buf[idx + 3] = 255;
                }
            }
            res.set_source(&device, &queue, mw, mh, wgpu::TextureFormat::Rgba8UnormSrgb, &buf);
        }
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("accum-enc") });
        accum_step(&mut enc, &mut res);
        queue.submit(std::iter::once(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // padded copy: bytes_per_row must be a multiple of 256
    let unpadded = rw * 4;
    let padded = ((unpadded + 255) / 256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * rh) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shot-enc") });
    draw_tube(&mut encoder, &res, &color_view, &depth_view);
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &readback,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(rh),
            },
        },
        wgpu::Extent3d {
            width: rw,
            height: rh,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();

    // Box-downsample SSxSS → 1, averaging in linear light (the buffer is sRGB).
    let srgb_to_lin = |c: u8| {
        let s = c as f32 / 255.0;
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    };
    let lin_to_srgb = |l: f32| {
        let s = if l <= 0.0031308 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
        (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    let inv = 1.0 / (SS * SS) as f32;
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for oy in 0..height {
        for ox in 0..width {
            let mut acc = [0.0f32; 4];
            for sy in 0..SS {
                let row = ((oy * SS + sy) * padded) as usize;
                for sx in 0..SS {
                    let p = row + ((ox * SS + sx) * 4) as usize;
                    acc[0] += srgb_to_lin(data[p]);
                    acc[1] += srgb_to_lin(data[p + 1]);
                    acc[2] += srgb_to_lin(data[p + 2]);
                    acc[3] += data[p + 3] as f32 / 255.0;
                }
            }
            pixels.push(lin_to_srgb(acc[0] * inv));
            pixels.push(lin_to_srgb(acc[1] * inv));
            pixels.push(lin_to_srgb(acc[2] * inv));
            pixels.push((acc[3] * inv * 255.0 + 0.5) as u8);
        }
    }
    let img = image::RgbaImage::from_raw(width, height, pixels).expect("image");
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    img.save(path).expect("save png");
    println!("wrote {path} ({width}x{height})");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();

    // `--preset trinitron|panasonic|slotmask` (default trinitron)
    let preset = args
        .iter()
        .position(|a| a == "--preset")
        .and_then(|i| args.get(i + 1))
        .map(|s| preset_by_name(s))
        .unwrap_or(TRINITRON);
    eprintln!("[preset] {}", preset.name);

    // Headless capture mode: `crtulum --shot out.png [WxH]`
    if let Some(i) = args.iter().position(|a| a == "--shot") {
        let path = args.get(i + 1).map(String::as_str).unwrap_or("shot.png");
        let (w, h) = args
            .get(i + 2)
            .and_then(|s| s.split_once('x'))
            .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
            .unwrap_or((1000, 800));
        save_shot(path, w, h, preset);
        return;
    }

    let capture = if args.iter().any(|a| a == "--capture") {
        eprintln!("[capture] starting — pick a window or screen in the portal dialog…");
        Some(capture::spawn())
    } else {
        None
    };

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("crtulum")
            .with_inner_size(winit::dpi::LogicalSize::new(1000.0, 800.0))
            .build(&event_loop)
            .unwrap(),
    );

    let mut state = pollster::block_on(State::new(window.clone(), capture, preset));

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
            match event {
                Event::WindowEvent { event, window_id } if window_id == state.window.id() => {
                    match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::KeyboardInput { event, .. } => {
                            if event.state == ElementState::Pressed {
                                match event.physical_key {
                                    PhysicalKey::Code(KeyCode::Escape) => elwt.exit(),
                                    // 1..9,0 pick a preset directly; Tab cycles through all.
                                    PhysicalKey::Code(KeyCode::Digit1) => state.set_preset(ALL_PRESETS[0]),
                                    PhysicalKey::Code(KeyCode::Digit2) => state.set_preset(ALL_PRESETS[1]),
                                    PhysicalKey::Code(KeyCode::Digit3) => state.set_preset(ALL_PRESETS[2]),
                                    PhysicalKey::Code(KeyCode::Digit4) => state.set_preset(ALL_PRESETS[3]),
                                    PhysicalKey::Code(KeyCode::Digit5) => state.set_preset(ALL_PRESETS[4]),
                                    PhysicalKey::Code(KeyCode::Digit6) => state.set_preset(ALL_PRESETS[5]),
                                    PhysicalKey::Code(KeyCode::Digit7) => state.set_preset(ALL_PRESETS[6]),
                                    PhysicalKey::Code(KeyCode::Digit8) => state.set_preset(ALL_PRESETS[7]),
                                    PhysicalKey::Code(KeyCode::Digit9) => state.set_preset(ALL_PRESETS[8]),
                                    PhysicalKey::Code(KeyCode::Digit0) => state.set_preset(ALL_PRESETS[9]),
                                    // P = power (warmup ↔ collapse); G = degauss.
                                    PhysicalKey::Code(KeyCode::KeyP) => state.toggle_power(),
                                    PhysicalKey::Code(KeyCode::KeyG) => {
                                        state.degauss_start = Some(std::time::Instant::now())
                                    }
                                    // [ / ] = trim exposure down/up (tune HDR on the panel).
                                    PhysicalKey::Code(KeyCode::BracketLeft) => {
                                        state.exposure = (state.exposure * 0.92).clamp(0.2, 5.0);
                                        eprintln!("[exposure] {:.2}", state.exposure);
                                    }
                                    PhysicalKey::Code(KeyCode::BracketRight) => {
                                        state.exposure = (state.exposure * 1.08).clamp(0.2, 5.0);
                                        eprintln!("[exposure] {:.2}", state.exposure);
                                    }
                                    // I = toggle 480i interlace vs 240p progressive.
                                    PhysicalKey::Code(KeyCode::KeyI) => {
                                        state.interlace = !state.interlace;
                                        eprintln!("[interlace] {}", if state.interlace { "480i" } else { "240p" });
                                    }
                                    PhysicalKey::Code(KeyCode::Tab) => {
                                        let i = ALL_PRESETS
                                            .iter()
                                            .position(|p| p.name == state.preset.name)
                                            .unwrap_or(0);
                                        state.set_preset(ALL_PRESETS[(i + 1) % ALL_PRESETS.len()]);
                                    }
                                    _ => {}
                                }
                            }
                        }
                        WindowEvent::Resized(size) => state.resize(size),
                        WindowEvent::MouseInput { state: s, button, .. } => {
                            if button == MouseButton::Left {
                                state.dragging = s == ElementState::Pressed;
                            }
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            let (px, py) = (position.x, position.y);
                            if state.dragging {
                                let dx = (px - state.last_cursor.0) as f32;
                                let dy = (py - state.last_cursor.1) as f32;
                                state.orbit.yaw -= dx * 0.005;
                                state.orbit.pitch =
                                    (state.orbit.pitch + dy * 0.005).clamp(-1.4, 1.4);
                            }
                            state.last_cursor = (px, py);
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                            let d = match delta {
                                MouseScrollDelta::LineDelta(_, y) => y,
                                MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                            };
                            state.orbit.distance = (state.orbit.distance - d * 0.2).clamp(1.2, 8.0);
                        }
                        WindowEvent::RedrawRequested => match state.render() {
                            Ok(()) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                state.resize(state.size)
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                            Err(e) => log::warn!("surface error: {e:?}"),
                        },
                        _ => {}
                    }
                }
                Event::AboutToWait => state.window.request_redraw(),
                _ => {}
            }
        })
        .unwrap();
}
