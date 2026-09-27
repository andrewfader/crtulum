// crtulum — a manipulable 3D CRT tube in a window.
//
//   cargo run              : test-pattern source
//   cargo run -- --capture : live source via the ScreenCast portal + PipeWire (M2)
//   cargo run -- --shot out.png 1000x800 : headless PNG render
//   cargo run -- --clip frames/ out/ 800x600 : run a frame sequence through the
//                                              tube, writing PNGs through the shared export renderer
//   cargo run -- --render clip.mp4 out.mp4 --script run.crts : scripted video
//                                              export (source → CRT → ffmpeg)
//   cargo run -- --render out.mp4 --script tas.crts : ditto, but the script also
//                                              drives a ROM frame-by-frame
//   cargo run -- --play game.sfc : play it, on the tube, with a controller
//
// Controls: left-drag orbit · scroll zoom · Esc quit

mod agent;
mod capture;
mod webcam;
mod font8x8;
mod glctx;
mod gpu;
mod gamescope;
mod vkctx;
mod libretro;
mod play;
mod video;
mod rom_cache;
mod phosphor;
mod benchmark;

use std::sync::Arc;

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowBuilder},
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

fn build_mesh(bulge: f32, cx: f32, cy: f32, cab: Cabinet) -> (Vec<Vertex>, Vec<u32>) {
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

    // --- 6. TV cabinet (material 3/5/6/7 = molded plastic, material 4 = speaker cloth) ---
    // The silhouette + finish are per-brand (see `Cabinet`), but every set shares the
    // same skeleton: the 4:3 screen recessed in the upper-centre, a top/side bezel, a
    // chin below the tube (speaker grille + controls on a consumer set; a thin uniform
    // strip on a pro monitor / PC display), and a tapered rear hump enclosing the funnel.
    // Proportions default to the deep, near-cubic Sony Trinitron KV-20TS (513 × 487 ×
    // 481 mm — a real CRT TV is almost as DEEP as it is wide, NOT a thin picture frame).
    let pl = cab.plastic; // body plastic material id
    let sb = cab.side_bezel;
    let tb = cab.top_bezel;
    let bc = cab.chin;
    let hw_cab = HALF_W + sb; //   outer half-width
    let cab_t = HALF_H + tb; //    top edge
    let cab_b = -(HALF_H + bc); // bottom edge
    let (cab_l, cab_r) = (-hw_cab, hw_cab);
    let (ox, oy) = (HALF_W * 1.018, HALF_H * 1.024); // screen opening (a touch over the glass)
    let z_front = bulge * 0.6 + 0.05; // front face plane, just behind the glass apex
    let z_rear = cab.rear_z; //   main box back (depth ≈ width → the real near-cube)
    let z_rear2 = z_rear - 0.48; // rear hump back (encloses the funnel/neck)
    // Edge chamfer: real injection-molded cabinets are never razor-edged — every outer
    // edge has a few-mm bevel that catches a bright highlight line. The front face is
    // inset by `cf`, then a 45° bevel (+ mitred corner facets) runs out to the full
    // cabinet extent where the side walls begin (at z = zc).
    let cf = cab.corner;
    let (fl, fr, fb, ft) = (cab_l + cf, cab_r - cf, cab_b + cf, cab_t - cf);
    let zc = z_front - cf;
    let quad = |v: &mut Vec<Vertex>, ix: &mut Vec<u32>, p: [[f32; 3]; 4], m: f32| push_quad(v, ix, p, m);
    // A rectangular panel recessed back from `zf` to `zr`: four plastic side walls + a
    // back face of material `mat` (mat 4 = speaker cloth, `pl` = a plain plastic recess).
    let recess_panel = |v: &mut Vec<Vertex>, ix: &mut Vec<u32>,
                        x0: f32, x1: f32, y0: f32, y1: f32, zf: f32, zr: f32, mat: f32| {
        quad(v, ix, [[x0, y0, zf], [x1, y0, zf], [x1, y0, zr], [x0, y0, zr]], pl); // bottom wall
        quad(v, ix, [[x0, y1, zf], [x1, y1, zf], [x1, y1, zr], [x0, y1, zr]], pl); // top wall
        quad(v, ix, [[x0, y0, zf], [x0, y1, zf], [x0, y1, zr], [x0, y0, zr]], pl); // left wall
        quad(v, ix, [[x1, y0, zf], [x1, y1, zf], [x1, y1, zr], [x1, y0, zr]], pl); // right wall
        quad(v, ix, [[x0, y0, zr], [x1, y0, zr], [x1, y1, zr], [x0, y1, zr]], mat); // back panel
    };
    // A short control knob (cylinder) protruding forward from `base_z` at (kx, ky).
    let knob = |v: &mut Vec<Vertex>, ix: &mut Vec<u32>, kx: f32, ky: f32, base_z: f32, kc: f32| {
        let kn = 16usize;
        let zk = base_z + 0.035;
        for k in 0..kn {
            let a0 = std::f32::consts::TAU * k as f32 / kn as f32;
            let a1 = std::f32::consts::TAU * (k + 1) as f32 / kn as f32;
            let p0 = [kx + kc * a0.cos(), ky + kc * a0.sin(), base_z];
            let p1 = [kx + kc * a1.cos(), ky + kc * a1.sin(), base_z];
            let q0 = [kx + kc * a0.cos(), ky + kc * a0.sin(), zk];
            let q1 = [kx + kc * a1.cos(), ky + kc * a1.sin(), zk];
            // Fractional material tag preserves the cabinet colour family while letting
            // the shader give frequently-handled knobs a slightly smoother finish.
            quad(v, ix, [p0, p1, q1, q0], pl + 0.15); // knob wall
            quad(v, ix, [q0, q1, [kx, ky, zk], [kx, ky, zk]], pl + 0.15); // knob top
        }
    };

    // 6a. Top bezel strip (always present, above the tube).
    quad(&mut verts, &mut indices,
        [[fl, oy, z_front], [fr, oy, z_front], [fr, ft, z_front], [fl, ft, z_front]], pl);
    // inner lip: recess the opening edge back to the glass block so the tube sits inset.
    let zl = -GLASS_T - 0.02;
    quad(&mut verts, &mut indices, [[-ox, oy, z_front], [ox, oy, z_front], [ox, oy, zl], [-ox, oy, zl]], pl);
    quad(&mut verts, &mut indices, [[-ox, -oy, z_front], [ox, -oy, z_front], [ox, -oy, zl], [-ox, -oy, zl]], pl);
    quad(&mut verts, &mut indices, [[-ox, -oy, z_front], [-ox, oy, z_front], [-ox, oy, zl], [-ox, -oy, zl]], pl);
    quad(&mut verts, &mut indices, [[ox, -oy, z_front], [ox, oy, z_front], [ox, oy, zl], [ox, -oy, zl]], pl);

    // 6a-bevel. 45° chamfer ring: the front face plane (z_front) out to the full cabinet
    // rectangle (zc), with four edge bevels + four mitred corner facets.
    quad(&mut verts, &mut indices, [[fl, ft, z_front], [fr, ft, z_front], [fr, cab_t, zc], [fl, cab_t, zc]], pl); // top
    quad(&mut verts, &mut indices, [[fl, fb, z_front], [fr, fb, z_front], [fr, cab_b, zc], [fl, cab_b, zc]], pl); // bottom
    quad(&mut verts, &mut indices, [[fl, fb, z_front], [fl, ft, z_front], [cab_l, ft, zc], [cab_l, fb, zc]], pl); // left
    quad(&mut verts, &mut indices, [[fr, fb, z_front], [fr, ft, z_front], [cab_r, ft, zc], [cab_r, fb, zc]], pl); // right
    quad(&mut verts, &mut indices, [[fr, ft, z_front], [fr, cab_t, zc], [cab_r, cab_t, zc], [cab_r, ft, zc]], pl); // TR corner
    quad(&mut verts, &mut indices, [[fl, ft, z_front], [fl, cab_t, zc], [cab_l, cab_t, zc], [cab_l, ft, zc]], pl); // TL corner
    quad(&mut verts, &mut indices, [[fr, fb, z_front], [fr, cab_b, zc], [cab_r, cab_b, zc], [cab_r, fb, zc]], pl); // BR corner
    quad(&mut verts, &mut indices, [[fl, fb, z_front], [fl, cab_b, zc], [cab_l, cab_b, zc], [cab_l, fb, zc]], pl); // BL corner

    let chin_top = -oy;

    // 6b. Side regions (tube left/right, oy..-oy): plain plastic on a bottom-speaker or
    // pro set; tall recessed speaker grilles flanking the tube on a Panasonic-style set.
    if cab.speakers == Speakers::Sides {
        for &sgn in &[-1.0f32, 1.0] {
            let (outer, inner) = if sgn < 0.0 { (fl, -ox) } else { (fr, ox) };
            let (xa, xb) = (outer.min(inner), outer.max(inner));
            let m = 0.028; // plastic border margin around the grille
            let (px0, px1, py0, py1) = (xa + m, xb - m, -oy + m, oy - m);
            quad(&mut verts, &mut indices, [[xa, -oy, z_front], [xb, -oy, z_front], [xb, py0, z_front], [xa, py0, z_front]], pl); // below
            quad(&mut verts, &mut indices, [[xa, py1, z_front], [xb, py1, z_front], [xb, oy, z_front], [xa, oy, z_front]], pl); // above
            quad(&mut verts, &mut indices, [[xa, py0, z_front], [px0, py0, z_front], [px0, py1, z_front], [xa, py1, z_front]], pl); // outer edge
            quad(&mut verts, &mut indices, [[px1, py0, z_front], [xb, py0, z_front], [xb, py1, z_front], [px1, py1, z_front]], pl); // inner edge
            recess_panel(&mut verts, &mut indices, px0, px1, py0, py1, z_front, z_front - 0.045, 4.0);
        }
    } else {
        quad(&mut verts, &mut indices, [[fl, -oy, z_front], [-ox, -oy, z_front], [-ox, oy, z_front], [fl, oy, z_front]], pl); // left
        quad(&mut verts, &mut indices, [[ox, -oy, z_front], [fr, -oy, z_front], [fr, oy, z_front], [ox, oy, z_front]], pl); // right
    }

    // 6c. Chin (below the tube), by speaker layout.
    match cab.speakers {
        Speakers::Bottom => {
            // A plastic frame around a recessed panel: speaker grille (left ~64%) + a
            // control plate with two knobs (right ~36%).
            let (rx0, rx1) = (fl + 0.05, fr - 0.05);
            let (ry0, ry1) = (fb + 0.075, chin_top - 0.06);
            let z_rec = z_front - 0.05;
            quad(&mut verts, &mut indices, [[fl, fb, z_front], [fr, fb, z_front], [fr, ry0, z_front], [fl, ry0, z_front]], pl); // below recess
            quad(&mut verts, &mut indices, [[fl, ry1, z_front], [fr, ry1, z_front], [fr, chin_top, z_front], [fl, chin_top, z_front]], pl); // above recess
            quad(&mut verts, &mut indices, [[fl, ry0, z_front], [rx0, ry0, z_front], [rx0, ry1, z_front], [fl, ry1, z_front]], pl); // left of recess
            quad(&mut verts, &mut indices, [[rx1, ry0, z_front], [fr, ry0, z_front], [fr, ry1, z_front], [rx1, ry1, z_front]], pl); // right of recess
            let gx1 = rx0 + (rx1 - rx0) * 0.64; // grille / controls split
            let div = 0.02;
            recess_panel(&mut verts, &mut indices, rx0, gx1, ry0, ry1, z_front, z_rec, 4.0); // speaker grille
            quad(&mut verts, &mut indices, [[gx1, ry0, z_rec], [gx1 + div, ry0, z_rec], [gx1 + div, ry1, z_rec], [gx1, ry1, z_rec]], pl); // divider
            recess_panel(&mut verts, &mut indices, gx1 + div, rx1, ry0, ry1, z_front, z_rec, pl); // control plate
            let ky = ry0 + (ry1 - ry0) * 0.5;
            knob(&mut verts, &mut indices, gx1 + (rx1 - gx1) * 0.36, ky, z_rec, 0.06);
            knob(&mut verts, &mut indices, gx1 + (rx1 - gx1) * 0.70, ky, z_rec, 0.06);
        }
        _ => {
            // Slim chin: a plain plastic panel (uniform bezel look). Consumer side-speaker
            // sets still carry a couple of control knobs on the lower right.
            quad(&mut verts, &mut indices, [[fl, fb, z_front], [fr, fb, z_front], [fr, chin_top, z_front], [fl, chin_top, z_front]], pl);
            if cab.speakers == Speakers::Sides {
                let ky = (fb + chin_top) * 0.5;
                knob(&mut verts, &mut indices, fr - 0.12, ky, z_front, 0.045);
                knob(&mut verts, &mut indices, fr - 0.26, ky, z_front, 0.045);
            }
        }
    }

    // 6b-badge. A molded brand plate proud of the bottom bezel, just under the tube.
    if cab.badge {
        let (bx, by, bhw, bhh) = (0.0, chin_top - 0.045, 0.145, 0.020);
        let (x0, x1, y0, y1) = (bx - bhw, bx + bhw, by - bhh, by + bhh);
        let zp = z_front + 0.012;
        quad(&mut verts, &mut indices, [[x0, y0, zp], [x1, y0, zp], [x1, y1, zp], [x0, y1, zp]], pl); // face
        quad(&mut verts, &mut indices, [[x0, y0, z_front], [x1, y0, z_front], [x1, y0, zp], [x0, y0, zp]], pl); // bottom
        quad(&mut verts, &mut indices, [[x0, y1, z_front], [x1, y1, z_front], [x1, y1, zp], [x0, y1, zp]], pl); // top
        quad(&mut verts, &mut indices, [[x0, y0, z_front], [x0, y1, z_front], [x0, y1, zp], [x0, y0, zp]], pl); // left
        quad(&mut verts, &mut indices, [[x1, y0, z_front], [x1, y1, z_front], [x1, y1, zp], [x1, y0, zp]], pl); // right
    }

    // 6d. Side walls: from the chamfer edge (zc) straight back to the main box.
    quad(&mut verts, &mut indices, [[cab_l, cab_t, zc], [cab_r, cab_t, zc], [cab_r, cab_t, z_rear], [cab_l, cab_t, z_rear]], pl); // top
    quad(&mut verts, &mut indices, [[cab_l, cab_b, zc], [cab_r, cab_b, zc], [cab_r, cab_b, z_rear], [cab_l, cab_b, z_rear]], pl); // bottom
    quad(&mut verts, &mut indices, [[cab_l, cab_b, zc], [cab_l, cab_t, zc], [cab_l, cab_t, z_rear], [cab_l, cab_b, z_rear]], pl); // left
    quad(&mut verts, &mut indices, [[cab_r, cab_b, zc], [cab_r, cab_t, zc], [cab_r, cab_t, z_rear], [cab_r, cab_b, z_rear]], pl); // right

    // 6e. Rear hump: taper the box in toward the tube axis and cap it (with a neck
    // hole). This is the classic bulging back of a CRT set enclosing the deflection bell.
    let (rhw, rht, rhb) = (hw_cab * 0.60, cab_t * 0.60, cab_b * 0.60);
    quad(&mut verts, &mut indices, [[cab_l, cab_t, z_rear], [cab_r, cab_t, z_rear], [rhw, rht, z_rear2], [-rhw, rht, z_rear2]], pl); // top taper
    quad(&mut verts, &mut indices, [[cab_l, cab_b, z_rear], [cab_r, cab_b, z_rear], [rhw, rhb, z_rear2], [-rhw, rhb, z_rear2]], pl); // bottom taper
    quad(&mut verts, &mut indices, [[cab_l, cab_b, z_rear], [cab_l, cab_t, z_rear], [-rhw, rht, z_rear2], [-rhw, rhb, z_rear2]], pl); // left taper
    quad(&mut verts, &mut indices, [[cab_r, cab_b, z_rear], [cab_r, cab_t, z_rear], [rhw, rht, z_rear2], [rhw, rhb, z_rear2]], pl); // right taper
    // rear face with a neck hole (ring of 4 strips around the hole)
    let nh = NECK_R * 1.6;
    quad(&mut verts, &mut indices, [[-rhw, rhb, z_rear2], [rhw, rhb, z_rear2], [nh, -nh, z_rear2], [-nh, -nh, z_rear2]], pl);
    quad(&mut verts, &mut indices, [[-rhw, rht, z_rear2], [rhw, rht, z_rear2], [nh, nh, z_rear2], [-nh, nh, z_rear2]], pl);
    quad(&mut verts, &mut indices, [[-rhw, rhb, z_rear2], [-rhw, rht, z_rear2], [-nh, nh, z_rear2], [-nh, -nh, z_rear2]], pl);
    quad(&mut verts, &mut indices, [[rhw, rhb, z_rear2], [rhw, rht, z_rear2], [nh, nh, z_rear2], [nh, -nh, z_rear2]], pl);

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
    glass: [f32; 4],  // faceplate thickness, reflection, vignette, mask triads across the face
    tone: [f32; 4],   // hdr flag, peak/white-point, tube drive (exposure), ntsc_strength
    scan: [f32; 4],   // beam math: beam_min, beam_max, beam_shape, beam_range
    env: [f32; 4],    // avg_r, avg_g, avg_b, apl  (screen-as-area-light bounce)
    look: [f32; 4],   // convergence, corner_radius, grain, ghost
    phys: [f32; 4],   // crt_gamma, reserved, glow_bounce, HV sag
    temporal: [f32; 4], // dt(sec), persist_mult, interlace, field_parity
    raster: [f32; 4], // measured response, shutter fraction, exposure group, reserved
    ptau: [f32; 4],   // per-phosphor decay tau: R, G, B (sec), w=power-law tail exponent
    geom: [f32; 4],   // raster geometry errors: pincushion, trapezoid, corner_pin, purity
    mono: [f32; 4],   // monochrome phosphor tint (rgb) + flag (w>0.5 = single-gun tube)
    cmat0: [f32; 4],  // CRT-phosphor → sRGB colour matrix, row 0 (real gamut + white pt)
    cmat1: [f32; 4],  // row 1
    cmat2: [f32; 4],  // row 2
    pwr: [f32; 4],    // power state: warmup, collapse, degauss, specular-glare enabled
    focus: [f32; 4],  // x=edge defocus (deflection spot growth), y=overscan (per side), z=roll rate, w=roll amp
    fx: [f32; 4],     // x=svm (scan-velocity crispen), y=diffusion (wide glass glow), z=subpixel mask flag, w=bfi screen mult
    camera: [f32; 4], // enabled, tan(horizontal FOV/2), aspect, reserved
    camera_ambient: [f32; 4], // linear mean camera radiance
    beam2: [f32; 4],  // x=spot profile exponent p, y=window-reflection enabled,
                      // z=ambient diffuse wash through the tinted faceplate, w=scatter redistribution
}

// ---------------------------------------------------------------------------
// CRT presets — observed geometry + phosphor of real monitor families.
// ---------------------------------------------------------------------------

// Where a set puts its loudspeakers — the single most brand-defining feature of the
// front silhouette. Bottom = a grille across the chin (Sony/RCA console TVs); Sides =
// vertical grilles flanking the tube (many Panasonic sets); None = a pro monitor / PC
// display / terminal with a thin uniform bezel and no speaker chin.
#[derive(Clone, Copy, PartialEq)]
enum Speakers {
    Bottom,
    Sides,
    None,
}

// Per-brand cabinet: the plastic finish + silhouette that wraps the bare tube. This is
// what makes a Trinitron read as a deep charcoal near-cube, a Panasonic as a silver set
// with side speakers, an RCA as a rounder warm-brown console, a PVM/PC monitor as a
// slim uniform bezel. `plastic` is the vertex material id (3 charcoal / 5 warm walnut /
// 6 silver / 7 beige); the rest are extents in screen-half-width units.
#[derive(Clone, Copy)]
struct Cabinet {
    plastic: f32,
    side_bezel: f32,
    top_bezel: f32,
    chin: f32,       // height of the space below the tube (speakers/controls)
    corner: f32,     // outer edge chamfer size (rounder = softer console look)
    rear_z: f32,     // z of the main box back (more negative = deeper set)
    speakers: Speakers,
    badge: bool,     // molded brand strip on the bottom bezel
}

// Connections are independent of the tube. Non-native connections represent an
// external decoder/tuner/converter; RGB and component share the clean path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum InputMode {
    #[default]
    Auto,
    Composite,
    Rf,
    SVideo,
    Rgb,
    Component,
}

impl InputMode {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "auto" | "default" => Ok(Self::Auto),
            "composite" => Ok(Self::Composite),
            "rf" => Ok(Self::Rf),
            "svideo" | "s-video" => Ok(Self::SVideo),
            "rgb" => Ok(Self::Rgb),
            "component" => Ok(Self::Component),
            _ => anyhow::bail!("unknown input `{value}` (use auto, composite, rf, s-video, rgb, component)"),
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Auto => Self::Composite,
            Self::Composite => Self::Rf,
            Self::Rf => Self::SVideo,
            Self::SVideo => Self::Rgb,
            Self::Rgb => Self::Component,
            Self::Component => Self::Auto,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Auto => "preset default",
            Self::Composite => "Composite",
            Self::Rf => "RF (NES-style)",
            Self::SVideo => "S-video",
            Self::Rgb => "RGB",
            Self::Component => "Component",
        }
    }

    fn signal(self, default: u8) -> u8 {
        match self {
            Self::Auto => default,
            Self::Rgb | Self::Component => 0,
            Self::SVideo => 1,
            Self::Composite => 2,
            Self::Rf => 3,
        }
    }
}

#[derive(Clone, Copy)]
struct Preset {
    name: &'static str,
    // geometry
    bulge: f32,
    curv_x: f32,
    curv_y: f32,
    cabinet: Cabinet,
    // optics
    mask_type: f32, // 0 aperture grille, 1 shadow (dot), 2 slot
    mask_strength: f32,
    halation: f32,
    // glass
    // Faceplate glass thickness, in world units, driving the refraction/dispersion trace.
    // The screen mesh is HALF_W 0.667 wide, i.e. 1.333 units for a 20" tube's ~400 mm of
    // visible picture, so ONE WORLD UNIT IS 300 mm and this number is the panel thickness
    // in units of that. Entertainment CRT faceplates run 10-15 mm at the centre (they are
    // thick because they are an implosion barrier), so a 20" panel is ~0.045 and a 25"
    // console ~0.055. The old 0.10-0.13 was 30-39 mm of glass — nearly three times a real
    // faceplate, which correspondingly tripled the parallax shift and the dispersion.
    parallax: f32,
    reflection: f32,
    vignette: f32,
    // Phosphor mask geometry, from specifications, measurements or estimates: stripe/dot pitch in mm and the tube's visible
    // picture width in mm. The shader wants the triad count across the face (screen_mm /
    // pitch_mm), which is the scale-free quantity — it is what decides whether the mask is
    // resolvable at a given zoom, and it is the number that differs between tubes. This
    // replaces a hand-set pitch in *output pixels*, which had no physical referent and got
    // the ordering wrong: it drew a broadcast PVM's grille coarser than a consumer
    // Trinitron's, where in truth the PVM has more than twice as many triads across its face.
    pitch_mm: f32,
    screen_mm: f32,
    // beam/geometry imperfections
    // RGB misregistration magnitude at the corners. The shader displaces red outward and
    // blue inward by cvec·|cvec|²·convergence·0.9, so peak red-to-blue separation in the
    // corner is 0.225 × this, in uv. On a 20" 4:3 tube (~406 mm of visible width) that is
    // 91 mm × convergence — which is the number to check against a service spec, because
    // misconvergence is always quoted in mm in a named corner zone. Sony's consumer spec
    // was ≤0.9 mm in the corner zone (0.5 mm centre); a studio PVM held ~0.3 mm; a tired,
    // never-adjusted consumer set or an arcade tube nobody has touched in a decade runs
    // 1.5-2.5 mm and looks visibly fringed. Anything past ~3 mm is not a character trait,
    // it is a set that needs a convergence strip and a service manual — which is where the
    // loose end of this range used to sit (0.055 = 5.0 mm, roughly double a bad real tube).
    convergence: f32,
    corner_radius: f32, // rounding of the active phosphor rectangle
    // raster deflection geometry errors (a pro monitor is near-perfect; consumer
    // sets bow and drift): [pincushion, trapezoid, corner_pincushion, purity]
    geom: [f32; 4],
    // guest/Megatron beam focus [beam_min, beam_max, beam_shape, beam_range] — this
    // is the tube's sharpness / TVL: a tight beam = a sharp PVM, a wide one = fuzzy.
    beam: [f32; 4],
    // Spot *profile* exponent for the generalized gaussian exp(-|d/w|^spot). The spot is
    // the gun's imaged cathode crossover convolved with the optics' aberration blur, so a
    // well-focused tube reads as a plateau with a steep falloff to the dark gap while a
    // soft/defocused one collapses to a plain bell (spot = 2). Higher = flatter-topped.
    spot: f32,
    // Faceplate light transmission. Entertainment CRT panels are tinted glass to lift
    // contrast: "clear" ≥75%, "gray" 60–75%, "tinted" ≤60%, and high-contrast tubes run
    // ~40%. It matters twice over: emitted phosphor light crosses the panel ONCE, while
    // room light reflected off the phosphor crosses it TWICE — so the ambient wash that
    // greys out a CRT's blacks in a lit room goes as T², which is why tinting works.
    glass_t: f32,
    // Phosphor persistence. A colour tube has three different phosphors, so this is a
    // multiplier on representative per-primary P22 decay constants (1.0 = baseline). A
    // single-gun mono tube has exactly ONE phosphor, so there is nothing to scale — this
    // is its absolute decay time in seconds and all three stored channels use it.
    persist: f32,
    // Default connection and legacy chassis calibration; input overrides only the signal.
    // 0=RGB/component (clean), 1=S-video (Y/C split), 2=composite.
    signal: u8,
    input: InputMode,
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
// Phosphor Technology Ltd CRT material specifications, grades QKL63/N-C1,
// GL29A/N-C1 and GL47/N-C2 (manufacturer CIE 1931 coordinates).
const PHOS_P22: [[f32; 2]; 3] = [[0.647, 0.343], [0.310, 0.594], [0.148, 0.062]];
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
    // Sony KV-20TS20 (1989): a deep charcoal near-cube; near-flat cylindrical face in a
    // fairly thick matte bezel, a distinctly tall chin carrying the speaker grille +
    // control door, and the SONY badge molded into the bottom bezel. (~photo refs)
    cabinet: Cabinet {
        plastic: 3.0,
        side_bezel: 0.150,
        top_bezel: 0.130,
        chin: 0.450,
        corner: 0.05,
        rear_z: -1.46,
        speakers: Speakers::Bottom,
        badge: true,
    },
    mask_type: 0.0,
    mask_strength: 0.90,
    halation: 0.35,
    parallax: 0.045,
    reflection: 0.50,
    vignette: 0.22,
    // 20" consumer Trinitron TV, aperture grille. Estimated pitch — crtdatabase
    // lists the KV-20TS20's tube (A51JUH50X) as an aperture grille but publishes no pitch, and
    // Sony did not spec it on consumer sets the way they did on monitors, because a TV runs one
    // scan rate and only has to resolve ~330 TVL. It is bounded from both sides though. The
    // repairfaq slot-mask measurements (13" 0.60, 19" 0.75, 25" 0.90 mm) are almost exactly
    // linear at 0.025 mm/inch, which puts a 20" consumer shadow/slot tube at ~0.78 mm; a
    // Trinitron grille of the same size runs finer than that, which was half the point of the
    // design. 0.66 mm over a 400 mm picture (a 20" 4:3 face is 406 mm wide) → 606 triads.
    pitch_mm: 0.66,
    screen_mm: 400.0,
    // a pro Trinitron/PVM is tightly converged with squarer corners.
    convergence: 0.010,
    corner_radius: 0.05,
    // studio-grade geometry: nearly straight, minimal impurity.
    geom: [0.008, 0.0, 0.010, 0.03],
    beam: [0.34, 0.74, 0.75, 1.0],
    spot: 3.0,      // well-focused consumer Sony: flat-topped scanline
    glass_t: 0.50,  // tinted consumer panel
    persist: 1.0,
    phos: 0,
    white_xy: [0.2831, 0.2971],
    signal: 1,
    input: InputMode::Auto,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// Panasonic-style consumer set: shadow (dot) mask, spherical bulge.
const PANASONIC: Preset = Preset {
    name: "panasonic",
    bulge: 0.13,
    curv_x: 0.50,
    curv_y: 0.50,
    // Panasonic consumer set: cool silver-grey plastic with tall vertical speaker
    // grilles flanking the tube (wide side bezels), a slim controls-only chin, and
    // softer edges than the Sony.
    cabinet: Cabinet {
        plastic: 6.0,
        side_bezel: 0.240,
        top_bezel: 0.110,
        chin: 0.240,
        corner: 0.045,
        rear_z: -1.40,
        speakers: Speakers::Sides,
        badge: true,
    },
    mask_type: 1.0,
    mask_strength: 0.85,
    halation: 0.42,
    parallax: 0.048,
    reflection: 0.45,
    vignette: 0.38,
    // Consumer dot-mask model borrowing the 0.75 mm horizontal pitch measured by
    // repairfaq on a 19" Samsung SLOT mask; not a Panasonic dot-pitch measurement.
    // The assumed 386 mm picture width gives 515 horizontal periods.
    pitch_mm: 0.75,
    screen_mm: 386.0,
    // consumer set: looser convergence, rounder tube corners.
    convergence: 0.024, // ~2.2 mm R–B in the corner: a consumer set well past its last alignment
    corner_radius: 0.12,
    // consumer geometry: visible pincushion + a little keystone and purity drift.
    geom: [0.055, 0.022, 0.060, 0.10],
    beam: [0.36, 0.78, 0.75, 1.0],
    spot: 2.4,      // ordinary consumer focus: nearly a plain bell
    glass_t: 0.52,
    persist: 1.0,
    phos: 0,
    white_xy: [0.2831, 0.2971],
    signal: 2,
    input: InputMode::Auto,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// Slot-mask consumer set (e.g. many 90s TVs).
const SLOTMASK: Preset = Preset {
    name: "slotmask",
    bulge: 0.10,
    curv_x: 0.42,
    curv_y: 0.38,
    // Generic 90s consumer set: charcoal, bottom speaker grille, mid proportions.
    cabinet: Cabinet {
        plastic: 3.0,
        side_bezel: 0.145,
        top_bezel: 0.125,
        chin: 0.400,
        corner: 0.05,
        rear_z: -1.44,
        speakers: Speakers::Bottom,
        badge: false,
    },
    mask_type: 2.0,
    mask_strength: 0.85,
    halation: 0.40,
    parallax: 0.052,
    reflection: 0.45,
    vignette: 0.34,
    // Large consumer slot mask: 0.90 mm, repairfaq's measurement of a 25" RCA, over that
    // tube's ~490 mm of visible picture → 544 triads.
    pitch_mm: 0.90,
    screen_mm: 490.0,
    convergence: 0.020, // ~1.8 mm: a slot-mask consumer set drifting, still short of a fault
    corner_radius: 0.10,
    geom: [0.040, -0.015, 0.040, 0.08],
    beam: [0.35, 0.76, 0.75, 1.0],
    spot: 2.4,
    glass_t: 0.52,
    persist: 1.0,
    phos: 0,
    white_xy: [0.2831, 0.2971],
    signal: 2,
    input: InputMode::Auto,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// RCA ColorTrak-style console TV: shadow mask, soft/fuzzy, warm, very curved. Old
// consumer sets were low-TVL (~300s) with a wide, unfocused beam and lots of bloom.
const RCA: Preset = Preset {
    name: "rca",
    bulge: 0.13,
    curv_x: 0.52,
    curv_y: 0.48, // deeply curved old spherical tube
    // RCA ColorTrak console: a bigger, rounder, warm walnut-brown box — a tall chin
    // with the speaker grille, softly chamfered edges, and the deepest cabinet here.
    cabinet: Cabinet {
        plastic: 5.0,
        side_bezel: 0.170,
        top_bezel: 0.150,
        chin: 0.500,
        corner: 0.090,
        rear_z: -1.52,
        speakers: Speakers::Bottom,
        badge: true,
    },
    mask_type: 1.0, // shadow-mask dot triads
    mask_strength: 0.62, // soft, low-contrast mask
    halation: 0.62, // glowy, blooms warm
    parallax: 0.055,
    reflection: 0.55,
    vignette: 0.40,
    // 25" console — repairfaq measured exactly this class of tube ("25 inch RCA - .9 mm.")
    // at 0.90 mm; ~490 mm of visible picture → 544 triads.
    pitch_mm: 0.90,
    screen_mm: 490.0,
    convergence: 0.026, // ~2.4 mm: the loosest a real, working consumer set gets (was 5.0 mm — a fault)
    corner_radius: 0.13,
    geom: [0.060, 0.025, 0.070, 0.14], // consumer bow + purity drift
    beam: [0.48, 0.98, 0.65, 1.0], // WIDE, unfocused beam = fuzzy / low TVL
    spot: 2.0,      // soft gun: aberration blur dominates → pure gaussian bell
    glass_t: 0.58,  // older, lighter-tinted console panel
    persist: 1.0,
    phos: 0,
    white_xy: [0.305, 0.322],
    signal: 2,
    input: InputMode::Auto,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// Sony PVM/BVM broadcast monitor: aperture grille, razor-sharp (600–800 TVL), fine
// stripe pitch, near-flat cylindrical face, studio-grade geometry, AR-coated glass.
const PVM: Preset = Preset {
    name: "pvm",
    bulge: 0.07,
    curv_x: 0.26,
    curv_y: 0.09, // near-flat, cylindrical
    // Sony PVM/BVM: a compact charcoal broadcast monitor — thin uniform bezel all
    // around, no consumer speaker chin.
    cabinet: Cabinet {
        plastic: 3.0,
        side_bezel: 0.095,
        top_bezel: 0.095,
        chin: 0.130,
        corner: 0.035,
        rear_z: -1.40,
        speakers: Speakers::None,
        badge: false,
    },
    mask_type: 0.0, // aperture grille
    mask_strength: 0.95,
    halation: 0.22, // low bloom
    parallax: 0.045,
    reflection: 0.34,
    vignette: 0.14,
    // Sony PVM-20L5 brochure: 0.31 mm grille, 388.4 mm effective picture width.
    pitch_mm: 0.31,
    screen_mm: 388.4,
    convergence: 0.008, // tight
    corner_radius: 0.04, // squarish pro face
    geom: [0.006, 0.0, 0.008, 0.02], // near-perfect
    beam: [0.26, 0.56, 0.85, 1.0], // TIGHT beam = sharp / high TVL
    spot: 4.0,      // razor focus: a real plateau with steep walls
    glass_t: 0.44,  // high-contrast tinted + AR-coated broadcast panel
    persist: 1.0,
    phos: 0,
    white_xy: [0.3127, 0.329],
    signal: 0,
    input: InputMode::Auto,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// 15 kHz arcade monitor (Wells Gardner / Hantarex chassis on a consumer-grade tube):
// shadow-mask triads, coarse pitch, big visible 240p scanlines, often misconverged.
const ARCADE: Preset = Preset {
    name: "arcade",
    bulge: 0.12,
    curv_x: 0.42,
    curv_y: 0.40,
    // Bare 15 kHz tube in a black metal chassis: charcoal, thin bezel, no speakers.
    cabinet: Cabinet {
        plastic: 3.0,
        side_bezel: 0.130,
        top_bezel: 0.120,
        chin: 0.170,
        corner: 0.04,
        rear_z: -1.44,
        speakers: Speakers::None,
        badge: false,
    },
    mask_type: 1.0, // shadow-mask triads
    mask_strength: 0.80,
    halation: 0.44,
    parallax: 0.045,
    reflection: 0.55, // bare (uncoated) glass
    vignette: 0.36,
    // 19" Wells Gardner K7000 class: the 25" K7000 measures 0.82 mm, which scales to
    // ~0.63 mm on the 19" tube → 613 triads. Note that lands finer than the 0.75 mm
    // repairfaq measured on a 19" TV: an arcade tube had to hold sharp pixel art, and the
    // K7000 measurement says it did so with a finer mask than a television of the same size.
    pitch_mm: 0.63,
    screen_mm: 386.0,
    convergence: 0.025, // ~2.3 mm: an arcade tube nobody has converged in ten years (was 4.1 mm)
    corner_radius: 0.11,
    geom: [0.050, 0.020, 0.050, 0.10],
    beam: [0.40, 0.90, 0.70, 1.0], // wide, strong scanline gaps
    spot: 2.6,
    glass_t: 0.62,  // bare consumer-grade tube, only lightly tinted
    persist: 1.0,
    phos: 0,
    white_xy: [0.2831, 0.2971],
    signal: 0,
    input: InputMode::Auto,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// NEC MultiSync-style VGA PC monitor: fine (~0.28 mm) shadow mask, flatter late-CRT
// face, high line count (subtle scanlines), good geometry, cool/bright white.
const VGA: Preset = Preset {
    name: "vga",
    bulge: 0.05,
    curv_x: 0.22,
    curv_y: 0.20, // flatter
    // NEC MultiSync PC monitor: cool silver-grey, slim uniform bezel, no speakers.
    cabinet: Cabinet {
        plastic: 6.0,
        side_bezel: 0.075,
        top_bezel: 0.075,
        chin: 0.110,
        corner: 0.04,
        rear_z: -1.30,
        speakers: Speakers::None,
        badge: false,
    },
    mask_type: 1.0, // fine shadow mask
    mask_strength: 0.85,
    halation: 0.28,
    parallax: 0.033,
    reflection: 0.42,
    vignette: 0.18,
    // 15" NEC MultiSync: 0.28 mm shadow mask — the coarse end of repairfaq's "typical high
    // resolution CRTs ... .25 to .28 mm" — over a 280 mm picture (13.8" viewable) → 1000 triads.
    pitch_mm: 0.28,
    screen_mm: 280.0,
    convergence: 0.014, // ~1.3 mm: a consumer-grade VGA monitor, looser than a pro tube
    corner_radius: 0.07,
    geom: [0.018, 0.005, 0.020, 0.04], // good geometry
    beam: [0.28, 0.60, 0.85, 1.0], // sharp
    spot: 3.6,
    glass_t: 0.60,
    persist: 1.0,
    phos: 2,
    white_xy: [0.2831, 0.2971],
    signal: 0,
    input: InputMode::Auto,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// NEC Diamondtron / FE-series "totally flat" aperture-grille PC monitor: very fine
// (~0.24 mm) stripe, dead-flat face, superbright, minimal scanlines, cool white.
const DIAMONDTRON: Preset = Preset {
    name: "diamondtron",
    bulge: 0.02,
    curv_x: 0.05,
    curv_y: 0.05, // dead flat
    // NEC Diamondtron FE flat PC monitor: silver-grey, razor-thin uniform bezel.
    cabinet: Cabinet {
        plastic: 6.0,
        side_bezel: 0.060,
        top_bezel: 0.060,
        chin: 0.090,
        corner: 0.03,
        rear_z: -1.25,
        speakers: Speakers::None,
        badge: false,
    },
    mask_type: 0.0, // aperture grille (has damper wires)
    mask_strength: 0.92,
    halation: 0.20,
    parallax: 0.036,
    reflection: 0.30, // AR coated
    vignette: 0.12,
    // Mitsubishi Diamond Pro 930SB service manual: 0.24 mm grille and a
    // 366 mm full-scan display width (356 mm factory underscan setting).
    pitch_mm: 0.24,
    screen_mm: 366.0,
    convergence: 0.010,
    corner_radius: 0.03,
    geom: [0.010, 0.0, 0.010, 0.02], // flat, well-corrected
    beam: [0.26, 0.55, 0.88, 1.0], // very sharp / bright
    spot: 4.2,      // the flattest-topped, best-focused gun here
    glass_t: 0.50,  // AR-coated, tinted for contrast
    persist: 1.0,
    phos: 2,
    white_xy: [0.2831, 0.2971],
    signal: 0,
    input: InputMode::Auto,
    mono: [0.0, 0.0, 0.0, 0.0],
};

// Monochrome green terminal model: P1-like tint with long persistence, a single gun, no
// colour mask, and long persistence — the lingering green afterglow of a VT-style
// text terminal / IBM 5151. Crisp text beam, warm glow, gently curved small tube.
const GREEN: Preset = Preset {
    name: "green",
    bulge: 0.09,
    curv_x: 0.34,
    curv_y: 0.30,
    // IBM 5151-style terminal: a beige/cream molded case, slim bezel, no speakers.
    cabinet: Cabinet {
        plastic: 7.0,
        side_bezel: 0.120,
        top_bezel: 0.110,
        chin: 0.160,
        corner: 0.06,
        rear_z: -1.30,
        speakers: Speakers::None,
        badge: false,
    },
    mask_type: 0.0,     // unused (mono skips the RGB triad mask)
    mask_strength: 0.0, // no colour mask on a single-phosphor tube
    halation: 0.55, // strong phosphor glow/bleed
    parallax: 0.030,
    reflection: 0.42,
    vignette: 0.28,
    // single-phosphor tube: no mask at all (mask_strength 0), value unused.
    pitch_mm: 1.00,
    screen_mm: 240.0,
    convergence: 0.0, // one gun → no RGB misconvergence
    corner_radius: 0.08,
    geom: [0.028, 0.0, 0.030, 0.0], // no purity error on a mono tube
    beam: [0.30, 0.62, 0.85, 1.0],  // fairly tight for readable text
    spot: 3.2,
    glass_t: 0.40,                  // terminals wore a dark contrast filter over the face
    persist: 0.021715,              // representative 50 ms to 10% / ln(10)
    phos: 3,
    white_xy: [0.3127, 0.329],
    signal: 0,
    input: InputMode::Auto,
    mono: [0.10, 1.0, 0.14, 1.0], // P1 green (CIE ~0.218,0.712) → sRGB, normalized
};

// Monochrome amber terminal (P3 amber phosphor): the easier-on-the-eyes amber of a
// Wyse/late-80s terminal. Same tube, warmer phosphor, a touch less persistence.
const AMBER: Preset = Preset {
    name: "amber",
    bulge: 0.09,
    curv_x: 0.34,
    curv_y: 0.30,
    // Same beige terminal case as the green phosphor sibling.
    cabinet: Cabinet {
        plastic: 7.0,
        side_bezel: 0.120,
        top_bezel: 0.110,
        chin: 0.160,
        corner: 0.06,
        rear_z: -1.30,
        speakers: Speakers::None,
        badge: false,
    },
    mask_type: 0.0,
    mask_strength: 0.0,
    halation: 0.52,
    parallax: 0.030,
    reflection: 0.42,
    vignette: 0.28,
    // no mask on a single-gun tube (unused).
    pitch_mm: 1.00,
    screen_mm: 240.0,
    convergence: 0.0,
    corner_radius: 0.08,
    geom: [0.028, 0.0, 0.030, 0.0],
    beam: [0.30, 0.62, 0.85, 1.0],
    spot: 3.2,
    glass_t: 0.40,
    persist: 0.005646,             // representative 13 ms to 10% / ln(10)
    phos: 3,
    white_xy: [0.3127, 0.329],
    signal: 0,
    input: InputMode::Auto,
    mono: [1.0, 0.44, 0.06, 1.0], // P3 amber (CIE ~0.523,0.469) → sRGB, normalized
};

fn preset_by_name(name: &str) -> anyhow::Result<Preset> {
    ALL_PRESETS.iter().find(|p| p.name.eq_ignore_ascii_case(name)).copied()
        .ok_or_else(|| anyhow::anyhow!("unknown preset `{name}` (have: {})",
            ALL_PRESETS.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")))
}

// Cycle order for the Tab key + digit selection (1..9, 0).
const ALL_PRESETS: [Preset; 10] =
    [TRINITRON, PANASONIC, SLOTMASK, RCA, PVM, ARCADE, VGA, DIAMONDTRON, GREEN, AMBER];

// ---------------------------------------------------------------------------
// Shared GPU resources (surface-independent, so the live window and the
// headless `--shot` path build them the same way)
// ---------------------------------------------------------------------------

struct Resources {
    camera_texture: wgpu::Texture,
    camera_view: wgpu::TextureView,
    camera: [f32; 4],
    camera_ambient: [f32; 4],
    hdr_output: u8,
    physical_phosphor: bool,
    shutter_fraction: f32,
    exposure_group: u32,
    phosphor_state: [wgpu::Buffer; 3],
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

// Independent positive reservoirs retain all decay components and their energy.
fn make_phosphor_state(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32) -> [wgpu::Buffer; 3] {
    // Split the reservoirs across bindings so native 4K capture does not exceed
    // Vulkan's per-buffer range. No source pixels or decay components are lost.
    let size = w as u64 * h as u64 * 11 * 16;
    assert!(size <= device.limits().max_storage_buffer_binding_size as u64,
        "source raster exceeds this GPU's phosphor-history capacity");
    let buffers = std::array::from_fn(|_| device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("phosphor reservoir bank"), size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }));
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clear phosphor state") });
    for b in &buffers {
        enc.clear_buffer(b, 0, None);
    }
    queue.submit(Some(enc.finish()));
    buffers
}

// An HDR phosphor plane (render target + sampleable) at the source's resolution.
fn make_phosphor(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("phosphor"),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: PHOSPHOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
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
        // Average emitted light/current, not gamma-encoded voltage.
        ar += (r as f64 / 255.0).powf(2.4);
        ag += (g as f64 / 255.0).powf(2.4);
        ab += (b as f64 / 255.0).powf(2.4);
        n += 1.0;
        i += step;
    }
    if n == 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let r = (ar / n) as f32;
    let g = (ag / n) as f32;
    let b = (ab / n) as f32;
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
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.camera_view) },
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
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.camera_view) },
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
                    wgpu::BindGroupEntry { binding: 4, resource: self.phosphor_state[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.phosphor_state[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.phosphor_state[2].as_entire_binding() },
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
                self.phosphor_state = make_phosphor_state(device, queue, width, height);
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
        let phosphor_state = make_phosphor_state(device, queue, tw, th);
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
                    binding: 7, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
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
                    binding: 7, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });

        // --- pipeline ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(concat!(include_str!("phosphor.wgsl"), "\n", include_str!("shader.wgsl")).into()),
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

        let camera_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("webcam environment"),
            size: wgpu::Extent3d { width: webcam::WIDTH, height: webcam::HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let camera_view = camera_texture.create_view(&Default::default());
        // Initial bind groups (rebuilt on any source resize via rebuild_binds).
        let mk_screen = |pv: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("screen_bind"),
                layout: &bind_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&camera_view) },
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
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&camera_view) },
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&source_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(pv) },
                    wgpu::BindGroupEntry { binding: 4, resource: phosphor_state[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: phosphor_state[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: phosphor_state[2].as_entire_binding() },
                ],
            })
        };
        let screen_bind = [mk_screen(&phosphor_view[0]), mk_screen(&phosphor_view[1])];
        let accum_bind = [mk_accum(&phosphor_view[0]), mk_accum(&phosphor_view[1])];

    // --- geometry buffers ---
    let (verts, indices) = build_mesh(preset.bulge, preset.curv_x, preset.curv_y, preset.cabinet);
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
        camera_texture, camera_view, camera: [0.0; 4], camera_ambient: [0.0; 4],
        hdr_output: 1,
        physical_phosphor: std::env::var("CRTULUM_PHOSPHOR").as_deref() != Ok("legacy"),
        shutter_fraction: 1.0,
        exposure_group: 0,
        phosphor_state,
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

// NTSC fields advance independently of presentation (including non-integer exports).
const FIELD_HZ: f64 = 60_000.0 / 1001.0;
// Full-wave 60 Hz supply ripple sampled by the field clock. Keep the signed
// difference: it determines the direction of the drifting interference pattern.
const HUM_BEAT_HZ: f64 = 2.0 * 60.0 - 2.0 * FIELD_HZ;
#[derive(Default)]
struct FieldClock { next: u64 }
impl FieldClock {
    fn through(&mut self, seconds: f64) -> std::ops::Range<u64> {
        let end = (seconds.max(0.0) * FIELD_HZ + 1e-7).floor() as u64 + 1;
        let start = self.next;
        self.next = self.next.max(end);
        start..self.next
    }
}

// Uniform noise in ±amplitude/2 has RMS amplitude/sqrt(12). SNR is relative
// to unit video voltage, before clipping and gun transfer; targets are model choices.
fn noise_peak_to_peak(snr_db: f32) -> f32 {
    12.0_f32.sqrt() * 10.0_f32.powf(-snr_db / 20.0)
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
    subpixel: bool,
    bfi_mul: f32,
    glare: bool,
    window_reflection: bool,
) {
    let (view_proj, eye) = orbit.view_proj(aspect);
    let cmat = preset_color_matrix(preset);
    let uniforms = Uniforms {
        camera: res.camera,
        camera_ambient: res.camera_ambient,
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
            // Reserved. This was the per-tube scanline mix; scanline depth is now derived
            // in the shader from the spot geometry and the display's ability to resolve the
            // line pitch, which is what actually determines it (see scan_reconstruct).
            0.0,
            preset.halation,
        ],
        glass: [
            preset.parallax,
            preset.reflection,
            preset.vignette,
            // Mask triads across the visible picture width. Scale-free, so the shader can
            // key the mask off the faceplate itself and let the pixel footprint decide
            // whether it resolves — a real grille magnifies when you lean in and integrates
            // away when you step back, and neither of those happens to a pattern locked to
            // output pixels.
            preset.screen_mm / preset.pitch_mm.max(0.01),
        ],
        // HDR path: on a scRGB swapchain, emit linear light with highlights >1.0
        // (peak/drive push the beam above white). On SDR, tonemap to `peak` white
        // point. tone.w = input signal path (0=clean, 1=S-video, 2=composite, 3=RF).
        // tone.y carries the exposure trim on the HDR path (scales SDR-white → panel
        // reference white) and the tonemap white-point × exposure on the SDR path, so
        // the [ and ] keys tune brightness identically in both.
        //
        // `beam_drive` (tone.z) is now the tube's overall drive — a straight exposure on the
        // reconstructed beam — because the reconstruction is energy-normalised and no longer
        // gains up bright rows by widening them. It went up (1.7→2.0 SDR) purely to put the
        // picture back where it was: the old unnormalised beam sum was worth ~1.8× on a mid
        // grey and carried an extra ~1.26 gamma with it, and both of those had to come out.
        // Because that hidden gain scaled with beam WIDTH, it also flattered the fuzzy tubes
        // most; with it gone the ten tubes sit much closer together in mean brightness, as
        // they should — a wide beam spreads the same light, it does not make more.
        //
        // It has now come back DOWN (2.0 → 1.30 SDR, 2.2 → 1.43 HDR) for the mirror-image
        // reason: the mask is normalised to unit mean in the shader, so the ~45% the old
        // hand-fitted mask compensation was quietly losing is no longer lost, and the
        // additive highlight bloom that was manufacturing light is gone. Both were being
        // paid for out of the drive. Measured on the reference shot, 1.30 puts mean screen
        // luminance at 0.245 against the pre-audit 0.246 — the same picture brightness,
        // arrived at without the two fudges.
        tone: if hdr {
            [res.hdr_output as f32, exposure, 1.43, preset.input.signal(preset.signal) as f32]
        } else {
            [0.0, 1.08 * exposure, 1.30, preset.input.signal(preset.signal) as f32] // ACES exposure (was Reinhard white pt)
        },
        // Guest/Megatron beam math (per-tube focus/TVL): per-channel beam half-width
        // runs from beam_min (dark → tight) to beam_max (bright → wide); beam_shape
        // curves the growth; beam_range = ± source rows summed. Tight = sharp PVM,
        // wide = fuzzy RCA.
        scan: preset.beam,
        // The screen radiates its average color/brightness onto the tube body.
        env: res.avg,
        // Signal noise is injected in voltage space before gun transfer/decay.
        look: [
            preset.convergence,
            preset.corner_radius,
            if preset.input.signal(preset.signal) == 0 && (preset.mono[3] > 0.5 || preset.phos >= 2) {
                noise_peak_to_peak(64.0) // representative clean TTL/VGA path
            } else {
                match preset.input.signal(preset.signal) {
                    3 => noise_peak_to_peak(30.0), // illustrative RF tuner noise
                    2 => noise_peak_to_peak(36.0), // representative noisy consumer feed
                    1 => noise_peak_to_peak(42.0), // representative S-video baseband
                    _ => noise_peak_to_peak(52.0), // representative RGB/component feed
                }
            },
            0.012,
        ],
        // Gun voltage → current exponent, native white handled once by cmat.
        // phys.w approximates supply regulation from average picture level.
        phys: [
            2.4,
            0.0,
            0.42,
            if preset.mono[3] > 0.5 || preset.phos >= 2 {
                0.05 // terminal / PC monitor: modest, steady raster, decent regulation
            } else {
                match preset.signal {
                    2 => 0.13, // composite consumer set — cheapest chassis, sags most
                    1 => 0.09, // better consumer / prosumer
                    _ => 0.04, // RGB/component studio monitor: tightly regulated
                }
            },
        ],
        // Phosphor persistence + interlace: dt drives per-frame decay; temporal.y is
        // the per-tube persistence multiplier; temporal.z = interlace amount, .w =
        // field parity (alternate fields excite alternate lines → 480i twitter).
        // A mono tube's `persist` is already an absolute tau, so it must not be applied
        // again as a multiplier on top of the flat ptau built below.
        temporal: [
            dt.max(0.0),
            if preset.mono[3] > 0.5 { 1.0 } else { preset.persist },
            interlace,
            field,
        ],
        // Effective field-scale decay constants preserve the visible color afterglow.
        // Red carries motion history across fields while green and blue fade first.
        // Mono presets use their own exponential time-to-10% conversion.
        raster: [if res.physical_phosphor { 1.0 } else { 0.0 }, res.shutter_fraction,
            if res.exposure_group == 0 { time } else { res.exposure_group as f32 }, 0.0],
        ptau: if preset.mono[3] > 0.5 {
            // tau = T10 / ln(10) requires exponential decay. Adding the color
            // tail here would lengthen T10 and make it depend on brightness and dt.
            [preset.persist, preset.persist, preset.persist, 0.0]
        } else {
            [0.018, 0.006, 0.004, 0.7]
        },
        // Raster deflection geometry errors, per tube (see Preset.geom).
        geom: [preset.geom[0], preset.geom[1], preset.geom[2],
               preset.geom[3] * (1.0 - pwr[3])],
        // Monochrome phosphor tint + flag (single-gun green/amber terminals).
        mono: preset.mono,
        // Real phosphor-gamut + white-point colour matrix (computed on the CPU).
        cmat0: cmat[0],
        cmat1: cmat[1],
        cmat2: cmat[2],
        // Power-on warmup / power-off collapse / degauss animation state.
        pwr: [pwr[0], pwr[1], pwr[2], if glare { 1.0 } else { 0.0 }],
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
            // All presets use FIELD_HZ, including monochrome terminals. Derive
            // the mains beat from that clock rather than inventing a slower
            // terminal rate without actually changing its scan timing.
            let roll_rate = HUM_BEAT_HZ as f32;
            // Amplitude. The hum bar is mains ripple on the video/HV rails getting past the
            // supply's filtering, so its size is set by how much ripple survives — on a set
            // in good health, not much. Measured, a healthy CRT's hum bar is a percent or
            // two of peak white: you can find it on a flat grey field if you look, and it
            // disappears into normal picture content. Only a set with dried-out filter caps
            // shows the textbook 5-10% band, and that is a fault, not the look of a working
            // tube. 0.05 was drawing the fault. 0.02 draws a healthy set that is still
            // visibly analog.
            [defocus, overscan, roll_rate, 0.02]
        },
        // fx: SVM + diffusion (both derived from the tube's character), subpixel-mask
        // toggle, and the per-frame BFI screen multiplier.
        fx: {
            // Scan-velocity modulation was a CONSUMER-set feature tied to the analog
            // signal chain: strongest on composite RF sets, milder on the S-video-fed
            // Trinitron, and absent on RGB/component broadcast PVMs, PC monitors, and
            // single-gun mono terminals (which had no VM circuit). Derive from signal.
            let svm = if preset.mono[3] > 0.5 {
                0.0
            } else {
                match preset.signal {
                    2 => 0.55, // composite consumer TV — pronounced VM haloing
                    1 => 0.35, // S-video Trinitron — milder
                    _ => 0.0,  // RGB/component: none
                }
            };
            // Diffusion (wide scatter haze in the faceplate glass) tracks the same glass
            // that drives halation — a thick, fuzzy consumer tube scatters more, a sharp
            // AR-coated PVM/PC monitor less. Scale off halation so it stays per-tube.
            let diffusion = preset.halation * 0.55;
            let subpix = if subpixel { 1.0 } else { 0.0 };
            [svm, diffusion, subpix, bfi_mul]
        },
        beam2: {
            // The spot profile is exp(-|d/w|^p), p = the tube's low-current focus quality.
            // The area normaliser 1/(2·Γ(1+1/p)) used to be computed here, once per frame;
            // it moved into the shader because p is no longer constant across the picture —
            // the profile relaxes toward a gaussian as the spot blooms (see
            // scan_reconstruct). Energy normalisation is the point either way: light output
            // is LINEAR in beam current (the ~2.4 CRT gamma comes from the gun's grid
            // transfer curve, not the phosphor), so widening the spot on a bright row has to
            // redistribute that row's light, never manufacture more of it.
            let p = preset.spot.max(1.2);
            // Ambient wash: room light reflected off the phosphor/mask back out at the
            // viewer. It crosses the tinted faceplate TWICE (hence T²) and bounces off a
            // screen whose effective albedo is low — the phosphor itself is a pale powder,
            // but the black matrix between the stripes swallows most of what lands on it.
            // Unlike the specular room reflection this is diffuse, so it lifts the black
            // floor evenly instead of mirroring, and it is the reason a CRT in a lit room
            // never actually reaches black. Note it scales as T here, not T²: the emitted
            // picture already loses its own single pass through the tint, and the exposure
            // downstream normalises that away, so what survives as a tube-to-tube
            // difference is the ratio T²/T. That ratio is the whole argument for tinting —
            // halving transmission costs half the brightness but quarters the wash, so you
            // buy a factor of two in contrast and win it back with more beam current.
            // Scaled for a dimly-lit room rather than a bright one: this lands the tubes
            // at ~60–90:1 in-room contrast, which is the right order for a CRT measured
            // with the lights on (a dark-room measurement gives several hundred to one,
            // and that is the number datasheets quoted). Turn it up and you get the
            // daylight-on-the-screen look, at the cost of every saturated colour.
            let t = preset.glass_t.clamp(0.2, 0.95);
            let wash = t * 0.09;
            // Scattering redistributes light rather than adding it, so halation/diffusion
            // take their share out of the direct term. Kept partial (not the full 1.0) —
            // some of what scatters forward still reaches the eye inside the same pixel.
            // .y independently gates the recognisable mullioned-window reflection.
            [p, if window_reflection { 1.0 } else { 0.0 }, wash, 0.55]
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
    profile: Option<benchmark::LiveProfile>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    res: Resources,
    depth_view: wgpu::TextureView,
    orbit: Orbit,
    start: std::time::Instant,
    field_clock: FieldClock,
    degaussed: bool,
    dragging: bool,
    last_cursor: (f64, f64),
    window: Arc<Window>,
    capture: Option<capture::SharedFrame>,
    webcam: Option<webcam::Webcam>,
    camera_status: CameraStatus,
    last_seq: u64,
    /// Live play: a libretro core running a game, driven by the clock and a pad.
    player: Option<play::Player>,
    preset: Preset,
    hdr: bool, // true = linear sRGB or BT.2020 HDR swapchain, false = SDR tone map
    power: PowerState,
    degauss_start: Option<std::time::Instant>,
    frame: u64,       // field counter for 480i interlace
    interlace: bool,  // 480i (alternating fields) vs 240p progressive
    exposure: f32,    // live HDR/SDR exposure trim ([ and ] keys) for tuning on the panel
    subpixel: bool,   // subpixel-accurate (Megatron) mask vs the resolution-independent one (M key)
    bfi: bool,        // black-frame insertion for CRT-impulse motion clarity (B key; needs a high-refresh panel)
    refresh_hz: f32,  // detected panel refresh (for the BFI safety gate / message)
    glare: bool,      // tight ceiling-light specular on the faceplate (L key)
    window_reflection: bool, // mullioned daylight reflection in the environment (R key)
}

// Read the color space actually selected by the vendored Vulkan swapchain code.
// Other backends use the usual linear-sRGB float surface contract.
fn surface_output_mode(surface: &mut wgpu::Surface<'_>) -> u8 {
    // Read the actual configured pair, including after surface reconfiguration.
    unsafe {
        surface.as_hal::<wgpu::hal::api::Vulkan, _, _>(|raw| {
            match raw.and_then(|s| s.configured_color_space()) {
                Some(ash::vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT) => 1,
                Some(ash::vk::ColorSpaceKHR::BT2020_LINEAR_EXT) => 2,
                Some(ash::vk::ColorSpaceKHR::HDR10_ST2084_EXT) => 3,
                _ => 0,
            }
        }).unwrap_or(0)
    }
}

fn output_mode_label(mode: u8) -> &'static str {
    match mode {
        1 => "HDR scRGB linear", 2 => "HDR BT.2020 linear", 3 => "HDR10 BT.2020 PQ",
        _ => "SDR sRGB/BT.709 (compositor exposes no supported HDR surface)",
    }
}

// Best-effort panel refresh detection. On Wayland current_monitor() is often None
// at startup (the surface hasn't entered an output yet via wl_surface.enter), so
// this also falls back to the fastest available monitor rather than assuming 60 Hz.
fn detect_refresh_hz(window: &Window) -> f32 {
    let from_mhz = |mhz: u32| mhz as f32 / 1000.0;
    window
        .current_monitor()
        .and_then(|m| m.refresh_rate_millihertz())
        .map(from_mhz)
        .or_else(|| {
            window
                .available_monitors()
                .filter_map(|m| m.refresh_rate_millihertz())
                .max()
                .map(from_mhz)
        })
        .unwrap_or(60.0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CameraStatus {
    Off,
    Opening,
    Live,
    Unavailable,
}

fn window_title(preset: &Preset, camera: CameraStatus) -> String {
    let connection = match preset.input {
        InputMode::Auto => format!("preset default ({})", match preset.signal {
            1 => "S-video", 2 => "Composite", _ => "RGB/component",
        }),
        input => input.label().to_string(),
    };
    let camera = match camera {
        CameraStatus::Off => "camera off",
        CameraStatus::Opening => "opening camera (F4 to stop)",
        CameraStatus::Live => "camera LIVE (F4 to stop)",
        CameraStatus::Unavailable => "camera unavailable (F4 to retry; see terminal)",
    };
    format!("crtulum — {} — {connection} — {camera}", preset.name)
}

impl State {
    async fn new(
        window: Arc<Window>,
        capture: Option<capture::SharedFrame>,
        preset: Preset,
    ) -> anyhow::Result<State> {
        let size = window.inner_size();

        let instance = gpu::instance();
        let mut surface = instance.create_surface(window.clone()).map_err(|error| {
            anyhow::anyhow!("could not create a Vulkan surface: {error}")
        })?;
        let adapter = gpu::adapter(&instance, Some(&surface))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: gpu::limits(&adapter),
                },
                None,
            )
            .await
            .map_err(|error| anyhow::anyhow!("failed to create GPU device: {error}"))?;

        let caps = surface.get_capabilities(&adapter);
        eprintln!("[surface] offered formats: {:?}", caps.formats);
        // Prefer a true HDR swapchain: Rgba16Float with linear sRGB or BT.2020
        // negotiated by the Vulkan backend. Match shader primaries after configure.
        // Surface capabilities determine availability, not Wayland protocol names.
        // Fall back to sRGB with an SDR presentation tone map when needed.
        let hdr_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Rgba16Float)
            .or_else(|| caps.formats.iter().copied().find(|f| *f == wgpu::TextureFormat::Rgb10a2Unorm));
        let format = hdr_format
            .or_else(|| caps.formats.iter().copied().find(|f| f.is_srgb()))
            .unwrap_or(caps.formats[0]);
        let hdr = hdr_format.is_some();
        eprintln!(
            "[surface] using {:?} — HDR output {}",
            format,
            if hdr { "ENABLED" } else { "unavailable → SDR tonemap" }
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

        let mut res = build_resources(&device, &queue, format, preset);
        res.hdr_output = surface_output_mode(&mut surface);
        anyhow::ensure!(!hdr || res.hdr_output != 0, "HDR format configured without an HDR color space");
        eprintln!("[surface] {}", output_mode_label(res.hdr_output));
        let depth_view = create_depth(&device, config.width, config.height);

        Ok(State {
            profile: benchmark::LiveProfile::new(),
            surface,
            device,
            queue,
            config,
            size,
            res,
            depth_view,
            orbit: Orbit {
                // A restrained three-quarter product view exposes the tube's real depth,
                // curved faceplate and cabinet edge highlights. Dead-front framing made
                // even the deep mesh read like a flat shader preview.
                yaw: 0.24,
                pitch: 0.18,
                distance: 2.65,
            },
            start: std::time::Instant::now(),
            field_clock: FieldClock::default(),
            degaussed: false,
            // Power up with a warmup + auto-degauss, like a real set switching on.
            power: PowerState::Warmup(std::time::Instant::now()),
            degauss_start: Some(std::time::Instant::now()),
            frame: 0,
            interlace: false,
            exposure: 1.0,
            subpixel: false,
            bfi: false,
            glare: true,
            window_reflection: true,
            webcam: None,
            camera_status: CameraStatus::Off,
            // Panel refresh, for the BFI gate: strobing only helps at ≥100 Hz (at 60 Hz
            // it just flickers at 30). Best effort — re-detected on the first BFI toggle
            // once the Wayland surface has entered an output.
            refresh_hz: detect_refresh_hz(&window),
            dragging: false,
            last_cursor: (0.0, 0.0),
            window,
            capture,
            last_seq: 0,
            player: None,
            preset,
            hdr,
        })
    }

    fn toggle_webcam(&mut self) {
        if self.webcam.take().is_some() {
            self.res.camera[0] = 0.0;
            self.set_camera_status(CameraStatus::Off);
            eprintln!("[webcam] off; camera released");
        } else {
            match webcam::Webcam::start() {
                Ok(camera) => {
                    self.webcam = Some(camera);
                    self.set_camera_status(CameraStatus::Opening);
                }
                Err(error) => {
                    self.set_camera_status(CameraStatus::Unavailable);
                    eprintln!("[webcam] {error:#}");
                }
            }
        }
    }

    fn poll_webcam(&mut self) {
        let Some(camera) = &mut self.webcam else { return };
        match camera.poll() {
            Ok(Some(frame)) => {
                write_source(&self.queue, &self.res.camera_texture, webcam::WIDTH, webcam::HEIGHT, &frame);
                // Average radiance in linear light, also used outside the camera FOV.
                let mut mean = [0.0; 4];
                let mut count = 0.0;
                for pixel in frame.chunks_exact(4).step_by(16) {
                    for c in 0..3 {
                        let v = pixel[c] as f32 / 255.0;
                        mean[c] += if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
                    }
                    count += 1.0;
                }
                for c in 0..3 { mean[c] /= count; }
                self.res.camera_ambient = mean;
                self.res.camera = [1.0, camera.tan_half_fov, webcam::WIDTH as f32 / webcam::HEIGHT as f32, 0.0];
                self.set_camera_status(CameraStatus::Live);
            }
            Ok(None) => {}
            Err(error) => {
                eprintln!("[webcam] {error:#}; restoring synthetic room");
                self.webcam = None;
                self.res.camera[0] = 0.0;
                self.set_camera_status(CameraStatus::Unavailable);
            }
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

    /// Advance the game and hand its newest frame to the phosphor plane.
    fn poll_player(&mut self) {
        let Some(player) = &mut self.player else { return };
        if let Err(e) = player.tick() {
            eprintln!("[play] {e:#}");
            self.player = None;
            return;
        }
        if !player.fresh || player.size.0 == 0 {
            return;
        }
        player.fresh = false;
        let (w, h) = player.size;
        // The borrow has to end before the upload, which needs &self.res.
        let frame = std::mem::take(&mut player.frame);
        self.res.set_source(
            &self.device,
            &self.queue,
            w,
            h,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &frame,
        );
        if let Some(player) = &mut self.player {
            player.frame = frame;
        }
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.res.hdr_output = surface_output_mode(&mut self.surface);
        self.hdr = self.res.hdr_output != 0;
        eprintln!("[surface] {}", output_mode_label(self.res.hdr_output));
        self.depth_view = create_depth(&self.device, self.config.width, self.config.height);
    }

    // Switch tube/mask preset live. Optics come from `self.preset` each frame, but
    // the curvature is baked into the mesh, so the geometry buffers are rebuilt.
    fn set_preset(&mut self, mut preset: Preset) {
        preset.input = self.preset.input;
        let (verts, indices) = build_mesh(preset.bulge, preset.curv_x, preset.curv_y, preset.cabinet);
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
        self.show_input();
    }

    fn set_camera_status(&mut self, status: CameraStatus) {
        if self.camera_status != status {
            self.camera_status = status;
            self.window.set_title(&window_title(&self.preset, status));
        }
    }

    fn show_input(&self) {
        self.window.set_title(&window_title(&self.preset, self.camera_status));
        eprintln!("[input] {} (signal {})", self.preset.input.label(), self.preset.input.signal(self.preset.signal));
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
                    self.degaussed = true;
                    0.0
                } else {
                    (-e / DEGAUSS_TAU).exp() // exponential AC burst — snaps then fades fast
                }
            }
            None => 0.0,
        };
        [warmup, collapse, degauss, if self.degaussed { 1.0 } else { 0.0 }]
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
        let profile_start = self.profile.as_ref().map(|_| std::time::Instant::now());
        self.poll_webcam();
        let time = self.start.elapsed().as_secs_f64();
        // Bootstrap the first field. Thereafter catch-up fields use the source
        // held at the previous presentation, before polling a newer live frame.
        if self.field_clock.next == 0 {
            self.poll_capture();
            self.poll_player();
        }
        let pwr = self.power_params();
        self.frame = self.frame.wrapping_add(1);
        let interlace = if self.interlace { 1.0 } else { 0.0 };
        let aspect = self.config.width as f32 / self.config.height as f32;
        // Submit before overwriting the uniform buffer: queue writes are not snapshots
        // captured by an encoded render pass.
        self.res.exposure_group = self.frame as u32 % 16_000_000 + 1;
        for field in self.field_clock.through(time) {
            write_uniforms(&self.queue, &self.res, &self.orbit, aspect,
                (field as f64 / FIELD_HZ) as f32, &self.preset, 1.0, self.hdr,
                (1.0 / FIELD_HZ) as f32, pwr, interlace, (field % 4) as f32,
                self.exposure, self.subpixel, 1.0, self.glare, self.window_reflection);
            let mut enc = self.device.create_command_encoder(&Default::default());
            accum_step(&mut enc, &mut self.res);
            self.queue.submit(Some(enc.finish()));
        }
        // BFI: strobe the emitted phosphor light dark on alternate refreshes so motion
        // reads as CRT-impulse rather than LCD sample-and-hold. Only the emission is
        // blanked (the glass keeps mirroring the room). 1.0 = lit frame, 0.0 = dark.
        let bfi_mul = if self.bfi && (self.frame & 1) == 1 { 0.0 } else { 1.0 };
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
            0.0,
            pwr,
            interlace,
            (self.field_clock.next.saturating_sub(1) % 4) as f32,
            self.exposure,
            self.subpixel,
            bfi_mul,
            self.glare,
            self.window_reflection,
        );

        let profile_accum = profile_start.map(|_| std::time::Instant::now());
        let frame = self.surface.get_current_texture()?;
        let profile_acquire = profile_start.map(|_| std::time::Instant::now());
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        // Present the latest completed field.
        draw_tube(&mut encoder, &self.res, &view, &self.depth_view);
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        let profile_present = profile_start.map(|_| std::time::Instant::now());
        // Latch the next source only after drawing the completed fields. In
        // particular, a stall must not retroactively scan the newest game frame
        // into every missed field (which also destroys monochrome motion history).
        self.poll_capture();
        self.poll_player();
        if let Some(profile) = &mut self.profile {
            profile.record(profile_start.unwrap(), profile_accum.unwrap(),
                profile_acquire.unwrap(), profile_present.unwrap());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Headless capture: render one frame to a PNG (`--shot out.png`)
// ---------------------------------------------------------------------------

fn save_shot(path: &str, width: u32, height: u32, preset: Preset) -> anyhow::Result<()> {
    // Supersample: the CRT's fine mask + scanline structure sits near the output
    // Nyquist limit, so render at SSxSS and box-downsample (in linear light) to
    // the requested size. Without this the fine detail aliases into flat color.
    const SS: u32 = 3;
    let rw = width * SS;
    let rh = height * SS;

    let instance = gpu::instance();
    let adapter = gpu::adapter(&instance, None)?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("headless-device"),
            required_features: wgpu::Features::empty(),
            required_limits: gpu::limits(&adapter),
        },
        None,
    ))
    .map_err(|error| anyhow::anyhow!("failed to create Vulkan device: {error}"))?;

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
    let dt = (1.0 / FIELD_HZ) as f32;
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
    // Subpixel mask only resolves at native res, so it SSAAs away in a shot — off by
    // default; CRTULUM_SUBPIXEL=1 forces it for structural checks. BFI is a live-only
    // motion effect (a still can't show a strobe), so the shot always renders lit.
    let subpixel = envf("CRTULUM_SUBPIXEL", 0.0) > 0.5;
    let glare = envf("CRTULUM_GLARE", 1.0) > 0.5;
    let window_reflection = envf("CRTULUM_WINDOW_REFLECTION", 1.0) > 0.5;

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
        let phase = ((field as u32 + s + 4 - (steps - 1) % 4) % 4) as f32;
        let time = shot_t - (steps - 1 - s) as f32 * dt;
        write_uniforms(&queue, &res, &orbit, width as f32 / height as f32, time,
            &preset, SS as f32, false, dt, pwr, interlace, phase, exposure,
            subpixel, 1.0, glare, window_reflection);
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
    img.save(path)?;
    println!("wrote {path} ({width}x{height})");
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print!("{}", video::USAGE);
        return;
    }

    match gamescope::launch(&args) {
        Ok(Some(status)) => std::process::exit(status.code().unwrap_or(1)),
        Ok(None) => {},
        Err(error) => { eprintln!("[gamescope] {error:#}"); std::process::exit(2); }
    }

    // `--preset trinitron|panasonic|slotmask` (default trinitron)
    let mut preset = match args.iter().position(|a| a == "--preset") {
        Some(i) => match args.get(i + 1).ok_or_else(|| anyhow::anyhow!("--preset needs a name"))
            .and_then(|name| preset_by_name(name)) {
            Ok(preset) => preset,
            Err(error) => { eprintln!("{error}"); std::process::exit(2); }
        },
        None => TRINITRON,
    };
    if let Some(i) = args.iter().position(|a| a == "--input") {
        preset.input = match args.get(i + 1).ok_or_else(|| anyhow::anyhow!("--input needs a mode"))
            .and_then(|value| InputMode::parse(value)) {
            Ok(input) => input,
            Err(error) => { eprintln!("{error}"); std::process::exit(2); }
        };
    }
    eprintln!("[preset] {}", preset.name);

    if args.iter().any(|a| a == "--benchmark") {
        if let Err(error) = benchmark::run(&args, preset) {
            eprintln!("[benchmark] {error:#}");
            std::process::exit(1);
        }
        return;
    }

    // Headless capture mode: `crtulum --shot out.png [WxH]`
    if let Some(i) = args.iter().position(|a| a == "--shot") {
        let path = args.get(i + 1).map(String::as_str).unwrap_or("shot.png");
        let (w, h) = args
            .get(i + 2)
            .and_then(|s| s.split_once('x'))
            .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
            .unwrap_or((1000, 800));
        if let Err(error) = save_shot(path, w, h, preset) {
            eprintln!("[shot] {error:#}");
            std::process::exit(1);
        }
        return;
    }

    // `crtulum --fetch-agent Merlin` — pull a character's sprite sheet and frame table
    // into the user's data directory, since we ship the reader and not the artwork.
    if let Some(i) = args.iter().position(|a| a == "--fetch-agent") {
        match args.get(i + 1) {
            Some(name) if !name.starts_with('-') => {
                if let Err(e) = agent::fetch(name) {
                    eprintln!("[agent] {e:#}");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!(
                    "--fetch-agent needs a character name: {}",
                    agent::KNOWN.join(", ")
                );
                std::process::exit(1);
            }
        }
        return;
    }

    // Scripted video export: `crtulum --render in.mp4 out.mp4 [--script run.crts]`.
    // Runs a whole source (file / URL / stills / a TAS through RetroArch) through the
    // tube and pipes the result into ffmpeg. See src/video.rs.
    if args.iter().any(|a| a == "--render" || a == "--clip") {
        let result = video::opts_from_args(&args, preset).and_then(video::render);
        if let Err(e) = result {
            eprintln!("[render] error: {e:#}");
            std::process::exit(1);
        }
        return;
    }

    // Live play: `crtulum --play game.sfc [--core snes9x] [--option k=v]`. The game
    // runs on the tube with a controller, rather than being rendered to a file.
    let player = if let Some(i) = args.iter().position(|a| a == "--play") {
        let rom = match args.get(i + 1) {
            Some(r) if !r.starts_with('-') => std::path::PathBuf::from(r),
            _ => {
                eprintln!("--play needs a ROM: crtulum --play game.sfc [--core NAME]");
                std::process::exit(1);
            }
        };
        let core = args
            .iter()
            .position(|a| a == "--core")
            .and_then(|i| args.get(i + 1))
            .cloned();
        let options: Vec<(String, String)> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == "--option")
            .filter_map(|(i, _)| args.get(i + 1))
            .filter_map(|kv| kv.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
            .collect();
        match play::Player::new(&rom, core.as_deref(), &options) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("[play] error: {e:#}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    let capture = if args.iter().any(|a| a == "--capture") {
        eprintln!("[capture] starting — pick a window or screen in the portal dialog…");
        Some(capture::spawn())
    } else {
        None
    };

    let event_loop = EventLoop::new().unwrap_or_else(|error| {
        eprintln!("[window] cannot connect to the desktop: {error}");
        std::process::exit(1);
    });
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("crtulum")
            .with_inner_size(winit::dpi::LogicalSize::new(1000.0, 800.0))
            .build(&event_loop)
            .unwrap_or_else(|error| {
                eprintln!("[window] cannot create a window: {error}");
                std::process::exit(1);
            }),
    );

    let mut state = match pollster::block_on(State::new(window.clone(), capture, preset)) {
        Ok(state) => state,
        Err(error) => {
            eprintln!("[gpu] startup failed: {error:#}");
            std::process::exit(1);
        }
    };
    if args.iter().any(|a| a == "--require-hdr") && !state.hdr {
        eprintln!("[surface] --require-hdr: this compositor exposes no supported HDR surface; try --gamescope-hdr-test for HDR rendering with SDR output");
        std::process::exit(2);
    }
    state.player = player;
    state.show_input();
    let verify_frames = args.iter().position(|a| a == "--verify-frames")
        .map(|i| args.get(i + 1).and_then(|v| v.parse::<u64>().ok()).filter(|n| *n > 0)
            .unwrap_or_else(|| { eprintln!("--verify-frames needs a positive integer"); std::process::exit(2); }));
    let mut presented_frames = 0u64;

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
            match event {
                Event::WindowEvent { event, window_id } if window_id == state.window.id() => {
                    match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::KeyboardInput { event, .. } => {
                            // While a game is running its buttons come first, so the
                            // controls it uses can't also flip the tube's settings.
                            let consumed = match (&mut state.player, event.physical_key) {
                                (Some(p), PhysicalKey::Code(code)) => {
                                    p.set_key(code, event.state == ElementState::Pressed)
                                }
                                _ => false,
                            };
                            if !consumed && event.state == ElementState::Pressed {
                                match event.physical_key {
                                    PhysicalKey::Code(KeyCode::Escape) => {
                                        if state.window.fullscreen().is_some() {
                                            state.window.set_fullscreen(None);
                                            eprintln!("[fullscreen] off");
                                        } else {
                                            elwt.exit();
                                        }
                                    }
                                    // F11 = borderless fullscreen; Escape leaves it first.
                                    PhysicalKey::Code(KeyCode::F11) if !event.repeat => {
                                        let fullscreen = state.window.fullscreen().is_none();
                                        state.window.set_fullscreen(if fullscreen {
                                            Some(Fullscreen::Borderless(
                                                state.window.current_monitor(),
                                            ))
                                        } else {
                                            None
                                        });
                                        eprintln!("[fullscreen] {}", if fullscreen { "on" } else { "off" });
                                    }
                                    PhysicalKey::Code(KeyCode::F4) if !event.repeat => state.toggle_webcam(),
                                    // L and R isolate the two strongest photographic glass cues.
                                    PhysicalKey::Code(KeyCode::KeyL) if !event.repeat => {
                                        state.glare = !state.glare;
                                        eprintln!("[glare] {}", if state.glare { "on" } else { "off" });
                                    }
                                    PhysicalKey::Code(KeyCode::KeyR) if !event.repeat => {
                                        state.window_reflection = !state.window_reflection;
                                        eprintln!("[window reflection] {}", if state.window_reflection { "on" } else { "off" });
                                    }
                                    PhysicalKey::Code(KeyCode::F3) if !event.repeat => {
                                        state.preset.input = state.preset.input.next();
                                        state.show_input();
                                    }
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
                                    PhysicalKey::Code(KeyCode::KeyP) if !event.repeat => state.toggle_power(),
                                    // Pause the game (the tube keeps running).
                                    PhysicalKey::Code(KeyCode::F2) if !event.repeat => {
                                        if let Some(p) = &mut state.player {
                                            p.toggle_pause();
                                        }
                                    }
                                    PhysicalKey::Code(KeyCode::KeyG) if !event.repeat => {
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
                                    // I = alternate fields vs progressive scanning of the current signal.
                                    PhysicalKey::Code(KeyCode::KeyI) if !event.repeat => {
                                        state.interlace = !state.interlace;
                                        eprintln!("[interlace] {}", if state.interlace { "alternate fields" } else { "progressive" });
                                    }
                                    // M = subpixel-accurate (Megatron) mask vs the resolution-
                                    // independent gaussian mask. Only looks right at native
                                    // resolution on an RGB-stripe panel.
                                    PhysicalKey::Code(KeyCode::KeyM) if !event.repeat => {
                                        state.subpixel = !state.subpixel;
                                        eprintln!("[mask] {}", if state.subpixel { "subpixel (Megatron)" } else { "gaussian" });
                                    }
                                    // B = black-frame insertion (CRT-impulse motion clarity).
                                    // Needs a ≥100 Hz panel to help instead of just flickering.
                                    PhysicalKey::Code(KeyCode::KeyB) if !event.repeat => {
                                        state.bfi = !state.bfi;
                                        // Re-detect: at startup on Wayland current_monitor()
                                        // is usually None, so refresh_hz may still be the
                                        // 60 Hz fallback. By now the surface is mapped.
                                        state.refresh_hz = detect_refresh_hz(&state.window);
                                        if state.bfi && state.refresh_hz < 100.0 {
                                            eprintln!("[bfi] on — WARNING: {:.0} Hz panel; BFI needs ≥100 Hz to reduce blur (will flicker)", state.refresh_hz);
                                        } else {
                                            eprintln!("[bfi] {} ({:.0} Hz)", if state.bfi { "on" } else { "off" }, state.refresh_hz);
                                        }
                                    }
                                    PhysicalKey::Code(KeyCode::Tab) if !event.repeat => {
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
                        WindowEvent::Focused(false) => {
                            if let Some(player) = &mut state.player { player.clear_keys(); }
                            state.dragging = false;
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
                            Ok(()) => {
                                presented_frames += 1;
                                if verify_frames.is_some_and(|limit| presented_frames >= limit) {
                                    eprintln!("[verify] presented {} frames; {}", presented_frames, output_mode_label(state.res.hdr_output));
                                    eprintln!("[verify] surface {}x{}; monitor {:?}; refresh {:.3} Hz",
                                        state.config.width, state.config.height,
                                        state.window.current_monitor().and_then(|m| m.name()),
                                        detect_refresh_hz(&state.window));
                                    elwt.exit();
                                }
                            }
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
        .unwrap_or_else(|error| {
            eprintln!("[window] event loop failed: {error}");
            std::process::exit(1);
        });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_preserves_connection_and_camera_status() {
        let mut preset = PVM;
        preset.input = InputMode::Composite;
        for status in [CameraStatus::Off, CameraStatus::Opening, CameraStatus::Live, CameraStatus::Unavailable] {
            let title = window_title(&preset, status);
            assert!(title.contains("pvm") && title.contains("Composite"), "{title}");
            assert_eq!(title.contains("camera LIVE"), status == CameraStatus::Live);
        }
        assert!(window_title(&TRINITRON, CameraStatus::Off).contains("preset default (S-video)"));
        assert!(window_title(&RCA, CameraStatus::Live).contains("preset default (Composite)"));
        assert!(preset_by_name("trinitrron").is_err());
        assert_eq!(preset_by_name("PVM").unwrap().name, "pvm");
    }

    #[test]
    fn input_modes_preserve_defaults_and_cycle_back() {
        for preset in ALL_PRESETS {
            assert_eq!(preset.input, InputMode::Auto);
            assert_eq!(preset.input.signal(preset.signal), preset.signal);
            assert_eq!(InputMode::Composite.signal(preset.signal), 2);
            assert_eq!(InputMode::Rf.signal(preset.signal), 3);
        }
        let mut input = InputMode::Auto;
        for _ in 0..6 { input = input.next(); }
        assert_eq!(input, InputMode::Auto);
        assert!(InputMode::parse("typo").is_err());
    }

    #[test]
    fn signal_filter_cutoffs_match_the_documented_bandwidths() {
        // Read the actual shader constants and measure the finite discrete kernel's
        // frequency response; a -6 dB/-3 dB confusion must fail this test.
        let constant = |name: &str| -> f64 {
            let prefix = format!("const {name}: ");
            concat!(include_str!("phosphor.wgsl"), "\n", include_str!("shader.wgsl")).lines().find(|l| l.starts_with(&prefix)).unwrap()
                .split_once('=').unwrap().1.split(';').next().unwrap().trim().parse().unwrap()
        };
        let rate = 320.0 / 52.6;
        let step = constant("NTSC_STEP");
        let taps = constant("NTSC_TAPS") as i32;
        for (name, cutoff) in [
            ("NTSC_SIG_I", 1.0 / (1.0_f64 / 1.3_f64.powi(2) + 1.0 / 0.5_f64.powi(2)).sqrt()),
            ("NTSC_SIG_Q", 1.0 / (1.0_f64 / 0.4_f64.powi(2) + 1.0 / 0.5_f64.powi(2)).sqrt()),
            ("NTSC_SIG_Y", 3.0), ("NTSC_SIG_TRAP", 3.579545 / 10.0 / 2.0),
            ("SVIDEO_SIG_Y", 4.0),
        ] {
            let sigma = constant(name);
            let support = if name == "SVIDEO_SIG_Y" { 36 } else { taps };
            let (mut dc, mut gain) = (0.0, 0.0);
            for k in -support..=support {
                let x = k as f64 * step;
                let weight = (-x * x / (2.0 * sigma * sigma)).exp();
                dc += weight;
                gain += weight * (std::f64::consts::TAU * cutoff / rate * x).cos();
            }
            let db = 20.0 * (gain / dc).abs().log10();
            assert!((db + 3.0103).abs() < 0.025, "{name}: {db} dB at {cutoff} MHz");
        }
        assert!((constant("NTSC_FSC") - 3.579545 / rate).abs() < 1e-5);
    }

    #[test]
    fn voltage_noise_matches_target_snr() {
        for snr in [36.0, 42.0, 52.0, 64.0] {
            let amplitude = noise_peak_to_peak(snr);
            // Integrate the uniform distribution, independently of the conversion.
            let variance = (0..10000).map(|i| {
                let v = amplitude * ((i as f32 + 0.5) / 10000.0 - 0.5);
                v * v
            }).sum::<f32>() / 10000.0;
            assert!((-10.0 * variance.log10() - snr).abs() < 0.001);
        }
    }

    #[test]
    fn average_picture_level_tracks_beam_current() {
        let grey = source_stats(&[128, 128, 128, 255], 1, 1, false);
        assert!((grey[3] - (128.0_f32 / 255.0).powf(2.4)).abs() < 1e-6);
        let mixed = source_stats(&[0, 0, 0, 255, 255, 255, 255, 255], 2, 1, false);
        assert!((mixed[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn supply_ripple_tracks_the_field_clock() {
        assert!((1.0 / HUM_BEAT_HZ - 1001.0 / 120.0).abs() < 1e-10);
        for field in 0..10_000 {
            let time = field as f64 / FIELD_HZ;
            let mains = (std::f64::consts::TAU * 120.0 * time).sin();
            let beat = (std::f64::consts::TAU * HUM_BEAT_HZ * time).sin();
            assert!((mains - beat).abs() < 1e-10, "field {field}");
        }
    }

    #[test]
    fn beam_and_mask_preserve_uniform_field_energy_on_gpu() {
        let (device, queue) = headless_device();
        let mut res = build_resources(&device, &queue, wgpu::TextureFormat::Rgba8UnormSrgb, RCA);
        res.set_source(&device, &queue, 8, 8, wgpu::TextureFormat::Rgba8UnormSrgb, &vec![128; 8 * 8 * 4]);
        write_uniforms(&queue, &res, &Orbit { yaw: 0.0, pitch: 0.0, distance: 3.0 },
            1.0, 0.0, &RCA, 1.0, false, 0.0, [1.0, 0.0, 0.0, 0.0],
            0.0, 0.0, 1.0, false, 1.0, false, false);
        let source = format!("{}\n{}", concat!(include_str!("phosphor.wgsl"), "\n", include_str!("shader.wgsl")), r#"
            @fragment fn fs_camera_right(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(room(normalize(vec3<f32>(0.3, 0.2, 1.0))), 1.0);
            }
            @fragment fn fs_camera_left(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(room(normalize(vec3<f32>(-0.3, 0.2, 1.0))), 1.0);
            }
            @fragment fn fs_camera_bottom(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(room(normalize(vec3<f32>(0.3, -0.2, 1.0))), 1.0);
            }
            @fragment fn fs_camera_behind(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(room(vec3<f32>(0.0, 0.0, -1.0)), 1.0);
            }
            @fragment fn fs_camera_outside(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(room(normalize(vec3<f32>(1.0, 0.0, 0.1))), 1.0);
            }
            @fragment fn fs_camera_mirror(in: FullOut) -> @location(0) vec4<f32> {
                // A sloping glass normal redirects a head-on view toward world +X.
                let n = normalize(vec3<f32>(0.15, 0.10, 1.0));
                return vec4<f32>(room(reflect(vec3<f32>(0.0, 0.0, -1.0), n)), 1.0);
            }
            @fragment fn fs_beam_probe(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(scan_reconstruct(in.uv, u.params.xy, 1.7, vec2<f32>(0.0)) / u.tone.z, 1.0);
            }
            @fragment fn fs_hdr_highlight(in: FullOut) -> @location(0) vec4<f32> {
                return output_color(vec3<f32>(4.0));
            }
            @fragment fn fs_hdr_pq(in: FullOut) -> @location(0) vec4<f32> {
                return output_color(vec3<f32>(1.0));
            }
            @fragment fn fs_hdr_srgb(in: FullOut) -> @location(0) vec4<f32> {
                return output_color(vec3<f32>(1.0, 0.0, 0.0));
            }
            @fragment fn fs_hdr_bt2020(in: FullOut) -> @location(0) vec4<f32> {
                return output_color(vec3<f32>(1.0, 0.0, 0.0));
            }
            @fragment fn fs_bounce(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(screen_bounce_color(), 1.0);
            }
            @fragment fn fs_bounce_bfi(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(screen_bounce_color(), 1.0);
            }
            @fragment fn fs_mask_near(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(phosphor3(in.uv.x, 0.0), 1.0);
            }
            @fragment fn fs_mask_middle(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(phosphor3(in.uv.x, 0.2), 1.0);
            }
            @fragment fn fs_mask_far(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(phosphor3(in.uv.x, 0.8), 1.0);
            }
            @fragment fn fs_hum(in: FullOut) -> @location(0) vec4<f32> {
                return vec4<f32>(
                    hum_modulation(in.uv.y, 0.0, u.focus.z, 0.02),
                    hum_modulation(in.uv.y, 2.0, u.focus.z, 0.02),
                    hum_modulation(in.uv.y, 7.0, u.focus.z, 0.02), 1.0);
            }
            @fragment fn fs_svm_step(in: FullOut) -> @location(0) vec4<f32> {
                let beam = svm_source(in.uv, u.params.xy);
                let trajectory = svm_trajectory(beam.x * u.params.x, floor(beam.y * u.params.y), u.params.xy);
                let light = textureSampleLevel(t_screen, s_screen, beam.xy, 0.0).r * beam.z;
                // Uniform deposited energy integrates inverse velocity over
                // screen position; velocity itself has no unit-mean invariant.
                return vec4<f32>(light, beam.z,
                    abs(beam.x * u.params.x + trajectory.x - in.uv.x * u.params.x), 1.0);
            }
        "#);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("beam-energy-test"), source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let binds = res.accum_pipeline.get_bind_group_layout(0);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&binds], push_constant_ranges: &[],
        });
        let decoded = ((128.0_f32 / 255.0 + 0.055) / 1.055).powf(2.4);
        let current = (128.0_f32 / 255.0).powf(2.4);
        for (entry, expected) in [("fs_beam_probe", [decoded; 3]), ("fs_mask_near", [0.26317; 3]),
            ("fs_mask_middle", [0.26317; 3]), ("fs_mask_far", [0.26317; 3]),
            ("fs_hdr_highlight", [4.0; 3]), ("fs_hdr_pq", [0.48586; 3]),
            ("fs_hdr_srgb", [1.0, 0.0, 0.0]), ("fs_hdr_bt2020", [0.6274, 0.0691, 0.0164]),
            ("fs_bounce", [current; 3]), ("fs_bounce_bfi", [0.0; 3]), ("fs_hum", [1.0; 3]),
            ("fs_svm_step", [0.5, 1.0, 0.0]),
            ("fs_camera_right", [1.0, 0.0, 0.0]), ("fs_camera_left", [0.0, 1.0, 0.0]),
            ("fs_camera_bottom", [0.0, 0.0, 1.0]), ("fs_camera_behind", [0.2, 0.3, 0.4]),
            ("fs_camera_outside", [0.2, 0.3, 0.4]), ("fs_camera_mirror", [1.0, 0.0, 0.0])] {
            if entry.starts_with("fs_camera") {
                let pixels: Vec<u8> = (0..webcam::HEIGHT).flat_map(|y| (0..webcam::WIDTH).flat_map(move |x| {
                    match (x < webcam::WIDTH / 2, y < webcam::HEIGHT / 2) {
                        (true, true) => [255, 0, 0, 255],
                        (false, true) => [0, 255, 0, 255],
                        (true, false) => [0, 0, 255, 255],
                        (false, false) => [255; 4],
                    }
                })).collect();
                write_source(&queue, &res.camera_texture, webcam::WIDTH, webcam::HEIGHT, &pixels);
                res.camera = [1.0, (70_f32.to_radians() * 0.5).tan(), 4.0 / 3.0, 0.0];
                res.camera_ambient = [0.2, 0.3, 0.4, 0.0];
                write_uniforms(&queue, &res, &Orbit { yaw: 0.0, pitch: 0.0, distance: 3.0 },
                    1.0, 0.0, &PVM, 1.0, false, 0.0, [1.0, 0.0, 0.0, 0.0],
                    0.0, 0.0, 1.0, false, 1.0, false, false);
            }
            if entry == "fs_svm_step" {
                let pixels: Vec<u8> = (0..64).flat_map(|i| [if i % 8 < 4 { 0 } else { 255 }; 4]).collect();
                res.set_source(&device, &queue, 8, 8, wgpu::TextureFormat::Rgba8UnormSrgb, &pixels);
                write_uniforms(&queue, &res, &Orbit { yaw: 0.0, pitch: 0.0, distance: 3.0 },
                    1.0, 0.0, &RCA, 1.0, false, 0.0, [1.0, 0.0, 0.0, 0.0],
                    0.0, 0.0, 1.0, false, 1.0, false, false);
            }
            if entry.starts_with("fs_hdr") || entry.starts_with("fs_bounce") {
                res.hdr_output = if entry == "fs_hdr_bt2020" { 2 } else if entry == "fs_hdr_pq" { 3 } else { 1 };
                write_uniforms(&queue, &res, &Orbit { yaw: 0.0, pitch: 0.0, distance: 3.0 },
                    1.0, 0.0, &PVM, 1.0, true, 0.0, [1.0, 0.0, 0.0, 0.0],
                    0.0, 0.0, 1.0, false, if entry == "fs_bounce_bfi" { 0.0 } else { 1.0 }, false, false);
            }
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None, layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: "vs_full", buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: entry,
                    targets: &[Some(wgpu::ColorTargetState { format: PHOSPHOR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                    compilation_options: Default::default() }),
                primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None,
            });
            let (target, view) = make_phosphor(&device, 32, 256);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 256 * 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None, color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view,
                        resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &res.accum_bind[0], &[]);
                pass.draw(0..3, 0..1);
            }
            enc.copy_texture_to_buffer(target.as_image_copy(), wgpu::ImageCopyBuffer { buffer: &buffer,
                layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(256) } },
                wgpu::Extent3d { width: 32, height: 256, depth_or_array_layers: 1 });
            queue.submit(Some(enc.finish()));
            let slice = buffer.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range();
            let mut sums = [0.0; 3];
            for (index, pixel) in data.chunks_exact(8).enumerate() {
                for c in 0..3 {
                    let h = u16::from_le_bytes([pixel[2 * c], pixel[2 * c + 1]]);
                    assert_eq!(h >> 15, 0);
                    let e = (h >> 10) & 31;
                    assert!(e < 31, "non-finite beam energy");
                    let value = if e == 0 { (h & 1023) as f32 * 2.0_f32.powi(-24) }
                        else { (1.0 + (h & 1023) as f32 / 1024.0) * 2.0_f32.powi(e as i32 - 15) };
                    if entry == "fs_mask_far" {
                        assert!((value - expected[c]).abs() < 0.0002, "unresolved mask aliases: {value}");
                    }
                    if entry == "fs_hum" {
                        let y = ((index / 32) as f64 + 0.5) / 256.0;
                        let phase = HUM_BEAT_HZ * [0.0, 2.0, 7.0][c] - 2.0 * y;
                        let expected = 1.0 + 0.02 * (std::f64::consts::TAU * phase).sin();
                        assert!((value as f64 - expected).abs() < 0.001, "hum phase/amplitude: {value} vs {expected}");
                    }
                    if entry == "fs_svm_step" && c == 1 {
                        assert!(value > 0.7 && value < 1.3, "beam folded or stalled: {value}");
                    }
                    if entry == "fs_svm_step" && c == 2 {
                        assert!(value < 1e-4, "beam trajectory inversion failed: {value}");
                    }
                    sums[c] += value;
                }
            }
            for (sum, expected) in sums.into_iter().zip(expected) {
                let mean = sum / (32 * 256) as f32;
                assert!((mean - expected).abs() < 0.002, "{entry} lost energy: {mean} vs {expected}");
            }
        }
    }

    #[test]
    fn field_clock_is_independent_of_presentation_rate() {
        for fps in [24.0, 25.0, 30.0, 50.0, 59.94, 60.0, 120.0, 144.0] {
            let mut clock = FieldClock::default();
            let mut fields = Vec::new();
            for frame in 0..=(fps * 10.0_f64).ceil() as u64 {
                fields.extend(clock.through((frame as f64 / fps).min(10.0)));
            }
            assert_eq!(fields, (0..600).collect::<Vec<_>>(), "{fps} fps");
            assert!(clock.through(10.0).is_empty());
        }
    }

    #[test]
    fn measured_phosphor_integrates_scan_timing_and_published_decay_on_gpu() {
        let (device, queue) = headless_device();
        let mut res = build_resources(&device, &queue, wgpu::TextureFormat::Rgba8UnormSrgb, PVM);
        let orbit = Orbit { yaw: 0.0, pitch: 0.0, distance: 3.0 };
        let period = 1.0 / FIELD_HZ;
        // Isolate phosphor integration from the adapter's approximate sRGB
        // texture decode. Linear UNORM 55 is close to decoded sRGB 128; apply
        // the shader's OETF and tube gamma analytically for the reference.
        let level = 55_u8;
        let voltage = 1.055 * (level as f64 / 255.).powf(1. / 2.4) - 0.055;
        let excitation = voltage.powf(2.4);
        let read = |res: &Resources| -> Vec<[f32; 3]> {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 256 * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_texture_to_buffer(res.phosphor[res.phos_cur].as_image_copy(),
                wgpu::ImageCopyBuffer { buffer: &buffer, layout: wgpu::ImageDataLayout {
                    offset: 0, bytes_per_row: Some(256), rows_per_image: Some(8) } },
                wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 });
            queue.submit(Some(enc.finish()));
            let slice = buffer.slice(..); slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range();
            (0..64).map(|pixel| std::array::from_fn(|c| {
                let offset = (pixel / 8) * 256 + (pixel % 8) * 8 + c * 2;
                let h = u16::from_le_bytes([data[offset], data[offset + 1]]);
                let e = (h >> 10) & 31;
                assert!(e < 31 && h >> 15 == 0);
                if e == 0 { (h & 1023) as f32 * 2_f32.powi(-24) }
                else { (1.0 + (h & 1023) as f32 / 1024.0) * 2_f32.powi(e as i32 - 15) }
            })).collect()
        };
        for field in 0..3 {
            let source_level = if field == 1 { level } else { 0 };
            res.set_source(&device, &queue, 8, 8, wgpu::TextureFormat::Rgba8Unorm, &vec![source_level; 256]);
            res.exposure_group = field + 1;
            write_uniforms(&queue, &res, &orbit, 1.0, field as f32 * period as f32, &PVM, 1.0,
                false, period as f32, [1.0, 0.0, 0.0, 0.0], 0.0, field as f32, 1.0, false, 1.0, false, false);
            queue.write_buffer(&res.ubuf, (std::mem::offset_of!(Uniforms, look) + 8) as u64,
                bytemuck::bytes_of(&0.0_f32));
            let mut enc = device.create_command_encoder(&Default::default());
            accum_step(&mut enc, &mut res); queue.submit(Some(enc.finish()));
            for (pixel, rgb) in read(&res).iter().enumerate() {
                let x = ((pixel % 8) as f64 + 0.5) / 8.;
                let horizontal = (63.5555556 - 52.6 + x * 52.6) / 63.5555556;
                let arrival = ((8. * (262.5 / 240. - 1.)) + (pixel / 8) as f64 + horizontal)
                    / (8. * 262.5 / 240.) * period;
                for c in 0..3 {
                    let expected = if field == 0 { 0. } else {
                        let end = field as f64 * period - arrival;
                        excitation * phosphor::energy(c, (end - period).max(0.), end)
                    };
                    assert!((rgb[c] as f64 - expected).abs() < 0.0005,
                        "field {field}, pixel {pixel}, channel {c}: {} vs {expected}", rgb[c]);
                }
            }
        }
        // A warm, static tube must preserve a neutral field's mean radiance;
        // short measured decay constants must not darken it or tint it red.
        let mut warm = build_resources(&device, &queue, wgpu::TextureFormat::Rgba8UnormSrgb, PVM);
        warm.set_source(&device, &queue, 8, 8, wgpu::TextureFormat::Rgba8Unorm, &vec![level; 256]);
        write_uniforms(&queue, &warm, &orbit, 1.0, 0.0, &PVM, 1.0, false, period as f32,
            [1.0, 0.0, 0.0, 0.0], 0.0, 0.0, 1.0, false, 1.0, false, false);
        queue.write_buffer(&warm.ubuf, (std::mem::offset_of!(Uniforms, look) + 8) as u64, bytemuck::bytes_of(&0.0_f32));
        let mut enc = device.create_command_encoder(&Default::default());
        accum_step(&mut enc, &mut warm); queue.submit(Some(enc.finish()));
        for pixel in read(&warm) { for value in pixel {
            assert!((value - excitation as f32).abs() < 0.0005, "warm field changed: {value}");
        }}
        // A short shutter resolves the moving raster instead of dimming every
        // row uniformly. Compare red against exact impulse-response integrals.
        warm.shutter_fraction = 0.25;
        warm.exposure_group = 2;
        write_uniforms(&queue, &warm, &orbit, 1.0, period as f32, &PVM, 1.0, false, period as f32,
            [1.0, 0.0, 0.0, 0.0], 0.0, 1.0, 1.0, false, 1.0, false, false);
        queue.write_buffer(&warm.ubuf, (std::mem::offset_of!(Uniforms, look) + 8) as u64, bytemuck::bytes_of(&0.0_f32));
        let mut enc = device.create_command_encoder(&Default::default());
        accum_step(&mut enc, &mut warm); queue.submit(Some(enc.finish()));
        let short = read(&warm);
        for (pixel, rgb) in short.iter().enumerate() {
            let x = ((pixel % 8) as f64 + 0.5) / 8.;
            let horizontal = (63.5555556 - 52.6 + x * 52.6) / 63.5555556;
            let arrival = (8. * (262.5/240. - 1.) + (pixel/8) as f64 + horizontal)
                / (8. * 262.5/240.) * period;
            let energy: f64 = (0..4).map(|age| {
                let end = (age + 1) as f64 * period - arrival;
                phosphor::energy(0, (end - period * 0.25).max(0.), end)
            }).sum();
            let expected = 4. * excitation * energy;
            assert!((rgb[0] as f64 - expected).abs() < 0.001, "short shutter at {pixel}: {} vs {expected}", rgb[0]);
        }
        assert!(short[0][0] < 0.001 && short[63][0] > 0.1, "short shutter did not resolve scanout");
    }

    // Read actual GPU phosphor values, before tone mapping or glass hides errors.
    #[test]
    fn legacy_phosphor_transfer_interlace_and_power() {
        let (device, queue) = headless_device();
        let mut res = build_resources(&device, &queue, wgpu::TextureFormat::Rgba8UnormSrgb, PVM);
        res.physical_phosphor = false;
        let pixels = vec![128u8; 4 * 4 * 4];
        res.set_source(&device, &queue, 4, 4, wgpu::TextureFormat::Rgba8UnormSrgb, &pixels);
        let orbit = Orbit { yaw: 0.0, pitch: 0.0, distance: 3.0 };
        let read = |res: &Resources| -> Vec<f32> {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 1024, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_texture_to_buffer(res.phosphor[res.phos_cur].as_image_copy(),
                wgpu::ImageCopyBuffer { buffer: &buffer, layout: wgpu::ImageDataLayout {
                    offset: 0, bytes_per_row: Some(256), rows_per_image: Some(4),
                } }, wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 });
            queue.submit(Some(enc.finish()));
            let slice = buffer.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range();
            (0..4).map(|y| {
                let h = u16::from_le_bytes([data[y * 256], data[y * 256 + 1]]);
                let exp = (h >> 10) & 31;
                let fraction = (h & 1023) as f32;
                if exp == 0 { fraction * 2.0_f32.powi(-24) }
                else { (1.0 + fraction / 1024.0) * 2.0_f32.powi(exp as i32 - 15) }
            }).collect()
        };
        for field in 0..2 {
            write_uniforms(&queue, &res, &orbit, 1.0, field as f32 / FIELD_HZ as f32,
                &PVM, 1.0, false, 1.0 / FIELD_HZ as f32, [1.0, 0.0, 0.0, 0.0],
                1.0, field as f32, 1.0, false, 1.0, false, false);
            // Isolate gun transfer from the separately tested voltage noise.
            queue.write_buffer(&res.ubuf, (std::mem::offset_of!(Uniforms, look) + 8) as u64,
                bytemuck::bytes_of(&0.0_f32));
            let mut enc = device.create_command_encoder(&Default::default());
            accum_step(&mut enc, &mut res);
            queue.submit(Some(enc.finish()));
            let rows = read(&res);
            for (y, value) in rows.iter().enumerate() {
                if y % 2 == field {
                    let expected = (128.0_f32 / 255.0).powf(2.4);
                    assert!((value - expected).abs() < 0.003, "gun transfer: {value} vs {expected}");
                } else if field == 0 {
                    assert!(*value < 0.0001, "inactive row {y} in field {field} was excited: {value}; rows={rows:?}");
                } else {
                    // Previously scanned rows keep a fading afterglow. They must
                    // neither be re-excited nor have their history erased.
                    let excitation = (128.0_f32 / 255.0).powf(2.4);
                    assert!(*value > 0.01 && *value < excitation * 0.9,
                        "inactive row {y} did not coast on its history: {value}");
                }
            }
        }
        write_uniforms(&queue, &res, &orbit, 1.0, 1.0, &PVM, 1.0, false, 0.1,
            [1.0, 1.0, 0.0, 0.0], 0.0, 0.0, 1.0, false, 1.0, false, false);
        let mut enc = device.create_command_encoder(&Default::default());
        accum_step(&mut enc, &mut res);
        queue.submit(Some(enc.finish()));
        let afterglow = read(&res);
        let excitation = (128.0_f32 / 255.0).powf(2.4);
        assert!(afterglow.iter().all(|v| *v > 0.0 && *v < excitation * 0.05),
            "power-off must stop excitation and preserve fading history: {afterglow:?}");
        write_uniforms(&queue, &res, &orbit, 1.0, 2.0, &PVM, 1.0, false, 1.0,
            [1.0, 1.0, 0.0, 0.0], 0.0, 0.0, 1.0, false, 1.0, false, false);
        let mut enc = device.create_command_encoder(&Default::default());
        accum_step(&mut enc, &mut res);
        queue.submit(Some(enc.finish()));
        assert!(read(&res).iter().all(|v| *v < 0.0001), "powered-off gun still excites phosphor");

        // A quoted time-to-10% must hold at different starting levels and step
        // sizes. A level-dependent tail on the converted tau breaks both.
        for (preset, t10) in [(GREEN, 0.050), (AMBER, 0.013)] {
            for level in [128u8, 255] {
                for steps in [1, 10] {
                    let mut mono = build_resources(&device, &queue, wgpu::TextureFormat::Rgba8UnormSrgb, preset);
                    mono.physical_phosphor = false;
                    mono.set_source(&device, &queue, 4, 4, wgpu::TextureFormat::Rgba8UnormSrgb, &vec![level; 64]);
                    for step in 0..=steps {
                        let power = if step == 0 { [1.0, 0.0, 0.0, 0.0] } else { [1.0, 1.0, 0.0, 0.0] };
                        write_uniforms(&queue, &mono, &orbit, 1.0, 0.0, &preset, 1.0,
                            false, if step == 0 { 0.0 } else { t10 / steps as f32 },
                            power, 0.0, 0.0, 1.0, false, 1.0, false, false);
                        let mut enc = device.create_command_encoder(&Default::default());
                        accum_step(&mut enc, &mut mono);
                        queue.submit(Some(enc.finish()));
                    }
                    let expected = (level as f32 / 255.0).powf(2.4) * 0.1;
                    for value in read(&mono) {
                        assert!((value / expected - 1.0).abs() < 0.02,
                            "{} T10 drift at level {level}, {steps} steps: {value} vs {expected}", preset.name);
                    }
                }
            }
        }
    }

    /// Γ(1+x) as the shader computes it — a cubic fit over x = 1/p, p ∈ [1.2, 5]. It moved
    /// into the shader when the spot exponent stopped being constant across the picture, so
    /// this mirrors it here to keep it pinned; a wrong Γ would silently re-expose the whole
    /// picture by a few percent per preset with nothing to see but "the tubes look a bit off".
    fn gamma1p(x: f32) -> f32 {
        ((-0.10654 * x + 0.58755) * x - 0.47554) * x + 0.99029
    }

    /// The property that actually matters: the beam reconstruction must make a flat field
    /// come back as exactly itself for every tube, at every drive level — including the
    /// drive-dependent spot exponent, which changes the normaliser as well as the profile.
    /// If it does not, brightness depends on beam width and every preset sits at its own
    /// exposure for no physical reason.
    #[test]
    fn beam_profile_normaliser_conserves_energy() {
        // x = 1/p, so the fit's domain is [0.2, 0.833] — p from 5 down to 1.2.
        for (x, want) in [(0.2f32, 0.918_169f32), (0.5, 0.886_227), (0.8, 0.931_384), (0.25, 0.906_402)] {
            let got = gamma1p(x);
            assert!((got - want).abs() < 1e-3, "gamma1p({x}) = {got}, want {want}");
        }

        for preset in ALL_PRESETS {
            let p0 = preset.spot.max(1.2);
            // Signal levels from black to full white; each must come back at unit gain.
            for c in [0.05f32, 0.25, 0.5, 0.75, 1.0] {
                let s = c.powf(preset.beam[2]);
                let w = preset.beam[0] + (preset.beam[1] - preset.beam[0]) * s;
                let p = p0 + (2.0 - p0) * s; // flat top relaxes to a gaussian as it blooms
                let norm = 1.0 / (2.0 * gamma1p(1.0 / p));
                // Average the reconstruction across one row's worth of subpixel phases:
                // the sum oscillates (that oscillation IS the scanline), its mean is the
                // settled brightness, and the mean is what has to equal the signal.
                let n = 64;
                let sum: f32 = (0..n)
                    .map(|i| {
                        let d0 = i as f32 / n as f32;
                        // ±(beam_range+1) rows, matching scan_reconstruct's loop.
                        (-(preset.beam[3] as i32)..=preset.beam[3] as i32 + 1)
                            .map(|k| {
                                let d = (d0 - k as f32).abs();
                                c * (-(d / w).powf(p)).exp() * (norm / w)
                            })
                            .sum::<f32>()
                    })
                    .sum::<f32>()
                    / n as f32;
                assert!(
                    (sum / c - 1.0).abs() < 0.01,
                    "{}: flat field {c} reconstructed at {:.4}x (spot {p}, width {w:.3}) — the \
                     beam is not energy-conserving, so brightness depends on beam width",
                    preset.name,
                    sum / c,
                );
            }
        }
    }

    /// The scanline profile must never invert. Summing overlapping rows is an aperture
    /// filter, and a flat-topped spot wider than half the line pitch has a NEGATIVE response
    /// at the line frequency — the raster peaks BETWEEN the scanlines instead of on them.
    /// That is what forced the spot exponent to relax with drive; without it the Trinitron
    /// inverted by 11% on a white field and the arcade tube flipped phase partway down a
    /// gradient. Assert the modulation stays non-negative and falls monotonically as the
    /// spot blooms, for every tube.
    #[test]
    fn scanline_modulation_never_inverts() {
        for preset in ALL_PRESETS {
            let p0 = preset.spot.max(1.2);
            let mut prev = f32::INFINITY;
            for c in [0.15f32, 0.3, 0.5, 0.7, 1.0] {
                let s = c.powf(preset.beam[2]);
                let w = preset.beam[0] + (preset.beam[1] - preset.beam[0]) * s;
                let p = p0 + (2.0 - p0) * s;
                let norm = 1.0 / (2.0 * gamma1p(1.0 / p));
                let at = |frac: f32| -> f32 {
                    (-4..=4)
                        .map(|k| (-((frac - k as f32).abs() / w).powf(p)).exp() * (norm / w))
                        .sum::<f32>()
                };
                let (centre, gap) = (at(0.0), at(0.5));
                let modulation = (centre - gap) / (centre + gap);
                assert!(
                    modulation > -0.02,
                    "{}: scanlines invert at drive {c} (modulation {modulation:.3}) — the \
                     raster is peaking between the lines",
                    preset.name,
                );
                assert!(
                    modulation <= prev + 0.02, // 1% wiggles from the Γ fit are not a rise
                    "{}: scanline depth rose at drive {c} ({modulation:.3} after {prev:.3}) — a \
                     brighter beam must merge the lines, never separate them",
                    preset.name,
                );
                prev = modulation;
            }
        }
    }

    /// The mask has to be normalised to unit mean or the tube drive silently absorbs its
    /// transmission loss, which differs per mask type and per strength — the ten tubes then
    /// sit at ten different brightnesses for no physical reason. mask_mean() in the shader is
    /// a closed form; check it against a numeric integration of the pattern it claims to
    /// average, and check the open-area figures it implies are the real ones.
    #[test]
    fn mask_mean_matches_the_pattern_and_real_open_areas() {
        let w = 0.105f32;
        let gauss = |t: f32, c: f32| (-(t - c) * (t - c) / (2.0 * w * w)).exp();
        let stripe = |t: f32| -> f32 {
            (-2..=2)
                .map(|k| {
                    let tk = t + k as f32;
                    gauss(tk, 1.0 / 6.0) + gauss(tk, 3.0 / 6.0) + gauss(tk, 5.0 / 6.0)
                })
                .sum::<f32>()
                / 3.0
        };
        let n = 4096;
        let mean = |f: &dyn Fn(f32, f32) -> f32| -> f32 {
            let mut acc = 0.0;
            for i in 0..n {
                for j in 0..n / 64 {
                    acc += f(i as f32 / n as f32, (j as f32 + 0.5) / (n / 64) as f32);
                }
            }
            acc / (n * (n / 64)) as f32
        };
        // grille: stripes only. shadow: × the dot row profile. slot: × the slot profile.
        let grille = mean(&|x, _| stripe(x));
        let shadow = mean(&|x, y| stripe(x) * (0.35 + 0.65 * (-(y - 0.5) * (y - 0.5) / (2.0 * 0.09)).exp()));
        let slot = mean(&|x, y| {
            let s = ((y / 0.12).clamp(0.0, 1.0)).powi(2) * (3.0 - 2.0 * (y / 0.12).clamp(0.0, 1.0))
                * (((1.0 - y) / 0.12).clamp(0.0, 1.0)).powi(2)
                    * (3.0 - 2.0 * ((1.0 - y) / 0.12).clamp(0.0, 1.0));
            stripe(x) * (0.45 + 0.55 * s)
        });
        for (name, got, want) in [
            ("grille", grille, 0.263_17),
            ("shadow", shadow, 0.263_17 * 0.792_20),
            ("slot", slot, 0.263_17 * 0.934_00),
        ] {
            assert!(
                (got - want).abs() < 0.006,
                "mask_mean({name}) = {want}, pattern integrates to {got} — the normaliser and \
                 the pattern have drifted apart, so this mask type is off-brightness",
            );
        }
        // And the transmissions those means imply are the published open areas: aperture
        // grille ~22-25%, slot ~20%, shadow mask ~15-18%.
        assert!((0.22..0.28).contains(&grille), "grille open area {grille}");
        assert!((0.19..0.26).contains(&slot), "slot open area {slot}");
        assert!((0.15..0.22).contains(&shadow), "shadow open area {shadow}");
    }

    /// The mask is a physical grille on the faceplate, so its triad count has to come from a
    /// documented pitch and screen-width inputs — and the ordering that falls out of those
    /// is the thing the old hand-set pitch-in-output-pixels got backwards: a broadcast PVM
    /// and a PC monitor have far MORE triads across the face than any consumer TV.
    #[test]
    fn mask_triad_counts_follow_measured_pitch() {
        let triads = |name: &str| -> f32 {
            let p = preset_by_name(name).unwrap();
            p.screen_mm / p.pitch_mm
        };
        for (name, lo, hi) in [
            ("trinitron", 450.0, 700.0),
            ("panasonic", 450.0, 700.0),
            ("rca", 450.0, 700.0),
            ("arcade", 450.0, 700.0),
            ("pvm", 1100.0, 1400.0),
            ("vga", 900.0, 1100.0),
            ("diamondtron", 1300.0, 1600.0),
        ] {
            let t = triads(name);
            assert!((lo..hi).contains(&t), "{name}: {t:.0} triads across, want {lo}..{hi}");
        }
        assert!(
            triads("pvm") > 2.0 * triads("trinitron"),
            "a PVM's 0.31 mm grille must be far finer than a consumer TV's",
        );
        assert!(
            triads("diamondtron") > triads("pvm"),
            "a 0.24 mm Diamondtron is the finest mask here",
        );
    }

    // Tests exercise the same mandatory physical Vulkan selection as the app.
    fn headless_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = gpu::instance();
        let adapter = gpu::adapter(&instance, None).expect("GPU test requires a physical Vulkan GPU");
        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test-device"),
                required_features: wgpu::Features::empty(),
                required_limits: gpu::limits(&adapter),
            },
            None,
        )).expect("create hardware Vulkan test device")
    }

    // Feed a sequence of source frames through the tube (phosphor history carried
    // across fields, exactly like `--clip`) and read back the final tube render as
    // RGBA8 at native resolution. `frames` are RGBA8 buffers of size sw*sh.
    fn render_sequence(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frames: &[Vec<u8>],
        sw: u32,
        sh: u32,
        ow: u32,
        oh: u32,
        dt: f32,
    ) -> Vec<u8> {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut res = build_resources(device, queue, format, TRINITRON);
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-color"),
            size: wgpu::Extent3d { width: ow, height: oh, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = create_depth(device, ow, oh);
        let orbit = Orbit { yaw: 0.82, pitch: 0.34, distance: 3.7 };
        let pwr = [1.0, 0.0, 0.0, 0.0];

        // One field per source frame — the phosphor plane retains the last few, so
        // moving content trails.
        let mut t = 0.0f32;
        let mut field = 0.0f32;
        for frame in frames {
            res.physical_phosphor = false;
        res.set_source(device, queue, sw, sh, format, frame);
            write_uniforms(
                queue, &res, &orbit, ow as f32 / oh as f32, t, &TRINITRON, 1.0, false, dt, pwr,
                0.0, field, 1.0, false, 1.0, true, true,
            );
            let mut enc = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test-accum") });
            accum_step(&mut enc, &mut res);
            queue.submit(std::iter::once(enc.finish()));
            device.poll(wgpu::Maintain::Wait);
            t += 1.0 / 60.0;
            field = 1.0 - field;
        }

        // Draw the final field and read it back (padded rows → 256 multiple).
        let unpadded = ow * 4;
        let padded = ((unpadded + 255) / 256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-readback"),
            size: (padded * oh) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test-draw") });
        draw_tube(&mut enc, &res, &color_view, &depth_view);
        enc.copy_texture_to_buffer(
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
                    rows_per_image: Some(oh),
                },
            },
            wgpu::Extent3d { width: ow, height: oh, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(enc.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut px = Vec::with_capacity((ow * oh * 4) as usize);
        for row in 0..oh {
            let start = (row * padded) as usize;
            px.extend_from_slice(&data[start..start + unpadded as usize]);
        }
        px
    }

    // A bright horizontal band on a dark field, at vertical position `y0`.
    fn band(sw: u32, sh: u32, y0: i32) -> Vec<u8> {
        let mut buf = vec![0u8; (sw * sh * 4) as usize];
        for y in 0..sh {
            let on = (y as i32 - y0).abs() < 7;
            for x in 0..sw {
                let idx = ((y * sw + x) * 4) as usize;
                let v = if on { 235 } else { 6 };
                buf[idx] = v;
                buf[idx + 1] = v;
                buf[idx + 2] = v;
                buf[idx + 3] = 255;
            }
        }
        buf
    }

    /// At video field cadence, moving bright content leaves a *decaying phosphor
    /// trail* on the tube, and that trail is red-dominant — because the red P22
    /// phosphor sits a whole EIA persistence class above green and blue and lingers
    /// well over an order of magnitude longer (see the decay constants in
    /// `write_uniforms`). We prove it by rendering the same final field two ways:
    /// once fed as a moving band (real motion history) and once fed as a still at
    /// the final position (phosphor converged, no history). The difference is the
    /// melt, and it must lead in red.
    #[test]
    fn legacy_waterfall_melts() {
        let (device, queue) = headless_device();

        let (sw, sh) = (240u32, 180u32);
        let (ow, oh) = (400u32, 320u32);
        let dt = (1.0 / FIELD_HZ) as f32;
        let n = 20;
        let y_end = (sh as i32) - 30; // final band position (near the bottom)

        // Motion: band scrolls top → bottom, ending at y_end.
        let motion: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let y = 20 + (y_end - 20) * i / (n - 1);
                band(sw, sh, y)
            })
            .collect();
        // Static: the final position, held — phosphor converges, no trail.
        let still: Vec<Vec<u8>> = (0..n).map(|_| band(sw, sh, y_end)).collect();

        let m = render_sequence(&device, &queue, &motion, sw, sh, ow, oh, dt);
        let s = render_sequence(&device, &queue, &still, sw, sh, ow, oh, dt);

        // Per-channel mean absolute difference (the trail lives in the delta).
        let (mut dr, mut dg, mut db) = (0.0f64, 0.0f64, 0.0f64);
        let px = (ow * oh) as usize;
        for i in 0..px {
            dr += (m[i * 4] as f64 - s[i * 4] as f64).abs();
            dg += (m[i * 4 + 1] as f64 - s[i * 4 + 1] as f64).abs();
            db += (m[i * 4 + 2] as f64 - s[i * 4 + 2] as f64).abs();
        }
        dr /= px as f64;
        dg /= px as f64;
        db /= px as f64;
        eprintln!("melt trail mean|Δ| per channel — R:{dr:.3} G:{dg:.3} B:{db:.3}");

        // A trail exists at all (motion differs from the converged still).
        assert!(dr > 0.05, "no persistence trail: moving content did not melt (ΔR={dr:.3})");
        // And it's the red-lingering phosphor signature, not a symmetric spatial
        // blur: red must clearly lead green and blue.
        assert!(dr > 1.2 * dg, "trail not red-dominant (R={dr:.3} vs G={dg:.3})");
        assert!(dr > 1.2 * db, "trail not red-dominant (R={dr:.3} vs B={db:.3})");
    }

    #[test]
    fn multi_monitor_geometry_and_buffer_init_e2e() {
        let (device, queue) = headless_device();
        let (ow, oh) = (640u32, 480u32);
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let orbit = Orbit { yaw: 0.0, pitch: 0.0, distance: 1.45 };
        let dt = (1.0 / FIELD_HZ) as f32;
        let pwr = [1.0, 0.0, 0.0, 0.0];

        // 1. Prove buffer clean-initialization: make_phosphor_state zeroing
        let state_buffers = make_phosphor_state(&device, &queue, 32, 32);
        for buf in &state_buffers {
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zero-check"),
                size: 32 * 32 * 11 * 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buf, 0, &readback, 0, 32 * 32 * 11 * 16);
            queue.submit(Some(enc.finish()));
            let slice = readback.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range();
            assert!(data.iter().all(|&b| b == 0), "phosphor state buffer contained non-zero garbage on allocation");
        }

        // Helper to render one full frame through the complete 3D CRT pipeline
        let render_crt = |sw: u32, sh: u32, frame: &[u8]| -> Vec<u8> {
            let mut res = build_resources(&device, &queue, format, PVM);
            res.physical_phosphor = false;
            res.set_source(&device, &queue, sw, sh, format, frame);
            write_uniforms(
                &queue, &res, &orbit, ow as f32 / oh as f32, 0.0, &PVM, 1.0, false, dt, pwr,
                0.0, 0.0, 1.0, false, 1.0, false, false,
            );
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("accum") });
            accum_step(&mut enc, &mut res);
            queue.submit(Some(enc.finish()));
            device.poll(wgpu::Maintain::Wait);

            let color = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("e2e-color"),
                size: wgpu::Extent3d { width: ow, height: oh, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_view = create_depth(&device, ow, oh);
            let unpadded = ow * 4;
            let padded = ((unpadded + 255) / 256) * 256;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("e2e-readback"),
                size: (padded * oh) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("draw") });
            draw_tube(&mut enc, &res, &color_view, &depth_view);
            enc.copy_texture_to_buffer(
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
                        rows_per_image: Some(oh),
                    },
                },
                wgpu::Extent3d { width: ow, height: oh, depth_or_array_layers: 1 },
            );
            queue.submit(Some(enc.finish()));
            let slice = readback.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
            device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range();
            let mut px = Vec::with_capacity((ow * oh * 4) as usize);
            for row in 0..oh {
                let start = (row * padded) as usize;
                px.extend_from_slice(&data[start..start + unpadded as usize]);
            }
            px
        };

        let sample = |px: &[u8], x: u32, y: u32| -> u8 {
            let idx = ((y * ow + x) * 4) as usize;
            px[idx] // Red channel
        };

        // 2. Test 16:9 Landscape Source (e.g. DP-2 2560x1440 or HDMI-1 3840x2160)
        let white_16_9 = vec![255u8; 320 * 180 * 4];
        let px_16_9 = render_crt(320, 180, &white_16_9);
        let center_16_9 = sample(&px_16_9, 320, 240);
        let top_bar_16_9 = sample(&px_16_9, 320, 50);
        let bottom_bar_16_9 = sample(&px_16_9, 320, 430);
        eprintln!("16:9 - Center: {center_16_9}, Top bar: {top_bar_16_9}, Bottom bar: {bottom_bar_16_9}");
        assert!(center_16_9 > 80, "16:9 center not lit: {center_16_9}");
        assert!(top_bar_16_9 < 35, "16:9 top letterbox bar leaked light: {top_bar_16_9}");
        assert!(bottom_bar_16_9 < 35, "16:9 bottom letterbox bar leaked light: {bottom_bar_16_9}");

        // 3. Test 9:16 Portrait Source (e.g. DP-3 1440x2560)
        let white_9_16 = vec![255u8; 180 * 320 * 4];
        let px_9_16 = render_crt(180, 320, &white_9_16);
        let center_9_16 = sample(&px_9_16, 320, 240);
        let left_bar_9_16 = sample(&px_9_16, 100, 240);
        let right_bar_9_16 = sample(&px_9_16, 540, 240);
        eprintln!("9:16 - Center: {center_9_16}, Left bar: {left_bar_9_16}, Right bar: {right_bar_9_16}");
        assert!(center_9_16 > 80, "9:16 center not lit: {center_9_16}");
        assert!(left_bar_9_16 < 35, "9:16 left pillarbox bar leaked light: {left_bar_9_16}");
        assert!(right_bar_9_16 < 35, "9:16 right pillarbox bar leaked light: {right_bar_9_16}");

        // 4. Test 4:3 Native Source (e.g. 320x240 or 640x480)
        let white_4_3 = vec![255u8; 320 * 240 * 4];
        let px_4_3 = render_crt(320, 240, &white_4_3);
        let center_4_3 = sample(&px_4_3, 320, 240);
        let mid_top_4_3 = sample(&px_4_3, 320, 100);
        let mid_left_4_3 = sample(&px_4_3, 160, 240);
        eprintln!("4:3 - Center: {center_4_3}, Mid-top: {mid_top_4_3}, Mid-left: {mid_left_4_3}");
        assert!(center_4_3 > 80, "4:3 center not lit: {center_4_3}");
        assert!(mid_top_4_3 > 60, "4:3 top raster missing: {mid_top_4_3}");
        assert!(mid_left_4_3 > 60, "4:3 left raster missing: {mid_left_4_3}");

        // Confirm letterboxed/pillarboxed regions in non-4:3 are significantly dimmer than in 4:3
        assert!(sample(&px_16_9, 320, 75) < sample(&px_4_3, 320, 75) / 2, "16:9 vertical letterbox did not darken outer area relative to 4:3");
        assert!(sample(&px_9_16, 140, 240) < sample(&px_4_3, 140, 240) / 2, "9:16 horizontal pillarbox did not darken outer area relative to 4:3");
    }
}
