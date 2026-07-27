// A headless OpenGL context, so hardware-rendering libretro cores have somewhere to
// draw.
//
// The software path in `libretro.rs` covers everything that ever went out over an RF
// modulator, but the 3D machines — N64, PlayStation's hardware renderers, Dreamcast,
// PSP, Saturn — are built around a GPU and hand back a texture, not a framebuffer.
// libretro's contract for those is simple enough:
//
//   * we create a GL context and keep it current on the thread that calls retro_run;
//   * we hand the core `get_current_framebuffer()` (an FBO to render into) and
//     `get_proc_address()` (so it can resolve GL entry points);
//   * we call its `context_reset()` once the context exists;
//   * from then on `video_refresh` is called with RETRO_HW_FRAME_BUFFER_VALID rather
//     than a pixel pointer, meaning "it's in the FBO".
//
// EGL's surfaceless platform gives us a context with no window and no X/Wayland
// connection, which is exactly right for an offline renderer. We read the FBO back
// with glReadPixels and feed it into the same phosphor pipeline as everything else —
// a copy per frame, but at 320x240 or 640x480 it doesn't register next to the CRT
// shader, and it keeps the core's GL entirely separate from crtulum's Vulkan.

use std::ffi::{c_void, CString};

use anyhow::{anyhow, bail, Context, Result};
use khronos_egl as egl;

type Egl = egl::DynamicInstance<egl::EGL1_5>;

/// Mesa's windowless platform — no display server needed.
const PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31DD;

pub struct GlContext {
    egl: Egl,
    display: egl::Display,
    context: egl::Context,
    fbo: u32,
    color: u32,
    depth: u32,
    size: (u32, u32),
    /// Scratch for the readback, kept around so we're not reallocating every frame.
    scratch: Vec<u8>,
}

/// The raw `eglGetProcAddress`, which is what a core's `get_proc_address` callback
/// has to forward to. Taken straight from the library so it can live in a plain
/// function pointer (the EGL instance itself isn't `Send`, and the callback has no
/// context to hang state off).
pub type ProcAddressFn = unsafe extern "C" fn(*const std::ffi::c_char) -> *const c_void;

pub fn raw_get_proc_address() -> Result<ProcAddressFn> {
    // Leaked on purpose: the pointer has to stay valid for the life of the core.
    let lib = unsafe { libloading::Library::new("libEGL.so.1") }
        .context("loading libEGL.so.1 for eglGetProcAddress")?;
    let f: ProcAddressFn = *unsafe { lib.get(b"eglGetProcAddress\0") }
        .context("libEGL has no eglGetProcAddress")?;
    std::mem::forget(lib);
    Ok(f)
}

impl GlContext {
    /// `gles` picks the client API; `major`/`minor` and the depth/stencil flags come
    /// straight from the core's `retro_hw_render_callback`.
    pub fn new(
        width: u32,
        height: u32,
        want_depth: bool,
        want_stencil: bool,
        major: u32,
        minor: u32,
        gles: bool,
        core_profile: bool,
    ) -> Result<GlContext> {
        let egl = unsafe { Egl::load_required_from_filename("libEGL.so.1") }
            .map_err(|e| anyhow!("loading libEGL.so.1: {e}"))?;

        // Surfaceless: no window, no display server.
        let display = unsafe {
            egl.get_platform_display(
                PLATFORM_SURFACELESS_MESA,
                egl::DEFAULT_DISPLAY,
                &[egl::ATTRIB_NONE],
            )
        }
        .context(
            "no surfaceless EGL display — the GPU driver has to expose \
             EGL_MESA_platform_surfaceless for headless GL",
        )?;
        let (ver_major, ver_minor) = egl.initialize(display).context("eglInitialize")?;

        let api = if gles { egl::OPENGL_ES_API } else { egl::OPENGL_API };
        egl.bind_api(api).context("eglBindAPI")?;

        let renderable = if gles {
            // GLES3 contexts are requested through the ES2 renderable bit plus a
            // version attribute; there is no separate ES3 bit in the base spec.
            egl::OPENGL_ES2_BIT
        } else {
            egl::OPENGL_BIT
        };
        let config = egl
            .choose_first_config(
                display,
                &[
                    egl::SURFACE_TYPE,
                    egl::PBUFFER_BIT,
                    egl::RENDERABLE_TYPE,
                    renderable,
                    egl::RED_SIZE,
                    8,
                    egl::GREEN_SIZE,
                    8,
                    egl::BLUE_SIZE,
                    8,
                    egl::ALPHA_SIZE,
                    8,
                    egl::NONE,
                ],
            )
            .context("eglChooseConfig")?
            .ok_or_else(|| anyhow!("no EGL config with an RGBA8 pbuffer"))?;

        let mut ctx_attribs = vec![
            egl::CONTEXT_MAJOR_VERSION,
            major.max(1) as i32,
            egl::CONTEXT_MINOR_VERSION,
            minor as i32,
        ];
        if !gles && core_profile {
            ctx_attribs.push(egl::CONTEXT_OPENGL_PROFILE_MASK);
            ctx_attribs.push(egl::CONTEXT_OPENGL_CORE_PROFILE_BIT);
        }
        ctx_attribs.push(egl::NONE);

        let context = egl
            .create_context(display, config, None, &ctx_attribs)
            .with_context(|| {
                format!(
                    "creating a {} {major}.{minor}{} context",
                    if gles { "GLES" } else { "OpenGL" },
                    if core_profile { " core" } else { "" }
                )
            })?;

        // EGL_KHR_surfaceless_context: current with no draw/read surface at all.
        egl.make_current(display, None, None, Some(context))
            .context("eglMakeCurrent (surfaceless)")?;

        // Now that a context is current, GL entry points can be resolved.
        gl::load_with(|s| match egl.get_proc_address(s) {
            Some(f) => f as *const c_void,
            None => std::ptr::null(),
        });

        let mut ctx = GlContext {
            egl,
            display,
            context,
            fbo: 0,
            color: 0,
            depth: 0,
            size: (width.max(1), height.max(1)),
            scratch: Vec::new(),
        };
        ctx.build_framebuffer(want_depth, want_stencil)?;

        let renderer = unsafe {
            let p = gl::GetString(gl::RENDERER);
            if p.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(p as *const _).to_string_lossy().into_owned()
            }
        };
        eprintln!(
            "[emu] hardware rendering: EGL {ver_major}.{ver_minor} surfaceless · {renderer} · \
             {}x{} target",
            ctx.size.0, ctx.size.1
        );
        Ok(ctx)
    }

    fn build_framebuffer(&mut self, want_depth: bool, want_stencil: bool) -> Result<()> {
        let (w, h) = (self.size.0 as i32, self.size.1 as i32);
        unsafe {
            gl::GenTextures(1, &mut self.color);
            gl::BindTexture(gl::TEXTURE_2D, self.color);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA8 as i32,
                w,
                h,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

            gl::GenFramebuffers(1, &mut self.fbo);
            gl::BindFramebuffer(gl::FRAMEBUFFER, self.fbo);
            gl::FramebufferTexture2D(
                gl::FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                self.color,
                0,
            );

            // Cores ask for depth and stencil separately, but a combined buffer is
            // what the hardware wants and satisfies either request.
            if want_depth || want_stencil {
                gl::GenRenderbuffers(1, &mut self.depth);
                gl::BindRenderbuffer(gl::RENDERBUFFER, self.depth);
                gl::RenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH24_STENCIL8, w, h);
                gl::FramebufferRenderbuffer(
                    gl::FRAMEBUFFER,
                    gl::DEPTH_STENCIL_ATTACHMENT,
                    gl::RENDERBUFFER,
                    self.depth,
                );
            }

            let status = gl::CheckFramebufferStatus(gl::FRAMEBUFFER);
            if status != gl::FRAMEBUFFER_COMPLETE {
                bail!("the render target came out incomplete (glCheckFramebufferStatus = {status:#x})");
            }
            gl::Viewport(0, 0, w, h);
            gl::ClearColor(0.0, 0.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        }
        Ok(())
    }

    /// The FBO name handed to the core each frame.
    pub fn framebuffer(&self) -> u32 {
        self.fbo
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Resolve a GL entry point for the core.
    pub fn proc_address(&self, name: &str) -> *const c_void {
        match CString::new(name).ok().and_then(|_| self.egl.get_proc_address(name)) {
            Some(f) => f as *const c_void,
            None => std::ptr::null(),
        }
    }

    /// Read `w`x`h` of the FBO back as RGBA8.
    ///
    /// GL's origin is bottom-left and libretro frames are top-down, so this flips as
    /// it copies (which is what `bottom_left_origin` on the core's callback means).
    pub fn read_rgba(&mut self, w: u32, h: u32, flip: bool, out: &mut Vec<u8>) {
        let (w, h) = (w.min(self.size.0).max(1), h.min(self.size.1).max(1));
        let row = (w * 4) as usize;
        let bytes = row * h as usize;
        self.scratch.resize(bytes, 0);
        out.resize(bytes, 0);
        unsafe {
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.fbo);
            gl::PixelStorei(gl::PACK_ALIGNMENT, 1);
            gl::ReadPixels(
                0,
                0,
                w as i32,
                h as i32,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                self.scratch.as_mut_ptr() as *mut c_void,
            );
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, 0);
        }
        if flip {
            for y in 0..h as usize {
                let src = (h as usize - 1 - y) * row;
                out[y * row..y * row + row].copy_from_slice(&self.scratch[src..src + row]);
            }
        } else {
            out.copy_from_slice(&self.scratch);
        }
    }
}

impl Drop for GlContext {
    fn drop(&mut self) {
        unsafe {
            if self.fbo != 0 {
                gl::DeleteFramebuffers(1, &self.fbo);
            }
            if self.color != 0 {
                gl::DeleteTextures(1, &self.color);
            }
            if self.depth != 0 {
                gl::DeleteRenderbuffers(1, &self.depth);
            }
        }
        let _ = self.egl.make_current(self.display, None, None, None);
        let _ = self.egl.destroy_context(self.display, self.context);
        let _ = self.egl.terminate(self.display);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole hardware path rests on this: a windowless GL context, an FBO, and a
    /// readback that lands the right way up. Draw two known bands with the scissor
    /// box and check they come back where they should.
    #[test]
    fn headless_gl_renders_and_reads_back() {
        let mut ctx = match GlContext::new(64, 32, true, true, 3, 3, false, true) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no headless GL here ({e:#})");
                return;
            }
        };
        assert_ne!(ctx.framebuffer(), 0, "an FBO should have been created");

        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, ctx.framebuffer());
            gl::Disable(gl::SCISSOR_TEST);
            gl::ClearColor(0.0, 0.0, 1.0, 1.0); // all blue
            gl::Clear(gl::COLOR_BUFFER_BIT);
            // …then a red band across the BOTTOM half in GL's coordinates.
            gl::Enable(gl::SCISSOR_TEST);
            gl::Scissor(0, 0, 64, 16);
            gl::ClearColor(1.0, 0.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::Disable(gl::SCISSOR_TEST);
            gl::Finish();
        }

        let mut buf = Vec::new();
        ctx.read_rgba(64, 32, true, &mut buf);
        assert_eq!(buf.len(), 64 * 32 * 4);
        fn px(buf: &[u8], x: usize, y: usize) -> [u8; 3] {
            let o = (y * 64 + x) * 4;
            [buf[o], buf[o + 1], buf[o + 2]]
        }
        // Flipped to top-down, GL's bottom band must come back as the BOTTOM rows.
        assert_eq!(px(&buf, 32, 4), [0, 0, 255], "top of the image should be the blue clear");
        assert_eq!(px(&buf, 32, 28), [255, 0, 0], "bottom should be the red scissored band");

        // …and unflipped, it's the other way up.
        ctx.read_rgba(64, 32, false, &mut buf);
        assert_eq!(px(&buf, 32, 4), [255, 0, 0]);
        assert_eq!(px(&buf, 32, 28), [0, 0, 255]);
    }
}
