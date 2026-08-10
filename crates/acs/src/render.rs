//! Compositing animation frames into straight-alpha RGBA buffers.

use std::collections::HashMap;

use crate::types::{Character, Frame, IndexedImage, MouthShape};
use crate::Error;

/// Bounding box of the non-transparent pixels of a frame, as
/// `(left, top, right, bottom)` with the right and bottom edges exclusive.
pub type Bounds = (u32, u32, u32, u32);

/// A composited frame, 8 bits per channel, non-premultiplied.
#[derive(Debug, Clone)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    /// Extent of the drawn artwork. Character canvases are usually much taller
    /// than the art they hold, so callers that need to position things against
    /// the character (a word balloon, say) want this rather than the canvas.
    pub bounds: Option<Bounds>,
}

impl RgbaImage {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; (width as usize) * (height as usize) * 4],
            bounds: None,
        }
    }

    pub fn stride(&self) -> usize {
        self.width as usize * 4
    }

    /// Widens the recorded artwork extent to include the given pixel.
    fn include(&mut self, x: u32, y: u32) {
        self.bounds = Some(match self.bounds {
            None => (x, y, x + 1, y + 1),
            Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1)),
        });
    }
}

/// Caches decoded images, which are shared freely between frames.
#[derive(Default)]
pub struct ImageCache {
    images: HashMap<u32, IndexedImage>,
}

impl ImageCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get<'a>(
        &'a mut self,
        character: &Character,
        index: u32,
    ) -> Result<&'a IndexedImage, Error> {
        if let std::collections::hash_map::Entry::Vacant(slot) = self.images.entry(index) {
            slot.insert(character.image(index as usize)?);
        }
        Ok(&self.images[&index])
    }

    pub fn clear(&mut self) {
        self.images.clear();
    }
}

impl Character {
    /// Composites a frame onto a transparent canvas the size of the character.
    ///
    /// When `mouth` is set and the frame carries a matching overlay, the mouth
    /// image is substituted in, which is what drives lip sync during speech.
    pub fn render_frame(
        &self,
        frame: &Frame,
        mouth: Option<MouthShape>,
        cache: &mut ImageCache,
    ) -> Result<RgbaImage, Error> {
        let mut canvas = RgbaImage::new(self.info.width as u32, self.info.height as u32);

        // Frames rarely define all seven mouth shapes, so fall back to the
        // closest one by openness rather than dropping the overlay entirely.
        let overlay = mouth.and_then(|m| {
            frame
                .overlays
                .iter()
                .filter_map(|o| o.shape.map(|s| (s, o)))
                .min_by_key(|(s, _)| s.openness().abs_diff(m.openness()))
                .map(|(_, o)| o)
        });
        let replace_top = overlay.map(|o| o.replace_top_image).unwrap_or(false);

        // Images composite last-to-first, so entry 0 ends up on top.
        for (i, fi) in frame.images.iter().enumerate().rev() {
            // The overlay stands in for the frame's top-most image when asked.
            if replace_top && i == 0 {
                continue;
            }
            if let Ok(img) = cache.get(self, fi.image_index) {
                blit(
                    &mut canvas,
                    img,
                    fi.x as i32,
                    fi.y as i32,
                    &self.info.palette,
                    self.info.transparent_index,
                );
            }
        }

        if let Some(o) = overlay {
            if let Ok(img) = cache.get(self, o.image_index as u32) {
                blit(
                    &mut canvas,
                    img,
                    o.x as i32,
                    o.y as i32,
                    &self.info.palette,
                    self.info.transparent_index,
                );
            }
        }

        Ok(canvas)
    }
}

fn blit(
    canvas: &mut RgbaImage,
    img: &IndexedImage,
    ox: i32,
    oy: i32,
    palette: &[crate::types::Rgb],
    transparent: u8,
) {
    let stride = img.stride();
    for y in 0..img.height as i32 {
        let dy = oy + y;
        if dy < 0 || dy >= canvas.height as i32 {
            continue;
        }
        // Source rows are stored bottom-up.
        let src_row = (img.height as i32 - 1 - y) as usize * stride;
        for x in 0..img.width as i32 {
            let dx = ox + x;
            if dx < 0 || dx >= canvas.width as i32 {
                continue;
            }
            let Some(&idx) = img.pixels.get(src_row + x as usize) else {
                continue;
            };
            if idx == transparent {
                continue;
            }
            let Some(c) = palette.get(idx as usize) else {
                continue;
            };
            let o = (dy as usize * canvas.width as usize + dx as usize) * 4;
            canvas.data[o] = c.r;
            canvas.data[o + 1] = c.g;
            canvas.data[o + 2] = c.b;
            canvas.data[o + 3] = 0xFF;
            canvas.include(dx as u32, dy as u32);
        }
    }
}
