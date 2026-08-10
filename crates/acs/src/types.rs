//! ACS version 2 structure definitions and parsing.

use crate::reader::{Cursor, Locator};
use crate::Error;

pub const ACS_SIGNATURE: u32 = 0xABCD_ABC3;
pub const ACF_SIGNATURE: u32 = 0xABCD_ABC4;
/// Leading bytes of an OLE compound document, used by MSAgent 1.5 characters.
pub const OLE_SIGNATURE: u32 = 0xE011_CFD0;

/// Voice output enabled bit within the character flags.
const FLAG_VOICE: u32 = 0x0000_0020;
/// Word balloon enabled bits.
const FLAG_BALLOON: u32 = 0x0000_0300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Clone)]
pub struct LocalizedInfo {
    pub language_id: u16,
    pub name: String,
    pub description: String,
    pub extra: String,
}

#[derive(Debug, Clone)]
pub struct VoiceInfo {
    pub speed: u32,
    pub pitch: u16,
    pub language_id: Option<u16>,
    pub dialect: Option<String>,
    /// 1 = female, 2 = male in the SAPI 4 convention these files were authored against.
    pub gender: Option<u16>,
    pub age: Option<u16>,
    pub style: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BalloonInfo {
    pub lines: u8,
    pub chars_per_line: u8,
    pub foreground: Rgb,
    pub background: Rgb,
    pub border: Rgb,
    pub font_name: String,
    pub font_height: i32,
    pub font_weight: i32,
    pub italic: bool,
}

#[derive(Debug, Clone)]
pub struct StateInfo {
    pub name: String,
    pub animations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CharacterInfo {
    pub major_version: u16,
    pub minor_version: u16,
    pub guid: [u8; 16],
    pub width: u16,
    pub height: u16,
    pub transparent_index: u8,
    pub flags: u32,
    pub voice: Option<VoiceInfo>,
    pub balloon: Option<BalloonInfo>,
    pub palette: Vec<Rgb>,
    pub states: Vec<StateInfo>,
    pub localized: Vec<LocalizedInfo>,
}

/// Windows primary language identifier for English.
const LANG_ENGLISH: u16 = 0x09;

/// Maps a POSIX locale such as `de_DE.UTF-8` to a Windows primary language id.
pub fn primary_language_from_locale(locale: &str) -> Option<u16> {
    let code = locale
        .split(['_', '-', '.', '@'])
        .next()?
        .to_ascii_lowercase();
    Some(match code.as_str() {
        "ar" => 0x01,
        "zh" => 0x04,
        "cs" => 0x05,
        "da" => 0x06,
        "de" => 0x07,
        "el" => 0x08,
        "en" => 0x09,
        "es" => 0x0A,
        "fi" => 0x0B,
        "fr" => 0x0C,
        "he" => 0x0D,
        "hu" => 0x0E,
        "it" => 0x10,
        "ja" => 0x11,
        "ko" => 0x12,
        "nl" => 0x13,
        "no" | "nb" | "nn" => 0x14,
        "pl" => 0x15,
        "pt" => 0x16,
        "ru" => 0x19,
        "hr" => 0x1A,
        "sk" => 0x1B,
        "sv" => 0x1D,
        "th" => 0x1E,
        "tr" => 0x1F,
        "sl" => 0x24,
        _ => return None,
    })
}

impl CharacterInfo {
    /// Picks the localized entry matching `primary` (a Windows primary language
    /// id), falling back to English and then to the first usable entry. Many
    /// characters ship twenty or more locales in arbitrary order, so taking the
    /// first one blindly gives surprising results.
    pub fn localized_preferred(&self, primary: Option<u16>) -> Option<&LocalizedInfo> {
        let by_primary = |p: u16| self.localized.iter().find(|l| l.language_id & 0x3FF == p);
        primary
            .and_then(by_primary)
            .or_else(|| by_primary(LANG_ENGLISH))
            .or_else(|| self.localized.iter().find(|l| !l.name.is_empty()))
            .or_else(|| self.localized.first())
    }

    pub fn name_for(&self, primary: Option<u16>) -> Option<&str> {
        self.localized_preferred(primary)
            .map(|l| l.name.as_str())
            .filter(|s| !s.is_empty())
    }

    pub fn description_for(&self, primary: Option<u16>) -> Option<&str> {
        self.localized_preferred(primary)
            .map(|l| l.description.as_str())
            .filter(|s| !s.is_empty())
    }

    pub fn name(&self) -> Option<&str> {
        self.name_for(None)
    }

    pub fn description(&self) -> Option<&str> {
        self.description_for(None)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrameImage {
    pub image_index: u32,
    pub x: i16,
    pub y: i16,
}

#[derive(Debug, Clone, Copy)]
pub struct Branch {
    pub frame_index: u16,
    pub probability: u16,
}

/// Mouth shapes a frame can substitute in while the character speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouthShape {
    Closed,
    WideOpen1,
    WideOpen2,
    WideOpen3,
    WideOpen4,
    Medium,
    Narrow,
}

impl MouthShape {
    /// How far open the mouth is, used to substitute the closest available
    /// shape when a frame does not define every overlay.
    pub fn openness(self) -> u8 {
        match self {
            Self::Closed => 0,
            Self::Narrow => 1,
            Self::Medium => 2,
            Self::WideOpen1 => 3,
            Self::WideOpen2 => 4,
            Self::WideOpen3 => 5,
            Self::WideOpen4 => 6,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Closed,
            1 => Self::WideOpen1,
            2 => Self::WideOpen2,
            3 => Self::WideOpen3,
            4 => Self::WideOpen4,
            5 => Self::Medium,
            6 => Self::Narrow,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Overlay {
    pub shape: Option<MouthShape>,
    pub replace_top_image: bool,
    pub image_index: u16,
    pub x: i16,
    pub y: i16,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub images: Vec<FrameImage>,
    /// Index into the audio table, or `None` when the frame is silent.
    pub audio_index: Option<u16>,
    /// Frame duration in hundredths of a second.
    pub duration: u16,
    pub exit_frame: i16,
    pub branches: Vec<Branch>,
    pub overlays: Vec<Overlay>,
}

impl Frame {
    pub fn duration_ms(&self) -> u64 {
        self.duration as u64 * 10
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    ReturnAnimation,
    ExitBranches,
    None,
}

#[derive(Debug, Clone)]
pub struct Animation {
    pub name: String,
    pub transition: Transition,
    pub return_animation: String,
    pub frames: Vec<Frame>,
}

impl Animation {
    /// Total run time in milliseconds if every frame plays once in order.
    pub fn duration_ms(&self) -> u64 {
        self.frames.iter().map(|f| f.duration_ms()).sum()
    }
}

/// A decoded 8-bit indexed image, stored bottom-up exactly as in the file.
#[derive(Debug, Clone)]
pub struct IndexedImage {
    pub width: u16,
    pub height: u16,
    pub pixels: Vec<u8>,
}

impl IndexedImage {
    pub fn stride(&self) -> usize {
        stride_for(self.width)
    }

    /// Palette index at the given top-left-origin coordinate.
    #[inline]
    pub fn index_at(&self, x: u32, y: u32) -> Option<u8> {
        if x >= self.width as u32 || y >= self.height as u32 {
            return None;
        }
        // Rows are stored bottom-up.
        let row = (self.height as u32 - 1 - y) as usize;
        self.pixels.get(row * self.stride() + x as usize).copied()
    }
}

/// Row stride of an 8bpp DIB: each row is padded to a 4-byte boundary.
pub fn stride_for(width: u16) -> usize {
    ((width as usize) + 3) & !3
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn read_rgbquad(c: &mut Cursor) -> Result<Rgb, Error> {
    // Stored blue first, matching the Windows RGBQUAD layout.
    let b = c.u8()?;
    let g = c.u8()?;
    let r = c.u8()?;
    c.u8()?; // reserved
    Ok(Rgb { r, g, b })
}

fn read_localized(c: &mut Cursor) -> Result<LocalizedInfo, Error> {
    Ok(LocalizedInfo {
        language_id: c.u16()?,
        name: c.string()?,
        description: c.string()?,
        extra: c.string()?,
    })
}

fn read_voice(c: &mut Cursor) -> Result<VoiceInfo, Error> {
    c.skip(16)?; // TTS engine GUID
    c.skip(16)?; // TTS mode GUID
    let speed = c.u32()?;
    let pitch = c.u16()?;
    let has_extra = c.bool()?;
    let mut v = VoiceInfo {
        speed,
        pitch,
        language_id: None,
        dialect: None,
        gender: None,
        age: None,
        style: None,
    };
    if has_extra {
        v.language_id = Some(c.u16()?);
        v.dialect = Some(c.string()?);
        v.gender = Some(c.u16()?);
        v.age = Some(c.u16()?);
        v.style = Some(c.string()?);
    }
    Ok(v)
}

fn read_balloon(c: &mut Cursor) -> Result<BalloonInfo, Error> {
    let lines = c.u8()?;
    let chars_per_line = c.u8()?;
    let foreground = read_rgbquad(c)?;
    let background = read_rgbquad(c)?;
    let border = read_rgbquad(c)?;
    let font_name = c.string()?;
    let font_height = c.i32()?;
    let font_weight = c.i32()?;
    let italic = c.bool()?;
    c.u8()?; // trailing unknown byte, observed as zero
    Ok(BalloonInfo {
        lines,
        chars_per_line,
        foreground,
        background,
        border,
        font_name,
        font_height,
        font_weight,
        italic,
    })
}

fn read_state(c: &mut Cursor) -> Result<StateInfo, Error> {
    let name = c.string()?;
    let animations = c.list(Cursor::count_u16, |c| c.string())?;
    Ok(StateInfo { name, animations })
}

/// Parses ACSCHARACTERINFO assuming the given optional sub-structures are
/// present. Whether VOICEINFO/BALLOONINFO appear is driven by the flags word,
/// and guessing wrong desynchronises everything after it, so the caller retries
/// other combinations if this yields implausible results.
fn read_character_info_with(
    data: &[u8],
    offset: usize,
    want_voice: bool,
    want_balloon: bool,
) -> Result<CharacterInfo, Error> {
    let mut c = Cursor::at(data, offset)?;
    let minor_version = c.u16()?;
    let major_version = c.u16()?;
    let localized_loc = c.locator()?;
    let mut guid = [0u8; 16];
    guid.copy_from_slice(&c.bytes(16)?);
    let width = c.u16()?;
    let height = c.u16()?;
    let transparent_index = c.u8()?;
    let flags = c.u32()?;
    let _anim_set_major = c.u16()?;
    let _anim_set_minor = c.u16()?;

    let voice = if want_voice {
        Some(read_voice(&mut c)?)
    } else {
        None
    };
    let balloon = if want_balloon {
        Some(read_balloon(&mut c)?)
    } else {
        None
    };

    let palette_count = c.count_u32()?;
    if palette_count == 0 || palette_count > 256 {
        return Err(Error::Parse(format!(
            "implausible palette size {}",
            palette_count
        )));
    }
    let mut palette = Vec::with_capacity(palette_count);
    for _ in 0..palette_count {
        palette.push(read_rgbquad(&mut c)?);
    }

    if c.bool()? {
        c.datablock()?; // monochrome tray icon bitmap
        c.datablock()?; // colour tray icon bitmap
    }

    let states = c.list(Cursor::count_u16, read_state)?;

    let mut localized = Vec::new();
    if localized_loc.offset != 0 && localized_loc.offset < data.len() {
        if let Ok(mut lc) = Cursor::at(data, localized_loc.offset) {
            localized = lc
                .list(Cursor::count_u16, read_localized)
                .unwrap_or_default();
        }
    }

    Ok(CharacterInfo {
        major_version,
        minor_version,
        guid,
        width,
        height,
        transparent_index,
        flags,
        voice,
        balloon,
        palette,
        states,
        localized,
    })
}

fn read_character_info(data: &[u8], offset: usize) -> Result<CharacterInfo, Error> {
    // Peek the flags to pick the most likely layout, then fall back.
    let flags = Cursor::at(data, offset)
        .and_then(|mut c| {
            c.skip(2 + 2 + 8 + 16 + 2 + 2 + 1)?;
            c.u32()
        })
        .unwrap_or(0);

    let preferred = (flags & FLAG_VOICE != 0, flags & FLAG_BALLOON != 0);
    let mut attempts = vec![preferred];
    for combo in [(true, true), (false, true), (true, false), (false, false)] {
        if combo != preferred {
            attempts.push(combo);
        }
    }

    let mut last = None;
    for (voice, balloon) in attempts {
        match read_character_info_with(data, offset, voice, balloon) {
            Ok(info) if !info.states.is_empty() || !info.palette.is_empty() => return Ok(info),
            Ok(info) => last = Some(Ok(info)),
            Err(e) => {
                if last.is_none() {
                    last = Some(Err(e));
                }
            }
        }
    }
    last.unwrap_or_else(|| Err(Error::Parse("could not parse character info".into())))
}

fn read_frame(c: &mut Cursor) -> Result<Frame, Error> {
    let images = c.list(Cursor::count_u16, |c| {
        Ok(FrameImage {
            image_index: c.u32()?,
            x: c.i16()?,
            y: c.i16()?,
        })
    })?;
    let audio_raw = c.u16()?;
    let duration = c.u16()?;
    let exit_frame = c.i16()?;
    let branches = c.list(Cursor::count_u8, |c| {
        Ok(Branch {
            frame_index: c.u16()?,
            probability: c.u16()?,
        })
    })?;
    let overlays = c.list(Cursor::count_u8, |c| {
        let shape = MouthShape::from_u8(c.u8()?);
        let replace_top_image = c.bool()?;
        let image_index = c.u16()?;
        let _unknown = c.u8()?;
        let has_region = c.bool()?;
        let x = c.i16()?;
        let y = c.i16()?;
        let _w = c.u16()?;
        let _h = c.u16()?;
        if has_region {
            c.datablock()?;
        }
        Ok(Overlay {
            shape,
            replace_top_image,
            image_index,
            x,
            y,
        })
    })?;

    Ok(Frame {
        images,
        // 0xFFFF marks "no sound" rather than audio entry 65535.
        audio_index: if audio_raw == u16::MAX {
            None
        } else {
            Some(audio_raw)
        },
        duration,
        exit_frame,
        branches,
        overlays,
    })
}

fn read_animation(data: &[u8], name: String, loc: Locator) -> Result<Animation, Error> {
    let mut c = Cursor::at(data, loc.offset)?;
    let _upper_name = c.string()?;
    let transition = match c.u8()? {
        0 => Transition::ReturnAnimation,
        1 => Transition::ExitBranches,
        _ => Transition::None,
    };
    let return_animation = c.string()?;
    let frames = c.list(Cursor::count_u16, read_frame)?;
    Ok(Animation {
        name,
        transition,
        return_animation,
        frames,
    })
}

/// A parsed character file. Image and audio payloads stay in the backing buffer
/// and are decoded on demand.
pub struct Character {
    data: Vec<u8>,
    pub info: CharacterInfo,
    pub animations: Vec<Animation>,
    image_locs: Vec<Locator>,
    audio_locs: Vec<Locator>,
}

impl Character {
    pub fn parse(data: Vec<u8>) -> Result<Self, Error> {
        let mut c = Cursor::new(&data);
        let signature = c.u32()?;
        match signature {
            ACS_SIGNATURE => {}
            ACF_SIGNATURE => {
                return Err(Error::Unsupported(
                    "this is an .acf file; its animations live in separate .aca files".into(),
                ))
            }
            OLE_SIGNATURE => {
                return Err(Error::Unsupported(
                    "this is a Microsoft Agent 1.5 character (OLE compound file), \
                     which uses a different layout than version 2"
                        .into(),
                ))
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "not an ACS file (signature 0x{:08X})",
                    other
                )))
            }
        }

        let character_loc = c.locator()?;
        let animation_loc = c.locator()?;
        let image_loc = c.locator()?;
        let audio_loc = c.locator()?;

        let info = read_character_info(&data, character_loc.offset)?;

        let mut ac = Cursor::at(&data, animation_loc.offset)?;
        let anim_refs = ac.list(Cursor::count_u32, |c| {
            let name = c.string()?;
            let loc = c.locator()?;
            Ok((name, loc))
        })?;
        let mut animations = Vec::with_capacity(anim_refs.len());
        for (name, loc) in anim_refs {
            // A single corrupt animation should not sink the whole character.
            match read_animation(&data, name.clone(), loc) {
                Ok(a) => animations.push(a),
                Err(_) => animations.push(Animation {
                    name,
                    transition: Transition::None,
                    return_animation: String::new(),
                    frames: Vec::new(),
                }),
            }
        }

        let mut ic = Cursor::at(&data, image_loc.offset)?;
        let image_locs = ic.list(Cursor::count_u32, |c| {
            let loc = c.locator()?;
            let _checksum = c.u32()?;
            Ok(loc)
        })?;

        let mut auc = Cursor::at(&data, audio_loc.offset)?;
        let audio_locs = auc.list(Cursor::count_u32, |c| {
            let loc = c.locator()?;
            let _checksum = c.u32()?;
            Ok(loc)
        })?;

        Ok(Character {
            data,
            info,
            animations,
            image_locs,
            audio_locs,
        })
    }

    pub fn image_count(&self) -> usize {
        self.image_locs.len()
    }

    pub fn audio_count(&self) -> usize {
        self.audio_locs.len()
    }

    /// Decodes one image from the image table.
    pub fn image(&self, index: usize) -> Result<IndexedImage, Error> {
        let loc = *self
            .image_locs
            .get(index)
            .ok_or_else(|| Error::Parse(format!("image index {} out of range", index)))?;
        let mut c = Cursor::at(&self.data, loc.offset)?;
        let _unknown = c.u8()?;
        let width = c.u16()?;
        let height = c.u16()?;
        let compressed = c.bool()?;

        let expected = stride_for(width) * height as usize;
        let raw = c.datablock()?;

        let pixels = if compressed {
            match crate::decompress::decompress(&raw, expected) {
                Ok(p) => p,
                // Some third-party characters set the flag but store the bits
                // verbatim; accept that when the size already matches.
                Err(_) if raw.len() >= expected => raw,
                Err(e) => return Err(e),
            }
        } else {
            raw
        };

        Ok(IndexedImage {
            width,
            height,
            pixels,
        })
    }

    /// Raw RIFF/WAVE bytes for an audio table entry.
    pub fn audio(&self, index: usize) -> Option<&[u8]> {
        let loc = self.audio_locs.get(index)?;
        self.data.get(loc.offset..loc.offset + loc.size)
    }

    pub fn animation_by_name(&self, name: &str) -> Option<&Animation> {
        self.animations
            .iter()
            .find(|a| a.name.eq_ignore_ascii_case(name))
    }
}
