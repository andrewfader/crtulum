//! Reader for Microsoft Agent version 2 character files (`.acs`).
//!
//! Layout follows the MSAgent Character Data Specification: a header of
//! locators pointing at the character description, animation table, image table
//! and audio table, with most payloads run through a proprietary bit-level
//! compressor.

pub mod decompress;
pub mod reader;
pub mod render;
pub mod types;

pub use render::{ImageCache, RgbaImage};
pub use types::{
    Animation, BalloonInfo, Branch, Character, CharacterInfo, Frame, FrameImage, IndexedImage,
    LocalizedInfo, MouthShape, Overlay, Rgb, StateInfo, Transition, VoiceInfo,
};

use std::fmt;
use std::path::Path;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Parse(String),
    Compression(String),
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{}", e),
            Error::Parse(m) => write!(f, "malformed character file: {}", m),
            Error::Compression(m) => write!(f, "decompression failed: {}", m),
            Error::Unsupported(m) => write!(f, "unsupported file: {}", m),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Reads and parses a character file from disk.
pub fn load<P: AsRef<Path>>(path: P) -> Result<Character, Error> {
    let data = std::fs::read(path)?;
    Character::parse(data)
}
