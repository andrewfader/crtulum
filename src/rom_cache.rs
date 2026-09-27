//! Lossless, frame-addressed libretro recordings. A cache entry is published only
//! after every frame and its PCM have been written successfully.
use std::{fs::{self, File}, io::{BufReader, BufWriter, Read, Write}, path::{Path, PathBuf}, hash::{Hash, Hasher}};
use anyhow::{bail, Context, Result};

pub struct Recording {
    path: PathBuf,
    pending: PathBuf,
    reader: Option<BufReader<File>>,
    writer: Option<BufWriter<File>>,
}

pub fn key(rom: &Path, core: &Path, system: &Path, fallback: &Path, options: &[(String, String)], inputs: &str) -> Result<String> {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    "crtulum-rom-recording-v1".hash(&mut hash);
    include_str!("libretro.rs").hash(&mut hash);
    options.hash(&mut hash);
    inputs.hash(&mut hash);
    fingerprint(rom, &mut hash, 0)?;
    fingerprint(core, &mut hash, 0)?;
    // The work directory is also the no-BIOS fallback; recordings themselves
    // must never enter their own key. Real system directories include BIOS data.
    if system != fallback && system.is_dir() { fingerprint(system, &mut hash, 0)?; }
    Ok(format!("{:016x}", hash.finish()))
}

fn fingerprint(path: &Path, hash: &mut impl Hasher, depth: usize) -> Result<()> {
    if depth > 16 { bail!("recursive media dependency at {}", path.display()); }
    path.canonicalize()?.hash(hash);
    if path.is_dir() {
        let mut entries = fs::read_dir(path)?.map(|e| e.map(|e| e.path())).collect::<std::io::Result<Vec<_>>>()?;
        entries.sort();
        for entry in entries { fingerprint(&entry, hash, depth + 1)?; }
    } else {
        let mut file = File::open(path)?;
        let mut buffer = [0; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 { break; }
            hash.write(&buffer[..n]);
        }
        match path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
            "cue" | "m3u" => {
                let cue = path.extension().unwrap().eq_ignore_ascii_case("cue");
                for line in fs::read_to_string(path)?.lines() {
                    let line = line.trim();
                    let dependency = if cue {
                        if !line.get(..5).is_some_and(|s| s.eq_ignore_ascii_case("FILE ")) { continue; }
                        let rest = line[5..].trim();
                        if let Some(rest) = rest.strip_prefix('"') { rest.split('"').next().unwrap_or("") }
                        else { rest.split_whitespace().next().unwrap_or("") }
                    } else {
                        if line.is_empty() || line.starts_with('#') { continue; }
                        line
                    };
                    fingerprint(&path.parent().unwrap_or(Path::new(".")).join(dependency), hash, depth + 1)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

impl Recording {
    pub fn open(work: &Path, key: &str, frames: u64) -> Result<Self> {
        let path = work.join(format!("rom-{key}-{frames}.bin"));
        let pending = work.join(format!("rom-{key}-{frames}-{}.partial", std::process::id()));
        if let Ok(file) = File::open(&path) {
            let mut reader = BufReader::new(file);
            let mut header = [0; 16];
            reader.read_exact(&mut header).context("reading ROM recording header")?;
            if &header[..8] != b"CRTROM01" || u64::from_le_bytes(header[8..].try_into().unwrap()) != frames {
                bail!("invalid ROM recording {}; remove it to record again", path.display());
            }
            eprintln!("[emu] reusing cached recording {}", path.display());
            return Ok(Self { path, pending, reader: Some(reader), writer: None });
        }
        let mut writer = BufWriter::new(File::create(&pending)?);
        writer.write_all(b"CRTROM01")?;
        writer.write_all(&frames.to_le_bytes())?;
        Ok(Self { path, pending, reader: None, writer: Some(writer) })
    }

    pub fn read(&mut self) -> Result<Option<(Vec<u8>, u32, u32, Vec<i16>)>> {
        let Some(reader) = &mut self.reader else { return Ok(None); };
        let mut header = [0; 12];
        reader.read_exact(&mut header).context("truncated ROM recording")?;
        let w = u32::from_le_bytes(header[..4].try_into().unwrap());
        let h = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let samples = u32::from_le_bytes(header[8..].try_into().unwrap()) as usize;
        if w == 0 || h == 0 || w > 8192 || h > 8192 || samples > 2_000_000 || samples % 2 != 0 {
            bail!("invalid frame dimensions/audio in ROM recording");
        }
        let mut pixels = vec![0; w as usize * h as usize * 4];
        reader.read_exact(&mut pixels)?;
        let mut bytes = vec![0; samples * 2];
        reader.read_exact(&mut bytes)?;
        let pcm = bytes.chunks_exact(2).map(|v| i16::from_le_bytes([v[0], v[1]])).collect();
        Ok(Some((pixels, w, h, pcm)))
    }

    pub fn write(&mut self, pixels: &[u8], w: u32, h: u32, pcm: &[i16]) -> Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.write_all(&w.to_le_bytes())?;
            writer.write_all(&h.to_le_bytes())?;
            writer.write_all(&(pcm.len() as u32).to_le_bytes())?;
            writer.write_all(pixels)?;
            for sample in pcm { writer.write_all(&sample.to_le_bytes())?; }
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
            writer.get_ref().sync_all()?;
            drop(writer);
            fs::rename(&self.pending, &self.path)?;
            eprintln!("[emu] cached recording {}", self.path.display());
        }
        Ok(())
    }
}

impl Drop for Recording {
    fn drop(&mut self) { let _ = fs::remove_file(&self.pending); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recordings_are_atomic_and_preserve_native_modes_and_pcm() {
        let dir = std::env::temp_dir().join(format!("crtulum-recording-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        {
            let mut interrupted = Recording::open(&dir, "interrupted", 2).unwrap();
            interrupted.write(&[1; 8], 2, 1, &[123, -456]).unwrap();
        }
        assert!(!dir.join("rom-interrupted-2.bin").exists());
        let mut recording = Recording::open(&dir, "complete", 2).unwrap();
        recording.write(&[1; 8], 2, 1, &[123, -456]).unwrap();
        recording.write(&[2; 12], 1, 3, &[]).unwrap();
        recording.finish().unwrap();
        let mut replay = Recording::open(&dir, "complete", 2).unwrap();
        assert_eq!(replay.read().unwrap().unwrap(), (vec![1; 8], 2, 1, vec![123, -456]));
        assert_eq!(replay.read().unwrap().unwrap(), (vec![2; 12], 1, 3, vec![]));
        assert!(replay.read().is_err());
        drop(recording); drop(replay);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cache_key_tracks_content_disc_tracks_bios_options_and_input() {
        let dir = std::env::temp_dir().join(format!("crtulum-cache-key-{}", std::process::id()));
        fs::create_dir_all(dir.join("bios")).unwrap();
        let rom = dir.join("disc.cue"); let core = dir.join("core.so");
        fs::write(&rom, "FILE \"track.bin\" BINARY\n").unwrap();
        fs::write(dir.join("track.bin"), "track one").unwrap();
        fs::write(&core, "core one").unwrap();
        fs::write(dir.join("bios/bios.bin"), "bios one").unwrap();
        let k = |options: &[(String, String)], input: &str| key(&rom, &core, &dir.join("bios"), &dir, options, input).unwrap();
        let initial = k(&[], "input");
        assert_ne!(initial, k(&[], "other input"));
        assert_ne!(initial, k(&[("renderer".into(), "hardware".into())], "input"));
        for path in [dir.join("track.bin"), core.clone(), dir.join("bios/bios.bin")] {
            let before = k(&[], "input");
            fs::write(path, "changed!").unwrap();
            assert_ne!(before, k(&[], "input"));
        }
        fs::remove_dir_all(dir).unwrap();
    }
}
