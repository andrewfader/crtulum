//! Little-endian cursor over the file image, with the primitive readers the
//! ACS structures are built from.

use crate::Error;

pub struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn at(data: &'a [u8], pos: usize) -> Result<Self, Error> {
        if pos > data.len() {
            return Err(Error::Parse(format!(
                "offset {} past end of file ({} bytes)",
                pos,
                data.len()
            )));
        }
        Ok(Self { data, pos })
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn seek(&mut self, pos: usize) -> Result<(), Error> {
        if pos > self.data.len() {
            return Err(Error::Parse(format!("seek past end of file: {}", pos)));
        }
        self.pos = pos;
        Ok(())
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::Parse("length overflow".into()))?;
        if end > self.data.len() {
            return Err(Error::Parse(format!(
                "unexpected end of file: wanted {} bytes at {}, file is {} bytes",
                n,
                self.pos,
                self.data.len()
            )));
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    pub fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    pub fn bool(&mut self) -> Result<bool, Error> {
        Ok(self.u8()? != 0)
    }

    pub fn u16(&mut self) -> Result<u16, Error> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn i16(&mut self) -> Result<i16, Error> {
        Ok(self.u16()? as i16)
    }

    pub fn u32(&mut self) -> Result<u32, Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn i32(&mut self) -> Result<i32, Error> {
        Ok(self.u32()? as i32)
    }

    pub fn bytes(&mut self, n: usize) -> Result<Vec<u8>, Error> {
        Ok(self.take(n)?.to_vec())
    }

    pub fn skip(&mut self, n: usize) -> Result<(), Error> {
        self.take(n)?;
        Ok(())
    }

    /// A STRING: a ULONG character count followed by that many UTF-16 code
    /// units plus a NUL terminator (the terminator is absent when the count is
    /// zero).
    pub fn string(&mut self) -> Result<String, Error> {
        let count = self.u32()? as usize;
        if count == 0 {
            return Ok(String::new());
        }
        let units = self.take((count + 1) * 2)?;
        let utf16: Vec<u16> = units[..count * 2]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(String::from_utf16_lossy(&utf16))
    }

    /// An ACSLOCATOR: absolute byte offset plus size.
    pub fn locator(&mut self) -> Result<Locator, Error> {
        Ok(Locator {
            offset: self.u32()? as usize,
            size: self.u32()? as usize,
        })
    }

    /// A DATABLOCK: a ULONG size followed by that many bytes.
    pub fn datablock(&mut self) -> Result<Vec<u8>, Error> {
        let n = self.u32()? as usize;
        self.bytes(n)
    }

    /// A COMPRESSED block: compressed size, uncompressed size, then the data. A
    /// compressed size of zero means the payload is stored verbatim.
    pub fn compressed(&mut self) -> Result<Vec<u8>, Error> {
        let csize = self.u32()? as usize;
        let usize_ = self.u32()? as usize;
        if csize == 0 {
            return self.bytes(usize_);
        }
        let raw = self.take(csize)?;
        crate::decompress::decompress(raw, usize_)
    }

    /// Reads a length-prefixed list, where `count` reads the prefix.
    pub fn list<T, C, F>(&mut self, count: C, mut item: F) -> Result<Vec<T>, Error>
    where
        C: FnOnce(&mut Self) -> Result<usize, Error>,
        F: FnMut(&mut Self) -> Result<T, Error>,
    {
        let n = count(self)?;
        let mut v = Vec::with_capacity(n.min(4096));
        for _ in 0..n {
            v.push(item(self)?);
        }
        Ok(v)
    }

    pub fn count_u8(&mut self) -> Result<usize, Error> {
        Ok(self.u8()? as usize)
    }
    pub fn count_u16(&mut self) -> Result<usize, Error> {
        Ok(self.u16()? as usize)
    }
    pub fn count_u32(&mut self) -> Result<usize, Error> {
        Ok(self.u32()? as usize)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Locator {
    pub offset: usize,
    pub size: usize,
}
