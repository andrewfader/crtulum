//! The proprietary LZ-style bit-level compressor used by Microsoft Agent.
//!
//! Layout of a compressed block: one `0x00` byte, the bit stream, then padding
//! `0xFF` bytes. Bits are consumed least-significant-bit first within each byte,
//! and multi-bit values are assembled with the first bit read as bit 0.
//!
//! Each sequence starts with a type bit: `0` introduces one literal byte, `1` a
//! back-reference. A back-reference encodes its offset width as a unary run of
//! up to three 1 bits (6/9/12/20 bits wide, with a bias added), then the length
//! as a unary run of N ones (contributing `2^N - 1`) followed by N more bits.
//!
//! The end-of-stream marker is a 20-bit offset field of all ones, which is why
//! the trailing `0xFF` bytes belong to the bit stream rather than being padding
//! to strip off.

use crate::Error;

struct BitReader<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit: 0 }
    }

    #[inline]
    fn read_bit(&mut self) -> Option<u32> {
        let byte = *self.data.get(self.bit >> 3)?;
        let v = (byte >> (self.bit & 7)) & 1;
        self.bit += 1;
        Some(v as u32)
    }

    /// Reads `n` bits, first bit read becomes the least significant bit.
    #[inline]
    fn read_bits(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for i in 0..n {
            v |= self.read_bit()? << i;
        }
        Some(v)
    }

    /// Counts consecutive 1 bits, stopping after `max`. The terminating 0 bit is
    /// consumed; if the run reaches `max` no further bit is read.
    #[inline]
    fn read_unary(&mut self, max: u32) -> Option<u32> {
        let mut n = 0;
        while n < max {
            if self.read_bit()? == 1 {
                n += 1;
            } else {
                return Some(n);
            }
        }
        Some(n)
    }
}

/// Offset field width and the bias added to the decoded value, indexed by the
/// length of the leading unary run.
const OFFSET_CLASS: [(u32, u32); 4] = [(6, 0x0001), (9, 0x0041), (12, 0x0241), (20, 0x1241)];

const END_MARKER: u32 = 0x000F_FFFF;

/// Decompresses an MSAgent compressed block into exactly `uncompressed_size`
/// bytes.
pub fn decompress(data: &[u8], uncompressed_size: usize) -> Result<Vec<u8>, Error> {
    if uncompressed_size == 0 {
        return Ok(Vec::new());
    }
    if data.is_empty() {
        return Err(Error::Compression("empty compressed block".into()));
    }
    if data[0] != 0x00 {
        return Err(Error::Compression(format!(
            "bad compression header: expected 0x00, got 0x{:02X}",
            data[0]
        )));
    }

    // The trailing 0xFF bytes carry the end-of-stream marker, so the whole
    // remainder of the block is bit stream.
    let mut r = BitReader::new(&data[1..]);
    let mut out: Vec<u8> = Vec::with_capacity(uncompressed_size);

    while out.len() < uncompressed_size {
        let Some(kind) = r.read_bit() else { break };

        if kind == 0 {
            let Some(b) = r.read_bits(8) else { break };
            out.push(b as u8);
            continue;
        }

        // Back-reference: offset width class first.
        let Some(class) = r.read_unary(3) else { break };
        let (width, bias) = OFFSET_CLASS[class as usize];
        let Some(raw) = r.read_bits(width) else { break };

        if width == 20 && raw == END_MARKER {
            break;
        }

        let offset = (raw + bias) as usize;
        // A 20-bit offset class encodes one extra byte of length.
        let mut count = if width == 20 { 3usize } else { 2usize };

        let Some(run) = r.read_unary(12) else { break };
        if run == 12 {
            return Err(Error::Compression(
                "invalid length prefix (12 set bits)".into(),
            ));
        }
        // A unary run of N ones contributes 2^N - 1, then N literal bits follow.
        count += ((1u32 << run) - 1) as usize;
        if run > 0 {
            let Some(extra) = r.read_bits(run) else { break };
            count += extra as usize;
        }

        if offset > out.len() {
            return Err(Error::Compression(format!(
                "back-reference offset {} exceeds {} bytes produced",
                offset,
                out.len()
            )));
        }

        // Ranges may overlap the insertion point, so copy one byte at a time.
        let mut src = out.len() - offset;
        let end = (out.len() + count).min(uncompressed_size);
        while out.len() < end {
            let b = out[src];
            out.push(b);
            src += 1;
        }
    }

    if out.len() < uncompressed_size {
        return Err(Error::Compression(format!(
            "truncated stream: produced {} of {} bytes",
            out.len(),
            uncompressed_size
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worked example from the specification: a compressed RGNDATAHEADER.
    #[test]
    fn spec_example() {
        let compressed = [
            0x00, 0x40, 0x00, 0x04, 0x10, 0xD0, 0x90, 0x80, 0x42, 0xED, 0x98, 0x01, 0xB7, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        ];
        let expected = [
            0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA8, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let got = decompress(&compressed, expected.len()).expect("decompress");
        assert_eq!(got, expected);
    }
}
