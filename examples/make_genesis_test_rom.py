#!/usr/bin/env python3
"""Build examples/inputtest.md — the Mega Drive twin of inputtest.nes.

Same idea, different CPU: every frame it reads pad 1 and paints the whole screen a
colour built from the buttons held, so a scripted run is a direct readout of its own
input timeline.

    Up/Down/Left → red shades      Right/B/C → green shades      nothing → black

No planes, no tiles: the VDP is set up for a backdrop-only screen (VRAM powers up
zeroed, so every tile pixel is transparent and the backdrop colour fills the frame),
and the program just writes CRAM entry 0 each frame.

Hand-assembled 68000. Mapper-less 68k ROM at $000000, code at $200.
"""

import pathlib
import struct

# ---------------------------------------------------------------------------
# The program. Offsets in comments are absolute addresses.
# ---------------------------------------------------------------------------
code = bytes.fromhex(
    # --- $200 take the CPU: no interrupts while we set the VDP up -------------
    "46FC 2700"                     # move.w #$2700,sr

    # --- VDP registers. Only what a backdrop-only screen needs. --------------
    "33FC 8004 00C0 0004"           # reg0  = $04  no H-interrupt
    "33FC 8144 00C0 0004"           # reg1  = $44  display on, Mega Drive mode
    "33FC 8700 00C0 0004"           # reg7  = $00  backdrop = CRAM entry 0
    "33FC 8C81 00C0 0004"           # reg12 = $81  H40 — a 320x224 frame

    # --- $224 main loop: read pad 1 -----------------------------------------
    # The pad multiplexes on TH: driving it HIGH selects the ??11CBRLDU half (the
    # low half only reports Up/Down/A/Start), so TH has to be both configured as an
    # output *and* actually driven high. Buttons read active LOW, hence the NOT.
    "13FC 0040 00A1 0009"           # move.b #$40,($A10009)  TH is an output
    "13FC 0040 00A1 0003"           # move.b #$40,($A10003)  …and drive it HIGH
    "1039 00A1 0003"                # move.b ($A10003),d0    read pad 1
    "4600"                          # not.b  d0              active low → high

    # --- colour = (U,D,L) in the red nibble | (Right,B,C) in the green -------
    # CRAM is ----BBB- GGG- RRR-, so the low three buttons shift into bits 1-3
    # and the high three into bits 5-7. Every single button gets its own colour.
    "3200"                          # move.w d0,d1
    "0241 0007"                     # andi.w #$0007,d1       U/D/L
    "E349"                          # lsl.w  #1,d1           → red
    "E648"                          # lsr.w  #3,d0
    "0240 0007"                     # andi.w #$0007,d0       Right/B/C
    "EB48"                          # lsl.w  #5,d0           → green
    "8041"                          # or.w   d1,d0

    # --- write it to CRAM[0] -------------------------------------------------
    "23FC C000 0000 00C0 0004"      # move.l #$C0000000,($C00004)  CRAM write, addr 0
    "33C0 00C0 0000"                # move.w d0,($C00000)
    "60C4"                          # bra.s  loop  ($224)
    .replace(" ", "")
)

CODE_ORG = 0x200
ROM_SIZE = 0x20000  # 128K, comfortably above the header + code

rom = bytearray(b"\xff" * ROM_SIZE)

# --- vectors: SP, PC, then every exception pointed at the entry so a stray one
#     can't wander off into unmapped space.
struct.pack_into(">II", rom, 0, 0x00FFFE00, CODE_ORG)
for v in range(2, 64):
    struct.pack_into(">I", rom, v * 4, CODE_ORG)

# --- the 256-byte cartridge header. Emulators key off "SEGA" at $100.
def field(off, text, width):
    rom[off : off + width] = text.encode("ascii").ljust(width)[:width]

field(0x100, "SEGA MEGA DRIVE ", 16)
field(0x110, "(C)CRTU 2026.JUL", 16)
field(0x120, "CRTULUM INPUT TEST", 48)
field(0x150, "CRTULUM INPUT TEST", 48)
field(0x180, "GM 00000000-00", 14)
struct.pack_into(">H", rom, 0x18E, 0)  # checksum — not verified by emulators
field(0x190, "J", 16)                  # I/O: 3-button pad
struct.pack_into(">II", rom, 0x1A0, 0x00000000, ROM_SIZE - 1)  # ROM start/end
struct.pack_into(">II", rom, 0x1A8, 0x00FF0000, 0x00FFFFFF)    # RAM start/end
field(0x1B0, " ", 12)                  # no SRAM
field(0x1BC, " ", 52)
field(0x1F0, "JUE", 16)                # region: anywhere

rom[CODE_ORG : CODE_ORG + len(code)] = code

out = pathlib.Path(__file__).with_name("inputtest.md")
out.write_bytes(bytes(rom))
print(f"wrote {out} ({len(rom)} bytes, {len(code)} bytes of code)")
