#!/usr/bin/env python3
"""Build examples/inputtest.nes — a tiny homebrew NES ROM for verifying input scripts.

It does one thing: every frame it reads controller 1 and paints the whole screen a
colour derived from which buttons are held. That makes a scripted run trivially
checkable — render a frame, look at the colour, and you know exactly which buttons
the script had down on that frame.

    none  $00 grey      A     $01 dark blue   B      $02 blue
    Sel   $04 purple    Start $08 olive       Up     $14 pink
    Down  $28 orange    Left  $10 grey-blue   Right  $20 white

The poll runs in the NMI handler, i.e. at a fixed point in every frame — the same
thing real games do, and a prerequisite for one-frame inputs to register reliably.

Hand-assembled 6502 (no external assembler needed). Mapper 0, 16K PRG at $C000.
"""

import pathlib

# ---------------------------------------------------------------------------
# The program. Addresses in comments are offsets into PRG (which maps to $C000).
# ---------------------------------------------------------------------------
code = bytes([
    # --- reset: mask interrupts, kill the APU frame IRQ, zero the PPU ---
    0x78,                    # 00  SEI
    0xD8,                    # 01  CLD
    0xA2, 0x40,              # 02  LDX #$40
    0x8E, 0x17, 0x40,        # 04  STX $4017      APU frame counter: no IRQ
    0xA2, 0xFF,              # 07  LDX #$FF
    0x9A,                    # 09  TXS            stack
    0xE8,                    # 0A  INX            X = 0
    0x8E, 0x00, 0x20,        # 0B  STX $2000      NMI off while we set up
    0x8E, 0x01, 0x20,        # 0E  STX $2001      rendering off (backdrop only)
    0x8E, 0x10, 0x40,        # 11  STX $4010      DMC off

    # --- the PPU needs two vblanks before it's warm ---
    0x2C, 0x02, 0x20,        # 14  BIT $2002
    0x10, 0xFB,              # 17  BPL -5         → 14
    0x2C, 0x02, 0x20,        # 19  BIT $2002
    0x10, 0xFB,              # 1C  BPL -5         → 19

    # --- hand the frame over to NMI and idle ---
    0xA9, 0x80,              # 1E  LDA #$80
    0x8D, 0x00, 0x20,        # 20  STA $2000      NMI on
    0x4C, 0x23, 0xC0,        # 23  JMP $C023      spin here forever

    # --- NMI: fires once per frame at the start of vblank, so the controller is
    #     polled at a FIXED phase. A spin-loop poller drifts against the frame
    #     boundary (the NES frame is 29780.5 CPU cycles, so the phase creeps by
    #     half a cycle a frame) and periodically misses a one-frame input — which
    #     is exactly why real games poll from NMI, and why this ROM must too if
    #     it's going to verify frame-exact input.
    0xA9, 0x01,              # 26  LDA #$01
    0x8D, 0x16, 0x40,        # 28  STA $4016      strobe high
    0xA9, 0x00,              # 2B  LDA #$00
    0x8D, 0x16, 0x40,        # 2D  STA $4016      strobe low → latched
    0xA2, 0x08,              # 30  LDX #$08
    0xA9, 0x00,              # 32  LDA #$00
    0x85, 0x00,              # 34  STA $00        buttons = 0
    0xAD, 0x16, 0x40,        # 36  LDA $4016      read a button into bit 0
    0x4A,                    # 39  LSR A          → carry
    0x66, 0x00,              # 3A  ROR $00        carry → bit 7, shifting right
    0xCA,                    # 3C  DEX
    0xD0, 0xF7,              # 3D  BNE -9         → 36
    # after 8 rotations: bit0=A bit1=B bit2=Select bit3=Start
    #                    bit4=Up bit5=Down bit6=Left bit7=Right

    # --- colour = ((buttons >> 4) << 2) EOR buttons, masked to the 64-colour
    #     palette. Folding the nibbles keeps all eight buttons distinguishable.
    0xA5, 0x00,              # 3F  LDA $00
    0x4A, 0x4A, 0x4A, 0x4A,  # 41  LSR A x4       high nibble (d-pad)
    0x0A, 0x0A,              # 45  ASL A x2
    0x45, 0x00,              # 47  EOR $00        mix in A/B/Select/Start
    0x29, 0x3F,              # 49  AND #$3F
    0x85, 0x01,              # 4B  STA $01

    # --- write it to palette entry 0 (the universal backdrop) ---
    0xAD, 0x02, 0x20,        # 4D  LDA $2002      reset the $2006 address latch
    0xA9, 0x3F,              # 50  LDA #$3F
    0x8D, 0x06, 0x20,        # 52  STA $2006
    0xA9, 0x00,              # 55  LDA #$00
    0x8D, 0x06, 0x20,        # 57  STA $2006      VRAM address = $3F00
    0xA5, 0x01,              # 5A  LDA $01
    0x8D, 0x07, 0x20,        # 5C  STA $2007      palette[0] = colour

    # --- park the VRAM address back on $3F00 so the backdrop shows it ---
    0xA9, 0x3F,              # 5F  LDA #$3F
    0x8D, 0x06, 0x20,        # 61  STA $2006
    0xA9, 0x00,              # 64  LDA #$00
    0x8D, 0x06, 0x20,        # 66  STA $2006
    0x40,                    # 69  RTI
])

NMI_ENTRY = 0xC026

PRG_SIZE = 16 * 1024
CHR_SIZE = 8 * 1024

prg = bytearray(b"\x00" * PRG_SIZE)
prg[: len(code)] = code
# Vectors at $FFFA/$FFFC/$FFFE. NMI is the whole point here; IRQ can't fire (SEI +
# no APU frame IRQ) but points at reset so a stray one can't wander off.
prg[PRG_SIZE - 6 : PRG_SIZE] = bytes(
    [NMI_ENTRY & 0xFF, NMI_ENTRY >> 8, 0x00, 0xC0, 0x00, 0xC0]  # NMI, RESET, IRQ
)

header = bytes([ord("N"), ord("E"), ord("S"), 0x1A, 1, 1, 0x00, 0x00]) + bytes(8)
rom = header + bytes(prg) + bytes(CHR_SIZE)

out = pathlib.Path(__file__).with_name("inputtest.nes")
out.write_bytes(rom)
print(f"wrote {out} ({len(rom)} bytes)")
