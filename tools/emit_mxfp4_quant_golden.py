#!/usr/bin/env python
"""emit_mxfp4_quant_golden.py - golden values for k3_mxfp4_quant.

The C requant kernel (src/core/k3_mxfp4_quant.c) converts fp32 -> MXFP4 packed +
E8M0 scales at GGUF cache-admit time (D2a). Its reference is the numpy
implementation here, which is the SAME math as tools/make_tiny_checkpoint.py's
mxfp4_quant (group 32, scale = floor(log2(max|w|)) - 2 biased by 127, nearest
E2M1 with ties toward the smaller magnitude, low nibble = even element). The C
test compares BIT-EXACTLY against the committed golden file, so no tolerance
argument is possible: either the bits match or the implementation is wrong.

The cases deliberately cover the boundary rules the implementation can get wrong:
  * exact tie points between E2M1 magnitudes (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    - numpy argmin picks the SMALLER value on a tie, and the C must too;
  * zero groups (scale must come out as floor(log2(1e-30))-2 = -102, i.e. byte 25,
    with all-zero nibbles, exactly as numpy's 1e-30 clamp produces);
  * all-zero and all-negative rows; -0.0 (no sign bit);
  * magnitudes straddling every E2M1 power boundary (0.5, 1, 1.5, 2, 3, 4, 6);
  * values above 6 and just below every threshold;
  * random rows at realistic weight magnitudes with both signs;
  * a w2-style row (tiny values ~1e-2..1e-3) so the no-clamp scale rule is gated.

The reference here is independent of the C: it uses numpy float64 log2/floor and
argmin exactly like make_tiny_checkpoint.py, and the C must reproduce it for
every f32 input by construction (frexp floor, exact power-of-two division, exact
f32 threshold comparisons).

GOLDEN FILE FORMAT (tests/fixtures/mxfp4_quant_golden.bin, all LE)
    "K3MX" u32 version u32 ncases
    per case:
      u32 rows u32 cols
      f32  w[rows*cols]
      u8   packed[rows*cols/2]
      u8   scales[rows*cols/32]

Usage: emit_mxfp4_quant_golden.py [out_path]
"""
from __future__ import annotations

import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "fixtures", "mxfp4_quant_golden.bin")

GROUP = 32
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def mxfp4_quant(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference: make_tiny_checkpoint.py::mxfp4_quant, unchanged semantics."""
    w = w.astype(np.float32)
    R, C = w.shape
    assert C % GROUP == 0
    g = C // GROUP
    blocks = w.reshape(R, g, GROUP)
    amax = np.abs(blocks).max(axis=2)
    exp = (np.floor(np.log2(np.maximum(amax, 1e-30))) - 2).astype(np.int32)
    if exp.min() < -127 or exp.max() > 127:
        raise SystemExit("exponent outside E8M0 range")
    scales = (exp + 127).astype(np.uint8)
    mult = np.exp2(exp.astype(np.float32)).reshape(R, g, 1)
    x = blocks / mult
    pos = np.abs(x)
    idx = np.argmin(np.abs(pos[..., None] - LUT), axis=-1).astype(np.uint8)
    nib = np.where(x < 0, idx | 0x8, idx).astype(np.uint8)
    packed = (nib[:, :, 0::2] | (nib[:, :, 1::2] << 4)).reshape(R, g * GROUP // 2)
    return packed, scales


def case(w: np.ndarray) -> bytes:
    packed, scales = mxfp4_quant(w)
    R, C = w.shape
    out = struct.pack("<II", R, C)
    out += w.astype(np.float32).tobytes()
    assert packed.shape == (R, C // 2)
    assert scales.shape == (R, C // GROUP)
    out += packed.tobytes()
    out += scales.tobytes()
    return out


def main() -> int:
    rng = np.random.RandomState(20260813)   # legacy generator, frozen across numpy
    cases = []

    # 1: a realistic random matrix (w1-like magnitudes)
    cases.append(rng.normal(0, 0.5, (8, 64)).astype(np.float32))
    # 2: tiny values (w2-like, ~0.02): exercises the unclamped scale rule
    cases.append(rng.normal(0, 0.02, (4, 96)).astype(np.float32))
    # 3: large values up to and past 6
    cases.append(rng.uniform(-8, 8, (3, 32)).astype(np.float32))
    # 4: values pinned exactly on every E2M1 tie point and every power boundary
    special = np.array([
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
        2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0,
        -0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75,
        -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -7.0,
    ], dtype=np.float32)
    cases.append(np.tile(special, (4, 1)))          # 128 cols
    # 5: powers of two times the tie points, so the scale changes but the ties stay
    scale2 = special * np.float32(2.0**9)
    cases.append(np.tile(scale2, (2, 2))[:, :64].astype(np.float32))
    # 6: all zeros - scale must still be emitted (byte 25), nibbles all zero
    cases.append(np.zeros((2, 32), dtype=np.float32))
    # 7: all-negative row (sign bits everywhere)
    cases.append(-np.abs(rng.normal(0, 0.3, (2, 64))).astype(np.float32))
    # 8: sub-ulp-of-scale values: everything quantises to 0 within a nonzero scale
    cases.append((rng.normal(0, 0.02, (2, 64)) * np.float32(1e-4)).astype(np.float32))
    # 9: a single row at full expert width (3584 cols = 112 groups)
    cases.append(rng.normal(0, 0.02, (1, 3584)).astype(np.float32))

    out = b"K3MX" + struct.pack("<II", 1, len(cases))
    for w in cases:
        out += case(w)

    path = sys.argv[1] if len(sys.argv) > 1 else OUT
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(out)
    print("wrote %s (%d cases, %d bytes)" % (path, len(cases), len(out)))

    # self-check: round trip must reconstruct the quantised values exactly
    for w in cases:
        packed, scales = mxfp4_quant(w)
        R, C = w.shape
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        val = np.where(lo & 0x8, -E2M1[lo & 7], E2M1[lo & 7]).astype(np.float32)
        val2 = np.where(hi & 0x8, -E2M1[hi & 7], E2M1[hi & 7]).astype(np.float32)
        out = np.empty((R, C), dtype=np.float32)
        out[:, 0::2] = val
        out[:, 1::2] = val2
        mult = np.exp2((scales.astype(np.int32) - 127).astype(np.float32))
        out *= np.repeat(mult, GROUP, axis=1)[:, : C]
        err = np.abs(out - w).max()
        assert err <= 3.0 * np.abs(out).max() + 1e-6, (err, w.max())
    print("round-trip self-check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
