#!/usr/bin/env python
"""
emit_gguf_dequant_golden.py - golden values for the C dequant unit test.

WHY THIS EXISTS
    tests/unit/test_gguf_dequant.c compares the C dequant BIT-EXACTLY against
    committed expected values. The dequant math (fp16 widen, int8 scale, IQ1_S
    grid lookup) is a handful of exactly-representable fp32 operations, so there
    is no tolerance to argue about: either the bits match the reference or the
    implementation is wrong.

    The golden bytes come from TWO sources, both embedded here so CI (no real
    file, no network) can regenerate everything:

      * REAL-FILE BLOCKS: the first blocks of `output.weight` (Q8_0) and of
        `blk.1.ffn_gate_exps.weight` (IQ1_S) from
        Kimi-K3-UD-IQ1_S-00002-of-00014.gguf, extracted once with bounded reads
        (a few hundred bytes). Running with --from-real-file re-extracts them and
        fails if they ever drift from what is embedded, so the provenance cannot
        rot silently.
      * SYNTHETIC BLOCKS: deterministic LCG bytes (numpy RandomState, fixed
        seeds; the legacy generator is frozen across numpy versions), covering
        shapes the real file never uses: block-boundary rows (ne0 not a multiple
        of QK), fp16 subnormal scales, every IQ1_S sub-block scale/sign combo.

    The reference dequant is implemented here in numpy INDEPENDENTLY of the C
    code (fp16->fp32 by IEEE construction, Q8_0 and IQ1_S by the llama.cpp
    formulas from ggml-quants.c:553/:2650), and the bf16 expectations use the
    same round-to-nearest-even conversion the C header does, cross-checked
    against torch's bfloat16 cast.

GOLDEN FILE FORMAT (tests/fixtures/gguf_dequant_golden.bin, all LE)
    "K3DG" u32 version u32 ncases
    per case:
      u32 ggml_type      u32 ndim      s64 ne[4]
      u64 nraw           u64 nvals     u32 flags   u32 reserved
      flags bit0: expected bf16 follows the fp32 block
      flags bit1: 3-D IQ1_S expert-slice case (test dequantizes slice 0 and 1)
      raw bytes[nraw]  f32 expected[nvals]  u16 expected_bf16[nvals]

Usage:
    tools/emit_gguf_dequant_golden.py [--from-real-file SHARD_DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIXTURES, ROOT

# -------------------------------------------------------------------- provenance --
# Real-file blocks, extracted 2026-08-13 from
# /workspace/unsloth/Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00002-of-00014.gguf.
# Q8_0: `output.weight`, data_offset=14048 (first tensor of shard 2), blocks 0..8.
# IQ1_S: `blk.1.ffn_gate_exps.weight`, data_offset=6089795680;
#        iq_a = blocks 0..3 (expert 0, row 0), iq_b = blocks at index
#        (7*3072 + 3)*14 .. +3 (expert 7, row 3).
REAL = {
    "shard": "Kimi-K3-UD-IQ1_S-00002-of-00014.gguf",
    "q8_tensor": "output.weight",
    "q8_off": 14048,
    "q8_blocks": 9,
    "iq_tensor": "blk.1.ffn_gate_exps.weight",
    "iq_off": 6089795680,
    "iq_a_blocks": 4,
    "iq_b_block0": (7 * 3072 + 3) * 14,          # expert 7, row 3, block 0
    "iq_b_blocks": 4,
}

Q8_RAW = bytes.fromhex(
    "060fe02400d42dfddcf845f134e44681f90507cb3eceffeb00d95bff44d5ecfe"
    "1c212a0df75656bb7848e8ebfdf377ba1e287f21ff50e0d7eb3be85609d6560e"
    "e42496579d0efa00d77fb91238fd04e06d1cda2afa18e2f4f744dee245f4fe5d"
    "e9b926d3dee7f21039152ebb1bdef5f4d2d481f3daaf2522fc0825df4b010fbb"
    "1ac4b4ec452cf5eeee0ee0f03b0317e30681f80ae1cc3e1c04d7ae400cd602f3"
    "1413c5280024750ddbecf21021ee07812d1155d1d2ecf62eeb2a3d25e803d7c5"
    "fd140805282412d840f2ceaebf0f05d722f9facb7ff75632bc290dd930dbe3f9"
    "bbc5ce05e820f814112625e1fbe26311c95c0981de2808101fe2e2039437fbf0"
    "f60a17f7f9f24dcff7fd1dd3e2ff11dd020d0a9a021e659388c2adbde61215d7"
    "151ed2a9ec5681e16c0e040d11265ee840db"
)
IQ_A_RAW = bytes.fromhex(
    "2c1cbb08cc03f1af179fc087eafab03da5af96f15cea12a0720da98c95d1bd6c"
    "281ecc4843489449ccf3525127c38f562b37891aaa0b97306eb51eb148158546"
    "4b19919441aa147715534f53207ba767a687bad4a0fa135c7ffac5529f764ad3"
    "c2fbf7e1fc1a974d330a05e14f92e3019b0802e26db7a563055d00f4e5fd00ef"
    "7eb3660553bd8fc8b4f6b6f915fe206bf4536a4360672b1b8305f99e4d288286"
    "4e4240cc189eedece0ff8751c6614b9faa20809b53182901b0551bdb3a43b1c1"
    "86fbb8ca025d8f50"
)
IQ_B_RAW = bytes.fromhex(
    "45069b9c3150ddc668064fdefceba5b4f103b4fbe2e5a6bba9dd5a2a86785221"
    "29bb77db9e51814ccbd6bfcc70af0e79ccbad10411342a2b5f47b53e6a554461"
    "be79e751bb9999a4596b039291ad6da803bee503fa7996e815d1a1e8e2db5955"
    "5c79dff9c904d9407bfb98cf30a0107f7a154b5ddfbd7c66c8eba3ed097618b4"
    "dc613f4b6a119bc3dcea7158d8669a77effca5ed6fea6f05cedd7069a62f3f08"
    "b7f2cfe47ea897230fdb0ae3911633ef8f4b888e1bee391edd4213e90558bcfb"
    "eb5661d52fe7dcd7"
)

Q8_SHA256 = "13d468b153bc995d52a47e097f7c879ff94cf14ab4c302e3a9419aa2f27e5f80"
IQ_A_SHA256 = "26a55870b782dd1a8300f6324b1b47172d65c5f62ff0d2f34d3c55bcd4433584"
IQ_B_SHA256 = "7366b908f3edd5b875264f93b218f2ffcd69157b55abc71ce90372b4b19e2928"

QK8, BS8 = 32, 34          # Q8_0 block geometry
QKI, BSI = 256, 50         # IQ1_S block geometry
DELTA = 0.125              # IQ1S_DELTA (ggml-common.h:1134)


# ----------------------------------------------------------------- reference math --
def fp16_to_fp32(h: np.ndarray) -> np.ndarray:
    """Exact fp16 -> fp32 by IEEE construction; subnormals via the exact value
    man*2^-24, which is representable in fp32, so the bit pattern is the same the
    C renormalization loop produces."""
    h = h.astype(np.uint32)
    sign = (h & 0x8000) << 16
    exp = (h >> 10) & 0x1F
    man = h & 0x3FF
    u = np.zeros_like(h)
    norm = (exp > 0) & (exp < 31)
    u[norm] = sign[norm] | ((exp[norm] - 15 + 127) << 23) | (man[norm] << 13)
    zero = (exp == 0) & (man == 0)
    u[zero] = sign[zero]
    sub = (exp == 0) & (man != 0)
    if sub.any():
        val = man[sub].astype(np.float32) * np.float32(2.0 ** -24)
        u[sub] = val.view(np.uint32)
    infnan = exp == 31
    u[infnan] = sign[infnan] | 0x7F800000 | (man[infnan] << 13)
    return u.view(np.float32)


def f32_to_bf16(f32: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even on the dropped 16 mantissa bits. Identical algorithm
    to k3_gguf_f32_to_bf16 in the C header (and to tools/qdq_trunk.py)."""
    u = f32.view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) >> 16).astype(np.uint16)


def load_iq1s_grid() -> np.ndarray:
    """Parse iq1s_grid out of src/core/iq1s_grid.h, the same file the C code
    compiles against, so the reference and the implementation share one table."""
    path = os.path.join(ROOT, "src", "core", "iq1s_grid.h")
    src = open(path, encoding="utf-8").read()
    vals = [int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", src)]
    assert len(vals) == 2048, len(vals)
    assert len(set(vals)) == 2048, "iq1s_grid has duplicate entries"
    return np.frombuffer(np.array(vals, dtype="<u8").tobytes(), dtype=np.int8)


GRID = load_iq1s_grid()


def dequant_q8_0(raw: bytes) -> np.ndarray:
    """y[j] = qs[j] * fp32(d), per 34-byte block (ggml-quants.c:553)."""
    assert len(raw) % BS8 == 0
    n = len(raw) // BS8
    y = np.empty(n * QK8, np.float32)
    for i in range(n):
        blk = raw[i * BS8:(i + 1) * BS8]
        d = fp16_to_fp32(np.frombuffer(blk[:2], np.uint16, 1))[0]
        qs = np.frombuffer(blk[2:], np.int8, QK8)
        y[i * QK8:(i + 1) * QK8] = qs.astype(np.float32) * d
    return y


def dequant_iq1_s(raw: bytes) -> np.ndarray:
    """Reference dequantize_row_iq1_s (ggml-quants.c:2650)."""
    assert len(raw) % BSI == 0
    n = len(raw) // BSI
    y = np.empty(n * QKI, np.float32)
    for i in range(n):
        blk = raw[i * BSI:(i + 1) * BSI]
        d = fp16_to_fp32(np.frombuffer(blk[:2], np.uint16, 1))[0]
        qs = np.frombuffer(blk[2:34], np.uint8, 32)
        qh = np.frombuffer(blk[34:50], np.uint16, 8)   # 8 sub-block scales, 16 bytes
        for ib in range(8):
            qhb = int(qh[ib])
            dl = d * np.float32(2 * ((qhb >> 12) & 7) + 1)
            delta = np.float32(-DELTA) if qhb & 0x8000 else np.float32(DELTA)
            for li in range(4):
                idx = int(qs[4 * ib + li]) | (((qhb >> (3 * li)) & 7) << 8)
                grid = GRID[idx * 8:(idx + 1) * 8].astype(np.float32)
                y[i * QKI + 32 * ib + 8 * li: i * QKI + 32 * ib + 8 * (li + 1)] = (
                    (grid + delta) * dl)
    return y


# ------------------------------------------------------------------- synthesis --
def lcg_bytes(seed: int, n: int) -> bytes:
    return np.random.RandomState(seed).randint(0, 256, n).astype(np.uint8).tobytes()


def lcg_floats(seed: int, n: int, lo=-0.5, hi=0.5) -> np.ndarray:
    return np.random.RandomState(seed).uniform(lo, hi, n).astype(np.float32)


def synth_q8_blocks(seed: int, nblocks: int) -> bytes:
    """Random-ish Q8_0 blocks; the last block's d is a deliberate fp16 subnormal
    (renormalization coverage, exact in fp32)."""
    b = bytearray(lcg_bytes(seed, nblocks * BS8))
    if nblocks:
        b[(nblocks - 1) * BS8:(nblocks - 1) * BS8 + 2] = struct.pack("<H", 0x03FF)
    return bytes(b)


def synth_iq1_blocks(seed: int, nblocks: int) -> bytes:
    """Random-ish IQ1_S blocks with every sub-block scale (0..7) and sign combo
    present, and a subnormal fp16 d in the last block."""
    b = bytearray(lcg_bytes(seed, nblocks * BSI))
    for i in range(nblocks):
        blk = i * BSI
        qh = b[blk + 34:blk + 50]                      # 8 x uint16 sub-block fields
        for ib in range(8):
            scale = (i * 8 + ib) % 8
            sign = 0x8000 if (i + ib) % 3 == 0 else 0
            cur = struct.unpack("<H", qh[2 * ib:2 * ib + 2])[0]
            struct.pack_into("<H", b, blk + 34 + 2 * ib, (cur & 0x0FFF) | (scale << 12) | sign)
    if nblocks:
        struct.pack_into("<H", b, (nblocks - 1) * BSI, 0x0001)   # min subnormal d
    return bytes(b)


# --------------------------------------------------------------------- fixtures --
def make_cases() -> list[dict]:
    """Every golden case. Shapes are GGUF order (ne0 fastest). nvals is the padded
    value count the dequant produces (ceil(ne0/QK)*QK * ne1*ne2*ne3)."""
    cases = []

    f32_3d = lcg_floats(11, 24)
    cases.append(dict(name="f32_3d", type=0, ne=(4, 3, 2), raw=f32_3d.tobytes(),
                      expect=f32_3d))
    f32_1d = lcg_floats(12, 7)
    cases.append(dict(name="f32_1d", type=0, ne=(7,), raw=f32_1d.tobytes(),
                      expect=f32_1d))
    cases.append(dict(name="f32_empty", type=0, ne=(0, 5), raw=b"",
                      expect=np.empty(0, np.float32)))
    cases.append(dict(name="q8_real_2d", type=8, ne=(96, 3), raw=Q8_RAW,
                      expect=dequant_q8_0(Q8_RAW)))
    q8_part = Q8_RAW[2 * BS8:4 * BS8]                       # blocks 2-3
    cases.append(dict(name="q8_partial_row", type=8, ne=(40, 1), raw=q8_part,
                      expect=dequant_q8_0(q8_part)))
    q8_syn = synth_q8_blocks(21, 4)
    cases.append(dict(name="q8_synth_weird", type=8, ne=(37, 2), raw=q8_syn,
                      expect=dequant_q8_0(q8_syn)))
    cases.append(dict(name="q8_empty", type=8, ne=(0, 7), raw=b"",
                      expect=np.empty(0, np.float32)))
    cases.append(dict(name="iq1_real_1d", type=19, ne=(1024, 1, 1), raw=IQ_A_RAW,
                      expect=dequant_iq1_s(IQ_A_RAW)))
    iq_slice = IQ_A_RAW[:2 * BSI] + IQ_B_RAW[:2 * BSI]       # 2 blocks expert0 + 2 expert7
    cases.append(dict(name="iq1_slice_3d", type=19, ne=(512, 1, 2), raw=iq_slice,
                      expect=dequant_iq1_s(iq_slice), slice_case=True))
    iq_syn = synth_iq1_blocks(31, 2)
    cases.append(dict(name="iq1_synth_partial", type=19, ne=(300, 1), raw=iq_syn,
                      expect=dequant_iq1_s(iq_syn)))
    cases.append(dict(name="iq1_empty", type=19, ne=(0, 1, 896), raw=b"",
                      expect=np.empty(0, np.float32)))
    return cases


def write_golden(path: str, cases: list[dict]) -> None:
    out = bytearray()
    out += b"K3DG"
    out += struct.pack("<II", 1, len(cases))
    for c in cases:
        nvals = len(c["expect"])
        assert nvals == c["expect"].size
        flags = 1 | (2 if c.get("slice_case") else 0)
        # The bf16 block is part of the format for every case, and the C test
        # refuses to compare a case whose flags bit0 is clear: keep the two sides
        # honest by construction.
        assert nvals == 0 or (flags & 1), (c["name"], "non-empty case without bf16 block")
        out += struct.pack("<II", c["type"], len(c["ne"]))
        out += struct.pack("<4q", *([int(x) for x in c["ne"]] + [0] * (4 - len(c["ne"]))))
        out += struct.pack("<QQII", len(c["raw"]), nvals, flags, 0)
        out += c["raw"]
        out += c["expect"].tobytes()
        out += f32_to_bf16(c["expect"]).tobytes()
    with open(path, "wb") as f:
        f.write(bytes(out))
    print("wrote %s (%d bytes, %d cases)" % (path, len(out), len(cases)))


def verify_real_blocks(shard_dir: str) -> None:
    """Re-extract the provenance blocks from the real file and require them to be
    byte-identical to what is embedded. Any drift is a fail, not a warning: the
    golden values would no longer describe the real file."""
    from gguf import GGUFReader  # venv only; not needed for regeneration

    shard = os.path.join(shard_dir, REAL["shard"])
    if not os.path.isfile(shard):
        sys.exit("no such shard: %s" % shard)
    r = GGUFReader(shard)
    by_name = {t.name: t for t in r.tensors}
    with open(shard, "rb") as f:
        def blocks(tname, off, nblk, bsz):
            t = by_name[tname]
            assert int(t.data_offset) == off, (tname, int(t.data_offset), off)
            f.seek(off)
            return f.read(nblk * bsz)
        q8 = blocks(REAL["q8_tensor"], REAL["q8_off"], REAL["q8_blocks"], BS8)
        iq_a = blocks(REAL["iq_tensor"], REAL["iq_off"], REAL["iq_a_blocks"], BSI)
        f.seek(REAL["iq_off"] + REAL["iq_b_block0"] * BSI)
        iq_b = f.read(REAL["iq_b_blocks"] * BSI)
    got = {"q8": hashlib.sha256(q8).hexdigest(),
           "iq_a": hashlib.sha256(iq_a).hexdigest(),
           "iq_b": hashlib.sha256(iq_b).hexdigest()}
    want = {"q8": Q8_SHA256, "iq_a": IQ_A_SHA256, "iq_b": IQ_B_SHA256}
    for k in got:
        status = "match" if got[k] == want[k] else "DRIFTED"
        print("  %-5s sha256 %s  (%s)" % (k, got[k], status))
    if got != want:
        sys.exit("embedded real-file blocks no longer match %s; re-embed them" % shard)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-real-file", metavar="SHARD_DIR",
                    help="verify the embedded real-file blocks against the 554 GB "
                         "file (bounded reads only) before regenerating")
    args = ap.parse_args()

    if args.from_real_file:
        print("verifying embedded blocks against %s" % args.from_real_file)
        verify_real_blocks(args.from_real_file)

    if not hasattr(np, "float32"):
        sys.exit("numpy too old")

    # Cross-check the bf16 rounding against torch when it is available; a drift
    # would break the C test's bit-exact bf16 comparison later, better here.
    try:
        import torch
        probe = np.array([0.0, -0.0, 1.0, 0.1, -3.75, 65504.0, 1e-8, 123.456], np.float32)
        ref = f32_to_bf16(probe)
        tor = torch.from_numpy(probe).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
        if not np.array_equal(ref, tor):
            print("WARNING: f32_to_bf16 disagrees with torch on:", probe, ref, tor)
    except ImportError:
        print("note: torch not available; bf16 rounding cross-check skipped")

    cases = make_cases()
    total = sum(len(c["expect"]) for c in cases)
    print("reference dequant computed for %d values across %d cases" % (total, len(cases)))
    write_golden(os.path.join(FIXTURES, "gguf_dequant_golden.bin"), cases)


if __name__ == "__main__":
    main()
