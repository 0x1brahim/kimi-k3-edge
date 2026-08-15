#!/usr/bin/env python
"""
ref_gguf_experts.py - python reference computed FROM THE SAME GGUF BYTES as the
engine (P3 test-dev; architect D2 hard gate + PARITY GATE 1 references).

WHY IT EXISTS
    The D2 gate's criterion is SELF-CONSISTENCY: the GGUF path's expert matmul
    output must agree with a python reference computed from the SAME GGUF IQ1_S
    bytes, NOT with the safetensors MXFP4 bytes (a different lossy encoding).
    And PARITY GATE 1's strongest form compares the engine's whole GGUF-path
    forward against a reference built from the same GGUF bytes, so any
    disagreement is an io/math bug, not quantization noise.

WHAT THIS FILE PROVIDES
    - a tiny, independent GGUF reader (GgufShards): header, metadata walk
      (including the D7 quirk arrays: type-5 elements are 4 bytes), tensor
      infos, per-shard offsets;
    - the engine's VIEW of every tensor, mirroring k3_gguf_map.c's finder
      transformations: Q8_0 -> fp32 -> round-to-nearest-even bf16 -> fp32,
      F32 -> bf16 for ssm_beta, the composite kv_b (attn_k_b transposed +
      attn_v_b per head), the folded *_res_score vectors (norm side) with the
      constant-1.0 proj side, A_log prefix UNFOLDED from the folded -exp(A_log)
      ssm_a; IQ1_S experts stay RAW (the fix wave's native path - no requant
      anywhere in the reference);
    - dequant_iq1_s / dequant_q8_0: the same block math as k3_gguf_dequant.c
      (the committed gguf_dequant_golden.bin is the independent check on both);
    - full-model forward through k3_ref.K3Model with the engine-view weights,
      plus a capture variant recording per-MoE-layer routed inputs (x, z) and
      router selections for the Gate-2 golden;
    - emit_gate2_golden: the committed tests/fixtures/gguf_gate2_golden.bin
      (v2, per expert: the routed input z and the RAW-dequant fp32 reference
      outputs gu/up/edn - NO requant anywhere in the reference path) that
      tests/unit/test_gguf_gate.c asserts the engine's NATIVE k3_matmul_iq1_s
      chain against; emit_gate2_real_golden does the same on bounded windows
      of REAL shard bytes (the K3_GGUF_REAL-gated leg).

usage:
    ref_gguf_experts.py <gguf_path> <ids> [out.json]
        full-model reference logits for a GGUF fixture (single file or shard dir)
    ref_gguf_experts.py --gate2 <gguf_path> <out.bin>
        emit the Gate-2 golden file for a GGUF fixture
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from k3_ref import K3Config, K3Model, apply_attn_res, situ_glu      # noqa: E402
import torch.nn.functional as F                                           # noqa: E402
from make_tiny_checkpoint import (mxfp4_dequant, mxfp4_quant,   # noqa: E402
                                              engine_name as mtc_engine_name)

PRE = "language_model.model."
QK8, BS8 = 32, 34          # Q8_0 block geometry
QKI, BSI = 256, 50         # IQ1_S block geometry
IQ1S_DELTA = 0.125
MXFP4_GROUP = 32

# -------------------------------------------------------------------- fp16 ----
def f16_to_f32(h: np.ndarray) -> np.ndarray:
    """fp16 -> fp32 by IEEE construction (exact; subnormals via man*2^-24,
    representable in fp32, so the bits match the C renormalization loop). Same
    math as k3_gguf_f16_to_f32 and the committed dequant golden generator."""
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


def f32_to_bf16_rne(w: np.ndarray) -> np.ndarray:
    """f32 -> bf16, round-to-nearest-even on the dropped 16 mantissa bits.

    Statement-for-statement mirror of k3_gguf_f32_to_bf16 (the add-0x7FFF+lsb
    trick, with uint32 wrap like the C). The engine dequantizes every trunk
    matrix to bf16 with this rounding, so the reference must reproduce it
    exactly for the same-bytes comparisons to be meaningful."""
    u = w.view(np.uint32)
    lsb = (u >> 16) & np.uint32(1)
    u = u + np.uint32(0x7FFF) + lsb
    return (u >> 16).astype(np.uint16)


def load_iq1s_grid() -> np.ndarray:
    """Parse iq1s_grid out of src/core/iq1s_grid.h - the same bytes the C uses
    (2048 x int8 in {-1,0,1}, little-endian byte order)."""
    path = os.path.join(ROOT, "src", "core", "iq1s_grid.h")
    with open(path) as f:
        text = f.read()
    hexes = re.findall(r"0x([0-9a-fA-F]{16})", text)
    assert len(hexes) == 2048, "iq1s_grid.h must hold exactly 2048 entries"
    vals = np.array([int(h, 16) for h in hexes], dtype=np.uint64)
    grid = vals.view(np.int8).reshape(2048, 8)      # LE: byte 0 is grid[0]
    assert grid.shape == (2048, 8)
    return grid


GRID = load_iq1s_grid()

# ----------------------------------------------------------------- dequant ----
def dequant_q8_0(raw: bytes, cols: int) -> np.ndarray:
    """Q8_0 tensor bytes -> fp32 [rows][cols]. y[j] = qs[j] * fp32(d), per
    34-byte block (k3_gguf_dequant.c k3_deq_q8_0)."""
    nblk = len(raw) // BS8
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nblk, BS8)
    d = f16_to_f32(b[:, :2].copy().view(np.uint16)[:, 0]).astype(np.float32)
    q = b[:, 2:].astype(np.int8).astype(np.float32)
    rows = nblk * QK8 // cols
    assert rows * cols == nblk * QK8, "Q8_0 rows must be block-multiples"
    return (q.reshape(rows, cols // QK8, QK8) * d.reshape(rows, cols // QK8, 1)
            ).reshape(rows, cols)


def dequant_iq1_s(raw: bytes, cols: int) -> np.ndarray:
    """IQ1_S tensor bytes -> fp32 [rows][cols], rows COMPACTED to cols (the
    dequant pads every row to the 256-block; the engine's kernel reads the
    padded rows straight from disk and skips the padding). Mirror of
    k3_deq_iq1_s:
      dl    = d * (2*((qh[ib] >> 12) & 7) + 1)
      delta = (qh[ib] & 0x8000) ? -0.125 : 0.125
      idx   = qs[4*ib + l] | (((qh[ib] >> 3*l) & 7) << 8)
      y     = dl * (grid[idx][j] + delta)"""
    nblk = len(raw) // BSI
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nblk, BSI)
    d = f16_to_f32(b[:, :2].copy().view(np.uint16)[:, 0]).astype(np.float32)
    qs = b[:, 2:34]
    qh = b[:, 34:50].copy().view(np.uint16)
    out = np.empty(nblk * QKI, dtype=np.float32)
    for i in range(nblk):
        for ib in range(8):
            qhb = int(qh[i, ib])
            dl = d[i] * np.float32(2 * ((qhb >> 12) & 7) + 1)
            delta = np.float32(-IQ1S_DELTA) if qhb & 0x8000 else np.float32(IQ1S_DELTA)
            for li in range(4):
                idx = int(qs[i, 4 * ib + li]) | (((qhb >> (3 * li)) & 7) << 8)
                g = GRID[idx].astype(np.float32)
                out[i * QKI + 32 * ib + 8 * li:i * QKI + 32 * ib + 8 * (li + 1)] = (
                    g + delta) * dl
    bpr = (cols + QKI - 1) // QKI
    rows = nblk // bpr
    return out.reshape(rows, bpr * QKI)[:, :cols].copy()


def expert_engine_view(w: np.ndarray) -> np.ndarray:
    """fp32 expert matrix -> the engine's OLD D2a view: MXFP4 requant, then
    widened back to fp32 (k3_mxfp4_quant's exact numpy mirror). Kept ONLY for
    the generator's requant-error measurement (make_tiny_gguf.py
    measure_quant_errors); the fix wave's Gate-2 reference path never uses it."""
    packed, scales = mxfp4_quant(w)
    return mxfp4_dequant(packed, scales)


# ------------------------------------------------------------ gguf reader ----
class GgufShards:
    """Minimal GGUF reader: header, metadata walk (D7 quirks included), tensor
    infos, relative-offset fixup. Serves every tensor by GGUF name with its
    stored type, shape (ne order) and raw bytes.

    lazy=True (the real-file gate): reads only the header + tensor infos of
    each shard (bounded: never the data section) and serves raw bytes with
    per-request pread windows - the 14-shard real file is 554 GB and must
    never be slurped into RAM. The fixture path keeps the whole-buffer read."""

    LAZY_HEAD = 64 << 20  # headers + infos are far smaller; safety cap

    def __init__(self, path: str, lazy: bool = False):
        self.tensors: dict[str, tuple[int, tuple, int, object, str]] = {}
        self.kv: dict[str, object] = {}
        self._kinds: dict[str, int] = {}
        self._shapes: dict[str, tuple] = {}
        self._lazy = lazy
        files = ([path] if path.endswith(".gguf")
                 else sorted(os.path.join(path, f)
                             for f in os.listdir(path) if f.endswith(".gguf")))
        if not files:
            raise SystemExit("no .gguf files under %s" % path)
        for p in files:
            self._parse(p)

    def _read_at(self, name: str, start: int, n: int) -> bytes:
        """Bytes [start, start+n) of a tensor's data section. Whole-buffer
        mode slices the in-memory shard; lazy mode preads a bounded window."""
        gtype, dims, off, b, p = self.tensors[name]
        del gtype, dims
        if isinstance(b, bytes):
            return bytes(b[off + start:off + start + n])
        with open(p, "rb") as f:
            f.seek(off + start)
            return f.read(n)

    # -- wire readers -------------------------------------------------------
    @staticmethod
    def _u16(b, o):
        return struct.unpack_from("<H", b, o)[0]

    @staticmethod
    def _u32(b, o):
        return struct.unpack_from("<I", b, o)[0]

    @staticmethod
    def _u64(b, o):
        return struct.unpack_from("<Q", b, o)[0]

    @staticmethod
    def _f32(b, o):
        return struct.unpack_from("<f", b, o)[0]

    def _str(self, b, o):
        n = self._u64(b, o)
        return b[o + 8:o + 8 + n].decode("utf-8"), o + 8 + n

    def _walk_value(self, b, o, vt, depth=0):
        if vt == 0:
            return self._u8(b, o), o + 1
        if vt == 1:
            return self._i8(b, o), o + 1
        if vt == 2:
            return self._u16(b, o), o + 2
        if vt == 3:
            return self._i16(b, o), o + 2
        if vt == 4:
            return self._u32(b, o), o + 4
        if vt == 5:
            return self._i32(b, o), o + 4
        if vt == 6:
            return self._f32(b, o), o + 4
        if vt == 7:
            return b[o], o + 1
        if vt == 8:
            return self._str(b, o)
        if vt == 9:    # array; the D7 quirk: type-5 elements are 4 bytes each
            et = self._u32(b, o)
            n = self._u64(b, o + 4)
            o += 12
            if et == 8:
                vals = []
                for _ in range(n):
                    s, o = self._str(b, o)
                    vals.append(s)
                return vals, o
            esz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                   10: 8, 11: 8, 12: 8}[et]
            vals = list(struct.unpack_from("<%d%s" % (n, {
                0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                7: "B", 10: "Q", 11: "q", 12: "d"}[et]), b, o))
            return vals, o + n * esz
        if vt == 10:
            return self._u64(b, o), o + 8
        if vt == 11:
            return struct.unpack_from("<q", b, o)[0], o + 8
        if vt == 12:
            return struct.unpack_from("<d", b, o)[0], o + 8
        raise ValueError("unknown metadata type %d" % vt)

    def _u8(self, b, o):
        return b[o]

    def _i8(self, b, o):
        return struct.unpack_from("<b", b, o)[0]

    def _i16(self, b, o):
        return struct.unpack_from("<h", b, o)[0]

    def _i32(self, b, o):
        return struct.unpack_from("<i", b, o)[0]

    def _parse(self, p: str):
        with open(p, "rb") as f:
            if self._lazy:
                b = f.read(min(os.path.getsize(p), self.LAZY_HEAD))
            else:
                b = f.read()
        assert b[:4] == b"GGUF", "bad magic in %s" % p
        version = self._u32(b, 4)
        assert version == 3, "GGUF version %d, expected 3" % version
        nt, nkv = self._u64(b, 8), self._u64(b, 16)
        o = 24
        for _ in range(nkv):
            key, o = self._str(b, o)
            vt = self._u32(b, o)
            val, o = self._walk_value(b, o + 4, vt)
            self.kv[key] = (vt, val)
        rel_off = []
        for _ in range(nt):
            name, o = self._str(b, o)
            ndim = self._u32(b, o)
            dims = tuple(self._u64(b, o + 4 + 8 * j) for j in range(ndim))
            o += 4 + 8 * ndim
            gtype = self._u32(b, o)
            off = self._u64(b, o + 4)
            o += 12
            assert name not in self.tensors, "duplicate tensor %s" % name
            self._kinds[name] = gtype
            self._shapes[name] = dims
            rel_off.append((name, gtype, dims, off))
        # data section start: the 32-aligned end of the tensor infos
        dstart = (o + 31) & ~31
        assert o <= dstart <= len(b), "tensor infos overrun in %s" % p
        for name, gtype, dims, off in rel_off:
            if self._lazy:
                self.tensors[name] = (gtype, dims, dstart + off, p, p)
            else:
                self.tensors[name] = (gtype, dims, dstart + off, b, p)

    # -- access -------------------------------------------------------------
    def raw(self, name: str) -> bytes:
        nb = self._nbytes(self._kinds[name], self._shapes[name])
        return self._read_at(name, 0, nb)

    @staticmethod
    def _nbytes(gtype: int, dims: tuple) -> int:
        if gtype == 0:
            qk, bsz = 1, 4
        elif gtype == 8:
            qk, bsz = QK8, BS8
        elif gtype == 19:
            qk, bsz = QKI, BSI
        else:
            raise ValueError("unsupported ggml type %d" % gtype)
        nb = ((dims[0] + qk - 1) // qk) * bsz
        for d in dims[1:]:
            nb *= d
        return nb

    def get_gguf(self, name: str) -> tuple[np.ndarray, int]:
        """Raw tensor -> fp32 [rows][cols...] as stored (dequantized, rows NOT
        padded for IQ1_S: compacted)."""
        gtype, dims = self._kinds[name], self._shapes[name]
        raw = self.raw(name)
        if gtype == 0:
            return np.frombuffer(raw, dtype=np.float32).reshape(dims[::-1]).copy(), 0
        if gtype == 8:
            return dequant_q8_0(raw, dims[0]).reshape(dims[::-1]), 8
        if gtype == 19:
            return dequant_iq1_s(raw, dims[0]).reshape(dims[::-1]), 19
        raise ValueError("unsupported ggml type %d" % gtype)


# ------------------------------------------------- engine-view translation ----
# Mirrors k3_gguf_map.c's contract table for the aligned tiny arch.
ONES = "ones"
KVB = "kvb"

def _layer_num(st_name: str) -> int:
    m = re.match(PRE + r"layers\.(\d+)\.", st_name)
    return int(m.group(1)) if m else -1


def resolve(st_name: str, is_mla: bool, is_dense: bool) -> tuple[str, object]:
    """Engine name -> (GGUF tensor name, kind). kind: "f32"/"q8" (bf16-rounded
    view), "beta" (F32 file, bf16 view), "alog", ONES, KVB."""
    if st_name == PRE + "embed_tokens.weight":
        return "token_embd.weight", "q8"
    if st_name == "language_model.lm_head.weight":
        return "output.weight", "q8"
    if st_name == PRE + "norm.weight":
        return "output_norm.weight", "f32"
    if st_name == PRE + "output_attn_res_norm.weight":
        return "output_res_score.weight", "f32"   # FOLDED vector (see below)
    if st_name == PRE + "output_attn_res_proj.weight":
        return ONES, "f32"
    L = _layer_num(st_name)
    body = st_name[len(PRE + "layers.%d." % L):]
    if body == "input_layernorm.weight":
        return "blk.%d.attn_norm.weight" % L, "f32"
    if body == "post_attention_layernorm.weight":
        return "blk.%d.ffn_norm.weight" % L, "f32"
    if body == "self_attention_res_norm.weight":
        return "blk.%d.attn_res_score.weight" % L, "f32"
    if body == "self_attention_res_proj.weight":
        return ONES, "f32"
    if body == "mlp_res_norm.weight":
        return "blk.%d.ffn_res_score.weight" % L, "f32"
    if body == "mlp_res_proj.weight":
        return ONES, "f32"
    if is_mla:
        mla = {
            "self_attn.q_a_proj.weight":        "blk.%d.attn_q_a.weight",
            "self_attn.q_a_layernorm.weight":   "blk.%d.attn_q_a_norm.weight",
            "self_attn.q_b_proj.weight":        "blk.%d.attn_q_b.weight",
            "self_attn.kv_a_proj_with_mqa.weight": "blk.%d.attn_kv_a_mqa.weight",
            "self_attn.kv_a_layernorm.weight":  "blk.%d.attn_kv_a_norm.weight",
            "self_attn.kv_b_proj.weight":       KVB,
            "self_attn.o_proj.weight":          "blk.%d.attn_output.weight",
            "self_attn.g_proj.weight":          "blk.%d.attn_gate.weight",
        }
        if body in mla:
            g = mla[body]
            return (g % L, "f32") if g != KVB else (KVB, "q8")
    else:
        kda = {
            "self_attn.q_proj.weight":          "blk.%d.attn_q.weight",
            "self_attn.k_proj.weight":          "blk.%d.attn_k.weight",
            "self_attn.v_proj.weight":          "blk.%d.attn_v.weight",
            "self_attn.g_proj.weight":          "blk.%d.ssm_g.weight",
            "self_attn.o_proj.weight":          "blk.%d.attn_output.weight",
            "self_attn.f_a_proj.weight":        "blk.%d.ssm_f_a.weight",
            "self_attn.f_b_proj.weight":        "blk.%d.ssm_f_b.weight",
            "self_attn.b_proj.weight":          "blk.%d.ssm_beta.weight",
            "self_attn.o_norm.weight":          "blk.%d.ssm_norm.weight",
        }
        if body in kda:
            kind = "beta" if body == "self_attn.b_proj.weight" else "q8"
            return kda[body] % L, kind
        if body == "self_attn.A_log":
            return "blk.%d.ssm_a" % L, "alog"
        if body == "self_attn.dt_bias":
            return "blk.%d.ssm_dt.bias" % L, "f32"
        if body == "self_attn.q_conv1d.weight":
            return "blk.%d.ssm_conv1d_q.weight" % L, "f32"
        if body == "self_attn.k_conv1d.weight":
            return "blk.%d.ssm_conv1d_k.weight" % L, "f32"
        if body == "self_attn.v_conv1d.weight":
            return "blk.%d.ssm_conv1d_v.weight" % L, "f32"
    if is_dense:
        dense = {
            "mlp.gate_proj.weight": "blk.%d.ffn_gate.weight",
            "mlp.up_proj.weight":   "blk.%d.ffn_up.weight",
            "mlp.down_proj.weight": "blk.%d.ffn_down.weight",
        }
        if body in dense:
            return dense[body] % L, "q8"
    else:
        moe = {
            "block_sparse_moe.gate.weight":               "blk.%d.ffn_gate_inp.weight",
            "block_sparse_moe.gate.e_score_correction_bias": "blk.%d.exp_probs_b.bias",
            "block_sparse_moe.routed_expert_down_proj.weight": "blk.%d.ffn_routed_down.weight",
            "block_sparse_moe.routed_expert_up_proj.weight": "blk.%d.ffn_routed_up.weight",
            "block_sparse_moe.routed_expert_norm.weight": "blk.%d.ffn_routed_norm.weight",
            "block_sparse_moe.shared_experts.gate_proj.weight": "blk.%d.ffn_gate_shexp.weight",
            "block_sparse_moe.shared_experts.up_proj.weight": "blk.%d.ffn_up_shexp.weight",
            "block_sparse_moe.shared_experts.down_proj.weight": "blk.%d.ffn_down_shexp.weight",
        }
        if body in moe:
            return moe[body] % L, "f32" if "gate_inp" in moe[body] or "bias" in moe[body] else "q8"
    raise KeyError("no contract entry for %s" % st_name)


def kvb_view(sh: GgufShards, L: int, cfg: K3Config) -> np.ndarray:
    """The composite kv_b: engine [n_heads*(qk_nope+v_head)][kv_lora], from
    attn_k_b (per-head TRANSPOSED) + attn_v_b (straight), mirroring gfind_kvb."""
    QN, VH, KL, H = (cfg.qk_nope_head_dim, cfg.v_head_dim,
                     cfg.kv_lora_rank, cfg.num_attention_heads)
    kb = sh.get_gguf("blk.%d.attn_k_b.weight" % L)[0]      # [H][KL][QN]
    vb = sh.get_gguf("blk.%d.attn_v_b.weight" % L)[0]      # [H][VH][KL]
    out = np.empty((H * (QN + VH), KL), dtype=np.float32)
    for h in range(H):
        out[h * (QN + VH):h * (QN + VH) + QN, :] = kb[h].T
        out[h * (QN + VH) + QN:(h + 1) * (QN + VH), :] = vb[h]
    return out


def get_engine_view(sh: GgufShards, st_name: str, cfg: K3Config) -> np.ndarray:
    """The ENGINE's fp32 view of one engine-named tensor (mirror of the GGUF
    finder): bf16-rounded for reqn matrices, exact fp32 for reqw vectors,
    composite kv_b, folded res scores with ones, A_log prefix, requantized
    experts (handled by get_expert_view, not here)."""
    L = _layer_num(st_name)
    is_mla = cfg.is_mla(L) if L >= 0 else False
    is_dense = cfg.is_dense(L) if L >= 0 else False
    gname, kind = resolve(st_name, is_mla, is_dense)
    if gname == ONES:
        return np.ones(cfg.hidden_size, dtype=np.float32)
    if gname == KVB:
        return kvb_view(sh, L, cfg)
    w, _gtype = sh.get_gguf(gname)
    if kind in ("q8", "beta"):
        # bf16 -> fp32 is a LEFT SHIFT: bf16 IS the top 16 bits of the fp32.
        return (f32_to_bf16_rne(w).astype(np.uint32) << 16).view(np.float32)
    if kind == "alog":
        # the file stores ssm_a FOLDED as -exp(A_log); the engine's KDA exp()s
        # A_log at runtime, so the engine view UNFOLDS: A_log = ln(-ssm_a),
        # exactly as k3_gguf_map.c's finder does at bind. Fail loud on a
        # non-negative value (corrupt file).
        a = w[:cfg.kda_num_heads].astype(np.float32)
        assert (a < 0).all(), "ssm_a must be negative (folded -exp(A_log))"
        return np.log(-a).astype(np.float32)
    return w.astype(np.float32)


def get_expert_view(sh: GgufShards, L: int, e: int, which: str,
                    cfg: K3Config) -> np.ndarray:
    """Expert (L, e) matrix `which` in {'w1','w2','w3'}: the fp32 reference
    view from a RAW IQ1_S dequant - NO requant anywhere (the fix wave's Gate-2
    reference independence: the engine now consumes the IQ1_S bytes natively,
    and the reference is the same raw dequant + fp32 matmul the oracle does).
    w1=gate_exps [I][L], w2=down_exps [L][I], w3=up_exps [I][L] (GGUF ne0 =
    cols)."""
    gname = {"w1": "blk.%d.ffn_gate_exps.weight",
             "w2": "blk.%d.ffn_down_exps.weight",
             "w3": "blk.%d.ffn_up_exps.weight"}[which] % L
    gtype, dims, off, b, _p = sh.tensors[gname]
    assert gtype == 19 and len(dims) == 3
    per = sh._nbytes(gtype, dims) // dims[2]
    raw = sh._read_at(gname, e * per, per)
    cols, rows = dims[0], dims[1]
    return dequant_iq1_s(raw, cols).reshape(rows, cols)


def get_expert_view_partial(sh: GgufShards, L: int, e: int, which: str,
                            cfg: K3Config, nrows: int) -> np.ndarray:
    """The first `nrows` rows of expert (L, e) matrix `which` from a RAW
    IQ1_S dequant (bounded real-file reads: a few KB per matrix, the Gate-2
    real leg's window)."""
    gname = {"w1": "blk.%d.ffn_gate_exps.weight",
             "w2": "blk.%d.ffn_down_exps.weight",
             "w3": "blk.%d.ffn_up_exps.weight"}[which] % L
    gtype, dims, off, b, _p = sh.tensors[gname]
    assert gtype == 19 and len(dims) == 3
    per = sh._nbytes(gtype, dims) // dims[2]
    cols = dims[0]
    bpr = (cols + QKI - 1) // QKI
    rowb = bpr * BSI
    raw = sh._read_at(gname, e * per, nrows * rowb)
    full = dequant_iq1_s(raw, cols)          # [nrows][padded cols]
    return full[:, :cols].copy()


# --------------------------------------------------------------- forward ----
def load_model(sh: GgufShards, cfg: K3Config) -> K3Model:
    """Build K3Model(cfg) and load every weight from the GGUF bytes through the
    engine-view translation, so the reference computes exactly the engine's
    weights (fp32 accumulation differences remain, which the repo budget
    covers)."""
    m = K3Model(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if ".mlp.experts." in name and name.endswith(".weight"):
                parts = name.split(".")
                # the ENGINE's view (fix wave): the RAW IQ1_S dequant - the
                # engine multiplies the IQ1_S bytes directly (k3_matmul_iq1_s),
                # no requant anywhere in the reference path
                w = get_expert_view(sh, int(parts[1]), int(parts[4]),
                                    parts[5], cfg)
            else:
                # k3_ref param names -> the engine's safetensors names, then the
                # GGUF engine-view translation (mirror of mtc.engine_name)
                w = get_engine_view(sh, mtc_engine_name(name), cfg)
            p.copy_(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)))
    return m


@torch.no_grad()
def forward_capture(m: K3Model, cfg: K3Config, ids: list[int]):
    """Full forward (K3Model.forward semantics, statement for statement), also
    recording per-MoE-layer the final position's mlp input x, latent z, and the
    router's selection (idx, weights). Returns (logits[final], captures)."""
    B, T, E = 1, len(ids), cfg.hidden_size
    h = m.embed_tokens(torch.tensor([ids]))
    block_residual = h.new_zeros(B * T, 0, E)
    caps: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    eps = cfg.rms_norm_eps
    for i, layer in enumerate(m.layers):
        prefix_sum = h
        if block_residual.shape[1] > 0:
            h = apply_attn_res(prefix_sum.view(-1, E), block_residual,
                               layer.self_attention_res_proj.weight.squeeze(0),
                               layer.self_attention_res_norm.weight,
                               eps).view(B, T, E)
        if i % cfg.attn_res_block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, E).unsqueeze(1)], dim=1)
            prefix_sum = None
        h = layer.input_layernorm(h)
        h, _ = layer.self_attn(h, None)
        prefix_sum = h if prefix_sum is None else prefix_sum + h
        h = apply_attn_res(prefix_sum.view(-1, E), block_residual,
                           layer.mlp_res_proj.weight.squeeze(0),
                           layer.mlp_res_norm.weight, eps).view(B, T, E)
        h = layer.post_attention_layernorm(h)
        if not cfg.is_dense(i):
            mlp = layer.mlp
            flat = h.view(-1, E)
            idx, w = mlp.route(flat)
            z = mlp.down(flat)
            # torch.topk(sorted=False) makes no order promise, but the C test
            # compares positionally against k3_router's order: descending
            # biased score, ties by lower expert index (repeated strict-max).
            # Reorder the captured selection to that contract before storing.
            with torch.no_grad():
                lg = F.linear(flat.float(), mlp.gate.weight.float())
                sc = lg.sigmoid()
                biased = sc + mlp.e_score_correction_bias.float()
            bi = biased[-1].cpu().numpy()
            ii = idx[-1].cpu().numpy()
            order = np.lexsort((ii, -bi[ii]))
            caps[i] = (flat[-1].detach().cpu().numpy().astype(np.float32),
                       z[-1].detach().cpu().numpy().astype(np.float32),
                       ii[order].astype(np.int32),
                       w[-1].detach().cpu().numpy().astype(np.float32)[order])
        h = layer.mlp(h)
        prefix_sum = h if prefix_sum is None else prefix_sum + h
        h = prefix_sum       # DecoderLayer returns the RESIDUAL SUM, not the mlp out
    h = apply_attn_res(h.view(-1, E), block_residual,
                       m.output_attn_res_proj.weight.squeeze(0),
                       m.output_attn_res_norm.weight, eps).view(B, T, E)
    logits = m.lm_head(m.norm(h))[0, -1].float().cpu().numpy()
    return logits, caps


# ------------------------------------------------------------- gate2 golden ----
G2_MAGIC = b"K3G2"
G2_VERSION = 2
# v2 (fix wave): the reference is a RAW IQ1_S dequant + fp32 matmul, NO requant
# anywhere in the reference path (the old v1 carried a requant mirror and a
# measured requant noise, the D2 self-consistency blind spot the bake-off
# caught). Case record: u32 layer, u32 expert, u32 rows, f32 z[L],
# f32 gu[rows], f32 up[rows], f32 edn[rows] (edn present iff flags & 1).
# rows = I (gate/up rows; tiny arch has I == L) in the fixture leg; the real
# leg's bounded window is rows = 16. Router records follow (fixture leg only).
G2_FLAG_HAS_EDN = 1


def _chain_outputs(z, w1, w3, w2, cfg, rows):
    """The first `rows` outputs of the expert chain from RAW dequant fp32
    matrices: gu = z@w1.T, up = z@w3.T, act = situ_glu, edn = act@w2.T (w2's
    first `rows` rows when rows < L). float32 throughout; numpy/torch matmul
    (BLAS) accumulation order - the fp32 class the C test's budget covers."""
    g = (z.astype(np.float32) @ w1.T.astype(np.float32)).astype(np.float32)
    u = (z.astype(np.float32) @ w3.T.astype(np.float32)).astype(np.float32)
    act = situ_glu(torch.from_numpy(np.stack([g, u]).reshape(1, -1)),
                   cfg.situ_beta, cfg.situ_linear_beta).numpy().astype(np.float32)
    edn = (act[:, :rows] @ w2[:rows, :rows].T.astype(np.float32)).astype(np.float32)
    return (g.astype(np.float32), u.astype(np.float32),
            edn.astype(np.float32))


def emit_gate2_golden(gguf_path: str, out_path: str, ids: list[int]):
    """The committed Gate-2 golden (v2): per (layer, expert) the routed input
    z, the RAW-dequant fp32 reference expert outputs (gu, up, edn - NO requant
    anywhere); plus per-MoE-layer router capture. The C test
    (test_gguf_gate.c) asserts the engine's NATIVE k3_matmul_iq1_s chain
    against this file within an fp32-accumulation budget."""
    sh = GgufShards(gguf_path)
    kv = sh.kv
    cfg = cfg_from_kv(kv)
    m = load_model(sh, cfg)
    logits, caps = forward_capture(m, cfg, ids)
    print("gate2 reference: %d layers, argmax %d" % (cfg.num_hidden_layers,
                                                     int(np.argmax(logits))))

    E = cfg.hidden_size
    L = cfg.routed_expert_hidden_size
    moe_i = cfg.moe_intermediate_size
    rows = moe_i                    # gate/up rows; tiny arch has I == L
    moe_layers = sorted(caps)
    cases = []
    for _li, layer in enumerate(moe_layers):
        z = caps[layer][1]
        for e in range(cfg.num_experts):
            w1 = get_expert_view(sh, layer, e, "w1", cfg)
            w3 = get_expert_view(sh, layer, e, "w3", cfg)
            w2 = get_expert_view(sh, layer, e, "w2", cfg)
            gu, up, edn = _chain_outputs(z, w1, w3, w2, cfg, rows)
            cases.append((layer, e, rows, z, gu, up, edn))
    print("gate2 cases: %d experts over %d MoE layers (rows %d)" %
          (len(cases), len(moe_layers), rows))

    with open(out_path, "wb") as f:
        f.write(G2_MAGIC)
        f.write(struct.pack("<IIIIIII", G2_VERSION, len(cases), len(moe_layers),
                            E, L, moe_i, G2_FLAG_HAS_EDN))
        for layer, e, r, z, gu, up, edn in cases:
            f.write(struct.pack("<III", layer, e, r))
            f.write(z.astype(np.float32).tobytes())
            f.write(gu.astype(np.float32).tobytes())
            f.write(up.astype(np.float32).tobytes())
            f.write(edn.astype(np.float32).tobytes())
        for layer in moe_layers:
            x, _z, idx, w = caps[layer]
            f.write(struct.pack("<I", layer))
            f.write(x.astype(np.float32).tobytes())
            f.write(idx.astype(np.int32).tobytes())
            f.write(w.astype(np.float32).tobytes())
    print("wrote %s" % out_path)
    return out_path


# ------------------------------------------------- gate2 real-bytes golden ----
def emit_gate2_real_golden(gguf_dir: str, out_path: str, layer: int = 1,
                           experts: tuple = (0, 1, 895), rows: int = 16):
    """The real-file Gate-2 golden (v2, flags & 1 == 0): RAW IQ1_S bytes from
    the REAL 14-shard file, bounded to a few KB per matrix (the first `rows`
    rows of each expert window), dequantized and dotted in fp32 (numpy) - the
    llama.cpp-math reference the engine's native kernel must sit at. z is a
    deterministic LCG vector stored in the golden, so the C test replays it.
    Router records are absent (nmoe = 0). The reader runs LAZY: headers +
    infos only, per-request pread windows - never the data section."""
    sh = GgufShards(gguf_dir, lazy=True)
    kv = sh.kv
    cfg = cfg_from_kv(kv)
    E = cfg.hidden_size
    L = cfg.routed_expert_hidden_size
    moe_i = cfg.moe_intermediate_size

    # deterministic z, same LCG the C test uses for its activation vectors
    z = np.empty(L, dtype=np.float32)
    st = np.uint32(0x12345678)
    for i in range(L):
        st = np.uint32((int(st) * 1664525 + 1013904223) & 0xFFFFFFFF)
        z[i] = (np.float32(st >> np.uint32(8)) / np.float32(16777216.0) -
                np.float32(0.5)) * np.float32(0.08)

    cases = []
    for e in experts:
        w1 = get_expert_view_partial(sh, layer, e, "w1", cfg, rows)
        w3 = get_expert_view_partial(sh, layer, e, "w3", cfg, rows)
        w2 = get_expert_view_partial(sh, layer, e, "w2", cfg, rows)
        gu, up, _edn = _chain_outputs(z, w1, w3, w2, cfg, rows)
        cases.append((layer, e, rows, z, gu, up))
    with open(out_path, "wb") as f:
        f.write(G2_MAGIC)
        f.write(struct.pack("<IIIIIII", G2_VERSION, len(cases), 0,
                            E, L, moe_i, 0))
        for layer, e, r, zz, gu, up in cases:
            f.write(struct.pack("<III", layer, e, r))
            f.write(zz.astype(np.float32).tobytes())
            f.write(gu.astype(np.float32).tobytes())
            f.write(up.astype(np.float32).tobytes())
    print("wrote %s (%d real-bytes cases, %d rows each)" %
          (out_path, len(cases), rows))
    return out_path


def forward_ref(path: str, ids: list[int], outp: str) -> int:
    """Full-model reference logits for a GGUF fixture, computed FROM ITS BYTES
    (engine-view weights; see load_model). Writes the same json shape as
    make_tiny_checkpoint/ref_forward: prompt_ids, argmax, vocab, logits_bits."""
    sh = GgufShards(path)
    kv = sh.kv
    cfg = cfg_from_kv(kv)
    m = load_model(sh, cfg)
    logits, _ = forward_capture(m, cfg, ids)
    tok = int(np.argmax(logits))
    with open(outp, "w") as f:
        json.dump({"prompt_ids": ids, "argmax": tok, "vocab": int(cfg.vocab_size),
                   "logits_bits": np.asarray(logits, dtype=np.float32)
                   .view(np.uint32).tolist()}, f)
    print("reference: argmax %d, wrote %s" % (tok, outp))
    return 0


def cfg_from_kv(kv: dict) -> K3Config:
    """K3Config from the shard-1 kimi-k3.* keys (mirror of k3_gguf_cfg)."""
    return K3Config(
        hidden_size=kv["kimi-k3.embedding_length"][1],
        num_hidden_layers=kv["kimi-k3.block_count"][1],
        vocab_size=kv["kimi-k3.vocab_size"][1],
        rms_norm_eps=kv["kimi-k3.attention.layer_norm_rms_epsilon"][1],
        kda_num_heads=kv["kimi-k3.attention.head_count"][1],
        kda_head_dim=kv["kimi-k3.kda.head_dim"][1],
        short_conv_kernel_size=kv["kimi-k3.ssm.conv_kernel"][1],
        gate_lower_bound=kv["kimi-k3.kda.gate_lower_bound"][1],
        num_attention_heads=kv["kimi-k3.attention.head_count"][1],
        q_lora_rank=kv["kimi-k3.attention.q_lora_rank"][1],
        kv_lora_rank=kv["kimi-k3.attention.kv_lora_rank"][1],
        qk_nope_head_dim=kv["kimi-k3.attention.key_length_mla"][1] -
                         kv["kimi-k3.rope.dimension_count"][1],
        qk_rope_head_dim=kv["kimi-k3.rope.dimension_count"][1],
        v_head_dim=kv["kimi-k3.attention.value_length_mla"][1],
        num_experts=kv["kimi-k3.expert_count"][1],
        num_experts_per_token=kv["kimi-k3.expert_used_count"][1],
        num_shared_experts=kv["kimi-k3.expert_shared_count"][1],
        routed_expert_hidden_size=kv["kimi-k3.expert_latent_length"][1],
        moe_intermediate_size=kv["kimi-k3.expert_feed_forward_length"][1],
        first_k_dense_replace=kv["kimi-k3.leading_dense_block_count"][1],
        intermediate_size=kv["kimi-k3.feed_forward_length"][1],
        attn_res_block_size=kv["kimi-k3.attn_res.block_size"][1],
        full_attn_layers=[i + 1 for i, v in
                          enumerate(kv["kimi-k3.attention.head_count_kv"][1]) if v],
        routed_scaling_factor=kv["kimi-k3.expert_weights_scale"][1],
        moe_renormalize=bool(kv["kimi-k3.expert_weights_norm"][1]),
    )


# -------------------------------------------------------------------- cli ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="GGUF fixture: a single .gguf file or a shard dir")
    ap.add_argument("ids", nargs="?", default="3,7,11,5,9",
                    help="comma-separated prompt ids (default 3,7,11,5,9)")
    ap.add_argument("out", nargs="?", default=None, help="output json")
    ap.add_argument("--gate2", metavar="OUT_BIN", default=None,
                    help="emit the Gate-2 golden file instead of logits")
    ap.add_argument("--gate2-real", metavar="OUT_BIN", default=None,
                    help="emit the Gate-2 golden from REAL shard bytes (bounded "
                         "windows, layer 1 experts 0/1/895, 16 rows)")
    a = ap.parse_args()
    ids = [int(v) for v in a.ids.split(",") if v != ""]
    if a.gate2:
        emit_gate2_golden(a.path, a.gate2, ids)
        return 0
    if a.gate2_real:
        emit_gate2_real_golden(a.path, a.gate2_real)
        return 0
    outp = a.out or "ref_logits_gguf.json"
    return forward_ref(a.path, ids, outp)


if __name__ == "__main__":
    sys.exit(main())
