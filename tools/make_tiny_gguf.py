#!/usr/bin/env python
"""make_tiny_gguf.py - tiny GGUF fixture generator for the GGUF parity gates (P3).

MIRRORS make_tiny_checkpoint.py: the SAME torch reference weights (same seed, same
init recipe) transformed to the on-disk dtypes of the REAL 554 GB file's production
path - trunk matrices Q8_0, the three merged MoE expert tensors IQ1_S (3-D along
the expert dim), small vectors F32, the router F32 - so the safetensors path and
the GGUF path load THE SAME weights, quantized through the two different lossy
encodings PARITY GATE 1 is designed to compare.

THE ALIGNED TINY ARCH
    make_tiny_checkpoint.ALIGNED_DIMS widens the oracle's head widths (kda_head_dim
    16, qk_nope 24, v_head 16) to 32-multiples. The GGUF map's Q8_0 dequant writes
    UNPADDED rows, so every row length must be a multiple of the 32-block, and the
    composite kv_b additionally needs qk_nope/v_head multiples of 32
    (k3_gguf_map.c row_aligned/kvb_validate). The real file's dims are all
    block-aligned; the oracle's 16/24 widths do not exist in it. The variant keeps
    every architectural feature and the named tiny dims (hidden 128, layers 13,
    vocab 256, experts 8 top2 shared2, latent 64, moe_inter 64, dense_inter 96,
    attn_res 3, first_dense 1, situ 4.0/25.0). The ST side of the parity pair is
    generated with make_tiny_checkpoint.py --aligned-dims from the SAME config.

QUANTIZERS (deterministic, documented)
    Q8_0  per 32-block: d = fp16(amax/127), q = rint(x/fp32(d)) clipped to [-127,
          127]; d = 0 for an all-zero block. (The C dequant only reads these
          bytes; the fixture writer's choice is documented, not contractual.)
    IQ1_S per 256-block: d = fp16(amax/24); per 32-value sub-block the scale k
          in 0..7 is searched over {the smallest k whose top level covers the
          sub-block max} union {k whose level 1.125*d*(2k+1) sits near 0.9*rms}
          plus its neighbours, choosing the lowest total SSE; the delta sign
          (+/-0.125, qh bit 15) is chosen per sub-block by lower SSE; per
          8-value group the grid index (0..2047) minimizing SSE over the FULL
          2048-entry iq1s_grid, sourced from src/core/iq1s_grid.h - the same
          bytes the C dequant reads. Ties resolve to the first minimum
          (np.argmin), so the encoder is bit-deterministic. Measured on this
          fixture: ~0.45 relative L2 vs the original fp32 - the format's own
          floor (4 levels spaced 1:9, per-value error bounded by 0.5*dl); the
          logits-level error is what the gates measure.

CONTAINER
    The GGUF v3 container is written by hand (the gguf 0.19.0 package cannot
    express IQ1_S tensors whose row length is not a 256-multiple - the tiny arch's
    latent/moe_inter are 64 - and the real writer's quirks must be mirrored
    exactly): metadata walk with the verified unsloth-fork quirks (ARRAY elements
    declared as type 5 are 4 bytes each - kimi-k3.attention.head_count_kv and
    tokenizer.ggml.token_type), split.* keys with their exact wire types, 32-byte
    alignment with zero padding. The metadata walk was cross-checked against the
    gguf 0.19.0 package's reader during development (it cannot read IQ1_S
    tensors with 64-wide rows, which is why the container is hand-written); the
    authoritative reader is the C one the test suite runs.

EMITS (all under tests/fixtures/, paths from tools/_paths.py)
    tiny_gguf/            single-shard fixture (split.* keys present, so both the
                          directory open and the single-file open accept it)
    tiny_gguf_multi/      3 shards, shard 1 metadata-only (llama.cpp convention),
                          the same weights split across shards 2 and 3
    tiny_gguf_bf16trunk/  single-shard BIT-EXACT SUB-GATE variant: the trunk is
                          written as F32 holding the ST path's bf16 values widened
                          (the engine's F32->bf16 RNE dequant is the identity on
                          bf16-representable values, so the engine's trunk bytes
                          equal the ST file's bf16 bytes bit-for-bit); experts and
                          reqw vectors unchanged
    gguf_gate2_golden.bin the PARITY GATE 2 golden (see ref_gguf_experts.py)
    <fixture>/ref_logits_gguf.json   full-model torch reference computed FROM THE
                          SAME GGUF BYTES (ref_gguf_experts.py)

The quantizer error is measured and printed (Q8_0 and IQ1_S relative L2 vs the
original fp32, plus the D2 requant noise: MXFP4(IQ1_S dequant) vs IQ1_S dequant).

usage: make_tiny_gguf.py [--seed N] [--prompt-ids a,b,c] [--out DIR]
       DIR defaults to tests/fixtures (tools/_paths.py).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import _paths                                                     # noqa: E402
import make_tiny_checkpoint as mtc                               # noqa: E402
import ref_gguf_experts as rge                                    # noqa: E402

QK8, BS8 = 32, 34          # Q8_0 block geometry
QKI, BSI = 256, 50         # IQ1_S block geometry
IQ1S_DELTA = 0.125


def quantize_q8_0(w: np.ndarray) -> bytes:
    """fp32 [rows][cols] -> Q8_0 bytes (cols must be a 32-multiple; every trunk
    row length of the aligned tiny arch is)."""
    w = w.astype(np.float32)
    rows, cols = w.shape
    assert cols % QK8 == 0
    g = cols // QK8
    blocks = w.reshape(rows, g, QK8)
    amax = np.abs(blocks).max(axis=2)
    d32 = np.where(amax == 0, 0.0, amax / 127.0)
    d16 = d32.astype(np.float16)
    d = d16.astype(np.float32)
    d_safe = np.where(d == 0, 1.0, d)
    q = np.rint(blocks / d_safe[..., None])
    q = np.where((amax == 0)[..., None], 0, q)
    q = np.clip(q, -127, 127).astype(np.int8)
    h = d16.view(np.uint16).astype(np.uint32)
    out = np.empty((rows, g, BS8), dtype=np.uint8)
    out[..., 0] = (h & 0xFF).astype(np.uint8)
    out[..., 1] = (h >> 8).astype(np.uint8)
    out[..., 2:] = q.view(np.uint8)
    return out.reshape(rows, g * BS8).tobytes()


# ------------------------------------------------------------- quant ---- (IQ1_S)
def _encode_iq1s_block(x: np.ndarray, grid: np.ndarray) -> bytes:
    """One 256-value IQ1_S block -> 50 bytes. See the module docstring for the
    exact rules; every choice is deterministic (first-minimum ties).

    The per-sub-block scale k is searched over a small candidate set: the
    smallest k whose top level covers the sub-block maximum, plus the k whose
    level 1.125*d*(2k+1) sits nearest 0.9*rms (the near-optimal scale for the
    4-level {+/-0.125, +/-1.125} grid; measured, the exhaustive 8-k search wins
    only ~0.002 rel-L2). d = amax/24 (measured slightly better than amax/16).
    The format's own floor on Gaussian data is ~0.45 rel-L2 (4 levels spaced
    1:9, per-value error bounded by 0.5*dl); see the report for the numbers."""
    assert x.size == QKI
    amax = float(np.abs(x).max())
    d16 = np.float16(amax / 24.0) if amax > 0 else np.float16(0.0)
    df = np.float32(d16)
    qs = np.zeros(32, dtype=np.uint8)
    qh = np.zeros(8, dtype=np.uint16)   # 8 sub-blocks of 32 values, 16 bytes

    gplus = (grid.astype(np.float32) + IQ1S_DELTA)   # (2048, 8) for dl = 1
    gminus = (grid.astype(np.float32) - IQ1S_DELTA)

    for ib in range(8):
        sub = x[ib * 32:(ib + 1) * 32]
        m = float(np.abs(sub).max())
        if m <= 0.0:
            continue                                # k=0, idx=0, sign=+ : zeros
        kcover = int(np.ceil(m / (1.125 * df) / 2.0 - 0.5))
        rms = float(np.sqrt((sub * sub).mean()))
        kt = round((0.9 * rms / df - 1.0) / 2.0)
        cands = sorted({max(0, min(7, k)) for k in (kcover, kt - 1, kt, kt + 1)})
        best = None
        for k in cands:
            dl = df * np.float32(2 * k + 1)
            # per-group argmin over the full grid, BOTH delta signs at once:
            # cost[group][sign] = min over grid of SSE. One vectorized pass.
            groups = sub.reshape(4, 8)
            cand = np.stack([gplus * dl, gminus * dl])      # (2, 2048, 8)
            diff = cand[None, :, :, :] - groups[:, None, None, :]
            cost = (diff * diff).sum(axis=3)                # (4, 2, 2048)
            best_idx = np.argmin(cost, axis=2)              # (4, 2)
            best_cost = np.take_along_axis(
                cost, best_idx[:, :, None], axis=2)[:, :, 0]
            total = float(best_cost.sum())
            if best is None or total < best[0]:
                best = (total, k, best_idx)
        _tot, k, best_idx = best
        dl = df * np.float32(2 * k + 1)
        cand = np.stack([gplus * dl, gminus * dl])
        diff = cand[None, :, :, :] - sub.reshape(4, 8)[:, None, None, :]
        cost = (diff * diff).sum(axis=3)
        best_cost = np.take_along_axis(
            cost, best_idx[:, :, None], axis=2)[:, :, 0]
        sign = int(np.argmin(best_cost.sum(axis=0)))        # + first on a tie
        for li in range(4):
            idx = int(best_idx[li, sign])
            qs[4 * ib + li] = idx & 0xFF
            qh[ib] |= np.uint16(((idx >> 8) & 0x7) << (3 * li))
        qh[ib] |= np.uint16(k << 12)
        if sign == 1:
            qh[ib] |= np.uint16(0x8000)
    out = bytearray(BSI)
    out[0:2] = struct.pack("<H", int(d16.view(np.uint16)))
    out[2:34] = qs.tobytes()
    out[34:50] = qh.astype(np.uint16).tobytes()
    return bytes(out)


def quantize_iq1_s(w: np.ndarray, grid: np.ndarray) -> bytes:
    """fp32 -> IQ1_S bytes: accepts [rows][cols] or [E][rows][cols] (merged
    experts; each expert's window is rows*ceil(cols/256) blocks, contiguous).
    Rows are padded with zero blocks (the engine discards the padding;
    deterministic)."""
    w = w.astype(np.float32)
    if w.ndim == 3:
        n_exp, rows, cols = w.shape
        w = w.reshape(n_exp * rows, cols)
    else:
        rows, cols = w.shape
    bpr = (cols + QKI - 1) // QKI
    out = bytearray()
    padded = np.zeros(bpr * QKI, dtype=np.float32)
    for r in range(w.shape[0]):
        padded[:cols] = w[r]
        for bi in range(bpr):
            out += _encode_iq1s_block(padded[bi * QKI:(bi + 1) * QKI], grid)
    return bytes(out)


# ------------------------------------------------------------ container ----
class GgufWriter:
    """Hand-written GGUF v3 container: mirrors the verified real-file layout
    (metadata walk with the D7 quirks, split keys with exact wire types, 32-byte
    alignment, zero padding, offsets relative to the data section)."""

    def __init__(self):
        self.kvs: list[tuple[str, int, bytes]] = []
        self.tensors: list[tuple[str, int, tuple, int, bytes]] = []

    def _kv(self, key, vt, payload):
        kb = struct.pack("<Q", len(key)) + key.encode()
        self.kvs.append((key, vt, kb + struct.pack("<I", vt) + payload))

    def add_u16(self, k, v):     self._kv(k, 2,  struct.pack("<H", v))
    def add_i32(self, k, v):     self._kv(k, 5,  struct.pack("<i", v))
    def add_u32(self, k, v):     self._kv(k, 4,  struct.pack("<I", v))
    def add_f32(self, k, v):     self._kv(k, 6,  struct.pack("<f", v))
    def add_bool(self, k, v):    self._kv(k, 7,  bytes([1 if v else 0]))
    def add_str(self, k, v):
        vb = v.encode()
        self._kv(k, 8, struct.pack("<Q", len(vb)) + vb)

    def add_arr_i32(self, k, vals):
        """ARRAY of type-5 (int32) elements, 4 bytes each on the wire - the
        VERIFIED unsloth-fork quirk (head_count_kv, token_type)."""
        payload = struct.pack("<I", 5) + struct.pack("<Q", len(vals))
        payload += b"".join(struct.pack("<i", int(v)) for v in vals)
        self._kv(k, 9, payload)

    def add_arr_str(self, k, vals):
        payload = struct.pack("<I", 8) + struct.pack("<Q", len(vals))
        for v in vals:
            vb = v.encode()
            payload += struct.pack("<Q", len(vb)) + vb
        self._kv(k, 9, payload)

    def add_tensor(self, name, ne, gtype, data):
        """ne is GGUF order (fastest first); gtype 0/8/19; data the raw bytes."""
        self.tensors.append((name, len(ne), tuple(int(d) for d in ne), gtype, data))

    def write(self, path, split_no, split_count, split_tensors):
        with open(path, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.kvs)))
            for _, _, rec in self.kvs:
                f.write(rec)
            off = 0
            for name, ndim, ne, gtype, data in self.tensors:
                f.write(struct.pack("<Q", len(name)) + name.encode())
                f.write(struct.pack("<I", ndim))
                for d in ne:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", gtype))
                f.write(struct.pack("<Q", off))
                off += (len(data) + 31) & ~31
            while f.tell() % 32:
                f.write(b"\x00")
            for _name, _ndim, _ne, _gtype, data in self.tensors:
                f.write(data)
                pad = (32 - len(data) % 32) % 32
                if pad:
                    f.write(b"\x00" * pad)
        # keep the object reusable for the next shard
        self.kvs = []
        self.tensors = []


def split_kvs(w: GgufWriter, no: int, count: int, tensors: int):
    w.add_u16("split.no", no)
    w.add_i32("split.tensors.count", tensors)
    w.add_u16("split.count", count)


# --------------------------------------------------------------- model ----
def build_tensors(cfg, model, grid, trunk_mode):
    """Lay out every tensor of the aligned tiny arch in GGUF order. trunk_mode:
    'q8' (production: trunk Q8_0) or 'bf16eq' (bit-exact sub-gate: trunk F32
    holding the ST path's bf16 values widened). Returns (name, ne, gtype, bytes)
    in deterministic file order: model level, then layers 0..12."""
    out = []

    def f32_of(p):
        return p.detach().float().cpu().numpy().astype(np.float32)

    def add(name, ne, gtype, data):
        out.append((name, ne, gtype, data))

    def add_mat(gguf_name, np_name, ne, reqn):
        """ne = GGUF order (cols, rows, ...). reqn trunk matrices: Q8_0 or the
        bf16-equivalent F32 variant; reqw vectors: exact F32."""
        w = f32_of(model.get_parameter(np_name))
        if reqn:
            if trunk_mode == "bf16eq":
                bf16 = (w.view(np.uint32) >> 16).astype(np.uint32)
                data = (bf16.astype(np.uint32) << 16).view(np.float32).tobytes()
                add(gguf_name, ne, 0, data)          # F32 = widened bf16
            else:
                add(gguf_name, ne, 8, quantize_q8_0(w))
        else:
            add(gguf_name, ne, 0, w.tobytes())

    H = cfg.hidden_size
    P = cfg.kda_num_heads * cfg.kda_head_dim
    D = cfg.kda_head_dim
    QL = cfg.q_lora_rank
    KVL = cfg.kv_lora_rank
    QN = cfg.qk_nope_head_dim
    QR = cfg.qk_rope_head_dim
    VH = cfg.v_head_dim
    NH = cfg.num_attention_heads
    E = cfg.num_experts
    lat = cfg.routed_expert_hidden_size
    inter = cfg.moe_intermediate_size
    DI = cfg.intermediate_size
    SI = inter * cfg.num_shared_experts
    CK = cfg.short_conv_kernel_size
    V = cfg.vocab_size

    # ---- model level ----
    add_mat("token_embd.weight", "embed_tokens.weight", (H, V), True)
    add_mat("output.weight", "lm_head.weight", (H, V), True)
    add("output_norm.weight", (H,), 0, f32_of(model.get_parameter("norm.weight")).tobytes())
    add("output_res_score.weight", (H,), 0,
        _folded(model, "output_attn_res_norm.weight", "output_attn_res_proj.weight"))

    for layer_idx in range(cfg.num_hidden_layers):
        pre = "layers.%d." % layer_idx
        is_mla = cfg.is_mla(layer_idx)
        is_dense = cfg.is_dense(layer_idx)

        def lp(s, pre=pre):  # parameter name helper (pre bound per iteration)
            return model.get_parameter(pre + s)

        add("blk.%d.attn_norm.weight" % layer_idx, (H,), 0,
            f32_of(lp("input_layernorm.weight")).tobytes())
        add("blk.%d.ffn_norm.weight" % layer_idx, (H,), 0,
            f32_of(lp("post_attention_layernorm.weight")).tobytes())
        add("blk.%d.attn_res_score.weight" % layer_idx, (H,), 0,
            _folded(model, pre + "self_attention_res_norm.weight",
                    pre + "self_attention_res_proj.weight"))
        add("blk.%d.ffn_res_score.weight" % layer_idx, (H,), 0,
            _folded(model, pre + "mlp_res_norm.weight", pre + "mlp_res_proj.weight"))

        if is_mla:
            add_mat("blk.%d.attn_q_a.weight" % layer_idx,
                    pre + "self_attn.q_a_proj.weight", (H, QL), True)
            add("blk.%d.attn_q_a_norm.weight" % layer_idx, (QL,), 0,
                f32_of(lp("self_attn.q_a_layernorm.weight")).tobytes())
            add_mat("blk.%d.attn_q_b.weight" % layer_idx,
                    pre + "self_attn.q_b_proj.weight", (QL, NH * (QN + QR)), True)
            add_mat("blk.%d.attn_kv_a_mqa.weight" % layer_idx,
                    pre + "self_attn.kv_a_proj_with_mqa.weight", (H, KVL + QR), True)
            add("blk.%d.attn_kv_a_norm.weight" % layer_idx, (KVL,), 0,
                f32_of(lp("self_attn.kv_a_layernorm.weight")).tobytes())
            # kv_b composite halves: k per head TRANSPOSED, v straight (the
            # engine reassembles; see k3_gguf_map.c gfind_kvb)
            kb = f32_of(lp("self_attn.kv_b_proj.weight"))            # [NH*(QN+VH)][KVL]
            k_b = np.stack([kb[h * (QN + VH):h * (QN + VH) + QN, :].T
                            for h in range(NH)])                     # [NH][KVL][QN]
            v_b = np.stack([kb[h * (QN + VH) + QN:(h + 1) * (QN + VH), :]
                            for h in range(NH)])                     # [NH][VH][KVL]
            if trunk_mode == "bf16eq":
                add("blk.%d.attn_k_b.weight" % layer_idx, (QN, KVL, NH), 0,
                    _bf16eq(k_b).tobytes())
                add("blk.%d.attn_v_b.weight" % layer_idx, (KVL, VH, NH), 0,
                    _bf16eq(v_b).tobytes())
            else:
                add("blk.%d.attn_k_b.weight" % layer_idx, (QN, KVL, NH), 8,
                    quantize_q8_0(k_b.reshape(NH * KVL, QN)))
                add("blk.%d.attn_v_b.weight" % layer_idx, (KVL, VH, NH), 8,
                    quantize_q8_0(v_b.reshape(NH * VH, KVL)))
            add_mat("blk.%d.attn_output.weight" % layer_idx,
                    pre + "self_attn.o_proj.weight", (P, H), True)
            add_mat("blk.%d.attn_gate.weight" % layer_idx,
                    pre + "self_attn.g_proj.weight", (H, P), True)
        else:
            add_mat("blk.%d.attn_q.weight" % layer_idx,
                    pre + "self_attn.q_proj.weight", (H, P), True)
            add_mat("blk.%d.attn_k.weight" % layer_idx,
                    pre + "self_attn.k_proj.weight", (H, P), True)
            add_mat("blk.%d.attn_v.weight" % layer_idx,
                    pre + "self_attn.v_proj.weight", (H, P), True)
            add_mat("blk.%d.ssm_g.weight" % layer_idx,
                    pre + "self_attn.g_proj.weight", (H, P), True)
            add_mat("blk.%d.attn_output.weight" % layer_idx,
                    pre + "self_attn.o_proj.weight", (P, H), True)
            for nm, gg in (("q", "q"), ("k", "k"), ("v", "v")):
                add("blk.%d.ssm_conv1d_%s.weight" % (layer_idx, gg), (CK, 1, P), 0,
                    f32_of(lp("self_attn.%s_conv1d.weight" % nm)).tobytes())
            add_mat("blk.%d.ssm_f_a.weight" % layer_idx,
                    pre + "self_attn.f_a_proj.weight", (H, D), True)
            add_mat("blk.%d.ssm_f_b.weight" % layer_idx,
                    pre + "self_attn.f_b_proj.weight", (D, P), True)
            bproj = f32_of(lp("self_attn.b_proj.weight"))
            if trunk_mode == "bf16eq":
                # the ST file stores b_proj as bf16 (truncated); the bf16trunk
                # variant must carry those exact values so the engine's F32->
                # bf16 RNE dequant is the identity on them
                bproj = _bf16eq(bproj)
            add("blk.%d.ssm_beta.weight" % layer_idx, (H, cfg.kda_num_heads), 0,
                bproj.tobytes())
            # The REAL checkpoint ships ssm_a FOLDED as -exp(A_log) (unsloth's
            # converter, conversion/kimi_k3.py:333-336: ssm_a = -exp(A_log));
            # llama.cpp consumes the fold directly. The engine's KDA exp()s
            # A_log at runtime (k3_kda_decay), so the map UNFOLDS at bind
            # (A_log = ln(-ssm_a), k3_gguf_map.c). The fixture must carry the
            # same folded bytes as the real file or the unfold is untestable.
            alog = f32_of(lp("self_attn.A_log"))
            add("blk.%d.ssm_a" % layer_idx, (cfg.kda_num_heads,), 0,
                (-np.exp(alog)).astype(np.float32).tobytes())
            add("blk.%d.ssm_dt.bias" % layer_idx, (P,), 0,
                f32_of(lp("self_attn.dt_bias")).tobytes())
            add("blk.%d.ssm_norm.weight" % layer_idx, (D,), 0,
                f32_of(lp("self_attn.o_norm.weight")).tobytes())

        if is_dense:
            add_mat("blk.%d.ffn_gate.weight" % layer_idx,
                    pre + "mlp.gate_proj.weight", (H, DI), True)
            add_mat("blk.%d.ffn_up.weight" % layer_idx,
                    pre + "mlp.up_proj.weight", (H, DI), True)
            add_mat("blk.%d.ffn_down.weight" % layer_idx,
                    pre + "mlp.down_proj.weight", (DI, H), True)
        else:
            add("blk.%d.ffn_gate_inp.weight" % layer_idx, (H, E), 0,
                f32_of(lp("mlp.gate.weight")).tobytes())
            add("blk.%d.exp_probs_b.bias" % layer_idx, (E,), 0,
                f32_of(lp("mlp.e_score_correction_bias")).tobytes())
            add_mat("blk.%d.ffn_routed_down.weight" % layer_idx,
                    pre + "mlp.down.weight", (H, lat), True)
            add_mat("blk.%d.ffn_routed_up.weight" % layer_idx,
                    pre + "mlp.up.weight", (lat, H), True)
            add("blk.%d.ffn_routed_norm.weight" % layer_idx, (lat,), 0,
                f32_of(lp("mlp.norm.weight")).tobytes())
            add_mat("blk.%d.ffn_gate_shexp.weight" % layer_idx,
                    pre + "mlp.shared.w1.weight", (H, SI), True)
            add_mat("blk.%d.ffn_up_shexp.weight" % layer_idx,
                    pre + "mlp.shared.w3.weight", (H, SI), True)
            add_mat("blk.%d.ffn_down_shexp.weight" % layer_idx,
                    pre + "mlp.shared.w2.weight", (SI, H), True)
            # merged experts: gate/up ne [latent, moe_inter, experts], down
            # [moe_inter, latent, experts]; expert e owns a contiguous window
            for which, gg, ne in (("w1", "ffn_gate_exps", (lat, inter, E)),
                                  ("w3", "ffn_up_exps", (lat, inter, E)),
                                  ("w2", "ffn_down_exps", (inter, lat, E))):
                mat = np.stack([f32_of(lp("mlp.experts.%d.%s.weight"
                                          % (e, which))) for e in range(E)])
                add("blk.%d.%s.weight" % (layer_idx, gg), ne, 19,
                    quantize_iq1_s(mat, grid))
    return out


def _bf16eq(w: np.ndarray) -> np.ndarray:
    """The ST path's bf16-truncated values, widened to fp32 (exactly
    representable, so the engine's F32->bf16 RNE dequant is the identity)."""
    return ((w.view(np.uint32) >> 16).astype(np.uint32) << 16).view(np.float32)


def _folded(model, norm_name, proj_name) -> bytes:
    """The *_res_score folded vector: fp32 norm.weight * proj.weight, computed
    from the exact fp32 values the ST checkpoint stores (the engine folds the
    same product at runtime; k3_attn_res)."""
    n = model.get_parameter(norm_name).detach().float().cpu().numpy().astype(np.float32)
    p = model.get_parameter(proj_name).detach().float().cpu().numpy().astype(np.float32)
    return (n * p.reshape(-1)).tobytes()


# ------------------------------------------------------------- fixtures ----
def config_kvs(cfg, ids) -> list:
    """The full verified kimi-k3.* inventory with the aligned tiny values, the
    tokenizer keys (D7 quirk arrays included) and the split keys (added per
    shard by the callers). Mirrors the real file's key set."""
    hck = [1 if (i + 1) in cfg.full_attn_layers else 0 for i in range(cfg.num_hidden_layers)]
    tokens = ["tok%03d" % i for i in range(cfg.vocab_size)]
    merges = ["tok%03d tok%03d" % (2 * i, 2 * i + 1) for i in range(cfg.vocab_size // 2 - 1)]
    return [
        ("general.architecture", "str", "kimi-k3"),
        ("general.file_type", "u32", 24),
        ("kimi-k3.block_count", "u32", cfg.num_hidden_layers),
        ("kimi-k3.context_length", "u32", 4096),
        ("kimi-k3.embedding_length", "u32", cfg.hidden_size),
        ("kimi-k3.feed_forward_length", "u32", cfg.intermediate_size),
        ("kimi-k3.attention.head_count", "u32", cfg.kda_num_heads),
        ("kimi-k3.attention.head_count_kv", "arr_i32", hck),
        ("kimi-k3.rope.freq_base", "f32", 10000.0),
        ("kimi-k3.attention.layer_norm_rms_epsilon", "f32", cfg.rms_norm_eps),
        ("kimi-k3.expert_count", "u32", cfg.num_experts),
        ("kimi-k3.expert_used_count", "u32", cfg.num_experts_per_token),
        ("kimi-k3.expert_group_used_count", "u32", 1),
        ("kimi-k3.expert_gating_func", "u32", 2),
        ("kimi-k3.attention.key_length", "u32", cfg.qk_nope_head_dim),
        ("kimi-k3.attention.value_length", "u32", cfg.v_head_dim),
        ("kimi-k3.vocab_size", "u32", cfg.vocab_size),
        ("kimi-k3.ssm.conv_kernel", "u32", cfg.short_conv_kernel_size),
        ("kimi-k3.kda.head_dim", "u32", cfg.kda_head_dim),
        ("kimi-k3.kda.gate_lower_bound", "f32", cfg.gate_lower_bound),
        ("kimi-k3.attention.q_lora_rank", "u32", cfg.q_lora_rank),
        ("kimi-k3.attention.kv_lora_rank", "u32", cfg.kv_lora_rank),
        ("kimi-k3.rope.dimension_count", "u32", cfg.qk_rope_head_dim),
        ("kimi-k3.attention.key_length_mla", "u32",
         cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),
        ("kimi-k3.attention.value_length_mla", "u32", cfg.v_head_dim),
        ("kimi-k3.expert_feed_forward_length", "u32", cfg.moe_intermediate_size),
        ("kimi-k3.expert_shared_count", "u32", cfg.num_shared_experts),
        ("kimi-k3.leading_dense_block_count", "u32", cfg.first_k_dense_replace),
        ("kimi-k3.expert_weights_scale", "f32", cfg.routed_scaling_factor),
        ("kimi-k3.expert_weights_norm", "bool", 1 if cfg.moe_renormalize else 0),
        ("kimi-k3.expert_latent_length", "u32", cfg.routed_expert_hidden_size),
        ("kimi-k3.activation.situ_beta", "f32", cfg.situ_beta),
        ("kimi-k3.activation.situ_linear_beta", "f32", cfg.situ_linear_beta),
        ("kimi-k3.attn_res.block_size", "u32", cfg.attn_res_block_size),
        ("tokenizer.ggml.model", "str", "gpt2"),
        ("tokenizer.ggml.pre", "str", "kimi-k2"),
        ("tokenizer.ggml.tokens", "arr_str", tokens),
        ("tokenizer.ggml.merges", "arr_str", merges),
        ("tokenizer.ggml.token_type", "arr_i32", [3, 1, 1] + [1] * (cfg.vocab_size - 3)),
        ("tokenizer.ggml.bos_token_id", "u32", 1),
        ("tokenizer.ggml.eos_token_id", "u32", 2),
        ("tokenizer.ggml.padding_token_id", "u32", 0),
    ]


def add_kvs(w: GgufWriter, kvs):
    for key, kind, val in kvs:
        getattr(w, "add_" + kind)(key, val)


def emit_single(path, kvs, tensors, total):
    w = GgufWriter()
    add_kvs(w, kvs)
    split_kvs(w, 0, 1, total)
    for t in tensors:
        w.add_tensor(*t)
    w.write(path, 0, 1, total)


def emit_multi(dirpath, kvs, tensors, total, split_layers, nshards=3):
    """Shard 1 metadata-only (llama.cpp convention), shards 2..n carry the
    tensors split by layer (model level + first split_layers layers in shard 2,
    the rest in shard 3)."""
    shard1 = GgufWriter()
    add_kvs(shard1, kvs)
    split_kvs(shard1, 0, nshards, total)
    shard1.write(os.path.join(dirpath, "tiny-k3-00001-of-00003.gguf"), 0, nshards, total)

    def layer_of(t):
        name = t[0]
        if name.startswith("blk."):
            return int(name.split(".")[1])
        return -1

    groups = [[], []]
    for t in tensors:
        groups[0 if layer_of(t) < split_layers else 1].append(t)
    for gi, grp in enumerate(groups):
        w = GgufWriter()
        split_kvs(w, gi + 1, nshards, total)
        for t in grp:
            w.add_tensor(*t)
        w.write(os.path.join(dirpath, "tiny-k3-00002-of-00003.gguf" if gi == 0
                             else "tiny-k3-00003-of-00003.gguf"), gi + 1, nshards, total)


# ------------------------------------------------------------ measurement ----
def measure_error(name, orig, deq):
    d = deq.ravel().astype(np.float64) - orig.ravel().astype(np.float64)
    rel = float(np.linalg.norm(d) / max(np.linalg.norm(orig.astype(np.float64)), 1e-30))
    mae = float(np.abs(d).max())
    return rel, mae


def measure_quant_errors(cfg, model, tensors, grid):
    """Q8_0 and IQ1_S quantization errors vs the original fp32, and the D2
    requant noise (MXFP4(IQ1_S dequant) vs IQ1_S dequant). Printed, and returned
    for the manifest."""
    stats = {"q8": [], "iq1s": [], "requant": []}
    for name, ne, gtype, data in tensors:
        if gtype == 8:
            w = _dequant_q8(data, ne[0])
            orig = _orig_mat(cfg, model, name)
            rel, mae = measure_error(name, orig, w)
            stats["q8"].append((name, rel, mae))
        elif gtype == 19:
            E = ne[2]
            per = len(data) // E
            for e in range(E):
                raw = data[e * per:(e + 1) * per]
                w = rge.dequant_iq1_s(raw, ne[0]).reshape(ne[1], ne[0])
                # original fp32 expert matrix from the model
                li = int(name.split(".")[1])
                which = {"ffn_gate_exps": "w1", "ffn_up_exps": "w3",
                         "ffn_down_exps": "w2"}[name.split(".")[2]]
                orig = model.get_parameter(
                    "layers.%d.mlp.experts.%d.%s.weight" % (li, e, which)
                ).detach().float().cpu().numpy().astype(np.float32)
                rel, mae = measure_error(name + "[%d]" % e, orig, w)
                stats["iq1s"].append((name + "[%d]" % e, rel, mae))
                rq = rge.expert_engine_view(w)
                rel2, _ = measure_error(name + "[%d].requant" % e, w, rq)
                stats["requant"].append((name + "[%d]" % e, rel2, 0.0))
    return stats


def _dequant_q8(data: bytes, cols: int) -> np.ndarray:
    nblk = len(data) // BS8
    b = np.frombuffer(data, dtype=np.uint8).reshape(nblk, BS8)
    d = rge.f16_to_f32(b[:, :2].copy().view(np.uint16)[:, 0]).astype(np.float32)
    q = b[:, 2:].astype(np.int8).astype(np.float32)
    rows = nblk * QK8 // cols
    return (q.reshape(rows, cols // QK8, QK8) * d.reshape(rows, cols // QK8, 1)
            ).reshape(rows, cols)


def _orig_mat(cfg, model, name):
    """The original fp32 values of a trunk tensor, by GGUF name."""
    if name == "token_embd.weight":
        return model.get_parameter("embed_tokens.weight").detach().float() \
            .cpu().numpy().astype(np.float32)
    if name == "output.weight":
        return model.get_parameter("lm_head.weight").detach().float() \
            .cpu().numpy().astype(np.float32)
    parts = name.split(".")
    L = int(parts[1])
    body = parts[2]
    if body in ("attn_k_b", "attn_v_b"):
        # the kv_b halves: slices of the model's kv_b_proj (fp32, exact)
        kb = model.get_parameter("layers.%d.self_attn.kv_b_proj.weight" % L) \
            .detach().float().cpu().numpy().astype(np.float32)
        QN, VH = cfg.qk_nope_head_dim, cfg.v_head_dim
        if body == "attn_k_b":
            return np.stack([kb[h * (QN + VH):h * (QN + VH) + QN, :].T
                             for h in range(cfg.num_attention_heads)])
        return np.stack([kb[h * (QN + VH) + QN:(h + 1) * (QN + VH), :]
                         for h in range(cfg.num_attention_heads)])
    p = {
        "attn_q_a": "layers.%d.self_attn.q_a_proj.weight",
        "attn_q_b": "layers.%d.self_attn.q_b_proj.weight",
        "attn_kv_a_mqa": "layers.%d.self_attn.kv_a_proj_with_mqa.weight",
        "attn_gate": "layers.%d.self_attn.g_proj.weight",
        "attn_q": "layers.%d.self_attn.q_proj.weight",
        "attn_k": "layers.%d.self_attn.k_proj.weight",
        "attn_v": "layers.%d.self_attn.v_proj.weight",
        "ssm_g": "layers.%d.self_attn.g_proj.weight",
        "ssm_f_a": "layers.%d.self_attn.f_a_proj.weight",
        "ssm_f_b": "layers.%d.self_attn.f_b_proj.weight",
        "ffn_routed_down": "layers.%d.mlp.down.weight",
        "ffn_routed_up": "layers.%d.mlp.up.weight",
        "ffn_gate_shexp": "layers.%d.mlp.shared.w1.weight",
        "ffn_up_shexp": "layers.%d.mlp.shared.w3.weight",
        "ffn_down_shexp": "layers.%d.mlp.shared.w2.weight",
        "ffn_gate": "layers.%d.mlp.gate_proj.weight",
        "ffn_up": "layers.%d.mlp.up_proj.weight",
        "ffn_down": "layers.%d.mlp.down_proj.weight",
    }
    if body == "attn_output":
        p = "layers.%d.self_attn.o_proj.weight"
    else:
        p = p[body]
    return model.get_parameter(p % L).detach().float().cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt-ids", default="3,7,11,5,9")
    ap.add_argument("--out", default=None, help="fixture root (default tests/fixtures)")
    a = ap.parse_args()

    fix = os.path.abspath(a.out) if a.out else _paths.FIXTURES
    ids = [int(v) for v in a.prompt_ids.split(",") if v != ""]

    cfg, model = mtc.build(a.seed, **mtc.ALIGNED_DIMS)
    print("model: hidden %d, layers %d, vocab %d, experts %d, moe_inter %d "
          "| kda_head_dim %d, qk_nope %d, v_head %d" %
          (cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size,
           cfg.num_experts, cfg.moe_intermediate_size, cfg.kda_head_dim,
           cfg.qk_nope_head_dim, cfg.v_head_dim))
    model.eval()

    grid = rge.load_iq1s_grid()

    dirs = {
        "single": os.path.join(fix, "tiny_gguf"),
        "multi": os.path.join(fix, "tiny_gguf_multi"),
        "bf16trunk": os.path.join(fix, "tiny_gguf_bf16trunk"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    kvs = config_kvs(cfg, ids)
    tensors_q8 = build_tensors(cfg, model, grid, "q8")
    tensors_bf16 = build_tensors(cfg, model, grid, "bf16eq")
    total = len(tensors_q8)
    print("tensors per fixture: %d" % total)

    emit_single(os.path.join(dirs["single"], "tiny-k3-00001-of-00001.gguf"),
                kvs, tensors_q8, total)
    emit_multi(dirs["multi"], kvs, tensors_q8, total, split_layers=6)
    emit_single(os.path.join(dirs["bf16trunk"], "tiny-k3-00001-of-00001.gguf"),
                kvs, tensors_bf16, total)

    # ---- quantization error measurement + manifest ----
    stats = measure_quant_errors(cfg, model, tensors_q8, grid)
    q8_rel = [s[1] for s in stats["q8"]]
    iq_rel = [s[1] for s in stats["iq1s"]]
    rq_rel = [s[1] for s in stats["requant"]]
    print("\nQUANTIZATION ERROR (relative L2 vs the original fp32)")
    print("  Q8_0 trunk  : %d tensors, rel-L2 min/mean/max = %.5f/%.5f/%.5f"
          % (len(q8_rel), min(q8_rel), float(np.mean(q8_rel)), max(q8_rel)))
    print("  IQ1_S experts: %d windows, rel-L2 min/mean/max = %.5f/%.5f/%.5f"
          % (len(iq_rel), min(iq_rel), float(np.mean(iq_rel)), max(iq_rel)))
    print("  D2 requant  : MXFP4(IQ1_S) vs IQ1_S, rel-L2 min/mean/max = "
          "%.5f/%.5f/%.5f"
          % (min(rq_rel), float(np.mean(rq_rel)), max(rq_rel)))
    assert max(iq_rel) < 0.6, "IQ1_S quantization error out of the 1-bit class"
    assert max(rq_rel) < 0.2, "requant error out of the expected class"

    # ---- references from THE SAME BYTES ----
    for key, d in dirs.items():
        path = os.path.join(d, "tiny-k3-00001-of-00001.gguf") if key != "multi" else d
        outp = os.path.join(d, "ref_logits_gguf.json")
        rge.forward_ref(path, ids, outp)
    gate2 = os.path.join(fix, "gguf_gate2_golden.bin")
    rge.emit_gate2_golden(os.path.join(dirs["single"], "tiny-k3-00001-of-00001.gguf"),
                          gate2, ids)

    manifest = {
        "arch": "aligned tiny (ALIGNED_DIMS)", "seed": a.seed,
        "prompt_ids": ids, "tensors": total,
        "q8_rel_l2": {"min": min(q8_rel), "mean": float(np.mean(q8_rel)),
                      "max": max(q8_rel)},
        "iq1s_rel_l2": {"min": min(iq_rel), "mean": float(np.mean(iq_rel)),
                        "max": max(iq_rel)},
        "requant_rel_l2": {"min": min(rq_rel), "mean": float(np.mean(rq_rel)),
                           "max": max(rq_rel)},
        "gate2_golden": os.path.basename(gate2),
    }
    with open(os.path.join(fix, "tiny_gguf_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("\nwrote fixtures under %s" % fix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
