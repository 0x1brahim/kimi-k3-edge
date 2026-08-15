# GGUF merged-expert ORIENTATION VERDICT (D4, wave-1 blocking)

Date: 2026-08-13. Author: dev-b. Evidence: real 14-shard file headers (all shards,
tensor-info only), `modeling_kimi_linear.py` (the expert definition used by
`modeling_kimi_k3.py`), `model.safetensors.index.json`, and the GGUF spec.

## VERDICT (one paragraph)

The GGUF merged expert tensors are the HF per-expert `w1/w2/w3` weights stacked
UNTRANSPOSED along the expert dimension. GGUF stores dimensions fastest-first, so
for `ffn_gate_exps`/`ffn_up_exps` `(ne0,ne1,ne2) = (3584,3072,896)` means: row
length ne0 = **in** (latent 3584), ne1 = **out** (3072), ne2 = **expert index**.
Each expert's slice is a CONTIGUOUS row-major `[out][in]` matrix — exactly the
engine's canonical layout (recon-model §1: `w1/w3[I=3072][L=3584]`, `w2[L=3584]
[I=3072]`). **NO transpose is needed at dequant.** For `ffn_down_exps` `(3072,
3584, 896)`: row length ne0 = in = 3072 (I), ne1 = out = 3584 (L) — again the
engine's `w2[out=L][in=I]`. IQ1_S block rows: 3584 = 14 × 256 blocks/row (gate/up),
3072 = 12 × 256 blocks/row (down); both dims are exact block multiples in every one
of the 276 expert tensors (verified across all 14 shards).

## Evidence

### E1. The HF side: what a per-expert weight IS (modeling_kimi_linear.py:242-260)

`KimiBlockSparseMLP.__init__` (the class instantiated per expert by
`KimiSparseMoeBlock`, modeling_kimi_linear.py:786-796, driven from
`modeling_kimi_k3.py`):

```python
self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)   # gate
self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)   # down
self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)   # up
```

With the Kimi K3 config (`config.json`, mirrored by the GGUF `kimi-k3.*` keys):
`hidden_dim = routed_expert_hidden_size = 3584` (latent L), `ffn_dim =
moe_intermediate_size = 3072` (I). `nn.Linear` weight shape is `[out_features,
in_features]`, row-major:

| HF tensor | Linear | weight shape | engine name/layout |
|---|---|---|---|
| `w1` (gate) | 3584 → 3072 | `[3072, 3584]` = `[out=I][in=L]` | `w1[I][L]` ✓ |
| `w3` (up)   | 3584 → 3072 | `[3072, 3584]` = `[out=I][in=L]` | `w3[I][L]` ✓ |
| `w2` (down) | 3072 → 3584 | `[3584, 3072]` = `[out=L][in=I]` | `w2[L][I]` ✓ |

### E2. The GGUF side: real-file dims (parsed from all 14 shard headers)

Every MoE layer N in 1..92 carries three IQ1_S tensors; dims are identical in every
shard (sample from shard 2; GGUF reader reports ne order, fastest first):

```
blk.N.ffn_gate_exps.weight  shape (3584, 3072, 896)   type 19 (IQ1_S)  1,926,758,400 B
blk.N.ffn_up_exps.weight    shape (3584, 3072, 896)   type 19 (IQ1_S)  1,926,758,400 B
blk.N.ffn_down_exps.weight  shape (3072, 3584, 896)   type 19 (IQ1_S)  1,926,758,400 B
```

Byte check: `ceil(3584/256)*50 * 3072 * 896 = 14*50*3072*896 = 1,926,758,400` ✓
and `12*50*3584*896 = 1,926,758,400` ✓ — exactly the file's n_bytes (and the
inter-tensor gaps), for BOTH dim orders.

### E3. The mapping is forced by E1+E2 (no ambiguity)

If the converter stacked HF weights as-is, `torch.stack([w1.weight for 896
experts])` has torch shape `[896, 3072, 3584]`, which GGUF serializes as
`ne = [3584, 3072, 896]` — **exactly** the file's `ffn_gate_exps` dims.
Transposing each expert first would give `ne = [3072, 3584, 896]` — which is
precisely the dims the file shows for `ffn_down_exps`, and vice versa. So the
three tensors are mutually consistent with untransposed stacking and mutually
inconsistent with every transposed variant: gate/up could only be `[3584,3072,…]`
if w1/w3 were stored as `[in,out]` (they are `[out,in]` per E1), and down could
only be `[3072,3584,…]` if w2 were stored as `[in,out]` (it is `[out,in]` per E1).
The file dims DISCRIMINATE the orientation; no other reading fits all three.

Cross-check with the engine's own convention: `output.weight` (Q8_0) is
`(7168, 163840)` in the file and `[vocab=163840][hidden=7168]` in the engine —
row length ne0 = in = hidden, i.e. the same "GGUF (in, out) == row-major
[out][in]" convention holds for every non-expert weight too.

### E4. Expert index and slice layout

- `model.safetensors.index.json` names per-expert tensors
  `language_model.model.layers.{N}.block_sparse_moe.experts.{E}.w{1,2,3}.
  {weight_packed,weight_scale}` with E = 0..895 in index order. The merged GGUF
  tensor's ne2 = 896 = expert count; stacking in ModuleList/index order is the
  only sane implementation and matches the dims. Expert E maps to ne2 = E.
- Slice layout: ne2 is the slowest-varying dim, so expert E's bytes are
  CONTIGUOUS: `[E * per_expert, (E+1) * per_expert)` with
  `per_expert = ceil(ne0/256)*50 * ne1` (14*50*3072 = 2,150,400 B for gate/up,
  12*50*3584 = 2,150,400 B for down). `k3_gguf_dequant_expert_slice` dequantizes
  exactly this window; the C test validates it against real blocks taken from
  expert 0 and expert 7 of `blk.1.ffn_gate_exps.weight`.
- Weight scaling: HF stores `weight_packed` (int8) + `weight_scale` (per-output-
  channel fp32, compressed-tensors convention). The GGUF converter dequantizes
  those to fp32 `[out][in]` and re-quantizes to IQ1_S row-wise over ne0; the
  orientation argument above is over the resulting fp32 layout, which is what
  matters for the engine. (The `.safetensors` files themselves are not present
  locally, so the packed/scale shapes could not be read; they are not needed for
  the verdict — the file dims + Linear shapes force it.)

### E5. What IQ1_S block-rows mean for 3072/3584

llama.cpp quantized tensors block along ne0 only (dequantize_row_iq1_s,
ggml-quants.c:2650). So for gate/up each row (one `out` row of 3584 in-values)
is 14 blocks of 256; for down each row is 12 blocks. A dequant writes
ceil(ne0/256)*256 values per row; with 3584/3072 both exact multiples of 256 the
padded and unpadded layouts coincide for every real tensor (verified: no
quantized tensor in any shard has ne0 % QK != 0).

## Consequences for wave 2 (expert path)

- `ffn_gate_exps` slice E dequantized to fp32 gives `w1[E][I=3072][L=3584]`
  row-major — feed directly into the MXFP4 requant at cache admit (D2) or the
  resident fp32 bank layout (`w1[expert][I][L]`, k3_ops.c:611-613).
- The shape-assert contract for these tensors is exactly
  `ne = [3584, 3072, 896]` / `[3072, 3584, 896]` (gate/up vs down), enforced by
  `k3_gguf_dequant`/`k3_gguf_dequant_expert_slice` on every call.

## Remaining ambiguity (explicit)

None in the orientation itself. One adjacent fact is UNVERIFIABLE locally: the
exact per-expert loop order inside unsloth's converter (E in order vs any
permutation). The dims prove the stacking was along the last dim in expert order;
the mapping `experts.E -> ne2=E` follows from the safetensors index naming and
ModuleList order, and no competing evidence exists. If a future parity run shows
mismatched expert outputs, the first thing to check is this index order, not the
orientation.
