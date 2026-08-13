/* k3_gguf_map.h - GGUF name mapping, per-dimension shape contracts and the
 * streamed layer/model bind for the GGUF path (D4a + D3a).
 *
 * THE PROBLEM
 *   The engine queries weights by safetensors names ("language_model.model.
 *   layers.5.self_attn.q_proj.weight", see k3_bind.c's plan and k3_load.c's
 *   EXPERT_FMT). The GGUF checkpoint stores DIFFERENT names ("blk.5.attn_q.
 *   weight", llama.cpp kimi-linear convention). This module is the find layer:
 *   the static table below translates every engine name to its GGUF name, states
 *   the per-dimension shape contract (GGUF ne order, fastest first) and the ggml
 *   type, and the bind finder validates BOTH before a byte is dequantized. The
 *   engine's numel-only checks are deliberately not enough for GGUF: a transposed
 *   matrix of equal numel must fail here, loudly.
 *
 * THE THREE LAYOUT TRAPS THIS TABLE ABSORBS (all verified against the real
 * 14-shard file and modeling_kimi_linear.py)
 *   1. kv_b_proj is SPLIT in GGUF: blk.N.attn_k_b.weight (ne=[qk_nope, kv_lora,
 *      n_heads]) holds the per-head k projection TRANSPOSED (llama.cpp's
 *      absorbed-query layout), and blk.N.attn_v_b.weight (ne=[kv_lora, v_head,
 *      n_heads]) holds v untransposed. The finder reassembles the engine's
 *      [n_heads*(qk_nope+v_head)][kv_lora] kv_b by transposing each head's k
 *      slice and concatenating k then v per head.
 *   2. The res_score tensors (attn_res_score/ffn_res_score/output_res_score) are
 *      the FOLDED norm.weight * proj.weight vectors the HF model computes at
 *      runtime. The engine folds norm*proj itself, so the finder hands the folded
 *      vector to the *_res_norm side and a constant-1.0 vector to the *_res_proj
 *      side: 1.0 * folded == folded.
 *   3. ssm_a ships kda_heads values but the engine's plan wants kda_head_dim
 *      (A_log takes a kda_heads prefix). The finder zero-pads to kda_head_dim.
 *
 * BUFFER DISCIPLINE (D3a)
 *   One reusable per-layer buffer holds every dequantized tensor of the current
 *   layer (bf16 matrices pointed at in place by k3_bind_layer_mem, fp32 vectors
 *   likewise); a fixed ones-vector lives at offset 0 and is rewritten every
 *   refill. Every tensor is re-pointed on every layer bind (k3_bind_layer_mem
 *   memsets K3LayerBind first), so no pointer survives a layer boundary. The
 *   buffer is sized for the LARGEST layer and k3_gguf_bind_alloc refuses to
 *   allocate beyond the --trunk-gb budget.
 */
#ifndef K3_GGUF_MAP_H
#define K3_GGUF_MAP_H

#include <stddef.h>
#include <stdint.h>

#include "k3.h"
#include "k3_bind.h"
#include "k3_gguf.h"

/* The finder context: the opened index, the reusable per-layer buffer (D3a),
 * the resident model buffer, and the raw-read/transpose scratch. Exposed like
 * the other engine structs (K3Cache, K3Trunk) so the CLI can own one on the
 * stack; the fields are only ever touched by k3_gguf_map.c. */
typedef struct K3GgufBind {
    const K3Gguf *g;
    const K3Cfg *c;
    int layer; /* current bind layer; -1 = model level    */
    int nl;    /* layers the buffer must cover            */

    unsigned char *lay_buf; /* reusable per-layer buffer            */
    size_t lay_cap;
    unsigned char *model_buf; /* resident model-level buffer          */
    size_t model_cap;

    unsigned char *buf; /* current destination (layer or model) */
    size_t buf_cap;

    unsigned char *widen; /* fp32 widen area for k3_bind_layer_mem */
    size_t widen_cap;

    unsigned char *raw; /* raw-read scratch (chunked dequant)    */
    size_t raw_cap;
    void *vals; /* scratch for the kv_b transpose        */
    size_t vals_cap;

    size_t used;         /* cursor into the current buffer        */
    double load_seconds; /* wall time inside k3_gguf_bind_layer   */
} K3GgufBind;

/* Create the finder context. No allocation: layer_bytes and validate_all can run
 * before any buffer exists. */
int k3_gguf_bind_init(K3GgufBind *b, const K3Gguf *g, const K3Cfg *c);

/* Dequantized bytes one layer needs (bf16 matrices + fp32 vectors + the ones
 * vector). Resolves every tensor name so a missing weight is caught here, at
 * planning time, like k3_bind_layer_bytes does for the safetensors path.
 * Returns -1 when any tensor is missing or violates its shape contract. */
int64_t k3_gguf_layer_bytes(const K3GgufBind *b, int layer);

/* Largest layer over [0, nl). -1 when any layer fails. */
int64_t k3_gguf_layer_bytes_max(const K3GgufBind *b, int nl);

/* Allocate the reusable layer buffer, the widen area and the raw-read scratch.
 * nl is the number of layers the buffer must cover. budget_bytes caps the layer
 * buffer: the biggest layer must fit or this fails loud (the --trunk-gb gate). */
int k3_gguf_bind_alloc(K3GgufBind *b, int nl, int64_t budget_bytes);

/* Bind layer L into out, refilling the reusable buffer. Safe to call repeatedly
 * (every token); every pointer in out is re-derived from the fresh buffer. */
int k3_gguf_bind_layer(K3GgufBind *b, int L, K3LayerBind *out);

/* Model-level weights (embed, norm, res folds, lm_head), resident: bound once
 * into their own buffer. want_lm_head = 0 skips the 2.35 GB lm_head. */
int k3_gguf_bind_model(K3GgufBind *b, int want_lm_head, K3ModelBind *m);

/* Wall time spent inside k3_gguf_bind_layer (reads + dequant), for the I/O-share
 * report. */
double k3_gguf_bind_load_seconds(const K3GgufBind *b);

/* Validate EVERY contract entry the engine can resolve (all layers, both
 * attention kinds, dense + MoE + model level, and the three merged expert
 * tensors per MoE layer) against the index: name present, per-dim shape, ggml
 * type, on-disk nbytes formula, block-aligned rows. Headers only - no tensor
 * data is ever read. Returns the number of tensors validated (2573 for the real
 * file), or -1 on the first violation. */
int k3_gguf_validate_all(const K3Gguf *g, const K3Cfg *c);

/* Translate one engine name to its GGUF name for layer L (-1 = model level).
 * Fills out with the GGUF name; the composite kv_b and the constant-ones entries
 * fill "(composite kv_b)" / "(const ones)" instead. Returns 0 on success, -1
 * when the name has no contract entry (a new family needs a table row, not a
 * guess). */
int k3_gguf_map_translate(const K3Cfg *c, int layer, const char *eng_name, char *out,
                          size_t outcap);

/* The per-dimension + ggml-type + nbytes contract check of ONE resolved tensor,
 * exposed for the unit tests; the bind path uses the same validation. layer is
 * the layer the engine name belongs to (-1 = model level). */
int k3_gguf_map_check(const K3Cfg *c, const K3Gguf *g, int layer, const char *eng_name,
                      const K3Tensor *t);

void k3_gguf_bind_free(K3GgufBind *b);

/* The GGUF names of the three merged expert tensors of one layer, for the expert
 * source and the tests. */
void k3_gguf_expert_names(int layer, char gate[64], char up[64], char down[64]);

#endif /* K3_GGUF_MAP_H */
