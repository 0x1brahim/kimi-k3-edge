/* k3_gguf_expert.h - K3ExpertSrc over the GGUF merged expert tensors (D2a).
 *
 * The GGUF checkpoint stores the 896 routed experts of each MoE layer as THREE
 * merged 3-D IQ1_S tensors (blk.N.ffn_{gate,up,down}_exps.weight, ne =
 * [latent, moe_inter, experts] for gate/up, [moe_inter, latent, experts] for
 * down; expert e owns the contiguous byte window [e*per, (e+1)*per)). The
 * engine's kernels consume experts as packed MXFP4 + E8M0 scales (K3ExpertQ,
 * k3.h), so this source dequantizes IQ1_S -> fp32 and RE-QUANTIZES fp32 ->
 * MXFP4 (k3_mxfp4_quant) at cache-admit time, exactly the architect's D2a
 * verdict: the K3ExpertSrc/K3ExpertQ/kernels stay 100% untouched, and the slot
 * holds the engine's canonical run order (w1.p w1.s w2.p w2.s w3.p w3.s) so a
 * fill is a plain offset lookup.
 *
 * w1 = ffn_gate_exps (gate), w3 = ffn_up_exps (up), w2 = ffn_down_exps (down):
 * checked against k3_load.h's run order and gguf_expert_orientation.md, where
 * the dims themselves discriminate the orientation (gate/up are [I][L] per
 * expert, down is [L][I] - exactly the engine's w1/w3 vs w2 layouts).
 *
 * CACHE DISCIPLINE (mirrors k3_cache): LRU slots, INFLIGHT reservation before a
 * fill, publish only after a successful fill, nslot >= topk+1 enforced so a
 * token's working set can never be evicted mid-matmul. The fills are SERIAL
 * (one reusable fp32 transient): the reads are page-cache hits and the
 * dequant+requant is CPU-bound, so a parallel fill would only multiply the
 * transient; the INFLIGHT state machine is still the real one, so a later
 * parallelization cannot break it.
 */
#ifndef K3_GGUF_EXPERT_H
#define K3_GGUF_EXPERT_H

#include <stdint.h>

#include "k3.h"
#include "k3_gguf.h"

typedef struct {
    K3ExpertSrc src; /* MUST be first: pass &s->src to K3MoeW */

    const K3Gguf *g;
    const K3Cfg *c;

    unsigned char *arena; /* nslot * slot_bytes                      */
    int64_t slot_bytes;
    int nslot;

    int32_t *slot_of; /* [n_layers*n_experts] -> slot, or -1     */
    int32_t *key_of;  /* [nslot] -> key, -1 empty, -2 INFLIGHT   */
    uint64_t *used_at;
    uint64_t clock;

    /* Per-matrix geometry (identical for every expert of every layer). */
    int rows[3], cols[3];       /* logical [out][in] per matrix      */
    int64_t p_off[3], s_off[3]; /* offsets of packed/scales in a slot*/
    int64_t p_bytes[3], s_bytes[3];

    /* One reusable transient: the raw IQ1_S window + the fp32 dequant rows. */
    unsigned char *raw;
    size_t raw_cap;
    float *f32;
    size_t f32_cap;

    /* stats, same field names as K3Cache so the run loop reads either */
    uint64_t hits, misses, evictions, bytes_read, prefetch_reads;
    double load_seconds;
} K3GgufExpertSrc;

/* budget_bytes sizes the arena; fails if it leaves fewer than topk+1 slots.
 * Also validates the three merged tensors of the first MoE layer (shape, ggml
 * type, on-disk bytes, per-expert window divisibility) and fails loud on any
 * violation, so a wrong expert geometry surfaces at startup, not mid-decode. */
int k3_gguf_expert_src_init(K3GgufExpertSrc *s, const K3Gguf *g, const K3Cfg *c,
                            int64_t budget_bytes);
void k3_gguf_expert_src_free(K3GgufExpertSrc *s);
void k3_gguf_expert_src_reset_stats(K3GgufExpertSrc *s);
void k3_gguf_expert_src_report(const K3GgufExpertSrc *s, const char *label);

#endif /* K3_GGUF_EXPERT_H */
