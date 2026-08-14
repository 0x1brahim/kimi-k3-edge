/* k3_gguf_expert.h - K3ExpertSrc over the GGUF merged expert tensors (fix wave:
 * native IQ1_S storage, replacing the D2a IQ1_S->MXFP4 requant admit).
 *
 * The GGUF checkpoint stores the 896 routed experts of each MoE layer as THREE
 * merged 3-D IQ1_S tensors (blk.N.ffn_{gate,up,down}_exps.weight, ne =
 * [latent, moe_inter, experts] for gate/up, [moe_inter, latent, experts] for
 * down; expert e owns the contiguous byte window [e*per, (e+1)*per)). The
 * engine's kernels consume experts in a backend-native encoding selected by
 * K3ExpertQ.wfmt: the ST cache stays MXFP4 + E8M0 scales (k3_cache.c,
 * untouched), and THIS source stores the IQ1_S slices AS-IS - no requant, no
 * transform - and tags every served expert K3_EXPERT_IQ1S so k3_moe dispatches
 * to k3_matmul_iq1_s. The architect's fix directive chose this over the D2a
 * requant: the requant was measured at 13.3% mean per-element error on the
 * real bytes (a transform the reference implementation does not have), and
 * removing it also deletes the per-admit dequant+requant compute and shrinks
 * expert storage ~2.72x (6,451,200 B per expert triple vs 17,547,264 B).
 *
 * w1 = ffn_gate_exps (gate), w3 = ffn_up_exps (up), w2 = ffn_down_exps (down):
 * checked against k3_load.h's run order and gguf_expert_orientation.md, where
 * the dims themselves discriminate the orientation (gate/up are [I][L] per
 * expert, down is [L][I] - exactly the engine's w1/w3 vs w2 layouts).
 *
 * SLOT LAYOUT: the canonical run order inside a slot is w1 then w2 then w3
 * (k3_load.h's measured layout; K3ExpertQ maps p1=w1, p2=w2, p3=w3), each a
 * raw IQ1_S byte span of rows * ceil(cols/256)*50 bytes (rows are padded to
 * the 256-block exactly as the on-disk bytes are; k3_matmul_iq1_s skips the
 * padding). A fill is a plain pread of the three windows straight into the
 * slot - no transient, no transform.
 *
 * CACHE DISCIPLINE (mirrors k3_cache): LRU slots, INFLIGHT reservation before a
 * fill, publish only after a successful fill, nslot >= topk+1 enforced so a
 * token's working set can never be evicted mid-matmul. The fills are SERIAL
 * (plain page-cache reads; the INFLIGHT state machine is the real one, so a
 * later parallelization cannot break it).
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
    int rows[3], cols[3];         /* logical [out][in] per matrix      */
    int64_t i_off[3], i_bytes[3]; /* IQ1_S byte span of each matrix in a slot */

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
