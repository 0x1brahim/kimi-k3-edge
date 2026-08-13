/* k3_gguf_map.c - see k3_gguf_map.h.
 *
 * The contract table below is the SINGLE source of truth for the GGUF path's
 * name mapping and shape expectations. It was built from a complete enumeration
 * of the real 14-shard file (2,573 tensor infos, headers only) cross-checked
 * against modeling_kimi_linear.py and the engine's plan_layer/plan_model:
 * every tensor the engine can resolve maps to exactly one entry, and the
 * real-file gate (k3_gguf_validate_all) proves the table covers ALL 2,573
 * tensors of the file - nothing is left to a name guess at runtime.
 *
 * FOUND VERIFIED FACTS THE TABLE ENCODES (deltas from the recon reports):
 *   - blk.N.attn_k_b.weight is ne=[qk_nope, kv_lora, n_heads] (128,512,96): the
 *     per-head k projection TRANSPOSED (llama.cpp's absorbed-query MLA layout),
 *     while attn_v_b.weight is ne=[kv_lora, v_head, n_heads] untransposed. The
 *     two halves tile kv_b_proj exactly (k_b + v_b numels == kv_b numel).
 *   - There is NO mlp_res_norm.weight anywhere in the file. The three
 *     *_res_score tensors carry the FOLDED norm*proj scoring vector, so both
 *     engine res entries resolve: *_res_norm -> the folded vector, *_res_proj ->
 *     a constant-1.0 vector (the engine folds norm*proj itself, and
 *     score*1.0 == score).
 *   - ssm_a is kda_heads values (96); the engine's plan wants kda_head_dim and
 *     takes a kda_heads prefix, so the finder zero-pads the tail.
 *   - ssm_beta (b_proj) ships F32; the engine keeps it narrow (bf16), so the
 *     finder converts F32 -> bf16 at dequant time.
 *   - ssm_conv1d_{q,k,v}.weight are ne=[conv_k, 1, P] F32: the engine's
 *     [P][conv_k] layout with the redundant middle axis dropped - same bytes.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "k3_gguf_map.h"
#include "k3_gguf_dequant.h"
#include "k3_st.h"

/* ------------------------------------------------------------------ errors */
static int mfail(const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "k3_gguf_map: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    return -1;
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static size_t align8(size_t x)
{
    return (x + 7u) & ~(size_t)7u;
}

/* ------------------------------------------------------------------ table */

/* Dimension codes: each contract dim is expressed in cfg terms, so a checkpoint
 * whose config disagrees with the file fails the shape assert instead of being
 * silently accepted. GGUF ne order, fastest first. */
enum {
    MD_H = 1, /* hidden                              */
    MD_P,     /* kda_heads * kda_head_dim            */
    MD_D,     /* kda_head_dim                        */
    MD_QL,    /* q_lora                              */
    MD_KVL,   /* kv_lora                             */
    MD_QN,    /* qk_nope                             */
    MD_QR,    /* qk_rope                             */
    MD_VH,    /* v_head                              */
    MD_HH,    /* kda_heads                           */
    MD_NH,    /* n_heads                             */
    MD_HQH,   /* n_heads * (qk_nope + qk_rope)       */
    MD_HKVD,  /* n_heads * (qk_nope + v_head)        */
    MD_KVW,   /* kv_lora + qk_rope                   */
    MD_SI,    /* moe_inter * n_shared                */
    MD_E,     /* n_experts                           */
    MD_L,     /* latent                              */
    MD_I,     /* moe_inter                           */
    MD_DI,    /* dense_inter                         */
    MD_CK,    /* conv_k                              */
    MD_VOC,   /* vocab                               */
    MD_ONE,   /* 1                                   */
};

static int64_t dimval(const K3Cfg *c, int code)
{
    switch (code) {
    case MD_H:
        return c->hidden;
    case MD_P:
        return (int64_t)c->kda_heads * c->kda_head_dim;
    case MD_D:
        return c->kda_head_dim;
    case MD_QL:
        return c->q_lora;
    case MD_KVL:
        return c->kv_lora;
    case MD_QN:
        return c->qk_nope;
    case MD_QR:
        return c->qk_rope;
    case MD_VH:
        return c->v_head;
    case MD_HH:
        return c->kda_heads;
    case MD_NH:
        return c->n_heads;
    case MD_HQH:
        return (int64_t)c->n_heads * (c->qk_nope + c->qk_rope);
    case MD_HKVD:
        return (int64_t)c->n_heads * (c->qk_nope + c->v_head);
    case MD_KVW:
        return (int64_t)c->kv_lora + c->qk_rope;
    case MD_SI:
        return (int64_t)c->moe_inter * c->n_shared;
    case MD_E:
        return c->n_experts;
    case MD_L:
        return c->latent;
    case MD_I:
        return c->moe_inter;
    case MD_DI:
        return c->dense_inter;
    case MD_CK:
        return c->conv_k;
    case MD_VOC:
        return c->vocab;
    case MD_ONE:
        return 1;
    }
    return -1;
}

/* Applicability of an entry to a layer. */
enum {
    AF_ALL   = 1 << 0,
    AF_MLA   = 1 << 1,
    AF_KDA   = 1 << 2,
    AF_MOE   = 1 << 3,
    AF_DENSE = 1 << 4,
    AF_MODEL = 1 << 5,
};

/* Entry flags. */
enum {
    MF_ONES  = 1 << 0, /* constant-1.0 vector (the folded res_score counterpart) */
    MF_PAD_D = 1 << 1, /* A_log: file has kda_heads values, engine wants kda_head_dim */
    MF_KVB   = 1 << 2, /* composite kv_b from attn_k_b (transposed) + attn_v_b */
};

typedef struct {
    const char *eng;  /* engine name; "%d" = layer (model entries have none) */
    const char *gguf; /* gguf name; "%d" = layer; NULL for ONES/KVB entries   */
    uint8_t ndim;
    uint8_t dim[4]; /* dim codes, GGUF ne order (fastest first)             */
    uint8_t gtype;  /* K3_GGUF_TYPE_*                                        */
    uint8_t out;    /* K3_GGUF_DEQ_BF16 or K3_GGUF_DEQ_F32                  */
    uint8_t appl;   /* AF_*                                                  */
    uint8_t flags;  /* MF_*                                                  */
} MapEntry;

/* The complete engine-name -> GGUF-name contract. Verified tensor-for-tensor
 * against the real file; k3_gguf_validate_all proves full coverage.
 *
 * RESIDUAL RISK (documented, not gated): the *_res_score tensors are assumed to
 * hold the FOLDED norm.weight * proj.weight scoring vector. The HF model computes
 * score_weight = norm * proj at runtime, the file carries exactly ONE tensor per
 * res point, and the 2,573-count arithmetic rules out a separate mlp_res_norm -
 * so the fold reading is the only self-consistent one. The finder hands the
 * folded vector to the *_res_norm side and a constant-1.0 vector to the
 * *_res_proj side, so the engine's own fold (norm*proj) evaluates to the file's
 * score. This is asserted structurally and, in the tests, at the byte level only
 * for self-consistency (norm side = file bytes, proj side = 1); a value-level
 * check against the HF checkpoint needs the safetensors shards, which do not
 * exist on this box. If the converter had stored the UNFOLDED proj vector, the
 * engine would silently compute proj*1 != norm*proj. Verify on the first
 * available HF checkpoint. */
static const MapEntry k3_map_table[] = {
    /* ---- model level ---- */
    {"language_model.model.embed_tokens.weight",
     "token_embd.weight",
     2,
     {MD_H, MD_VOC, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MODEL,
     0},
    {"language_model.lm_head.weight",
     "output.weight",
     2,
     {MD_H, MD_VOC, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MODEL,
     0},
    {"language_model.model.norm.weight",
     "output_norm.weight",
     1,
     {MD_H, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_MODEL,
     0},
    {"language_model.model.output_attn_res_norm.weight",
     "output_res_score.weight",
     1,
     {MD_H, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_MODEL,
     0},
    {"language_model.model.output_attn_res_proj.weight",
     NULL,
     1,
     {MD_H, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_MODEL,
     MF_ONES},

    /* ---- every layer ---- */
    {"language_model.model.layers.%d.input_layernorm.weight",
     "blk.%d.attn_norm.weight",
     1,
     {MD_H, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_ALL,
     0},
    {"language_model.model.layers.%d.post_attention_layernorm.weight",
     "blk.%d.ffn_norm.weight",
     1,
     {MD_H, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_ALL,
     0},
    {"language_model.model.layers.%d.self_attention_res_norm.weight",
     "blk.%d.attn_res_score.weight",
     1,
     {MD_H, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_ALL,
     0},
    {"language_model.model.layers.%d.self_attention_res_proj.weight",
     NULL,
     1,
     {MD_H, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_ALL,
     MF_ONES},
    /* The file has NO mlp_res_norm: ffn_res_score is the folded mlp pair. */
    {"language_model.model.layers.%d.mlp_res_norm.weight",
     "blk.%d.ffn_res_score.weight",
     1,
     {MD_H, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_ALL,
     0},
    {"language_model.model.layers.%d.mlp_res_proj.weight",
     NULL,
     1,
     {MD_H, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_ALL,
     MF_ONES},

    /* ---- MLA layers ---- */
    {"language_model.model.layers.%d.self_attn.q_a_proj.weight",
     "blk.%d.attn_q_a.weight",
     2,
     {MD_H, MD_QL, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MLA,
     0},
    {"language_model.model.layers.%d.self_attn.q_a_layernorm.weight",
     "blk.%d.attn_q_a_norm.weight",
     1,
     {MD_QL, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_MLA,
     0},
    {"language_model.model.layers.%d.self_attn.q_b_proj.weight",
     "blk.%d.attn_q_b.weight",
     2,
     {MD_QL, MD_HQH, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MLA,
     0},
    {"language_model.model.layers.%d.self_attn.kv_a_proj_with_mqa.weight",
     "blk.%d.attn_kv_a_mqa.weight",
     2,
     {MD_H, MD_KVW, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MLA,
     0},
    {"language_model.model.layers.%d.self_attn.kv_a_layernorm.weight",
     "blk.%d.attn_kv_a_norm.weight",
     1,
     {MD_KVL, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_MLA,
     0},
    {"language_model.model.layers.%d.self_attn.kv_b_proj.weight",
     NULL,
     3,
     {MD_QN, MD_KVL, MD_NH, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MLA,
     MF_KVB},
    {"language_model.model.layers.%d.self_attn.o_proj.weight",
     "blk.%d.attn_output.weight",
     2,
     {MD_P, MD_H, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MLA,
     0},
    {"language_model.model.layers.%d.self_attn.g_proj.weight",
     "blk.%d.attn_gate.weight",
     2,
     {MD_H, MD_P, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MLA,
     0},

    /* ---- KDA layers ---- */
    {"language_model.model.layers.%d.self_attn.q_proj.weight",
     "blk.%d.attn_q.weight",
     2,
     {MD_H, MD_P, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_KDA,
     0},
    {"language_model.model.layers.%d.self_attn.k_proj.weight",
     "blk.%d.attn_k.weight",
     2,
     {MD_H, MD_P, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_KDA,
     0},
    {"language_model.model.layers.%d.self_attn.v_proj.weight",
     "blk.%d.attn_v.weight",
     2,
     {MD_H, MD_P, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_KDA,
     0},
    {"language_model.model.layers.%d.self_attn.g_proj.weight",
     "blk.%d.ssm_g.weight",
     2,
     {MD_H, MD_P, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_KDA,
     0},
    {"language_model.model.layers.%d.self_attn.o_proj.weight",
     "blk.%d.attn_output.weight",
     2,
     {MD_P, MD_H, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_KDA,
     0},
    {"language_model.model.layers.%d.self_attn.q_conv1d.weight",
     "blk.%d.ssm_conv1d_q.weight",
     3,
     {MD_CK, MD_ONE, MD_P, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_KDA,
     0},
    {"language_model.model.layers.%d.self_attn.k_conv1d.weight",
     "blk.%d.ssm_conv1d_k.weight",
     3,
     {MD_CK, MD_ONE, MD_P, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_KDA,
     0},
    {"language_model.model.layers.%d.self_attn.v_conv1d.weight",
     "blk.%d.ssm_conv1d_v.weight",
     3,
     {MD_CK, MD_ONE, MD_P, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_KDA,
     0},
    {"language_model.model.layers.%d.self_attn.f_a_proj.weight",
     "blk.%d.ssm_f_a.weight",
     2,
     {MD_H, MD_D, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_KDA,
     0},
    {"language_model.model.layers.%d.self_attn.f_b_proj.weight",
     "blk.%d.ssm_f_b.weight",
     2,
     {MD_D, MD_P, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_KDA,
     0},
    /* ssm_beta ships F32; the engine keeps b_proj narrow (bf16), so the finder
     * converts. The shape is the engine's [kda_heads][hidden]. */
    {"language_model.model.layers.%d.self_attn.b_proj.weight",
     "blk.%d.ssm_beta.weight",
     2,
     {MD_H, MD_HH, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_BF16,
     AF_KDA,
     0},
    {"language_model.model.layers.%d.self_attn.A_log",
     "blk.%d.ssm_a",
     1,
     {MD_HH, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_KDA,
     MF_PAD_D},
    {"language_model.model.layers.%d.self_attn.dt_bias",
     "blk.%d.ssm_dt.bias",
     1,
     {MD_P, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_KDA,
     0},
    {"language_model.model.layers.%d.self_attn.o_norm.weight",
     "blk.%d.ssm_norm.weight",
     1,
     {MD_D, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_KDA,
     0},

    /* ---- MoE layers ---- */
    {"language_model.model.layers.%d.block_sparse_moe.gate.weight",
     "blk.%d.ffn_gate_inp.weight",
     2,
     {MD_H, MD_E, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_MOE,
     0},
    {"language_model.model.layers.%d.block_sparse_moe.gate.e_score_correction_bias",
     "blk.%d.exp_probs_b.bias",
     1,
     {MD_E, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_MOE,
     0},
    {"language_model.model.layers.%d.block_sparse_moe.routed_expert_down_proj.weight",
     "blk.%d.ffn_routed_down.weight",
     2,
     {MD_H, MD_L, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MOE,
     0},
    {"language_model.model.layers.%d.block_sparse_moe.routed_expert_up_proj.weight",
     "blk.%d.ffn_routed_up.weight",
     2,
     {MD_L, MD_H, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MOE,
     0},
    {"language_model.model.layers.%d.block_sparse_moe.routed_expert_norm.weight",
     "blk.%d.ffn_routed_norm.weight",
     1,
     {MD_L, 0, 0, 0},
     K3_GGUF_TYPE_F32,
     K3_GGUF_DEQ_F32,
     AF_MOE,
     0},
    {"language_model.model.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight",
     "blk.%d.ffn_gate_shexp.weight",
     2,
     {MD_H, MD_SI, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MOE,
     0},
    {"language_model.model.layers.%d.block_sparse_moe.shared_experts.up_proj.weight",
     "blk.%d.ffn_up_shexp.weight",
     2,
     {MD_H, MD_SI, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MOE,
     0},
    {"language_model.model.layers.%d.block_sparse_moe.shared_experts.down_proj.weight",
     "blk.%d.ffn_down_shexp.weight",
     2,
     {MD_SI, MD_H, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_MOE,
     0},

    /* ---- the single dense layer ---- */
    {"language_model.model.layers.%d.mlp.gate_proj.weight",
     "blk.%d.ffn_gate.weight",
     2,
     {MD_H, MD_DI, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_DENSE,
     0},
    {"language_model.model.layers.%d.mlp.up_proj.weight",
     "blk.%d.ffn_up.weight",
     2,
     {MD_H, MD_DI, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_DENSE,
     0},
    {"language_model.model.layers.%d.mlp.down_proj.weight",
     "blk.%d.ffn_down.weight",
     2,
     {MD_DI, MD_H, 0, 0},
     K3_GGUF_TYPE_Q8_0,
     K3_GGUF_DEQ_BF16,
     AF_DENSE,
     0},
};

#define N_MAP_ENTRIES (int)(sizeof k3_map_table / sizeof *k3_map_table)

static int map_applies(const MapEntry *e, const K3Cfg *c, int L)
{
    if (e->appl == AF_ALL) return 1;
    if (e->appl & AF_MODEL) return 0;
    if (e->appl & AF_MLA) return k3_is_mla(c, L);
    if (e->appl & AF_KDA) return k3_is_kda(c, L);
    if (e->appl & AF_MOE) return !k3_is_dense(c, L);
    if (e->appl & AF_DENSE) return k3_is_dense(c, L);
    return 0;
}

static int entry_dims(const K3Cfg *c, const MapEntry *e, int64_t ne[4])
{
    ne[0] = ne[1] = ne[2] = ne[3] = 1;
    for (int i = 0; i < e->ndim; i++) {
        ne[i] = dimval(c, e->dim[i]);
        if (ne[i] < 1)
            return mfail("dim code %d of %s evaluates to %lld", e->dim[i], e->eng,
                         (long long)ne[i]);
    }
    return 0;
}

static int64_t entry_numel(const K3Cfg *c, const MapEntry *e)
{
    int64_t ne[4];
    if (entry_dims(c, e, ne)) return -1;
    if (e->flags & MF_PAD_D) return c->kda_head_dim; /* zero-padded tail */
    return ne[0] * ne[1] * ne[2] * ne[3];
}

/* ------------------------------------------------------------------ bind ctx */

/* The struct body lives in k3_gguf_map.h (the CLI owns one on the stack). */

/* ------------------------------------------------------------------ checks */

/* The finder's unpadded row layout requires the row length to be an exact block
 * multiple (every real tensor satisfies this; a non-conforming file must fail
 * loud rather than silently misplace columns). */
static int row_aligned(const MapEntry *e, const int64_t ne0)
{
    const int64_t qk = e->gtype == K3_GGUF_TYPE_Q8_0    ? K3_GGUF_Q8_0_QK
                       : e->gtype == K3_GGUF_TYPE_IQ1_S ? K3_GGUF_IQ1_S_QK
                                                        : 1;
    return ne0 % qk == 0;
}

/* Per-dimension + ggml-type + nbytes contract check (D4a). The engine's
 * numel-only check is not enough for GGUF: a transposed matrix of equal numel
 * must fail here, loudly, with the file name and both shapes. */
static int map_validate(const K3GgufBind *b, const MapEntry *e, const K3Tensor *t)
{
    int64_t ne[4];
    if (entry_dims(b->c, e, ne)) return -1;

    if (t->ndim != e->ndim)
        return mfail("%s: %d dims, contract says %d", t->name, t->ndim, e->ndim);
    for (int i = 0; i < e->ndim; i++)
        if (t->shape[i] != ne[i])
            return mfail("%s: ne[%d] = %lld, contract says %lld (engine tensor %s)",
                         t->name, i, (long long)t->shape[i], (long long)ne[i], e->eng);
    const int gtype = b->g->gtype[t - b->g->t];
    if (gtype != e->gtype)
        return mfail("%s: ggml type %d, contract says %d (engine tensor %s)", t->name,
                     gtype, e->gtype, e->eng);
    const int64_t nb = k3_gguf_dequant_nbytes(e->gtype, e->ndim, ne);
    if (nb != t->nbytes)
        return mfail("%s: %lld bytes on disk, the type/shape formula gives %lld", t->name,
                     (long long)t->nbytes, (long long)nb);
    /* The bind dequantizes into an UNPADDED layout, so the row length must be
     * an exact block multiple. Enforced here (not only at bind) so validate_all
     * rejects the same files the bind would. */
    if (!row_aligned(e, ne[0]))
        return mfail("%s: row length %lld is not a block multiple; the dequant "
                     "would pad rows and misplace every column",
                     t->name, (long long)ne[0]);
    return 0;
}

static size_t ones_bytes(const K3Cfg *c)
{
    return (size_t)c->hidden * 4;
}

/* The constant-1.0 vector at offset 0, rewritten on every refill so a partially
 * overwritten buffer can never serve a stale fold. */
static void ones_write(K3GgufBind *b)
{
    const size_t n = (size_t)b->c->hidden;
    float *o       = (float *)b->buf;
    for (size_t i = 0; i < n; i++) o[i] = 1.0f;
}

int k3_gguf_bind_init(K3GgufBind *b, const K3Gguf *g, const K3Cfg *c)
{
    memset(b, 0, sizeof *b);
    b->g     = g;
    b->c     = c;
    b->layer = -1;
    return 0;
}

void k3_gguf_bind_free(K3GgufBind *b)
{
    free(b->lay_buf);
    free(b->model_buf);
    free(b->widen);
    free(b->raw);
    free(b->vals);
    memset(b, 0, sizeof *b);
}

double k3_gguf_bind_load_seconds(const K3GgufBind *b)
{
    return b->load_seconds;
}

/* ------------------------------------------------------------------ finder */

static const MapEntry *map_lookup(const K3GgufBind *b, const char *name)
{
    char want[256];
    for (int i = 0; i < N_MAP_ENTRIES; i++) {
        const MapEntry *e = &k3_map_table[i];
        if (b->layer < 0) {
            if (!(e->appl & AF_MODEL)) continue;
            if (strcmp(name, e->eng)) continue;
            return e;
        }
        if (e->appl & AF_MODEL) continue;
        if (!map_applies(e, b->c, b->layer)) continue;
        snprintf(want, sizeof want, e->eng, b->layer);
        if (!strcmp(name, want)) return e;
    }
    return NULL;
}

/* A true K3St view of the GGUF index (identical field layout by construction):
 * the D1a seam - the stock read helpers run unchanged, and dfd[] all -1 sends
 * them down the buffered path. Built as a copy rather than a cast so there is
 * no aliasing question. */
static K3St st_view(const K3Gguf *g)
{
    K3St s;
    memset(&s, 0, sizeof s);
    s.fd      = g->fd;
    s.dfd     = g->dfd;
    s.path    = g->path;
    s.nshard  = g->nshard;
    s.t       = g->t;
    s.nt      = g->nt;
    s.bucket  = g->bucket;
    s.nbucket = g->nbucket;
    s.strpool = g->strpool;
    s.strcap  = g->strcap;
    s.strlen_ = g->strlen_;
    return s;
}

#define MAP_CHUNK (64u << 20) /* raw bytes read per chunk: ~31 MB dequantized */

/* Dequantize the tensor t (already contract-validated) into dst in row chunks,
 * so a 1.25 GB embed never materialises a full-size raw transient. Rows are
 * block-aligned (enforced by the caller), so the padded and unpadded layouts
 * coincide and the destination stride is ne0 elements. */
static int dequant_chunked(K3GgufBind *b, const K3Tensor *t, const MapEntry *e,
                           const int64_t ne[4], unsigned char *dst)
{
    const int64_t qk      = e->gtype == K3_GGUF_TYPE_Q8_0    ? K3_GGUF_Q8_0_QK
                            : e->gtype == K3_GGUF_TYPE_IQ1_S ? K3_GGUF_IQ1_S_QK
                                                             : 1;
    const int64_t bsz     = e->gtype == K3_GGUF_TYPE_Q8_0    ? K3_GGUF_Q8_0_BSZ
                            : e->gtype == K3_GGUF_TYPE_IQ1_S ? K3_GGUF_IQ1_S_BSZ
                                                             : 4;
    const int64_t per_row = ne[0] / qk * bsz;
    const int64_t rows    = ne[1] * ne[2] * ne[3];
    const int esz         = e->out == K3_GGUF_DEQ_BF16 ? 2 : 4;

    const K3St s    = st_view(b->g);
    int64_t row_off = 0;
    while (row_off < rows) {
        int64_t chunk_rows  = rows - row_off;
        int64_t chunk_bytes = chunk_rows * per_row;
        if (chunk_bytes > (int64_t)MAP_CHUNK) {
            chunk_rows  = MAP_CHUNK / per_row;
            chunk_bytes = chunk_rows * per_row;
        }
        /* per_row > MAP_CHUNK would make chunk_rows 0 and the loop below spin
         * forever with an empty chunk. A row that large cannot be chunked; fail
         * loud instead of hanging on a crafted header. */
        if (chunk_rows <= 0)
            return mfail("%s: one row is %.1f MB, larger than the %.1f MB chunk; "
                         "this tensor cannot be read",
                         t->name, (double)per_row / 1e6, (double)MAP_CHUNK / 1e6);
        if ((size_t)chunk_bytes > b->raw_cap) {
            size_t nc         = (size_t)chunk_bytes;
            unsigned char *np = (unsigned char *)realloc(b->raw, nc);
            if (!np)
                return mfail("out of memory for the %.1f MB raw scratch",
                             (double)nc / 1e6);
            b->raw     = np;
            b->raw_cap = nc;
        }

        K3Tensor rt = *t;
        rt.off += row_off * per_row;
        rt.nbytes   = chunk_bytes;
        rt.ndim     = 2;
        rt.shape[0] = ne[0];
        rt.shape[1] = chunk_rows;
        rt.shape[2] = rt.shape[3] = 1;
        if (k3_st_read(&s, &rt, b->raw) != chunk_bytes)
            return mfail("short read of %s at row %lld", t->name, (long long)row_off);

        const int64_t sne[4] = {ne[0], chunk_rows, 1, 1};
        K3GgufTensor gt;
        memset(&gt, 0, sizeof gt);
        gt.ggml_type = e->gtype;
        gt.ndim      = 2;
        memcpy(gt.ne, sne, sizeof sne);
        gt.nbytes = chunk_bytes;
        gt.data   = b->raw;
        K3GgufExpect ex;
        memset(&ex, 0, sizeof ex);
        ex.ndim = 2;
        memcpy(ex.ne, sne, sizeof sne);
        if (k3_gguf_dequant(&gt, e->out,
                            dst + (size_t)row_off * (size_t)ne[0] * (size_t)esz,
                            &ex) != 0)
            return -1;
        row_off += chunk_rows;
    }
    return 0;
}

/* Header-only contract check for the kv_b halves (no buffer, no data reads). */
static int kvb_validate(const K3Gguf *g, const K3Cfg *c, int layer)
{
    char kname[96], vname[96];
    snprintf(kname, sizeof kname, "blk.%d.attn_k_b.weight", layer);
    snprintf(vname, sizeof vname, "blk.%d.attn_v_b.weight", layer);
    const K3Tensor *kt = k3_gguf_find(g, kname);
    const K3Tensor *vt = k3_gguf_find(g, vname);
    if (!kt || !vt)
        return mfail("layer %d: kv_b needs %s and %s, one is absent from the index",
                     layer, kname, vname);

    const int64_t QN = c->qk_nope, VH = c->v_head, KL = c->kv_lora, H = c->n_heads;
    const int64_t kne[4] = {QN, KL, H, 1}, vne[4] = {KL, VH, H, 1};
    if (kt->ndim != 3 || kt->shape[0] != QN || kt->shape[1] != KL || kt->shape[2] != H)
        return mfail("%s: shape [%lld %lld %lld], contract says [%lld %lld %lld]", kname,
                     (long long)kt->shape[0], (long long)kt->shape[1],
                     (long long)kt->shape[2], (long long)QN, (long long)KL, (long long)H);
    if (vt->ndim != 3 || vt->shape[0] != KL || vt->shape[1] != VH || vt->shape[2] != H)
        return mfail("%s: shape [%lld %lld %lld], contract says [%lld %lld %lld]", vname,
                     (long long)vt->shape[0], (long long)vt->shape[1],
                     (long long)vt->shape[2], (long long)KL, (long long)VH, (long long)H);
    if (g->gtype[kt - g->t] != K3_GGUF_TYPE_Q8_0 ||
        g->gtype[vt - g->t] != K3_GGUF_TYPE_Q8_0)
        return mfail("kv_b halves must be Q8_0 (layer %d)", layer);
    /* The dequant pads rows to the 32-block and the assembly scratch is sized
     * unpadded, so a head width that is not a block multiple would overflow
     * the scratch. Fail loud, mirroring the layer path's row_aligned rule. */
    if (QN % K3_GGUF_Q8_0_QK != 0 || VH % K3_GGUF_Q8_0_QK != 0)
        return mfail("kv_b halves need qk_nope %d and v_head %d to be multiples "
                     "of the %d-block (layer %d)",
                     (int)QN, (int)VH, K3_GGUF_Q8_0_QK, layer);
    if (k3_gguf_dequant_nbytes(K3_GGUF_TYPE_Q8_0, 3, kne) != kt->nbytes ||
        k3_gguf_dequant_nbytes(K3_GGUF_TYPE_Q8_0, 3, vne) != vt->nbytes)
        return mfail("kv_b halves: on-disk bytes disagree with the shape formula");
    return 0;
}

/* The composite kv_b: reassemble the engine's [n_heads*(qk_nope+v_head)]
 * [kv_lora] matrix from attn_k_b (per-head TRANSPOSED) + attn_v_b (straight).
 * See the header for the layout trap. */
static int gfind_kvb(K3GgufBind *b, const MapEntry *e, int64_t *off, int64_t *nbytes,
                     int *dtype)
{
    (void)e; /* the kv_b contract is fixed by kvb_validate; the entry is only */
             /* the marker that this composite is being asked for             */
    if (kvb_validate(b->g, b->c, b->layer) != 0) return -1;

    const K3Cfg *c   = b->c;
    const int64_t QN = c->qk_nope, VH = c->v_head, KL = c->kv_lora, H = c->n_heads;
    char kname[96], vname[96];
    snprintf(kname, sizeof kname, "blk.%d.attn_k_b.weight", b->layer);
    snprintf(vname, sizeof vname, "blk.%d.attn_v_b.weight", b->layer);
    const K3Tensor *kt   = k3_gguf_find(b->g, kname);
    const K3Tensor *vt   = k3_gguf_find(b->g, vname);
    const int64_t kne[4] = {QN, KL, H, 1}, vne[4] = {KL, VH, H, 1};

    const int64_t out_rows  = H * (QN + VH);
    const int64_t out_bytes = out_rows * KL * 2; /* bf16 */
    const size_t dst_off    = align8(b->used);
    if (dst_off + (size_t)out_bytes > b->buf_cap)
        return mfail("layer buffer overflow binding kv_b (layer %d)", b->layer);
    uint16_t *dst = (uint16_t *)(b->buf + dst_off);

    /* k part: read the whole tensor, dequant to bf16 scratch, transpose per head:
     * k_b[h][j][i] is engine row (h*(QN+VH)+i), column j. */
    {
        const size_t need = (size_t)(H * KL * QN) * 2;
        if (need > b->vals_cap) {
            void *np = realloc(b->vals, need);
            if (!np) return mfail("out of memory for the kv_b transpose scratch");
            b->vals     = np;
            b->vals_cap = need;
        }
        const K3St s = st_view(b->g);
        if (k3_st_read(&s, kt, b->raw) != kt->nbytes)
            return mfail("short read of %s", kname);
        K3GgufTensor kgt;
        memset(&kgt, 0, sizeof kgt);
        kgt.ggml_type = K3_GGUF_TYPE_Q8_0;
        kgt.ndim      = 3;
        memcpy(kgt.ne, kne, sizeof kne);
        kgt.nbytes = kt->nbytes;
        kgt.data   = b->raw;
        K3GgufExpect kex;
        memset(&kex, 0, sizeof kex);
        kex.ndim = 3;
        memcpy(kex.ne, kne, sizeof kne);
        if (k3_gguf_dequant(&kgt, K3_GGUF_DEQ_BF16, b->vals, &kex) != 0) return -1;
        const uint16_t *ks = (const uint16_t *)b->vals;
        for (int64_t h = 0; h < H; h++)
            for (int64_t i = 0; i < QN; i++)
                for (int64_t j = 0; j < KL; j++)
                    dst[(size_t)(h * (QN + VH) + i) * KL + j] =
                        ks[(size_t)(h * KL + j) * QN + i];
    }
    /* v part: straight copy per head. */
    {
        const K3St s = st_view(b->g);
        if (k3_st_read(&s, vt, b->raw) != vt->nbytes)
            return mfail("short read of %s", vname);
        K3GgufTensor vgt;
        memset(&vgt, 0, sizeof vgt);
        vgt.ggml_type = K3_GGUF_TYPE_Q8_0;
        vgt.ndim      = 3;
        memcpy(vgt.ne, vne, sizeof vne);
        vgt.nbytes = vt->nbytes;
        vgt.data   = b->raw;
        K3GgufExpect vex;
        memset(&vex, 0, sizeof vex);
        vex.ndim = 3;
        memcpy(vex.ne, vne, sizeof vne);
        if (k3_gguf_dequant(&vgt, K3_GGUF_DEQ_BF16, b->vals, &vex) != 0) return -1;
        const uint16_t *vs = (const uint16_t *)b->vals;
        for (int64_t h = 0; h < H; h++)
            for (int64_t i = 0; i < VH; i++)
                memcpy(dst + (size_t)(h * (QN + VH) + QN + i) * KL,
                       vs + (size_t)(h * VH + i) * KL, (size_t)KL * 2);
    }

    b->used = dst_off + (size_t)out_bytes;
    *off    = (int64_t)dst_off;
    *nbytes = out_bytes;
    *dtype  = K3_DT_BF16;
    return 0;
}

/* The K3MemSrc finder: translate, validate, dequantize into the current buffer,
 * report (off, nbytes, dtype) exactly as the trunk's finder does. */
static int gfind(void *ctx, const char *name, int64_t *off, int64_t *nbytes, int *dtype)
{
    K3GgufBind *b = (K3GgufBind *)ctx;

    const MapEntry *e = map_lookup(b, name);
    if (!e)
        return mfail("no contract entry for engine tensor '%s' (layer %d); a new "
                     "weight family needs a table row, not a guess",
                     name, b->layer);

    if (e->flags & MF_ONES) {
        /* The folded res_score counterpart: 1.0 * folded == folded. */
        *off    = 0;
        *nbytes = (int64_t)b->c->hidden * 4;
        *dtype  = K3_DT_F32;
        return 0;
    }
    if (e->flags & MF_KVB) return gfind_kvb(b, e, off, nbytes, dtype);

    char gname[160];
    snprintf(gname, sizeof gname, e->gguf, b->layer);
    const K3Tensor *t = k3_gguf_find(b->g, gname);
    if (!t)
        return mfail("layer %d: engine tensor %s needs %s, absent from the index",
                     b->layer, name, gname);
    if (map_validate(b, e, t) != 0) return -1;

    int64_t ne[4];
    if (entry_dims(b->c, e, ne)) return -1;
    if (!row_aligned(e, ne[0]))
        return mfail("%s: row length %lld is not a block multiple; the dequant "
                     "would pad rows and misplace every column",
                     t->name, (long long)ne[0]);

    const int esz     = e->out == K3_GGUF_DEQ_BF16 ? 2 : 4;
    int64_t out_numel = ne[0] * ne[1] * ne[2] * ne[3];
    if (e->flags & MF_PAD_D) out_numel = b->c->kda_head_dim; /* A_log tail */
    const int64_t out_bytes = out_numel * esz;
    const size_t dst_off    = align8(b->used);
    if (dst_off + (size_t)out_bytes > b->buf_cap)
        return mfail("layer buffer overflow binding %s", name);
    unsigned char *dst = b->buf + dst_off;

    if (dequant_chunked(b, t, e, ne, dst) != 0) return -1;
    if (e->flags & MF_PAD_D && b->c->kda_head_dim > ne[0])
        memset(dst + (size_t)ne[0] * esz, 0, (size_t)(b->c->kda_head_dim - ne[0]) * esz);

    b->used = dst_off + (size_t)out_bytes;
    *off    = (int64_t)dst_off;
    *nbytes = out_bytes;
    *dtype  = esz == 2 ? K3_DT_BF16 : K3_DT_F32;
    return 0;
}

/* ------------------------------------------------------------------ sizing */

int64_t k3_gguf_layer_bytes(const K3GgufBind *b, int layer)
{
    if (layer < 0 || layer >= b->c->n_layers)
        return mfail("layer %d out of range", layer);

    /* Mirror the bind cursor exactly: every tensor starts at align8(used), so
     * the sizing must reserve the same per-tensor slack or a valid configuration
     * would be refused at bind time. (The real model's dims are all even, so the
     * slack is zero there; the arithmetic must still agree by construction.) */
    int64_t total = (int64_t)ones_bytes(b->c);
    for (int i = 0; i < N_MAP_ENTRIES; i++) {
        const MapEntry *e = &k3_map_table[i];
        if (e->appl & AF_MODEL) continue;
        if (!map_applies(e, b->c, layer)) continue;
        if (e->flags & MF_ONES) continue; /* the ones vector is shared */
        total = align8((size_t)total);
        if (e->flags & MF_KVB) {
            total += (int64_t)b->c->n_heads * (b->c->qk_nope + b->c->v_head) *
                     b->c->kv_lora * 2;
            continue;
        }
        char gname[160];
        snprintf(gname, sizeof gname, e->gguf, layer);
        const K3Tensor *t = k3_gguf_find(b->g, gname);
        if (!t)
            return mfail("layer %d needs %s (engine tensor %s), absent from the "
                         "index",
                         layer, gname, e->eng);
        if (map_validate(b, e, t) != 0) return -1;
        const int64_t n = entry_numel(b->c, e);
        if (n < 0) return -1;
        total += n * (e->out == K3_GGUF_DEQ_BF16 ? 2 : 4);
    }
    return total;
}

int64_t k3_gguf_layer_bytes_max(const K3GgufBind *b, int nl)
{
    int64_t best = -1;
    for (int L = 0; L < nl; L++) {
        const int64_t n = k3_gguf_layer_bytes(b, L);
        if (n < 0) return -1;
        if (n > best) best = n;
    }
    return best;
}

static int64_t model_bytes(const K3GgufBind *b, int want_lm_head)
{
    const K3Cfg *c = b->c;
    int64_t total  = (int64_t)ones_bytes(c);
    for (int i = 0; i < N_MAP_ENTRIES; i++) {
        const MapEntry *e = &k3_map_table[i];
        if (!(e->appl & AF_MODEL)) continue;
        if (e->flags & MF_ONES) continue;
        if (!want_lm_head && !strcmp(e->eng, "language_model.lm_head.weight")) continue;
        total = align8((size_t)total); /* same cursor arithmetic as the bind */
        const K3Tensor *t = k3_gguf_find(b->g, e->gguf);
        if (!t)
            return mfail("model-level %s (%s) absent from the index", e->eng, e->gguf);
        if (map_validate(b, e, t) != 0) return -1;
        const int64_t n = entry_numel(b->c, e);
        if (n < 0) return -1;
        total += n * (e->out == K3_GGUF_DEQ_BF16 ? 2 : 4);
    }
    return total;
}

/* ------------------------------------------------------------------ binds */

int k3_gguf_bind_alloc(K3GgufBind *b, int nl, int64_t budget_bytes)
{
    const int64_t laymax = k3_gguf_layer_bytes_max(b, nl);
    if (laymax < 0) return -1;
    if (laymax > budget_bytes) {
        fprintf(stderr,
                "k3_gguf: the largest layer needs %.2f GB dequantized (bf16), but "
                "the --trunk-gb budget is %.2f GB.\n"
                "  The GGUF path streams ONE layer at a time into a reusable buffer "
                "sized for the\n"
                "  biggest layer; raise --trunk-gb to at least %.2f GB.\n",
                (double)laymax / 1e9, (double)budget_bytes / 1e9, (double)laymax / 1e9);
        return -1;
    }
    b->nl      = nl;
    b->lay_cap = (size_t)laymax;
    if (posix_memalign((void **)&b->lay_buf, 4096, b->lay_cap) != 0) {
        fprintf(stderr, "k3_gguf: cannot allocate the %.2f GB layer buffer\n",
                (double)laymax / 1e9);
        return -1;
    }
    /* Raw-read scratch sized for one chunk up front: the composite kv_b and the
     * chunked dequant both read through it, and neither may realloc under the
     * bind's nose. */
    b->raw_cap = MAP_CHUNK;
    b->raw     = (unsigned char *)malloc(b->raw_cap);
    if (!b->raw) {
        fprintf(stderr, "k3_gguf: cannot allocate the %.1f MB raw scratch\n",
                (double)b->raw_cap / 1e6);
        return -1;
    }
    b->widen_cap = k3_bind_widen_bytes(b->c);
    b->widen     = (unsigned char *)malloc(b->widen_cap);
    if (!b->widen) {
        fprintf(stderr, "k3_gguf: cannot allocate the %.1f MB widen area\n",
                (double)b->widen_cap / 1e6);
        return -1;
    }
    return 0;
}

int k3_gguf_bind_layer(K3GgufBind *b, int L, K3LayerBind *out)
{
    if (!b->lay_buf) return mfail("layer buffer not allocated");
    b->layer   = L;
    b->buf     = b->lay_buf;
    b->buf_cap = b->lay_cap;
    b->used    = 0;
    ones_write(b);
    b->used = (size_t)ones_bytes(b->c);

    const double t0 = now_s();
    K3MemSrc src;
    src.find          = gfind;
    src.ctx           = b;
    size_t widen_used = 0;
    const int rc      = k3_bind_layer_mem(b->c, L, out, b->lay_buf, &src, b->widen,
                                          b->widen_cap, &widen_used);
    b->load_seconds += now_s() - t0;
    return rc;
}

int k3_gguf_bind_model(K3GgufBind *b, int want_lm_head, K3ModelBind *m)
{
    memset(m, 0, sizeof *m);
    const int64_t need = model_bytes(b, want_lm_head);
    if (need < 0) return -1;
    if (!b->model_buf) {
        if (posix_memalign((void **)&b->model_buf, 4096, (size_t)need) != 0) {
            fprintf(stderr,
                    "k3_gguf: cannot allocate %.2f GB for model-level "
                    "weights\n",
                    (double)need / 1e9);
            return -1;
        }
        b->model_cap = (size_t)need;
    }

    b->layer   = -1;
    b->buf     = b->model_buf;
    b->buf_cap = b->model_cap;
    b->used    = 0;
    ones_write(b);
    b->used = (size_t)ones_bytes(b->c);

    /* Same plan order as k3_bind.c's plan_model: embed, norm, res folds, lm_head. */
    int64_t off = 0, nb = 0;
    int dt = 0;
    if (gfind(b, "language_model.model.embed_tokens.weight", &off, &nb, &dt) != 0)
        return -1;
    m->embed = b->model_buf + off;
    if (gfind(b, "language_model.model.norm.weight", &off, &nb, &dt) != 0) return -1;
    m->norm = (const float *)(b->model_buf + off);
    if (gfind(b, "language_model.model.output_attn_res_norm.weight", &off, &nb, &dt) != 0)
        return -1;
    m->out_res_norm = (const float *)(b->model_buf + off);
    if (gfind(b, "language_model.model.output_attn_res_proj.weight", &off, &nb, &dt) != 0)
        return -1;
    m->out_res_proj = (const float *)(b->model_buf + off);
    if (want_lm_head) {
        if (gfind(b, "language_model.lm_head.weight", &off, &nb, &dt) != 0) return -1;
        m->lm_head = b->model_buf + off;
    }
    m->nbytes = b->used;
    m->wdt    = K3_WBF16;
    return 0;
}

/* ------------------------------------------------------------ validate_all */

int k3_gguf_map_translate(const K3Cfg *c, int layer, const char *eng_name, char *out,
                          size_t outcap)
{
    K3GgufBind b;
    memset(&b, 0, sizeof b);
    b.c               = c;
    b.layer           = layer;
    const MapEntry *e = map_lookup(&b, eng_name);
    if (!e)
        return mfail("no contract entry for engine tensor '%s' (layer %d)", eng_name,
                     layer);
    if (e->flags & MF_ONES)
        snprintf(out, outcap, "(const ones)");
    else if (e->flags & MF_KVB)
        snprintf(out, outcap, "(composite kv_b)");
    else if (layer < 0)
        snprintf(out, outcap, "%s", e->gguf);
    else
        snprintf(out, outcap, e->gguf, layer);
    return 0;
}

int k3_gguf_map_check(const K3Cfg *c, const K3Gguf *g, int layer, const char *eng_name,
                      const K3Tensor *t)
{
    K3GgufBind b;
    memset(&b, 0, sizeof b);
    b.g               = g;
    b.c               = c;
    b.layer           = layer;
    const MapEntry *e = map_lookup(&b, eng_name);
    if (!e)
        return mfail("no contract entry for engine tensor '%s' (layer %d)", eng_name,
                     layer);
    if (e->flags & (MF_ONES | MF_KVB))
        return mfail("'%s' is a composite/const entry, not a file tensor", eng_name);
    return map_validate(&b, e, t);
}

void k3_gguf_expert_names(int layer, char gate[64], char up[64], char down[64])
{
    snprintf(gate, 64, "blk.%d.ffn_gate_exps.weight", layer);
    snprintf(up, 64, "blk.%d.ffn_up_exps.weight", layer);
    snprintf(down, 64, "blk.%d.ffn_down_exps.weight", layer);
}

/* The three merged expert tensors: w1 = gate (ne [latent, moe_inter, experts]),
 * w3 = up (same), w2 = down (ne [moe_inter, latent, experts]); all IQ1_S. */
static int validate_experts(const K3Gguf *g, const K3Cfg *c, int layer)
{
    char gate[64], up[64], down[64];
    k3_gguf_expert_names(layer, gate, up, down);
    const char *names[3];
    names[0]            = gate;
    names[1]            = up;
    names[2]            = down;
    const int64_t le[4] = {c->latent, c->moe_inter, c->n_experts, 1};
    const int64_t de[4] = {c->moe_inter, c->latent, c->n_experts, 1};
    for (int i = 0; i < 3; i++) {
        const K3Tensor *t = k3_gguf_find(g, names[i]);
        if (!t) return mfail("layer %d needs %s, absent from the index", layer, names[i]);
        const int64_t *want = i == 2 ? de : le;
        if (t->ndim != 3 || t->shape[0] != want[0] || t->shape[1] != want[1] ||
            t->shape[2] != want[2])
            return mfail("%s: shape [%lld %lld %lld], contract says [%lld %lld %lld]",
                         names[i], (long long)t->shape[0], (long long)t->shape[1],
                         (long long)t->shape[2], (long long)want[0], (long long)want[1],
                         (long long)want[2]);
        if (g->gtype[t - g->t] != K3_GGUF_TYPE_IQ1_S)
            return mfail("%s: ggml type %d, contract says IQ1_S", names[i],
                         g->gtype[t - g->t]);
        const int64_t nb = k3_gguf_dequant_nbytes(K3_GGUF_TYPE_IQ1_S, 3, want);
        if (nb != t->nbytes)
            return mfail("%s: %lld bytes on disk, the shape formula gives %lld", names[i],
                         (long long)t->nbytes, (long long)nb);
        /* Per-expert windows must be whole: ne2 is the slowest dim, so every
         * expert owns nbytes/ne2 contiguous bytes. */
        if (t->nbytes % t->shape[2] != 0)
            return mfail("%s: %lld bytes does not divide evenly into %lld experts",
                         names[i], (long long)t->nbytes, (long long)t->shape[2]);
    }
    return 0;
}

int k3_gguf_validate_all(const K3Gguf *g, const K3Cfg *c)
{
    K3GgufBind b;
    memset(&b, 0, sizeof b);
    b.g     = g;
    b.c     = c;
    b.layer = -1;

    int checked = 0;
    for (int i = 0; i < N_MAP_ENTRIES; i++) {
        const MapEntry *e = &k3_map_table[i];
        if (!(e->appl & AF_MODEL)) continue;
        if (e->flags & MF_ONES) continue;
        const K3Tensor *t = k3_gguf_find(g, e->gguf);
        if (!t) return mfail("model-level %s absent from the index", e->gguf);
        if (map_validate(&b, e, t) != 0) return -1;
        checked++;
    }
    for (int L = 0; L < c->n_layers; L++) {
        for (int i = 0; i < N_MAP_ENTRIES; i++) {
            const MapEntry *e = &k3_map_table[i];
            if (e->appl & AF_MODEL) continue;
            if (!map_applies(e, c, L)) continue;
            if (e->flags & MF_ONES) continue;
            if (e->flags & MF_KVB) {
                if (kvb_validate(g, c, L) != 0) return -1;
                checked += 2;
                continue;
            }
            char gname[160];
            snprintf(gname, sizeof gname, e->gguf, L);
            const K3Tensor *t = k3_gguf_find(g, gname);
            if (!t)
                return mfail("layer %d needs %s (engine tensor %s)", L, gname, e->eng);
            b.layer = L;
            if (map_validate(&b, e, t) != 0) return -1;
            checked++;
        }
        if (!k3_is_dense(c, L)) {
            if (validate_experts(g, c, L) != 0) return -1;
            checked += 3;
        }
    }
    return checked;
}
