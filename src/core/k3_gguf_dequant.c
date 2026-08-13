/* k3_gguf_dequant.c - GGUF weight dequantization for the Kimi K3 engine.
 *
 * Scope: exactly the three ggml types present in the real 14-shard file - F32,
 * Q8_0, IQ1_S. Everything else fails loud. The math is a faithful C99 port of
 * llama.cpp's dequantize_row_{q8_0,iq1_s} (ggml-quants.c:553 and :2650), with the
 * iq1s_grid lookup table dumped verbatim from ggml-common.h (see iq1s_grid.h).
 *
 * FLOATING-POINT CONTRACT: -ffp-contract=off (both build systems set it) and plain
 * scalar fp32 ops in the exact order of the reference. The dequant is a handful of
 * multiplies and adds that are all exactly representable (d is a widened fp16, the
 * sub-block scale 2*k+1 is a small integer, grid[j]+delta is a multiple of 0.125),
 * so the output is BIT-IDENTICAL to the numpy reference the golden fixtures are
 * generated with, and tests compare bits, not tolerances.
 *
 * SHAPE ASSERT POLICY (D4): every entry point takes the caller's per-dimension
 * expectation and rejects any mismatch - wrong dims, negative dims, nbytes that
 * disagree with the block-size formula, unknown types, int64 overflow. The engine's
 * numel-only checks are not enough for GGUF (a transposed matrix of equal numel
 * would pass them); this module is where that failure class dies.
 */
#include "k3_gguf_dequant.h"

#include "iq1s_grid.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* ----------------------------------------------------------------- fail loud ---- */
static int k3_deq_fail(const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "k3_gguf_dequant: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    return -1;
}

/* ------------------------------------------------------------------ helpers ---- */
/* a*b with int64 overflow detection. All quantities here are non-negative, so a
 * single upper bound is the whole check. */
static int64_t k3_mul_i64(int64_t a, int64_t b, int *err)
{
    if (a == 0 || b == 0) return 0;
    if (a > INT64_MAX / b) { *err = 1; return 0; }
    return a * b;
}

/* Little-endian uint16 read. All supported platforms are LE, but the read is
 * written byte-wise so the code is portable by construction, not by assumption. */
static uint16_t k3_rd16(const uint8_t *p)
{
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

/* ceil(ne0/qk) without the ne0+qk-1 overflow trap. */
static int64_t k3_blocks_row(int64_t ne0, int64_t qk)
{
    return ne0 / qk + (ne0 % qk != 0);
}

/* Validate the tensor view and the caller's expectation; compute the expected
 * nbytes and the padded value count. Returns 0 on success (with *nvals filled)
 * or -1 after printing what failed. */
static int k3_deq_validate(const K3GgufTensor *t, const K3GgufExpect *expect,
                           int64_t *nvals)
{
    int err = 0;

    if (!t) return k3_deq_fail("NULL tensor");
    if (!expect) return k3_deq_fail("NULL shape expectation (D4: the caller must "
                                    "state the per-dimension contract on every call)");
    if (t->ndim < 1 || t->ndim > 4)
        return k3_deq_fail("ndim %d out of range 1..4", t->ndim);
    if (expect->ndim < 1 || expect->ndim > 4)
        return k3_deq_fail("expected ndim %d out of range 1..4", expect->ndim);
    for (int i = 0; i < 4; i++) {
        if (i < t->ndim && t->ne[i] < 0)
            return k3_deq_fail("negative dim ne[%d]=%lld", i, (long long)t->ne[i]);
        if (i < expect->ndim && expect->ne[i] < 0)
            return k3_deq_fail("negative expected dim ne[%d]=%lld", i,
                               (long long)expect->ne[i]);
    }
    if (t->ndim != expect->ndim)
        return k3_deq_fail("shape mismatch: tensor is %d-D, expectation is %d-D",
                           t->ndim, expect->ndim);
    for (int i = 0; i < t->ndim; i++)
        if (t->ne[i] != expect->ne[i])
            return k3_deq_fail("shape mismatch: ne[%d]=%lld but expectation is %lld",
                               i, (long long)t->ne[i], (long long)expect->ne[i]);

    int64_t qk, bsz;
    switch (t->ggml_type) {
    case K3_GGUF_TYPE_F32:   qk = 1;   bsz = 4;  break;
    case K3_GGUF_TYPE_Q8_0:  qk = 32;  bsz = 34; break;
    case K3_GGUF_TYPE_IQ1_S: qk = 256; bsz = 50; break;
    default:
        return k3_deq_fail("unsupported ggml_type %d (dispatch table holds exactly "
                           "F32=0, Q8_0=8, IQ1_S=19)", t->ggml_type);
    }

    /* nbytes = ceil(ne0/qk)*bsz * ne1*ne2*ne3, checked at every multiply. */
    int64_t per_row = k3_mul_i64(k3_blocks_row(t->ne[0], qk), bsz, &err);
    int64_t nb = per_row;
    for (int i = 1; i < t->ndim && !err; i++) nb = k3_mul_i64(nb, t->ne[i], &err);
    if (err)
        return k3_deq_fail("nbytes computation overflows int64 for ne=[%lld %lld %lld %lld]",
                           (long long)t->ne[0], (long long)t->ne[1],
                           (long long)t->ne[2], (long long)t->ne[3]);

    if (t->nbytes != nb)
        return k3_deq_fail("nbytes mismatch: tensor header says %lld, the "
                           "type/shape block formula gives %lld (ne=[%lld %lld %lld %lld])",
                           (long long)t->nbytes, (long long)nb,
                           (long long)t->ne[0], (long long)t->ne[1],
                           (long long)t->ne[2], (long long)t->ne[3]);

    /* Padded value count. For IQ1_S this is 5.12x the byte count, so it can
     * overflow where nbytes did not; check it independently rather than wrapping. */
    int64_t vpr = k3_mul_i64(k3_blocks_row(t->ne[0], qk), qk, &err);
    int64_t nv = vpr;
    for (int i = 1; i < t->ndim && !err; i++) nv = k3_mul_i64(nv, t->ne[i], &err);
    if (err)
        return k3_deq_fail("value count overflows int64 for ne=[%lld %lld %lld %lld]",
                           (long long)t->ne[0], (long long)t->ne[1],
                           (long long)t->ne[2], (long long)t->ne[3]);

    if (t->nbytes != 0 && !t->data)
        return k3_deq_fail("NULL data pointer for a %lld-byte tensor",
                           (long long)t->nbytes);

    *nvals = nv;
    return 0;
}

static void k3_emit(K3GgufDeqOut out, void *dst, int64_t i, float v)
{
    if (out == K3_GGUF_DEQ_BF16) ((uint16_t *)dst)[i] = k3_gguf_f32_to_bf16(v);
    else                         ((float *)dst)[i] = v;
}

/* --------------------------------------------------------------- per-type ---- */
/* F32: the engine's fp32 layout is the checkpoint's own bytes - trivial copy.
 * bf16 output is the same values rounded to the engine's narrow dtype. */
static int k3_deq_f32(const K3GgufTensor *t, K3GgufDeqOut out, void *dst)
{
    int64_t nv = 1;
    for (int i = 0; i < t->ndim; i++) nv *= t->ne[i];
    const float *src = (const float *)t->data;
    if (out == K3_GGUF_DEQ_BF16) {
        for (int64_t i = 0; i < nv; i++)
            ((uint16_t *)dst)[i] = k3_gguf_f32_to_bf16(src[i]);
    } else {
        memcpy(dst, src, (size_t)nv * sizeof(float));
    }
    return 0;
}

/* Q8_0: block = ggml_half d + int8 qs[32]; y[j] = qs[j] * fp32(d).
 * Reference: dequantize_row_q8_0, ggml-quants.c:553. */
static int k3_deq_q8_0(const K3GgufTensor *t, K3GgufDeqOut out, void *dst)
{
    const int64_t nblk = t->nbytes / K3_GGUF_Q8_0_BSZ;
    const uint8_t *p = (const uint8_t *)t->data;
    for (int64_t b = 0; b < nblk; b++) {
        const uint8_t *blk = p + b * K3_GGUF_Q8_0_BSZ;
        const float d = k3_gguf_f16_to_f32(k3_rd16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2);
        for (int j = 0; j < K3_GGUF_Q8_0_QK; j++)
            k3_emit(out, dst, b * K3_GGUF_Q8_0_QK + j, (float)qs[j] * d);
    }
    return 0;
}

/* IQ1_S: block = ggml_half d + qs[32] + qh[16]; 8 sub-blocks of 32 values.
 *   dl    = d * (2*((qh[ib] >> 12) & 7) + 1)          sub-block scale, 3 bits
 *   delta = (qh[ib] & 0x8000) ? -0.125 : 0.125        sign bit 15
 *   idx   = qs[4*ib + l] | (((qh[ib] >> 3*l) & 7) << 8)   11-bit grid index
 *   y     = dl * (grid[idx][j] + delta)
 * Reference: dequantize_row_iq1_s, ggml-quants.c:2650; IQ1S_DELTA = 0.125f,
 * grid in iq1s_grid.h (verbatim from ggml-common.h:1135). */
static int k3_deq_iq1_s(const K3GgufTensor *t, K3GgufDeqOut out, void *dst)
{
    const int64_t nblk = t->nbytes / K3_GGUF_IQ1_S_BSZ;
    const uint8_t *p = (const uint8_t *)t->data;
    for (int64_t b = 0; b < nblk; b++) {
        const uint8_t *blk = p + b * K3_GGUF_IQ1_S_BSZ;
        const float d = k3_gguf_f16_to_f32(k3_rd16(blk));
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 2 + 32;
        for (int ib = 0; ib < 8; ib++) {
            const uint16_t qhb = k3_rd16(qh + 2 * ib);
            const float dl = d * (float)(2 * ((qhb >> 12) & 7) + 1);
            const float delta = (qhb & 0x8000u) ? -0.125f : 0.125f;
            for (int l = 0; l < 4; l++) {
                const uint32_t idx = (uint32_t)qs[4 * ib + l] |
                                     (((uint32_t)(qhb >> (3 * l)) & 7u) << 8);
                const int8_t *grid = (const int8_t *)(iq1s_grid + idx);
                for (int j = 0; j < 8; j++)
                    k3_emit(out, dst, b * K3_GGUF_IQ1_S_QK + 32 * ib + 8 * l + j,
                            dl * ((float)grid[j] + delta));
            }
        }
    }
    return 0;
}

/* ----------------------------------------------------------------- dispatch ---- */
/* Extensible (type -> function) table. EXACTLY three entries: the three types the
 * real 554 GB file contains. No speculative entries: an unknown type is a loud
 * error, never a silent zero. */
typedef int (*k3_dequant_fn)(const K3GgufTensor *t, K3GgufDeqOut out, void *dst);

static const struct {
    int           type;
    const char   *name;
    k3_dequant_fn fn;
} k3_gguf_dequant_table[] = {
    { K3_GGUF_TYPE_F32,   "F32",   k3_deq_f32 },
    { K3_GGUF_TYPE_Q8_0,  "Q8_0",  k3_deq_q8_0 },
    { K3_GGUF_TYPE_IQ1_S, "IQ1_S", k3_deq_iq1_s },
};

static k3_dequant_fn k3_deq_lookup(int type)
{
    for (size_t i = 0; i < sizeof k3_gguf_dequant_table / sizeof *k3_gguf_dequant_table; i++)
        if (k3_gguf_dequant_table[i].type == type)
            return k3_gguf_dequant_table[i].fn;
    return NULL;
}

/* -------------------------------------------------------------- entry points ---- */
int64_t k3_gguf_dequant_nbytes(int ggml_type, int ndim, const int64_t ne[4])
{
    if (!ne) {
        k3_deq_fail("NULL dims array");
        return -1;
    }
    if (ndim < 1 || ndim > 4) {
        k3_deq_fail("ndim %d out of range 1..4", ndim);
        return -1;
    }
    for (int i = 0; i < ndim; i++)
        if (ne[i] < 0) {
            k3_deq_fail("negative dim ne[%d]=%lld", i, (long long)ne[i]);
            return -1;
        }
    int64_t qk, bsz;
    switch (ggml_type) {
    case K3_GGUF_TYPE_F32:   qk = 1;   bsz = 4;  break;
    case K3_GGUF_TYPE_Q8_0:  qk = 32;  bsz = 34; break;
    case K3_GGUF_TYPE_IQ1_S: qk = 256; bsz = 50; break;
    default:
        k3_deq_fail("unsupported ggml_type %d", ggml_type);
        return -1;
    }
    int err = 0;
    int64_t nb = k3_mul_i64(k3_blocks_row(ne[0], qk), bsz, &err);
    for (int i = 1; i < ndim && !err; i++) nb = k3_mul_i64(nb, ne[i], &err);
    if (err) {
        k3_deq_fail("nbytes computation overflows int64");
        return -1;
    }
    return nb;
}

int k3_gguf_dequant(const K3GgufTensor *t, K3GgufDeqOut out, void *dst,
                    const K3GgufExpect *expect)
{
    int64_t nvals;
    if (k3_deq_validate(t, expect, &nvals) != 0) return -1;

    if (out != K3_GGUF_DEQ_F32 && out != K3_GGUF_DEQ_BF16)
        return k3_deq_fail("unknown output dtype %d", (int)out);

    if (nvals == 0) return 0;               /* empty tensor: nothing to write */

    k3_dequant_fn fn = k3_deq_lookup(t->ggml_type);
    if (!fn) return k3_deq_fail("unsupported ggml_type %d", t->ggml_type);

    if (!dst) return k3_deq_fail("NULL destination for a %lld-element tensor",
                                 (long long)nvals);

    return fn(t, out, dst);
}

int k3_gguf_dequant_expert_slice(const K3GgufTensor *t, int64_t expert,
                                 K3GgufDeqOut out, void *dst,
                                 const K3GgufExpect *expect)
{
    int64_t nvals;
    if (k3_deq_validate(t, expect, &nvals) != 0) return -1;

    if (t->ggml_type != K3_GGUF_TYPE_IQ1_S)
        return k3_deq_fail("expert slice requires IQ1_S, tensor is ggml_type %d",
                           t->ggml_type);
    if (t->ndim != 3)
        return k3_deq_fail("expert slice requires a 3-D tensor, got %d-D", t->ndim);
    if (expert < 0 || expert >= t->ne[2])
        return k3_deq_fail("expert %lld out of range [0, %lld)", (long long)expert,
                           (long long)t->ne[2]);
    if (out != K3_GGUF_DEQ_F32 && out != K3_GGUF_DEQ_BF16)
        return k3_deq_fail("unknown output dtype %d", (int)out);

    /* nbytes == 0 exactly when the tensor is empty (all factors non-negative),
     * so this guard is the non-empty NULL-dst rejection, mirroring
     * k3_gguf_dequant. */
    if (t->nbytes != 0 && !dst)
        return k3_deq_fail("NULL destination for an expert slice");

    /* Empty tensor: succeed without touching dst. This must precede the slice
     * pointer math below - data may be NULL, and NULL + 0 is UB even though the
     * loop would never run. */
    if (nvals == 0) return 0;

    /* Per-expert bytes: ne1 rows of ceil(ne0/256) blocks, contiguous (ne2 is the
     * slowest dim, so slices do not interleave). Exact by construction: validation
     * above proved t->nbytes == per_expert * ne2 with all factors non-negative. */
    int64_t per_expert = t->nbytes / t->ne[2];

    K3GgufTensor sub = *t;
    sub.data = (const uint8_t *)t->data + expert * per_expert;
    sub.nbytes = per_expert;
    sub.ndim = 2;
    sub.ne[2] = sub.ne[3] = 1;
    return k3_deq_iq1_s(&sub, out, dst);
}
