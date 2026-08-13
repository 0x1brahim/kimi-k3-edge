/* k3_gguf_dequant.h - dequantize GGUF weights into the engine's canonical layouts.
 *
 * The real Kimi K3 GGUF (14 shards, 554 GB) contains EXACTLY three ggml types:
 * F32 (norms, biases, router, SSM params), Q8_0 (all dense weights) and IQ1_S (the
 * three merged MoE expert matrices per layer). The dispatch table below implements
 * precisely those three; any other ggml_type is a hard error. The table is an
 * extensible (type -> function) list, but no speculative types are added ahead of
 * a real file containing them.
 *
 * LAYOUT CONTRACT (what the dequant produces)
 *   GGUF stores dimensions fastest-first: ne[0] is the row length, and quantized
 *   blocks always run along ne[0] (llama.cpp convention). Dequant therefore writes,
 *   for every row, ceil(ne[0]/QK)*QK values: rows are padded to the block size
 *   exactly like the on-disk bytes are. dst must hold that many elements
 *   (padded_ne0 * ne1 * ne2 * ne3), as float (K3_GGUF_DEQ_F32) or uint16_t bf16
 *   (K3_GGUF_DEQ_BF16). Callers who need an unpadded canonical [out][in] matrix
 *   copy rows out; the real file's tensors all have block-aligned ne0, so the
 *   padded and unpadded layouts coincide there.
 *
 *   For the merged 3-D expert tensors (ne0 = in, ne1 = out, ne2 = expert) each
 *   expert's slice is a contiguous row-major [out][in] block: see
 *   tests/fixtures/gguf_expert_orientation.md for the verification.
 *
 * SHAPE ASSERTS (D4, mandatory)
 *   Every dequant call takes an expected per-dimension contract (K3GgufExpect) and
 *   fails loud on ANY mismatch: wrong dims, negative dims, a tensor nbytes that
 *   disagrees with the type's block-size formula, an unknown type, or an overflow
 *   in the nbytes computation. The engine's numel-only checks are deliberately not
 *   enough for GGUF (a transposed matrix of equal numel must not pass).
 *
 * All functions return 0 on success and -1 on failure, printing one diagnostic
 * line to stderr. Fail loud > silent anything.
 */
#ifndef K3_GGUF_DEQUANT_H
#define K3_GGUF_DEQUANT_H

#include <stdint.h>

/* ggml_type enum values the dispatch table accepts. */
#define K3_GGUF_TYPE_F32   0
#define K3_GGUF_TYPE_Q8_0  8
#define K3_GGUF_TYPE_IQ1_S 19

/* Quant block geometry (llama.cpp ggml-common.h block structs). */
#define K3_GGUF_Q8_0_QK   32    /* values per block  */
#define K3_GGUF_Q8_0_BSZ  34    /* bytes per block: ggml_half d + int8 qs[32] */
#define K3_GGUF_IQ1_S_QK  256   /* values per block  */
#define K3_GGUF_IQ1_S_BSZ 50    /* bytes per block: ggml_half d + qs[32] + qh[16] */

/* Dequant output element type. The engine's canonical narrow dtype is bf16; fp32 is
 * the widened form. bf16 conversion is round-to-nearest-even (see k3_gguf_f32_to_bf16). */
typedef enum {
    K3_GGUF_DEQ_F32 = 0,
    K3_GGUF_DEQ_BF16 = 1,
} K3GgufDeqOut;

/* A tensor as the GGUF reader hands it to the dequant: type, dimensions in GGUF
 * order (ne[0] fastest, row length), raw byte size on disk, and the raw bytes. */
typedef struct {
    int         ggml_type;   /* one of the K3_GGUF_TYPE_* values                 */
    int         ndim;        /* 1..4                                             */
    int64_t     ne[4];       /* ne[0] fastest (row length) ... ne[3] slowest     */
    int64_t     nbytes;      /* raw byte size on disk, as the reader parsed it   */
    const void *data;        /* the raw quantized bytes                          */
} K3GgufTensor;

/* The caller's per-dimension expectation. For a 2-D trunk weight: ne[0] = in
 * (cols), ne[1] = out (rows). For a merged expert tensor: ne[0] = in, ne[1] = out,
 * ne[2] = expert count. Every dim must match EXACTLY or the call fails. */
typedef struct {
    int     ndim;
    int64_t ne[4];
} K3GgufExpect;

/* Exact on-disk byte count for a tensor of this type and shape, or -1 on an unknown
 * type, bad ndim, a negative dim, or an int64 overflow (which would otherwise wrap
 * silently). The formula is ceil(ne[0]/QK)*block_bytes * ne[1]*ne[2]*ne[3], with
 * QK/block_bytes per type: F32 1/4, Q8_0 32/34, IQ1_S 256/50. */
int64_t k3_gguf_dequant_nbytes(int ggml_type, int ndim, const int64_t ne[4]);

/* Dequant one whole tensor into dst. expect is REQUIRED (never NULL): the shape
 * contract must be stated by the caller on every call, per D4. t->nbytes must equal
 * the block-size formula for (type, ne); t->data must be non-NULL unless the tensor
 * is empty (numel 0), in which case dst may be NULL and nothing is written. */
int k3_gguf_dequant(const K3GgufTensor *t, K3GgufDeqOut out, void *dst,
                    const K3GgufExpect *expect);

/* Dequant ONE expert's contiguous slice (ne[2] == expert) of a 3-D IQ1_S merged
 * expert tensor. Slice layout: per-expert bytes are contiguous in the tensor, each
 * expert is ne[1] rows of ceil(ne[0]/256)*50 bytes. The expected contract describes
 * the FULL tensor (including ne[2] = expert count); expert must be in [0, ne[2]).
 * Only IQ1_S is accepted here - it is the only type the real file uses for experts. */
int k3_gguf_dequant_expert_slice(const K3GgufTensor *t, int64_t expert,
                                 K3GgufDeqOut out, void *dst,
                                 const K3GgufExpect *expect);

/* f32 -> bf16, round-to-nearest-even on the dropped 16 mantissa bits. The standard
 * add-0x7FFF+lsb trick; exact for every finite value, and the dequant's inputs are
 * finite by construction (d is a widened fp16, dl*(grid+delta) is a few multiplies).
 * The engine's narrow dtype is bf16 (recon-model §1); see k3_bf16f/k3_bf16_to_f32
 * for the reverse direction. */
static inline uint16_t k3_gguf_f32_to_bf16(float f)
{
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t u = v.u;
    uint16_t lsb = (uint16_t)((u >> 16) & 1u);
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

/* fp16 -> fp32 is EXACT (no rounding): the engine already carries this conversion in
 * k3_st.c's F16 widen path; this is the same math, exposed for the dequant. */
static inline float k3_gguf_f16_to_f32(uint16_t h)
{
    union { uint32_t u; float f; } v;
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    if (exp == 0) {
        if (man == 0) v.u = sign;
        else { /* subnormal: renormalise */
            int sh = 0;
            while (!(man & 0x400u)) { man <<= 1; sh++; }
            man &= 0x3FFu;
            v.u = sign | ((uint32_t)(127 - 15 - sh + 1) << 23) | (man << 13);
        }
    } else if (exp == 31) v.u = sign | 0x7F800000u | (man << 13);
    else v.u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    return v.f;
}

#endif /* K3_GGUF_DEQUANT_H */
