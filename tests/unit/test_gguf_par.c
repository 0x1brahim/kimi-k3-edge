/* test_gguf_par.c - bit-identity gate for the OpenMP-parallel dequant/requant
 * loops (architect P3 decision A).
 *
 * The Q8_0/IQ1_S block loops in k3_gguf_dequant.c and the MXFP4 row loop in
 * k3_mxfp4_quant.c are block/row-parallel under OpenMP. The acceptance contract
 * is BIT-IDENTICAL output vs the single-thread path, so this test runs every
 * committed golden fixture and the tiny GGUF fixtures' tensors through BOTH
 * paths (omp_set_num_threads(1) vs 8) and requires byte-for-byte equality:
 *
 *   - tests/fixtures/gguf_dequant_golden.bin  (Q8_0/IQ1_S/F32, fp32 AND bf16,
 *     plus the 3-D expert-slice cases), and each non-empty quantized case
 *     TILED to thousands of blocks so real multi-thread chunking is exercised;
 *   - tests/fixtures/mxfp4_quant_golden.bin   (packed nibbles AND scales), plus
 *     a synthetic expert-width matrix;
 *   - every tensor of tests/fixtures/tiny_gguf and tiny_gguf_bf16trunk (the
 *     aligned tiny checkpoint: F32, Q8_0, IQ1_S).
 *
 * The thread-count edge case (1 thread) IS the serial side: omp_set_num_threads
 * changes the ICV before each run, so the two sides differ only in thread count.
 * Under a non-OpenMP build the pragmas compile out and both sides are the same
 * scalar code - the golden comparisons still verify correctness, and the
 * equality checks are trivially satisfied, which is exactly the promise of the
 * fallback build.
 *
 * usage: test_gguf_par <fixtures-dir>
 */
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_gguf.h"
#include "k3_gguf_dequant.h"
#include "k3_mxfp4_quant.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define PAR_THREADS 8 /* the "parallel" side of every comparison */

static int nfail;

static void fail(const char *fmt, ...)
{
    va_list ap;
    nfail++;
    fprintf(stderr, "  FAIL: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void set_threads(int n)
{
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n; /* fallback build: always serial, by construction */
#endif
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
           (uint32_t)p[3] << 24;
}

static uint64_t rd64(const uint8_t *p)
{
    return (uint64_t)rd32(p) | (uint64_t)rd32(p + 4) << 32;
}

/* ----------------------------------------------------------------- dequant -- */

typedef struct {
    int        type, ndim;
    int64_t    ne[4];
    uint64_t   nraw, nvals;
    uint32_t   flags;
    const uint8_t *raw, *f32, *bf16;
} DeqCase;

static const uint8_t *parse_deq_case(const uint8_t *p, const uint8_t *end, DeqCase *c)
{
    if ((size_t)(end - p) < 64) return NULL;
    c->type = (int)rd32(p); c->ndim = (int)rd32(p + 4); p += 8;
    for (int i = 0; i < 4; i++) { c->ne[i] = (int64_t)rd64(p); p += 8; }
    c->nraw = rd64(p); c->nvals = rd64(p + 8);
    c->flags = rd32(p + 16); p += 24;
    if (c->nraw > (uint64_t)(end - p)) return NULL;
    c->raw = p; p += c->nraw;
    if (c->nvals > (uint64_t)(end - p) / 4) return NULL;
    c->f32 = p; p += c->nvals * 4;
    if (c->nvals > (uint64_t)(end - p) / 2) return NULL;
    c->bf16 = p; p += c->nvals * 2;
    return p;
}

/* Dequant once at the given thread count into f32/bf16 buffers. */
static void dequant_at(const K3GgufTensor *t, const K3GgufExpect *ex, int threads,
                       float *f32, uint16_t *bf16)
{
    set_threads(threads);
    if (k3_gguf_dequant(t, K3_GGUF_DEQ_F32, f32, ex) != 0)
        fail("threads=%d: fp32 dequant returned error", threads);
    if (k3_gguf_dequant(t, K3_GGUF_DEQ_BF16, bf16, ex) != 0)
        fail("threads=%d: bf16 dequant returned error", threads);
}

/* One whole-tensor serial-vs-parallel comparison; nvals elements per side. */
static void cmp_tensor(const char *what, const K3GgufTensor *t,
                       const K3GgufExpect *ex, int64_t nvals)
{
    float *f1 = nvals ? (float *)malloc((size_t)nvals * sizeof(float)) : NULL;
    float *f2 = nvals ? (float *)malloc((size_t)nvals * sizeof(float)) : NULL;
    uint16_t *b1 = nvals ? (uint16_t *)malloc((size_t)nvals * sizeof(uint16_t)) : NULL;
    uint16_t *b2 = nvals ? (uint16_t *)malloc((size_t)nvals * sizeof(uint16_t)) : NULL;
    if (nvals && (!f1 || !f2 || !b1 || !b2)) {
        fail("%s: out of memory (%lld elems)", what, (long long)nvals);
        goto out;
    }
    dequant_at(t, ex, 1, f1, b1);
    dequant_at(t, ex, PAR_THREADS, f2, b2);
    if (nvals && memcmp(f1, f2, (size_t)nvals * sizeof(float)) != 0)
        fail("%s: fp32 output differs between 1 and %d threads", what, PAR_THREADS);
    if (nvals && memcmp(b1, b2, (size_t)nvals * sizeof(uint16_t)) != 0)
        fail("%s: bf16 output differs between 1 and %d threads", what, PAR_THREADS);
out:
    free(f1); free(f2); free(b1); free(b2);
}

/* One expert-slice serial-vs-parallel comparison through the expert-slice entry
 * (which shares the parallel block loop with the whole-tensor path). */
static void cmp_slice(const char *what, const K3GgufTensor *t,
                      const K3GgufExpect *ex, int64_t expert, int64_t nvals)
{
    float *f1 = nvals ? (float *)malloc((size_t)nvals * sizeof(float)) : NULL;
    float *f2 = nvals ? (float *)malloc((size_t)nvals * sizeof(float)) : NULL;
    uint16_t *b1 = nvals ? (uint16_t *)malloc((size_t)nvals * sizeof(uint16_t)) : NULL;
    uint16_t *b2 = nvals ? (uint16_t *)malloc((size_t)nvals * sizeof(uint16_t)) : NULL;
    if (nvals && (!f1 || !f2 || !b1 || !b2)) {
        fail("%s: out of memory (%lld elems)", what, (long long)nvals);
        goto out;
    }
    set_threads(1);
    if (k3_gguf_dequant_expert_slice(t, expert, K3_GGUF_DEQ_F32, f1, ex) != 0 ||
        k3_gguf_dequant_expert_slice(t, expert, K3_GGUF_DEQ_BF16, b1, ex) != 0)
        fail("%s: serial expert slice returned error", what);
    set_threads(PAR_THREADS);
    if (k3_gguf_dequant_expert_slice(t, expert, K3_GGUF_DEQ_F32, f2, ex) != 0 ||
        k3_gguf_dequant_expert_slice(t, expert, K3_GGUF_DEQ_BF16, b2, ex) != 0)
        fail("%s: parallel expert slice returned error", what);
    if (nvals && memcmp(f1, f2, (size_t)nvals * sizeof(float)) != 0)
        fail("%s: slice fp32 differs between 1 and %d threads", what, PAR_THREADS);
    if (nvals && memcmp(b1, b2, (size_t)nvals * sizeof(uint16_t)) != 0)
        fail("%s: slice bf16 differs between 1 and %d threads", what, PAR_THREADS);
out:
    free(f1); free(f2); free(b1); free(b2);
}

/* The tiled stress case: repeat the case's raw blocks to TILE_BLOCKS blocks in
 * one row, so the parallel loop really splits work across threads. Tiling is
 * exact because every quant block is self-contained. */
static void cmp_tiled(const DeqCase *c, int idx)
{
    const int64_t qk = c->type == K3_GGUF_TYPE_Q8_0    ? K3_GGUF_Q8_0_QK
                       : c->type == K3_GGUF_TYPE_IQ1_S ? K3_GGUF_IQ1_S_QK
                                                       : 1;
    const int64_t bsz = c->type == K3_GGUF_TYPE_Q8_0    ? K3_GGUF_Q8_0_BSZ
                        : c->type == K3_GGUF_TYPE_IQ1_S ? K3_GGUF_IQ1_S_BSZ
                                                        : 4;
    const int64_t nblk = (int64_t)c->nraw / bsz;
    if (nblk <= 0 || c->type == K3_GGUF_TYPE_F32) return; /* nothing to tile */

    const int64_t TILE = 4096;
    const int64_t tblocks = nblk * TILE;
    uint8_t *raw = (uint8_t *)malloc((size_t)(tblocks * bsz));
    if (!raw) { fail("case %d: tiled raw alloc failed", idx); return; }
    for (int64_t i = 0; i < TILE; i++)
        memcpy(raw + i * (size_t)(nblk * bsz), c->raw, (size_t)(nblk * bsz));

    K3GgufTensor t;
    memset(&t, 0, sizeof t);
    t.ggml_type = c->type;
    t.ndim      = 2;
    t.ne[0]     = tblocks * qk;
    t.ne[1]     = 1;
    t.ne[2]     = t.ne[3] = 1;
    t.nbytes    = tblocks * bsz;
    t.data      = raw;
    K3GgufExpect ex;
    memset(&ex, 0, sizeof ex);
    ex.ndim = 2;
    memcpy(ex.ne, t.ne, sizeof t.ne);

    cmp_tensor("tiled case", &t, &ex, tblocks * qk);
    free(raw);
}

static void run_dequant_golden(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fail("cannot open %s", path); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fail("alloc failed for %s", path); fclose(f); return; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fail("short read on %s", path);
        free(buf); fclose(f); return;
    }
    fclose(f);
    if (memcmp(buf, "K3DG", 4) != 0 || rd32(buf + 4) != 1) {
        fail("%s: bad magic/version", path);
        free(buf); return;
    }
    const int ncases = (int)rd32(buf + 8);
    printf("  gguf dequant golden: %d cases, serial(1) vs parallel(%d)\n",
           ncases, PAR_THREADS);

    const uint8_t *p = buf + 12;
    for (int i = 0; i < ncases; i++) {
        DeqCase c;
        p = parse_deq_case(p, buf + sz, &c);
        if (!p) { fail("golden case %d: truncated record", i); break; }

        K3GgufTensor t;
        memset(&t, 0, sizeof t);
        t.ggml_type = c.type;
        t.ndim      = c.ndim;
        for (int d = 0; d < 4; d++) t.ne[d] = c.ne[d];
        t.nbytes = (int64_t)c.nraw;
        t.data   = c.raw;
        K3GgufExpect ex;
        memset(&ex, 0, sizeof ex);
        ex.ndim = c.ndim;
        for (int d = 0; d < 4; d++) ex.ne[d] = c.ne[d];

        cmp_tensor("golden case", &t, &ex, (int64_t)c.nvals);
        cmp_tiled(&c, i);

        /* 3-D IQ1_S slice case: every expert's contiguous slice through the
         * expert-slice entry, which shares the parallel block loop. */
        if (c.flags & 2) {
            const int64_t prow = (c.ne[0] + K3_GGUF_IQ1_S_QK - 1) / K3_GGUF_IQ1_S_QK;
            const int64_t svals = prow * K3_GGUF_IQ1_S_QK * c.ne[1];
            for (int64_t e = 0; e < c.ne[2]; e++) {
                char what[64];
                snprintf(what, sizeof what, "expert slice e=%lld of case %d",
                         (long long)e, i);
                cmp_slice(what, &t, &ex, e, svals);
            }
        }
    }
    free(buf);
}

/* ----------------------------------------------------------------- requant -- */

static void run_mxfp4_golden(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fail("cannot open %s", path); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fail("alloc failed for %s", path); fclose(f); return; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fail("short read on %s", path);
        free(buf); fclose(f); return;
    }
    fclose(f);
    if (memcmp(buf, "K3MX", 4) != 0 || rd32(buf + 4) != 1) {
        fail("%s: bad magic/version", path);
        free(buf); return;
    }
    const int ncases = (int)rd32(buf + 8);
    printf("  mxfp4 requant golden: %d cases, serial(1) vs parallel(%d)\n",
           ncases, PAR_THREADS);

    const uint8_t *p = buf + 12;
    for (int i = 0; i < ncases; i++) {
        if ((size_t)(buf + sz - p) < 8) { fail("mxfp4 case %d: truncated", i); break; }
        const int rows = (int)rd32(p), cols = (int)rd32(p + 4);
        p += 8;
        const size_t nw = (size_t)rows * cols;
        if (nw > (size_t)(buf + sz - p) / 4) {
            fail("mxfp4 case %d: w truncated", i);
            break;
        }
        const float *w = (const float *)p;
        p += nw * 4;
        const size_t npk = nw / 2, nsc = nw / K3_MXFP4_GROUP;
        if (npk > (size_t)(buf + sz - p) || nsc > (size_t)(buf + sz - p - npk)) {
            fail("mxfp4 case %d: golden output truncated", i);
            break;
        }
        const uint8_t *want_pk = p, *want_sc = p + npk;
        p += npk + nsc; /* advance past the golden output for the next case */

        unsigned char *pk1 = (unsigned char *)malloc(npk);
        unsigned char *sc1 = (unsigned char *)malloc(nsc);
        unsigned char *pk2 = (unsigned char *)malloc(npk);
        unsigned char *sc2 = (unsigned char *)malloc(nsc);
        if (!pk1 || !sc1 || !pk2 || !sc2) { fail("mxfp4 case %d: alloc", i); break; }

        set_threads(1);
        if (k3_mxfp4_quant(pk1, sc1, w, rows, cols) != 0)
            fail("mxfp4 case %d: serial quant returned error", i);
        set_threads(PAR_THREADS);
        if (k3_mxfp4_quant(pk2, sc2, w, rows, cols) != 0)
            fail("mxfp4 case %d: parallel quant returned error", i);

        if (memcmp(pk1, pk2, npk) != 0)
            fail("mxfp4 case %d: packed differs between 1 and %d threads",
                 i, PAR_THREADS);
        if (memcmp(sc1, sc2, nsc) != 0)
            fail("mxfp4 case %d: scales differ between 1 and %d threads",
                 i, PAR_THREADS);
        /* and both must match the committed numpy golden, bit for bit */
        if (memcmp(pk1, want_pk, npk) != 0 || memcmp(sc1, want_sc, nsc) != 0)
            fail("mxfp4 case %d: parallel output differs from the golden", i);

        free(pk1); free(sc1); free(pk2); free(sc2);
    }

    /* A synthetic expert-width matrix (3584 cols = 112 groups, 512 rows): the
     * row-parallel split gets real work at every thread count. Deterministic
     * LCG values around w2-like magnitudes. */
    {
        const int rows = 512, cols = 3584;
        float *w = (float *)malloc((size_t)rows * cols * sizeof(float));
        unsigned char *pk1 = (unsigned char *)malloc((size_t)rows * cols / 2);
        unsigned char *sc1 = (unsigned char *)malloc((size_t)rows * cols / 32);
        unsigned char *pk2 = (unsigned char *)malloc((size_t)rows * cols / 2);
        unsigned char *sc2 = (unsigned char *)malloc((size_t)rows * cols / 32);
        if (!w || !pk1 || !sc1 || !pk2 || !sc2) { fail("mxfp4 stress alloc"); return; }
        uint32_t st = 12345;
        for (size_t i = 0; i < (size_t)rows * cols; i++) {
            st = st * 1664525u + 1013904223u;
            w[i] = ((float)(st >> 8) / 16777216.0f - 0.5f) * 0.04f;
        }
        set_threads(1);
        const int rc1 = k3_mxfp4_quant(pk1, sc1, w, rows, cols);
        set_threads(PAR_THREADS);
        const int rc2 = k3_mxfp4_quant(pk2, sc2, w, rows, cols);
        if (rc1 != 0 || rc2 != 0)
            fail("mxfp4 stress: quant returned %d/%d", rc1, rc2);
        else if (memcmp(pk1, pk2, (size_t)rows * cols / 2) != 0 ||
                 memcmp(sc1, sc2, (size_t)rows * cols / 32) != 0)
            fail("mxfp4 stress (%dx%d): 1-thread and %d-thread outputs differ",
                 rows, cols, PAR_THREADS);
        else
            printf("  mxfp4 stress: %dx%d matrix bit-identical across threads\n",
                   rows, cols);
        free(w); free(pk1); free(sc1); free(pk2); free(sc2);
    }
    free(buf);
}

/* ----------------------------------------------------------- tiny GGUF ---- */

/* Every tensor of one tiny GGUF file: read the raw bytes, dequant at 1 thread
 * and at PAR_THREADS, require byte-identical output. Unknown ggml types fail
 * loud: the fixture must stay inside the three supported types. */
static void run_tiny_gguf(const char *file)
{
    K3Gguf g;
    memset(&g, 0, sizeof g);
    if (k3_gguf_open_file(&g, file) != 0) {
        fail("tiny GGUF open failed: %s", file);
        return;
    }
    int checked = 0, empty = 0;
    for (int i = 0; i < g.nt; i++) {
        const K3Tensor *kt = &g.t[i];
        const int gt = g.gtype[i];
        int64_t qk;
        switch (gt) {
        case K3_GGUF_TYPE_F32:   qk = 1;   break;
        case K3_GGUF_TYPE_Q8_0:  qk = 32;  break;
        case K3_GGUF_TYPE_IQ1_S: qk = 256; break;
        default:
            fail("%s: tensor %s has unsupported ggml_type %d", file, kt->name, gt);
            continue;
        }
        if (kt->nbytes == 0) { empty++; continue; }

        uint8_t *raw = (uint8_t *)malloc((size_t)kt->nbytes);
        if (!raw) { fail("%s: alloc for %s", file, kt->name); break; }
        if (k3_gguf_read(&g, kt, raw) != kt->nbytes) {
            fail("%s: short read of %s", file, kt->name);
            free(raw);
            break;
        }

        K3GgufTensor t;
        memset(&t, 0, sizeof t);
        t.ggml_type = gt;
        t.ndim      = kt->ndim;
        for (int d = 0; d < 4; d++) t.ne[d] = kt->shape[d];
        t.nbytes = kt->nbytes;
        t.data   = raw;
        K3GgufExpect ex;
        memset(&ex, 0, sizeof ex);
        ex.ndim = kt->ndim;
        for (int d = 0; d < 4; d++) ex.ne[d] = kt->shape[d];

        /* The engine's runtime dtype mapping: F32 stays fp32, the quantized
         * types go to bf16. nvals = padded rows, exactly what dequant writes.
         * (The reader leaves trailing dims 0, so multiply only up to ndim.) */
        int64_t nvals = ((kt->shape[0] + qk - 1) / qk) * qk;
        for (int d = 1; d < kt->ndim; d++) nvals *= kt->shape[d];
        cmp_tensor(kt->name, &t, &ex, nvals);
        checked++;
        free(raw);
    }
    printf("  tiny GGUF %s: %d tensors checked (%d empty), bit-identical\n",
           file, checked, empty);
    k3_gguf_close(&g);
}

/* ------------------------------------------------------------------ main -- */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: test_gguf_par <fixtures-dir>\n");
        return 2;
    }
#ifdef _OPENMP
    /* The gate is only meaningful if the "parallel" side really runs on more
     * than one thread. Fail loud if the environment cannot provide it (e.g. a
     * nested-disabled context or a hostile ICV override) instead of silently
     * comparing serial against serial. */
    omp_set_num_threads(PAR_THREADS);
    if (omp_get_max_threads() < 2) {
        fprintf(stderr,
                "FAIL: omp_set_num_threads(%d) yields max %d threads; the "
                "parallel side would not be parallel\n",
                PAR_THREADS, omp_get_max_threads());
        return 1;
    }
    omp_set_num_threads(1); /* reset: every comparison sets its own count */
#endif
    char path[1024];
    printf("gguf parallel bit-identity gate\n");

    snprintf(path, sizeof path, "%s/gguf_dequant_golden.bin", argv[1]);
    run_dequant_golden(path);
    snprintf(path, sizeof path, "%s/mxfp4_quant_golden.bin", argv[1]);
    run_mxfp4_golden(path);
    snprintf(path, sizeof path, "%s/tiny_gguf/tiny-k3-00001-of-00001.gguf", argv[1]);
    run_tiny_gguf(path);
    snprintf(path, sizeof path, "%s/tiny_gguf_bf16trunk/tiny-k3-00001-of-00001.gguf",
             argv[1]);
    run_tiny_gguf(path);

    printf("\n%s (%d failures)\n", nfail == 0 ? "PARALLEL BIT-IDENTITY PASSED"
                                              : "PARALLEL BIT-IDENTITY FAILED",
           nfail);
    return nfail == 0 ? 0 : 1;
}
