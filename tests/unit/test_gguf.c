/* test_gguf.c - unit tests for the GGUF shard-set reader (k3_gguf.c).
 *
 * Two halves:
 *   (a) synthetic GGUF byte fixtures built in memory and written to a temp dir:
 *       the happy path (v3, tensors, every wire type 0..12, both unsloth-fork
 *       quirks) plus an adversarial set that must ALL fail loud (bad magic, wrong
 *       version, truncation, unknown types, misalignment, duplicates, split
 *       mismatches, offsets past EOF, nonzero padding);
 *   (b) a real-file gate: when K3_GGUF_REAL points at the UD-IQ1_S directory (or
 *       the known default path exists), open all 14 shards and assert the verified
 *       facts (64 KV on shard 1, 3 on the tensor shards, 2,573 tensors,
 *       split.count 14, per-shard tensor counts, the K3Cfg mapping). When the path
 *       is absent the gate prints SKIPPED and exits 0, so CI stays green without
 *       the 554 GB model.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "k3_gguf.h"

/* ------------------------------------------------------------------ builder */

typedef struct { unsigned char *b; size_t n, cap; } Buf;

static void bneed(Buf *b, size_t n)
{
    if (b->n + n > b->cap) {
        size_t nc = b->cap ? b->cap : 1 << 16;
        while (b->n + n > nc) nc *= 2;
        b->b = (unsigned char *)realloc(b->b, nc);
        b->cap = nc;
    }
}

static void bput(Buf *b, const void *p, size_t n)
{
    bneed(b, n);
    memcpy(b->b + b->n, p, n);
    b->n += n;
}

static void bu8(Buf *b, uint8_t v)    { bput(b, &v, 1); }
static void bu16(Buf *b, uint16_t v)  { bput(b, &v, 2); }
static void bu32(Buf *b, uint32_t v)  { bput(b, &v, 4); }
static void bu64(Buf *b, uint64_t v)  { bput(b, &v, 8); }
static void bf32(Buf *b, float v)     { bput(b, &v, 4); }

static void bstr(Buf *b, const char *s)
{
    bu64(b, strlen(s));
    bput(b, s, strlen(s));
}

/* One shard under construction. nkv is patched into the header at finish. */
typedef struct {
    Buf    b;
    uint32_t nkv;
    size_t info_end;       /* byte offset of the pad after the tensor infos */
} Shard;

static void shard_init(Shard *s, uint32_t version, uint64_t ntensors)
{
    memset(s, 0, sizeof *s);
    bput(&s->b, "GGUF", 4);
    bu32(&s->b, version);
    bu64(&s->b, ntensors);
    bu64(&s->b, 0);                    /* nkv, patched at finish */
}

static void shard_finish(Shard *s)
{
    uint64_t nkv = s->nkv;
    memcpy(s->b.b + 16, &nkv, 8);
    s->info_end = s->b.n;
    while (s->b.n % 32) bu8(&s->b, 0); /* zero padding, as the real file has */
}

static void skey(Shard *s, const char *k) { bstr(&s->b, k); s->nkv++; }

static void skv_u8(Shard *s, const char *k, uint8_t v)
{ skey(s, k); bu32(&s->b, 0); bu8(&s->b, v); }
static void skv_i8(Shard *s, const char *k, int8_t v)
{ skey(s, k); bu32(&s->b, 1); bu8(&s->b, (uint8_t)v); }
static void skv_u16(Shard *s, const char *k, uint16_t v)
{ skey(s, k); bu32(&s->b, 2); bu16(&s->b, v); }
static void skv_i16(Shard *s, const char *k, int16_t v)
{ skey(s, k); bu32(&s->b, 3); bu16(&s->b, (uint16_t)v); }
static void skv_u32(Shard *s, const char *k, uint32_t v)
{ skey(s, k); bu32(&s->b, 4); bu32(&s->b, v); }
static void skv_i32(Shard *s, const char *k, int32_t v)
{ skey(s, k); bu32(&s->b, 5); bu32(&s->b, (uint32_t)v); }
static void skv_f32(Shard *s, const char *k, float v)
{ skey(s, k); bu32(&s->b, 6); bf32(&s->b, v); }
static void skv_bool(Shard *s, const char *k, uint8_t v)
{ skey(s, k); bu32(&s->b, 7); bu8(&s->b, v); }
static void skv_str(Shard *s, const char *k, const char *v)
{ skey(s, k); bu32(&s->b, 8); bstr(&s->b, v); }
static void skv_u64(Shard *s, const char *k, uint64_t v)
{ skey(s, k); bu32(&s->b, 10); bu64(&s->b, v); }
static void skv_i64(Shard *s, const char *k, int64_t v)
{ skey(s, k); bu32(&s->b, 11); bu64(&s->b, (uint64_t)v); }
static void skv_f64(Shard *s, const char *k, double v)
{ skey(s, k); bu32(&s->b, 12); bput(&s->b, &v, 8); }

/* Array with the VERIFIED quirk: elements declared as type 5 are 4 bytes each on
 * the wire (observed on kimi-k3.attention.head_count_kv and
 * tokenizer.ggml.token_type in the real file). */
static void skv_arr_i32(Shard *s, const char *k, const uint32_t *v, uint32_t n)
{
    skey(s, k);
    bu32(&s->b, 9);
    bu32(&s->b, 5);
    bu64(&s->b, n);
    bput(&s->b, v, (size_t)n * 4);
}

static void skv_arr_str(Shard *s, const char *k, const char *const *v, uint32_t n)
{
    skey(s, k);
    bu32(&s->b, 9);
    bu32(&s->b, 8);
    bu64(&s->b, n);
    for (uint32_t i = 0; i < n; i++) bstr(&s->b, v[i]);
}

static void ssplit(Shard *s, uint16_t no, uint16_t count, int32_t tensors)
{
    skv_u16(s, "split.no", no);
    skv_i32(s, "split.tensors.count", tensors);
    skv_u16(s, "split.count", count);
}

static void stensor(Shard *s, const char *name, uint32_t ndim,
                    const uint64_t *dims, uint32_t gtype, uint64_t off)
{
    bstr(&s->b, name);
    bu32(&s->b, ndim);
    for (uint32_t i = 0; i < ndim; i++) bu64(&s->b, dims[i]);
    bu32(&s->b, gtype);
    bu64(&s->b, off);
}

/* ------------------------------------------------------------------ harness */

static int fails = 0;

static void ck(int cond, const char *what)
{
    if (cond) printf("  ok    %s\n", what);
    else { printf("  FAIL  %s\n", what); fails++; }
}

static char g_dir[256];

typedef struct { const char *name; const unsigned char *d; size_t n; } GFile;

static int write_fixture(const GFile *files, int nf)
{
    snprintf(g_dir, sizeof g_dir, "/tmp/k3ggufXXXXXX");
    if (!mkdtemp(g_dir)) return -1;
    for (int i = 0; i < nf; i++) {
        char path[512];
        snprintf(path, sizeof path, "%s/%s", g_dir, files[i].name);
        FILE *f = fopen(path, "wb");
        if (!f) return -1;
        fwrite(files[i].d, 1, files[i].n, f);
        fclose(f);
    }
    return 0;
}

static void cleanup_fixture(const GFile *files, int nf)
{
    for (int i = 0; i < nf; i++) {
        char path[512];
        snprintf(path, sizeof path, "%s/%s", g_dir, files[i].name);
        unlink(path);
    }
    rmdir(g_dir);
}

/* Open the fixture set; expect_open says whether the open must succeed. checks()
 * runs on the opened index (before close) when it succeeds. */
static void run_open(const char *label, const GFile *files, int nf, int expect_open,
                     void (*checks)(const K3Gguf *))
{
    if (write_fixture(files, nf) != 0) {
        printf("  FAIL  %-30s cannot write fixture\n", label);
        fails++;
        return;
    }
    K3Gguf g;
    int rc = k3_gguf_open(&g, g_dir);
    if (expect_open) {
        if (rc != 0) {
            printf("  FAIL  %-30s open rejected a valid fixture\n", label);
            fails++;
        }
        else {
            printf("  ok    %-30s open\n", label);
            if (checks) checks(&g);
            k3_gguf_close(&g);
        }
    } else {
        if (rc == 0) {
            printf("  FAIL  %-30s opened a corrupt fixture\n", label);
            fails++;
        } else
            printf("  ok    %-30s correctly rejected\n", label);
    }
    cleanup_fixture(files, nf);
}

/* ------------------------------------------------------------------ fixtures */

/* Minimal valid two-shard set: s1 metadata-only, s2 with one F32 tensor. Every
 * adversarial case starts here and injects one defect. Callers own the buffers
 * (free2 at the end of each case); this never frees, so the first call on a fresh
 * stack struct is safe. */
static void base_set(Shard *s1, Shard *s2)
{
    shard_init(s1, 3, 0);
    ssplit(s1, 0, 2, 1);
    shard_finish(s1);

    shard_init(s2, 3, 1);
    ssplit(s2, 1, 2, 1);
    stensor(s2, "x.weight", 1, (const uint64_t[]){ 8 }, 0, 0);
    shard_finish(s2);
    unsigned char data[32];
    memset(data, 7, sizeof data);
    bput(&s2->b, data, sizeof data);
}

static void free2(Shard *s1, Shard *s2)
{
    free(s1->b.b);
    free(s2->b.b);
}

static void base_files(GFile *f, const Shard *s1, const Shard *s2)
{
    f[0] = (GFile){ "s1.gguf", s1->b.b, s1->b.n };
    f[1] = (GFile){ "s2.gguf", s2->b.b, s2->b.n };
}

/* The full verified kimi-k3.* inventory with small, distinctive values. The
 * variant parameter lets the negative tests drop or corrupt exactly one thing. */
enum { CFG_OK = 0, CFG_OMIT_EXPERT_COUNT, CFG_UNKNOWN_KEY, CFG_HCK_SHORT,
       CFG_HCK_BADVAL, CFG_BLOCK_COUNT_F32, CFG_OMIT_FREQ_BASE,
       CFG_FREQ_BASE_U32 };

static void cfg_fixture(Shard *s1, Shard *s2, int variant)
{
    static const uint32_t hck[13] = { 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1 };

    shard_init(s1, 3, 0);
    skv_str(s1, "general.architecture", "kimi-k3");
    skv_u32(s1, "kimi-k3.block_count", 13);
    skv_u32(s1, "kimi-k3.context_length", 4096);
    skv_u32(s1, "kimi-k3.embedding_length", 128);
    skv_u32(s1, "kimi-k3.feed_forward_length", 128);
    skv_u32(s1, "kimi-k3.attention.head_count", 4);
    if (variant == CFG_HCK_SHORT)
        skv_arr_i32(s1, "kimi-k3.attention.head_count_kv", hck, 12);
    else if (variant == CFG_HCK_BADVAL) {
        uint32_t bad[13];
        memcpy(bad, hck, sizeof bad);
        bad[0] = 2;
        skv_arr_i32(s1, "kimi-k3.attention.head_count_kv", bad, 13);
    } else
        skv_arr_i32(s1, "kimi-k3.attention.head_count_kv", hck, 13);
    if (variant != CFG_OMIT_FREQ_BASE)
        skv_f32(s1, "kimi-k3.rope.freq_base", 10000.0f);
    skv_f32(s1, "kimi-k3.attention.layer_norm_rms_epsilon", 1e-5f);
    if (variant != CFG_OMIT_EXPERT_COUNT)
        skv_u32(s1, "kimi-k3.expert_count", 8);
    skv_u32(s1, "kimi-k3.expert_used_count", 2);
    skv_u32(s1, "kimi-k3.expert_group_used_count", 1);
    skv_u32(s1, "kimi-k3.expert_gating_func", 2);
    skv_u32(s1, "kimi-k3.attention.key_length", 24);
    skv_u32(s1, "kimi-k3.attention.value_length", 16);
    skv_u32(s1, "kimi-k3.vocab_size", 256);
    skv_u32(s1, "kimi-k3.ssm.conv_kernel", 4);
    skv_u32(s1, "kimi-k3.kda.head_dim", 16);
    skv_f32(s1, "kimi-k3.kda.gate_lower_bound", -5.0f);
    skv_u32(s1, "kimi-k3.attention.q_lora_rank", 32);
    skv_u32(s1, "kimi-k3.attention.kv_lora_rank", 16);
    skv_u32(s1, "kimi-k3.rope.dimension_count", 8);
    skv_u32(s1, "kimi-k3.attention.key_length_mla", 24);
    skv_u32(s1, "kimi-k3.attention.value_length_mla", 16);
    skv_u32(s1, "kimi-k3.expert_feed_forward_length", 32);
    skv_u32(s1, "kimi-k3.expert_shared_count", 2);
    skv_u32(s1, "kimi-k3.leading_dense_block_count", 1);
    skv_f32(s1, "kimi-k3.expert_weights_scale", 1.0f);
    skv_bool(s1, "kimi-k3.expert_weights_norm", 1);
    skv_u32(s1, "kimi-k3.expert_latent_length", 64);
    skv_f32(s1, "kimi-k3.activation.situ_beta", 4.0f);
    skv_f32(s1, "kimi-k3.activation.situ_linear_beta", 25.0f);
    skv_u32(s1, "kimi-k3.attn_res.block_size", 3);
    if (variant == CFG_UNKNOWN_KEY)
        skv_u32(s1, "kimi-k3.bogus.field", 1);
    ssplit(s1, 0, 2, 1);
    shard_finish(s1);

    if (variant == CFG_BLOCK_COUNT_F32 || variant == CFG_FREQ_BASE_U32) {
        /* re-type an f32/u32 key of the same 4-byte width: the walk stays aligned
         * and the open succeeds; only the cfg load must fail. */
        const char *needle = variant == CFG_BLOCK_COUNT_F32 ? "kimi-k3.block_count"
                                                            : "kimi-k3.rope.freq_base";
        unsigned char newtype = variant == CFG_BLOCK_COUNT_F32 ? 6 : 4;
        size_t nlen = strlen(needle);
        for (size_t j = 0; j + 8 + nlen + 4 <= s1->b.n; j++) {
            if (!memcmp(s1->b.b + j + 8, needle, nlen)) {
                s1->b.b[j + 8 + nlen] = newtype;
                break;
            }
        }
    }

    shard_init(s2, 3, 1);
    ssplit(s2, 1, 2, 1);
    stensor(s2, "x.weight", 1, (const uint64_t[]){ 8 }, 0, 0);
    shard_finish(s2);
    unsigned char data[32];
    memset(data, 7, sizeof data);
    bput(&s2->b, data, sizeof data);
}

/* ------------------------------------------------------------------ checks */

static void check_happy(const K3Gguf *g)
{
    ck(g->nshard == 3, "3 shards");
    ck(g->nt == 3, "3 tensors");
    ck(g->alignment == 32, "alignment defaults to 32");
    ck(g->shard_kv[0] == 20, "shard 1 has 20 metadata pairs");
    ck(g->shard_kv[1] == 3 && g->shard_kv[2] == 3, "tensor shards have 3 pairs");

    const K3Tensor *a = k3_gguf_find(g, "alpha.weight");
    ck(a != NULL, "find alpha.weight");
    if (a) {
        ck(a->shard == 1, "alpha in shard 2");
        ck(a->dtype == K3_DT_F32, "alpha dtype F32");
        ck(a->ndim == 2 && a->shape[0] == 4 && a->shape[1] == 4, "alpha shape (4,4)");
        ck(a->nbytes == 64, "alpha nbytes 64");
        ck(a->off % 32 == 0, "alpha offset 32-aligned");
    }
    const K3Tensor *b = k3_gguf_find(g, "beta.weight");
    ck(b != NULL, "find beta.weight");
    if (b) {
        ck(b->dtype == K3_DT_U8 && g->gtype[b - g->t] == 8, "beta U8/Q8_0");
        ck(b->nbytes == 34, "beta nbytes 34 (one Q8_0 block)");
        ck(b->off == a->off + 64, "beta follows alpha");
    }
    const K3Tensor *gm = k3_gguf_find(g, "gamma.weight");
    ck(gm != NULL, "find gamma.weight");
    if (gm) {
        ck(g->gtype[gm - g->t] == 19, "gamma gtype IQ1_S");
        ck(gm->nbytes == 50, "gamma nbytes 50 (one IQ1_S block)");
        ck(gm->shard == 2, "gamma in shard 3");
    }

    int bad = 0;
    for (int i = 0; i < g->nt; i++)
        if (k3_gguf_find(g, g->t[i].name) != &g->t[i]) bad++;
    ck(bad == 0, "round trip: every name resolves to itself");
    ck(k3_gguf_find(g, "no.such.tensor") == NULL, "absent name returns NULL");

    /* every wire type 0..12, including the two quirks */
    const K3GgufKV *k;
    k = k3_gguf_kv(g, "test.u8");
    ck(k && k->type == 0 && k->v.u == 0x5A, "u8 value");
    k = k3_gguf_kv(g, "test.i8");
    ck(k && k->type == 1 && k->v.u == (uint64_t)(int64_t)(int8_t)-3, "i8 value");
    k = k3_gguf_kv(g, "test.u16");
    ck(k && k->type == 2 && k->v.u == 0xBEEF, "u16 value");
    k = k3_gguf_kv(g, "test.i16");
    ck(k && k->type == 3 && k->v.u == (uint64_t)(int64_t)(int16_t)-1234, "i16 value");
    k = k3_gguf_kv(g, "test.u32");
    ck(k && k->type == 4 && k->v.u == 42, "u32 value");
    k = k3_gguf_kv(g, "test.i32");
    ck(k && k->type == 5 && k->v.u == (uint64_t)(int64_t)(int32_t)-7, "i32 value");
    k = k3_gguf_kv(g, "test.f32");
    ck(k && k->type == 6 && k->v.f == 3.5, "f32 value");
    k = k3_gguf_kv(g, "test.bool");
    ck(k && k->type == 7 && k->v.u == 1, "bool value");
    k = k3_gguf_kv(g, "test.str");
    ck(k && k->type == 8 && k->v.s.n == 5 && !memcmp(k->v.s.p, "hello", 5),
       "string value");
    k = k3_gguf_kv(g, "test.empty");
    ck(k && k->type == 8 && k->v.s.n == 0, "empty string value");
    k = k3_gguf_kv(g, "test.u64");
    ck(k && k->type == 10 && k->v.u == 0xDEADBEEFCAFEBABEull, "u64 value");
    k = k3_gguf_kv(g, "test.i64");
    ck(k && k->type == 11 && k->v.u == (uint64_t)-1234567890123ll, "i64 value");
    k = k3_gguf_kv(g, "test.f64");
    ck(k && k->type == 12 && k->v.f == 2.718281828, "f64 value");
    k = k3_gguf_kv(g, "test.arr_i32");
    ck(k && k->type == 9 && k->v.a.et == 5 && k->v.a.n == 4 && k->v.a.nbytes == 16 &&
       k->v.a.d && ((const uint32_t *)k->v.a.d)[0] == 1 &&
       ((const uint32_t *)k->v.a.d)[3] == 4,
       "array quirk: type-5 elements are 4 bytes each");
    k = k3_gguf_kv(g, "test.arr_str");
    ck(k && k->type == 9 && k->v.a.et == 8 && k->v.a.n == 3 && k->v.a.d == NULL,
       "string array walked, not materialized");

    unsigned char buf[128];
    ck(k3_gguf_read(g, a, buf) == 64 && !memcmp(buf, "\x00\x01\x02", 3), "read alpha");
    ck(k3_gguf_read(g, b, buf) == 34 && buf[33] == 97, "read beta");
}

/* ------------------------------------------------------------------ main */

int main(void)
{
    printf("== gguf reader (synthetic fixtures) ==\n");

    /* happy path: three shards, 4 tensors, every wire type, both quirks */
    {
        static const uint32_t arr4[4] = { 1, 2, 3, 4 };
        static const char *const arrs[3] = { "a", "bb", "ccc" };
        Shard s1, s2, s3;
        shard_init(&s1, 3, 0);
        skv_str(&s1, "general.architecture", "kimi-k3");
        skv_u32(&s1, "general.file_type", 24);
        skv_u8(&s1, "test.u8", 0x5A);
        skv_i8(&s1, "test.i8", -3);
        skv_u16(&s1, "test.u16", 0xBEEF);
        skv_i16(&s1, "test.i16", -1234);
        skv_u32(&s1, "test.u32", 42);
        skv_i32(&s1, "test.i32", -7);
        skv_f32(&s1, "test.f32", 3.5f);
        skv_bool(&s1, "test.bool", 1);
        skv_str(&s1, "test.str", "hello");
        skv_str(&s1, "test.empty", "");        /* empty string values are legal */
        skv_arr_i32(&s1, "test.arr_i32", arr4, 4);
        skv_arr_str(&s1, "test.arr_str", arrs, 3);
        skv_u64(&s1, "test.u64", 0xDEADBEEFCAFEBABEull);
        skv_i64(&s1, "test.i64", -1234567890123ll);
        skv_f64(&s1, "test.f64", 2.718281828);
        ssplit(&s1, 0, 3, 3);
        shard_finish(&s1);

        static const uint64_t d2d[2] = { 4, 4 };
        shard_init(&s2, 3, 2);
        ssplit(&s2, 1, 3, 3);
        stensor(&s2, "alpha.weight", 2, d2d, 0, 0);              /* F32 4x4 */
        stensor(&s2, "beta.weight", 1, (const uint64_t[]){ 32 }, 8, 64); /* Q8_0 */
        shard_finish(&s2);
        unsigned char data2[98];
        for (size_t i = 0; i < sizeof data2; i++) data2[i] = (unsigned char)i;
        bput(&s2.b, data2, sizeof data2);   /* alpha 64 B then beta 34 B */

        shard_init(&s3, 3, 1);
        ssplit(&s3, 2, 3, 3);
        stensor(&s3, "gamma.weight", 1, (const uint64_t[]){ 256 }, 19, 0); /* IQ1_S */
        shard_finish(&s3);
        unsigned char data3[50];
        for (size_t i = 0; i < sizeof data3; i++) data3[i] = (unsigned char)(i * 3);
        bput(&s3.b, data3, sizeof data3);

        GFile f[3] = {
            { "s1.gguf", s1.b.b, s1.b.n },
            { "s2.gguf", s2.b.b, s2.b.n },
            { "s3.gguf", s3.b.b, s3.b.n },
        };
        run_open("happy path", f, 3, 1, check_happy);
        free2(&s1, &s2);
        free(s3.b.b);
    }

    /* --- adversarial: every case must FAIL LOUD --- */
    printf("== gguf reader (adversarial fixtures) ==\n");

    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);

        s1.b.b[1] = 'X';
        base_files(f, &s1, &s2);
        run_open("bad magic", f, 2, 0, NULL);
        free2(&s1, &s2);

        base_set(&s1, &s2);
        s1.b.b[4] = 2;                      /* version u32 at offset 4 */
        base_files(f, &s1, &s2);
        run_open("wrong version", f, 2, 0, NULL);
        free2(&s1, &s2);

        base_set(&s1, &s2);
        s2.b.b[0] = 'X';                    /* bad magic on a later shard */
        base_files(f, &s1, &s2);
        run_open("bad magic on shard 2", f, 2, 0, NULL);
        free2(&s1, &s2);

        base_set(&s1, &s2);
        s1.b.n = 20;                        /* cut inside the header */
        base_files(f, &s1, &s2);
        run_open("truncated header", f, 2, 0, NULL);
        free2(&s1, &s2);

        base_set(&s1, &s2);
        s1.b.n = 24 + 12;                   /* header claims 3 KV, file ends early */
        base_files(f, &s1, &s2);
        run_open("truncated metadata", f, 2, 0, NULL);
        free2(&s1, &s2);

        base_set(&s1, &s2);
        s2.b.n = s2.info_end - 10;          /* cut inside the tensor infos */
        base_files(f, &s1, &s2);
        run_open("truncated tensor infos", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* unknown metadata value type: 13 is outside the 0..12 enum */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s1.b.b);
        shard_init(&s1, 3, 0);
        ssplit(&s1, 0, 2, 1);
        skey(&s1, "bad.type");
        bu32(&s1.b, 13);
        bu32(&s1.b, 1);
        shard_finish(&s1);
        base_files(f, &s1, &s2);
        run_open("unknown metadata type", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* a bool that is neither 0 nor 1 */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s1.b.b);
        shard_init(&s1, 3, 0);
        ssplit(&s1, 0, 2, 1);
        skv_bool(&s1, "bad.bool", 2);
        shard_finish(&s1);
        base_files(f, &s1, &s2);
        run_open("bool value 2", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* string value claiming more bytes than the file holds */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s1.b.b);
        shard_init(&s1, 3, 0);
        ssplit(&s1, 0, 2, 1);
        skey(&s1, "bad.str");
        bu32(&s1.b, 8);
        bu64(&s1.b, 1000);                  /* no bytes follow */
        shard_finish(&s1);
        base_files(f, &s1, &s2);
        run_open("truncated string value", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* metadata walk misalignment: the second key's length field is garbage */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        /* layout: header 24, split.no record is 22 bytes -> split.tensors.count
         * length field starts at 46 */
        memset(s1.b.b + 46, 0xFF, 8);
        base_files(f, &s1, &s2);
        run_open("misaligned metadata walk", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* a shard with no split keys at all */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s2.b.b);
        shard_init(&s2, 3, 0);
        shard_finish(&s2);
        base_files(f, &s1, &s2);
        run_open("missing split keys", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* split.count type mismatch (u32 where u16 is required) */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s2.b.b);
        shard_init(&s2, 3, 0);
        skv_u32(&s2, "split.no", 1);
        skv_i32(&s2, "split.tensors.count", 1);
        skv_u32(&s2, "split.count", 2);
        shard_finish(&s2);
        base_files(f, &s1, &s2);
        run_open("mistyped split key", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* split.count disagrees with the number of files */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s1.b.b);
        shard_init(&s1, 3, 0);
        ssplit(&s1, 0, 14, 1);
        shard_finish(&s1);
        base_files(f, &s1, &s2);
        run_open("shard-count mismatch", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* split.no disagrees with the sorted file position */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s2.b.b);
        shard_init(&s2, 3, 1);
        ssplit(&s2, 5, 2, 1);
        stensor(&s2, "x.weight", 1, (const uint64_t[]){ 8 }, 0, 0);
        shard_finish(&s2);
        unsigned char data[32];
        memset(data, 7, sizeof data);
        bput(&s2.b, data, sizeof data);
        base_files(f, &s1, &s2);
        run_open("split.no mismatch", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* the per-shard tensor counts do not sum to split.tensors.count */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s1.b.b);
        shard_init(&s1, 3, 0);
        ssplit(&s1, 0, 2, 99);
        shard_finish(&s1);
        base_files(f, &s1, &s2);
        run_open("split.tensors.count mismatch", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* an unsupported ggml type */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s2.b.b);
        shard_init(&s2, 3, 1);
        ssplit(&s2, 1, 2, 1);
        stensor(&s2, "q4.weight", 1, (const uint64_t[]){ 32 }, 3, 0); /* Q4_0 */
        shard_finish(&s2);
        unsigned char data[32];
        memset(data, 7, sizeof data);
        bput(&s2.b, data, sizeof data);
        base_files(f, &s1, &s2);
        run_open("unknown ggml type", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* tensor offset not aligned to the data-section alignment */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s2.b.b);
        shard_init(&s2, 3, 1);
        ssplit(&s2, 1, 2, 1);
        stensor(&s2, "x.weight", 1, (const uint64_t[]){ 8 }, 0, 8);
        shard_finish(&s2);
        unsigned char data[64];
        memset(data, 7, sizeof data);
        bput(&s2.b, data, sizeof data);
        base_files(f, &s1, &s2);
        run_open("misaligned tensor offset", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* a zero dimension */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s2.b.b);
        shard_init(&s2, 3, 1);
        ssplit(&s2, 1, 2, 1);
        stensor(&s2, "x.weight", 1, (const uint64_t[]){ 0 }, 0, 0);
        shard_finish(&s2);
        unsigned char data[32];
        memset(data, 7, sizeof data);
        bput(&s2.b, data, sizeof data);
        base_files(f, &s1, &s2);
        run_open("zero dimension", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* more than four dimensions */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s2.b.b);
        shard_init(&s2, 3, 1);
        ssplit(&s2, 1, 2, 1);
        stensor(&s2, "x.weight", 5, (const uint64_t[]){ 1, 1, 1, 1, 1 }, 0, 0);
        shard_finish(&s2);
        base_files(f, &s1, &s2);
        run_open("ndim > 4", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* tensor span ends past EOF */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s2.b.b);
        shard_init(&s2, 3, 1);
        ssplit(&s2, 1, 2, 1);
        stensor(&s2, "x.weight", 1, (const uint64_t[]){ 8 }, 0, 200);
        shard_finish(&s2);              /* no data appended: file ends at dstart */
        base_files(f, &s1, &s2);
        run_open("tensor past EOF", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* second tensor overlaps the first */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s2.b.b);
        shard_init(&s2, 3, 2);
        ssplit(&s2, 1, 2, 2);
        stensor(&s2, "a.weight", 1, (const uint64_t[]){ 8 }, 0, 0);
        stensor(&s2, "b.weight", 1, (const uint64_t[]){ 8 }, 0, 8);
        shard_finish(&s2);
        unsigned char data[64];
        memset(data, 7, sizeof data);
        bput(&s2.b, data, sizeof data);
        base_files(f, &s1, &s2);
        run_open("overlapping tensors", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* the same tensor name in two shards */
    {
        Shard s1 = { 0 }, s2 = { 0 }, s3 = { 0 };
        GFile f[3];
        shard_init(&s1, 3, 0);
        ssplit(&s1, 0, 3, 2);
        shard_finish(&s1);
        shard_init(&s2, 3, 1);
        ssplit(&s2, 1, 3, 2);
        stensor(&s2, "x.weight", 1, (const uint64_t[]){ 8 }, 0, 0);
        shard_finish(&s2);
        unsigned char data[32];
        memset(data, 7, sizeof data);
        bput(&s2.b, data, sizeof data);
        shard_init(&s3, 3, 1);
        ssplit(&s3, 2, 3, 2);
        stensor(&s3, "x.weight", 1, (const uint64_t[]){ 8 }, 0, 0);
        shard_finish(&s3);
        bput(&s3.b, data, sizeof data);
        f[0] = (GFile){ "s1.gguf", s1.b.b, s1.b.n };
        f[1] = (GFile){ "s2.gguf", s2.b.b, s2.b.n };
        f[2] = (GFile){ "s3.gguf", s3.b.b, s3.b.n };
        run_open("duplicate tensor name", f, 3, 0, NULL);
        free2(&s1, &s2);
        free(s3.b.b);
    }

    /* a nonzero byte in the pad between tensor infos and data */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        if (s2.info_end % 32 == 0) {
            printf("  FAIL  nonzero-pad fixture unexpectedly aligned\n");
            fails++;
        } else {
            s2.b.b[s2.info_end] = 1;
            base_files(f, &s1, &s2);
            run_open("nonzero pad byte", f, 2, 0, NULL);
        }
        free2(&s1, &s2);
    }

    /* the D7 boundary check itself: a nonzero byte in the pad immediately after
     * the last metadata KV of the metadata-only shard must fail the walk
     * (crit-a-1 F3: this exact rejection path had no negative test) */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        if (s1.info_end % 32 == 0) {
            printf("  FAIL  boundary fixture unexpectedly aligned\n");
            fails++;
        } else {
            s1.b.b[s1.info_end] = 1;
            base_files(f, &s1, &s2);
            run_open("boundary check after metadata", f, 2, 0, NULL);
        }
        free2(&s1, &s2);
    }

    /* duplicate metadata key within one shard (crit-a-2 #3) */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s1.b.b);
        shard_init(&s1, 3, 0);
        ssplit(&s1, 0, 2, 1);
        skv_u32(&s1, "test.dup", 1);
        skv_u32(&s1, "test.dup", 2);
        shard_finish(&s1);
        base_files(f, &s1, &s2);
        run_open("duplicate metadata key", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* nbytes computation overflow (dims product through mul_u64) */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s2.b.b);
        shard_init(&s2, 3, 1);
        ssplit(&s2, 1, 2, 1);
        stensor(&s2, "x.weight", 2,
                (const uint64_t[]){ 0x4000000000000000ull, 0x4000000000000000ull },
                0, 0);
        shard_finish(&s2);
        base_files(f, &s1, &s2);
        run_open("nbytes overflow", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* caps on lying headers: absurd kv/tensor counts, over-long array and string */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        uint64_t nkv = (1ull << 24) + 1;     /* nkv field at offset 16 */
        memcpy(s1.b.b + 16, &nkv, 8);
        base_files(f, &s1, &s2);
        run_open("absurd metadata count", f, 2, 0, NULL);
        free2(&s1, &s2);

        base_set(&s1, &s2);
        uint64_t nten = (1ull << 24) + 1;    /* tensor_count field at offset 8 */
        memcpy(s1.b.b + 8, &nten, 8);
        base_files(f, &s1, &s2);
        run_open("absurd tensor count", f, 2, 0, NULL);
        free2(&s1, &s2);

        base_set(&s1, &s2);
        free(s1.b.b);
        shard_init(&s1, 3, 0);
        ssplit(&s1, 0, 2, 1);
        skey(&s1, "test.big");
        bu32(&s1.b, 9);                     /* array */
        bu32(&s1.b, 5);                     /* elem type int32 */
        bu64(&s1.b, (1ull << 22) + 1);      /* over the 2^22 element cap */
        shard_finish(&s1);
        base_files(f, &s1, &s2);
        run_open("array over element cap", f, 2, 0, NULL);
        free2(&s1, &s2);

        base_set(&s1, &s2);
        free(s1.b.b);
        shard_init(&s1, 3, 0);
        ssplit(&s1, 0, 2, 1);
        skey(&s1, "bad.str");
        bu32(&s1.b, 8);
        bu64(&s1.b, (1ull << 24) + 1);      /* over the 16 MB string cap */
        shard_finish(&s1);
        base_files(f, &s1, &s2);
        run_open("string over length cap", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* invalid general.alignment: 0 and a non-power of two */
    {
        Shard s1, s2;
        GFile f[2];
        base_set(&s1, &s2);
        free(s1.b.b);
        shard_init(&s1, 3, 0);
        skv_u32(&s1, "general.alignment", 0);
        ssplit(&s1, 0, 2, 1);
        shard_finish(&s1);
        base_files(f, &s1, &s2);
        run_open("alignment zero", f, 2, 0, NULL);
        free2(&s1, &s2);

        base_set(&s1, &s2);
        free(s1.b.b);
        shard_init(&s1, 3, 0);
        skv_u32(&s1, "general.alignment", 7);
        ssplit(&s1, 0, 2, 1);
        shard_finish(&s1);
        base_files(f, &s1, &s2);
        run_open("alignment not power of two", f, 2, 0, NULL);
        free2(&s1, &s2);
    }

    /* an empty directory */
    {
        if (write_fixture(NULL, 0) == 0) {
            K3Gguf g;
            int rc = k3_gguf_open(&g, g_dir);
            ck(rc != 0, "empty directory rejected");
            if (rc == 0) k3_gguf_close(&g);
            cleanup_fixture(NULL, 0);
        } else {
            printf("  FAIL  cannot write empty fixture dir\n");
            fails++;
        }
    }

    /* --- config: happy + negative --- */
    {
        Shard s1 = { 0 }, s2 = { 0 };
        GFile f[2];
        cfg_fixture(&s1, &s2, CFG_OK);
        base_files(f, &s1, &s2);
        if (write_fixture(f, 2) != 0) {
            printf("  FAIL  cfg: cannot write fixture\n");
            fails++;
        } else {
            K3Gguf g;
            K3Cfg c;
            static int fa[128];
            if (k3_gguf_open(&g, g_dir) != 0) {
                printf("  FAIL  cfg: open\n");
                fails++;
            } else {
                int ok = k3_gguf_cfg(&g, &c, fa, 128);
                ck(ok == 1, "cfg loads");
                if (ok) {
                    ck(c.hidden == 128, "cfg hidden");
                    ck(c.n_layers == 13, "cfg layers");
                    ck(c.vocab == 256, "cfg vocab");
                    ck(c.kda_heads == 4 && c.n_heads == 4, "cfg heads");
                    ck(c.kda_head_dim == 16, "cfg kda head_dim");
                    ck(c.conv_k == 4, "cfg conv_k");
                    ck(c.gate_lb == -5.0f, "cfg gate_lb");
                    ck(c.q_lora == 32 && c.kv_lora == 16, "cfg q/kv lora");
                    ck(c.qk_nope == 16 && c.qk_rope == 8, "cfg qk_nope/qk_rope derived");
                    ck(c.v_head == 16, "cfg v_head");
                    ck(c.mla_out_gate == 1 && c.latent_norm == 1,
                       "cfg documented defaults");
                    ck(c.n_experts == 8 && c.topk == 2 && c.n_shared == 2,
                       "cfg experts");
                    ck(c.latent == 64 && c.moe_inter == 32,
                       "cfg latent/moe_inter");
                    ck(c.routed_scale == 1.0f && c.moe_renorm == 1,
                       "cfg routed scale/renorm");
                    ck(c.first_dense == 1 && c.dense_inter == 128, "cfg dense");
                    ck(c.attn_res_block == 3, "cfg attn_res_block");
                    ck(c.situ_b1 == 4.0f && c.situ_b2 == 25.0f, "cfg situ betas");
                    ck(c.rms_eps == 1e-5f, "cfg rms_eps");
                    ck(c.n_full_attn == 4, "cfg 4 MLA layers");
                    ck(fa[0] == 3 && fa[1] == 6 && fa[2] == 9 && fa[3] == 13,
                       "cfg one-based fa list");
                    /* the layer classifier must work on a gguf-derived cfg
                     * (crit-a-2 BLOCKER: full_attn was never assigned) */
                    ck(k3_is_mla(&c, 2) && k3_is_mla(&c, 12), "k3_is_mla true on MLA");
                    ck(!k3_is_mla(&c, 0) && !k3_is_mla(&c, 1), "k3_is_mla false on KDA");
                    ck(k3_is_kda(&c, 0) && k3_is_dense(&c, 0), "k3_is_kda/dense");
                }
                k3_gguf_close(&g);
            }
            cleanup_fixture(f, 2);
        }

        const char *neg_labels[] = {
            "cfg rejects missing key",
            "cfg rejects unknown key",
            "cfg rejects short head_count_kv",
            "cfg rejects bad head_count_kv value",
            "cfg rejects mistyped key",
            "cfg rejects missing rope.freq_base",
            "cfg rejects mistyped rope.freq_base",
        };
        const int neg_variants[] = { CFG_OMIT_EXPERT_COUNT, CFG_UNKNOWN_KEY,
                                     CFG_HCK_SHORT, CFG_HCK_BADVAL,
                                     CFG_BLOCK_COUNT_F32, CFG_OMIT_FREQ_BASE,
                                     CFG_FREQ_BASE_U32 };
        for (int i = 0; i < 7; i++) {
            free(s1.b.b);
            free(s2.b.b);
            cfg_fixture(&s1, &s2, neg_variants[i]);
            base_files(f, &s1, &s2);
            if (write_fixture(f, 2) != 0) {
                printf("  FAIL  %s: cannot write fixture\n", neg_labels[i]);
                fails++;
                continue;
            }
            K3Gguf g;
            K3Cfg c;
            static int fa[128];
            if (k3_gguf_open(&g, g_dir) != 0) {
                printf("  FAIL  %s: open should succeed, cfg must fail\n", neg_labels[i]);
                fails++;
            } else {
                int ok = k3_gguf_cfg(&g, &c, fa, 128);
                ck(ok == 0, neg_labels[i]);
                k3_gguf_close(&g);
            }
            cleanup_fixture(f, 2);
        }
        free2(&s1, &s2);
    }

    if (fails) {
        printf("\n%s\n", "GGUF SYNTHETIC: FAIL");
        return 1;
    }
    printf("\nGGUF SYNTHETIC: PASS\n");

    /* ------------------------------------------------------------------ */
    printf("== gguf reader (real-file gate) ==\n");

    const char *dir = getenv("K3_GGUF_REAL");
    if (!dir || !dir[0]) dir = "/workspace/unsloth/Kimi-K3-GGUF/UD-IQ1_S";
    if (access(dir, R_OK) != 0) {
        printf("SKIPPED: no real GGUF at %s (set K3_GGUF_REAL to enable)\n", dir);
        return 0;
    }

    K3Gguf g;
    if (k3_gguf_open(&g, dir) != 0) { printf("REAL-FILE GATE: FAIL (open)\n"); return 1; }

    ck(g.nshard == 14, "14 shards");
    ck(g.nt == 2573, "2573 tensors total");
    ck(g.alignment == 32, "alignment 32");
    ck(g.nkv == 64, "shard 1 carries 64 metadata pairs");
    ck(g.shard_kv[0] == 64, "shard 1 kv count");
    int kv3 = 1;
    for (int i = 1; i < 14; i++)
        if (g.shard_kv[i] != 3) kv3 = 0;
    ck(kv3, "tensor shards carry 3 metadata pairs each");

    static const int want_per_shard[14] = { 0, 225, 213, 214, 220, 198, 214,
                                            220, 203, 214, 220, 203, 219, 10 };
    int got_per_shard[14] = { 0 };
    for (int i = 0; i < g.nt; i++) got_per_shard[g.t[i].shard]++;
    int per_ok = 1;
    for (int i = 0; i < 14; i++)
        if (got_per_shard[i] != want_per_shard[i]) per_ok = 0;
    ck(per_ok, "per-shard tensor counts match recon");

    const K3GgufKV *k = k3_gguf_kv(&g, "split.no");
    ck(k && k->type == 2 && k->v.u == 0, "split.no u16 0");
    k = k3_gguf_kv(&g, "split.count");
    ck(k && k->type == 2 && k->v.u == 14, "split.count u16 14");
    k = k3_gguf_kv(&g, "split.tensors.count");
    ck(k && k->type == 5 && k->v.u == 2573, "split.tensors.count i32 2573");
    k = k3_gguf_kv(&g, "general.architecture");
    ck(k && k->type == 8 && k->v.s.n == 7 && !memcmp(k->v.s.p, "kimi-k3", 7),
       "general.architecture kimi-k3");

    /* the head_count_kv quirk on real bytes: type-5 array, 4-byte elements */
    k = k3_gguf_kv(&g, "kimi-k3.attention.head_count_kv");
    if (k && k->type == 9 && k->v.a.et == 5 && k->v.a.n == 93 && k->v.a.nbytes == 372) {
        const uint32_t *hc = (const uint32_t *)k->v.a.d;
        int ones = 0, ok5 = 1;
        for (uint32_t i = 0; i < 93; i++) {
            if (hc[i] > 1) ok5 = 0;
            if (hc[i] == 1) ones++;
        }
        ck(ok5 && ones == 24, "head_count_kv: 93 int32 values, 24 MLA layers");
        ck(hc[0] == 0 && hc[1] == 0 && hc[2] == 0 && hc[3] == 1 && hc[4] == 0,
           "head_count_kv first five values {0,0,0,1,0}");
    } else {
        ck(0, "head_count_kv quirk shape");
    }

    /* verified tensor facts */
    const K3Tensor *t;
    t = k3_gguf_find(&g, "output.weight");
    ck(t && t->shard == 1 && g.gtype[t - g.t] == 8 && t->nbytes == 1247805440 &&
       t->off == 14048 && t->ndim == 2 && t->shape[0] == 7168 && t->shape[1] == 163840,
       "output.weight Q8_0 (7168,163840) at +14048");
    t = k3_gguf_find(&g, "token_embd.weight");
    /* relative offset 1247862784 + data-section start 14048 (recon-gguf-spec
     * §1.1 verified the same arithmetic on shard 2) */
    ck(t && t->nbytes == 1247805440 && t->off == 1247862784 + 14048,
       "token_embd.weight follows output.weight");
    t = k3_gguf_find(&g, "output_norm.weight");
    ck(t && t->dtype == K3_DT_F32 && t->nbytes == 28672, "output_norm.weight F32");
    t = k3_gguf_find(&g, "blk.1.ffn_down_exps.weight");
    ck(t && g.gtype[t - g.t] == 19 && t->nbytes == 1926758400 && t->ndim == 3 &&
       t->shape[0] == 3072 && t->shape[1] == 3584 && t->shape[2] == 896,
       "blk.1.ffn_down_exps.weight IQ1_S (3072,3584,896)");

    /* a bounded sanity peek: one Q8_0 block of output.weight; the fp16 super-scale
     * d = 0x0f06 was verified against the on-disk bytes during recon. A direct
     * pread of 34 bytes, never k3_gguf_read: that would pull the whole 1.2 GB
     * tensor. */
    t = k3_gguf_find(&g, "output.weight");
    if (t) {
        unsigned char peek[34];
        ssize_t got = pread(g.fd[t->shard], peek, sizeof peek, (off_t)t->off);
        ck(got == 34 && peek[0] == 0x06 && peek[1] == 0x0f,
           "output.weight first block bytes match recon");
    }

    /* config derived from shard 1 must equal the released constants */
    {
        K3Cfg c;
        static int fa[128];
        if (!k3_gguf_cfg(&g, &c, fa, 128)) { printf("  FAIL  real cfg load\n"); fails++; }
        else {
            ck(c.hidden == 7168, "real hidden");
            ck(c.n_layers == 93, "real layers");
            ck(c.vocab == 163840, "real vocab");
            ck(c.kda_heads == 96 && c.n_heads == 96, "real heads");
            ck(c.kda_head_dim == 128 && c.conv_k == 4, "real kda dims");
            ck(c.gate_lb == -5.0f, "real gate_lb");
            ck(c.q_lora == 1536 && c.kv_lora == 512, "real lora ranks");
            ck(c.qk_nope == 128 && c.qk_rope == 64, "real qk dims derived");
            ck(c.v_head == 128, "real v_head");
            ck(c.n_experts == 896 && c.topk == 16 && c.n_shared == 2, "real experts");
            ck(c.latent == 3584 && c.moe_inter == 3072, "real latent/moe_inter");
            ck(c.routed_scale == 1.0f && c.moe_renorm == 1, "real routed scale/renorm");
            ck(c.first_dense == 1 && c.dense_inter == 33792, "real dense");
            ck(c.attn_res_block == 12, "real attn_res_block");
            ck(c.situ_b1 == 4.0f && c.situ_b2 == 25.0f, "real situ betas");
            ck(c.n_full_attn == 24, "real 24 MLA layers");
            ck(fa[0] == 4 && fa[22] == 92 && fa[23] == 93, "real fa list");
            int nmla = 0, nkda = 0;
            for (int i = 0; i < c.n_layers; i++)
                k3_is_mla(&c, i) ? nmla++ : nkda++;
            ck(nmla == 24 && nkda == 69, "real k3_is_mla 24/69 split");
            ck(k3_is_mla(&c, 91) && k3_is_mla(&c, 92), "real layers 91,92 MLA");
            ck(!k3_is_mla(&c, 0), "real layer 0 not MLA");
        }
    }

    int bad = 0;
    for (int i = 0; i < g.nt; i++)
        if (k3_gguf_find(&g, g.t[i].name) != &g.t[i]) bad++;
    ck(bad == 0, "real round trip: 2573/2573 resolve to themselves");
    ck(k3_gguf_find(&g, "language_model.model.layers.1.self_attn.A_log") == NULL,
       "safetensors-prefixed name correctly absent");

    k3_gguf_close(&g);

    printf("\n%s\n", fails ? "REAL-FILE GATE: FAIL" : "REAL-FILE GATE: PASS");
    return fails ? 1 : 0;
}
