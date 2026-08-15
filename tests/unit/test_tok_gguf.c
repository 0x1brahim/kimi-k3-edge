/* test_tok_gguf.c - GGUF-metadata tokenizer tests (D6, dev-d).
 *
 * Three halves:
 *   (a) synthetic GGUF fixtures built in memory: a small byte-level vocab written
 *       with the VERIFIED writer quirks (token_type declared as a type-5 array
 *       with 4-byte elements, per-value boundary checks), loaded through both
 *       loadable paths (tiktoken rankbpe vs the gguf merges list), plus an
 *       adversarial set that must ALL fail loud (missing/mistyped tokenizer keys,
 *       wrong model/pre, a type-11 array where the writer's type-5 is required,
 *       malformed merges, unknown control names, wrong reserved-token names,
 *       vocab/merges count violations);
 *   (b) a small byte-level vocab round-trip: the SAME tokens written as a
 *       tiktoken.model + tokenizer_config.json pair AND as a gguf fixture; the
 *       existing loader and the gguf loader must agree token-for-token on a
 *       corpus, and both must round-trip byte-exact. A second fixture proves the
 *       two loadable paths are DISTINCT (rankbpe and merges diverge on it);
 *   (c) a real-file gate: when the UD-IQ1_S shard set and the released HF
 *       tokenizer files exist, load both tokenizers and compare token-for-token
 *       over a corpus (the tok_parity cases plus specials, CJK, emoji, and the
 *       repo README when readable). SKIPPED with exit 0 when the paths are
 *       absent, so CI stays weightless-green.
 *
 * The `encode`/`encode-m`/`roundtrip` modes drive the python oracle
 * (tools/tok_parity_gguf.py) exactly like test_tok drives tok_parity.py.
 *
 * usage:
 *   test_tok_gguf                                  synthetic (+ real gate at default
 *                                                  paths when present)
 *   test_tok_gguf <gguf_path> <hf_tok_dir>         + real gate at given paths
 *   test_tok_gguf <gguf_path> <hf_tok_dir> encode|encode-m <textfile>
 *   test_tok_gguf <gguf_path> <hf_tok_dir> roundtrip <textfile>
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "k3_tok.h"

#define MAXIDS  (1 << 16)
#define MAXTEXT (1 << 22)

/* ------------------------------------------------------------- gguf builder -- */

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

static void bu8(Buf *b, uint8_t v)   { bput(b, &v, 1); }
static void bu16(Buf *b, uint16_t v) { bput(b, &v, 2); }
static void bu32(Buf *b, uint32_t v) { bput(b, &v, 4); }
static void bu64(Buf *b, uint64_t v) { bput(b, &v, 8); }

static void bstr(Buf *b, const char *s)
{
    bu64(b, strlen(s));
    bput(b, s, strlen(s));
}

typedef struct {
    Buf      b;
    uint32_t nkv;
    size_t   info_end;
} Shard;

static void shard_init(Shard *s, uint64_t ntensors)
{
    memset(s, 0, sizeof *s);
    bput(&s->b, "GGUF", 4);
    bu32(&s->b, 3);          /* version 3, the verified writer's */
    bu64(&s->b, ntensors);
    bu64(&s->b, 0);          /* nkv, patched at finish */
}

static void shard_finish(Shard *s)
{
    memcpy(s->b.b + 16, &s->nkv, 8);
    s->info_end = s->b.n;
    while (s->b.n % 32) bu8(&s->b, 0);   /* zero padding, as the real file has */
}

static void skey(Shard *s, const char *k) { bstr(&s->b, k); s->nkv++; }

static void skv_u32(Shard *s, const char *k, uint32_t v)
{ skey(s, k); bu32(&s->b, 4); bu32(&s->b, v); }
static void skv_str(Shard *s, const char *k, const char *v)
{ skey(s, k); bu32(&s->b, 8); bstr(&s->b, v); }

/* The VERIFIED quirk: an array declaring type-5 (int32) elements carries 4 bytes
 * per element (observed on tokenizer.ggml.token_type). */
static void skv_arr_i32(Shard *s, const char *k, const int32_t *v, uint32_t n)
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
    skey(s, "split.no"); bu32(&s->b, 2); bu16(&s->b, no);
    skey(s, "split.tensors.count"); bu32(&s->b, 5); bu32(&s->b, (uint32_t)tensors);
    skey(s, "split.count"); bu32(&s->b, 2); bu16(&s->b, count);
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

/* -------------------------------------------------------------- test state -- */

static int fails = 0;

static void ck(int cond, const char *what)
{
    if (cond) printf("  ok    %s\n", what);
    else { printf("  FAIL  %s\n", what); fails++; }
}

static char g_dir[256];

typedef struct { const char *name; const void *d; size_t n; } GFile;

static int write_fixture(const GFile *files, int nf)
{
    snprintf(g_dir, sizeof g_dir, "/tmp/k3tokgXXXXXX");
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

/* ------------------------------------------------------------------ vocab ---- */
/* The fixture vocab lives twice: as raw bytes (for tiktoken.model) and as the
 * GPT-2 byte-level strings (for the gguf tokens array). A dummy Tok provides the
 * bytemap both conversions need. */

static Tok VM;                 /* bytemap carrier; k3_bytelevel uses T->byte2str */

static char g_byte[256][3];    /* byte-level string of byte b */

static void vm_init(void)
{
    tk_build_bytemap(&VM);
    for (int b = 0; b < 256; b++) {
        memcpy(g_byte[b], VM.byte2str[b], (size_t)VM.byte2cp_len[b] + 1);
    }
}

/* byte-level string for a raw byte sequence; returns a pointer to static storage
 * sized for the longest token used by the fixtures. */
static char g_bl[64];

static const char *bl_of(const unsigned char *raw, int n)
{
    int l = k3_bytelevel(&VM, raw, n, g_bl);
    (void)l;
    return g_bl;
}

/* The 13 merged tokens (ids 256..268): raw bytes for tiktoken.model, merge
 * entries in rank order. */
static const char *const G_MERGE_PAIRS[13] = {
    "l l", "ll o", "h e", "he llo", "\xc4\xa0 he",
    "w o", "r l", "rl d", "wo rld", "\xc4\xa0 world",
    "e n", "en d", "o f",
};
static const unsigned char *const G_MERGED_RAW[13] = {
    (const unsigned char *)"ll", (const unsigned char *)"llo",
    (const unsigned char *)"he", (const unsigned char *)"hello",
    (const unsigned char *)" he", (const unsigned char *)"wo",
    (const unsigned char *)"rl", (const unsigned char *)"rld",
    (const unsigned char *)"world", (const unsigned char *)" world",
    (const unsigned char *)"en", (const unsigned char *)"end",
    (const unsigned char *)"of",
};

/* The 16 added-token names (ids 269..284), exactly tokenizer_config.json's
 * added_tokens_decoder. */
static const char *const G_ADDED[16] = {
    "[BOS]", "[EOS]", "<|end_of_msg|>", "<|open|>", "<|close|>", "<|sep|>",
    "[start_header_id]", "[end_header_id]", "[EOT]",
    "<|media_begin|>", "<|media_content|>", "<|media_end|>", "<|media_pad|>",
    "<osagent_mode>", "[UNK]", "[PAD]",
};
/* special:true flags for the 16, matching the released config. */
static const int G_ADDED_SPECIAL[16] = {
    1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};

#define G_NBASE    256
#define G_NMERGED  13
#define G_NADDED   16
#define G_NRES     4                       /* reserved_token_285..288 */
#define G_NTOK     (G_NBASE + G_NMERGED + G_NADDED + G_NRES)   /* 289 */

static int g_ncontrol(void) { return G_NADDED + G_NRES; }

/* Fill tokens[]/ttype[]/merges[] for the standard fixture. tokens[i] is the
 * byte-level string, id == index. */
static void std_vocab(const char **tokens, int32_t *ttype, const char **merges)
{
    for (int b = 0; b < G_NBASE; b++) tokens[b] = g_byte[b];
    /* bl_of shares one static buffer: copy each result out IMMEDIATELY, or every
     * slot below aliases the last conversion. */
    static char copied[G_NMERGED][64];
    for (int i = 0; i < G_NMERGED; i++) {
        const char *bl = bl_of(G_MERGED_RAW[i],
                               (int)strlen((const char *)G_MERGED_RAW[i]));
        snprintf(copied[i], sizeof copied[i], "%s", bl);
        tokens[G_NBASE + i] = copied[i];
    }
    for (int i = 0; i < G_NADDED; i++) tokens[G_NBASE + G_NMERGED + i] = G_ADDED[i];
    static char res[G_NRES][48];   /* static: tokens[] keeps these pointers */
    for (int i = 0; i < G_NRES; i++) {
        snprintf(res[i], sizeof res[i], "<|reserved_token_%d|>",
                 G_NBASE + G_NMERGED + G_NADDED + i);
        tokens[G_NBASE + G_NMERGED + G_NADDED + i] = res[i];
    }
    for (int i = 0; i < G_NTOK; i++) ttype[i] = i < G_NBASE + G_NMERGED ? 1 : 3;
    for (int i = 0; i < G_NMERGED; i++) merges[i] = G_MERGE_PAIRS[i];
}

/* Build the two-file gguf fixture. variant mutates exactly one thing for the
 * negative tests (see enum below). */
enum { FIX_OK = 0, FIX_NO_TOKENS, FIX_NO_MODEL, FIX_BAD_MODEL, FIX_BAD_PRE,
       FIX_TTYPE_SHORT, FIX_TTYPE_I64, FIX_VOCAB_MISMATCH, FIX_NO_BOS,
       FIX_BAD_MERGE, FIX_UNKNOWN_CONTROL, FIX_BAD_RESERVED, FIX_BAD_BOS_NAME,
       FIX_MERGE_COUNT, FIX_MISALIGN, FIX_TOKEN_NUL, FIX_ADDED_COUNT };

static const char *const FIX_NAMES[] = {
    "ok", "missing tokens key", "missing model key", "wrong model",
    "wrong pre", "short token_type", "token_type as int64 array",
    "vocab_size mismatch", "missing bos id", "malformed merge",
    "unknown control token", "wrong reserved name", "bos name mismatch",
    "merge count violation", "unknown value type (open)", "NUL in token string",
    "not all 16 added names are CONTROL",
};

static void fix_gguf(Shard *s1, Shard *s2, int variant)
{
    static const char *tokens[G_NTOK];
    static int32_t ttype[G_NTOK];
    static const char *merges[G_NMERGED];
    std_vocab(tokens, ttype, merges);

    shard_init(s1, 0);
    skv_str(s1, "general.architecture", "kimi-k3");
    if (variant != FIX_NO_MODEL)
        skv_str(s1, "tokenizer.ggml.model",
                variant == FIX_BAD_MODEL ? "gpt-4" : "gpt2");
    skv_str(s1, "tokenizer.ggml.pre",
            variant == FIX_BAD_PRE ? "llama" : "kimi-k2");
    if (variant != FIX_NO_TOKENS)
        skv_arr_str(s1, "tokenizer.ggml.tokens", tokens, G_NTOK);
    if (variant == FIX_TTYPE_SHORT)
        skv_arr_i32(s1, "tokenizer.ggml.token_type", ttype, G_NTOK - 1);
    else if (variant == FIX_TTYPE_I64) {
        /* The legacy-enum trap: an array DECLARING type 11 (int64) where the
         * verified writer uses type 5 with 4-byte elements must fail loud. */
        skey(s1, "tokenizer.ggml.token_type");
        bu32(&s1->b, 9);
        bu32(&s1->b, 11);
        bu64(&s1->b, G_NTOK);
        for (int i = 0; i < G_NTOK; i++) bu64(&s1->b, (uint64_t)(int64_t)ttype[i]);
    } else
        skv_arr_i32(s1, "tokenizer.ggml.token_type", ttype, G_NTOK);
    if (variant == FIX_BAD_MERGE) {
        static const char *bad[G_NMERGED];
        for (int i = 0; i < G_NMERGED; i++) bad[i] = merges[i];
        bad[3] = "he  llo";                 /* two spaces */
        skv_arr_str(s1, "tokenizer.ggml.merges", bad, G_NMERGED);
    } else
        skv_arr_str(s1, "tokenizer.ggml.merges", merges, G_NMERGED);
    if (variant != FIX_NO_BOS) {
        skv_u32(s1, "tokenizer.ggml.bos_token_id", G_NBASE + G_NMERGED);       /* 269 */
        skv_u32(s1, "tokenizer.ggml.eos_token_id", G_NBASE + G_NMERGED + 2);   /* 271 */
        skv_u32(s1, "tokenizer.ggml.padding_token_id",
                G_NBASE + G_NMERGED + G_NADDED - 1);                           /* 284 */
    }
    skv_u32(s1, "kimi-k3.vocab_size", variant == FIX_VOCAB_MISMATCH
                                        ? G_NTOK + 1 : G_NTOK);

    if (variant == FIX_UNKNOWN_CONTROL) {
        /* patch id 285's name to a 22-char name that is neither reserved-shaped
         * (digit count differs from the id) nor in the 16-name table, so ONLY
         * the loader's control-name rule can reject it. Same length both sides:
         * a shorter literal would copy a NUL in and trip the NUL guard instead
         * (crit-d-1 #1). */
        for (size_t j = 0; j + 22 <= s1->b.n; j++)
            if (!memcmp(s1->b.b + j, "<|reserved_token_285|>", 22)) {
                memcpy(s1->b.b + j, "<|reserved_token_28a|>", 22);
                break;
            }
    } else if (variant == FIX_BAD_RESERVED) {
        /* reserved name whose embedded id is NOT its own id (22 chars) */
        for (size_t j = 0; j + 22 <= s1->b.n; j++)
            if (!memcmp(s1->b.b + j, "<|reserved_token_286|>", 22)) {
                memcpy(s1->b.b + j, "<|reserved_token_999|>", 22);
                break;
            }
    } else if (variant == FIX_BAD_BOS_NAME) {
        for (size_t j = 0; j + 5 <= s1->b.n; j++)
            if (!memcmp(s1->b.b + j, "[BOS]", 5)) {
                memcpy(s1->b.b + j, "[BOZ]", 5);
                break;
            }
    } else if (variant == FIX_MERGE_COUNT) {
        /* A separate, consistently-encoded fixture (rebuild s1 from scratch):
         * 256 base + "ab"(256) + "bc"(257) + the full 16-name control set
         * (258..273), but only ONE merge for the shape's expected two
         * (274-256-16). The open walk succeeds and the 16-name assertion
         * passes; only the loader's gpt2 count invariant
         * (nmerges == ntok-256-n_control) can reject (crit-d-1 #2). */
        free(s1->b.b);
        shard_init(s1, 0);
        skv_str(s1, "tokenizer.ggml.model", "gpt2");
        skv_str(s1, "tokenizer.ggml.pre", "kimi-k2");
        static const char *mct[274];
        static int32_t mctt[274];
        for (int b = 0; b < 256; b++) mct[b] = g_byte[b];
        mct[256] = "ab";
        mct[257] = "bc";
        for (int i = 0; i < G_NADDED; i++) mct[258 + i] = G_ADDED[i];
        for (int i = 0; i < 258; i++) mctt[i] = 1;
        for (int i = 258; i < 274; i++) mctt[i] = 3;
        static const char *mcm[1] = { "a b" };
        skv_arr_str(s1, "tokenizer.ggml.tokens", mct, 274);
        skv_arr_i32(s1, "tokenizer.ggml.token_type", mctt, 274);
        skv_arr_str(s1, "tokenizer.ggml.merges", mcm, 1);
        skv_u32(s1, "tokenizer.ggml.bos_token_id", 258);
        skv_u32(s1, "tokenizer.ggml.eos_token_id", 260);
        skv_u32(s1, "tokenizer.ggml.padding_token_id", 273);
        skv_u32(s1, "kimi-k3.vocab_size", 274);
        ssplit(s1, 0, 2, 1);
        shard_finish(s1);
    } else if (variant == FIX_ADDED_COUNT) {
        /* Reuse the divergence fixture's layout with [PAD] (id 274) marked
         * NORMAL instead of CONTROL: the walk succeeds, the name rules pass,
         * and only the loader's "exactly 16 added names are CONTROL"
         * assertion can reject. */
        free(s1->b.b);
        shard_init(s1, 0);
        skv_str(s1, "tokenizer.ggml.model", "gpt2");
        skv_str(s1, "tokenizer.ggml.pre", "kimi-k2");
        static const char *act[275];
        static int32_t actt[275];
        for (int b = 0; b < 256; b++) act[b] = g_byte[b];
        act[256] = "ab";
        act[257] = "bc";
        act[258] = "abc";
        for (int i = 0; i < G_NADDED; i++) act[259 + i] = G_ADDED[i];
        for (int i = 0; i < 259; i++) actt[i] = 1;
        for (int i = 259; i < 274; i++) actt[i] = 3;
        actt[274] = 1;   /* [PAD] demoted: 15 of 16 names CONTROL */
        static const char *acm[4] = { "a b", "b c", "a bc", "x y" };
        skv_arr_str(s1, "tokenizer.ggml.tokens", act, 275);
        skv_arr_i32(s1, "tokenizer.ggml.token_type", actt, 275);
        skv_arr_str(s1, "tokenizer.ggml.merges", acm, 4);
        skv_u32(s1, "tokenizer.ggml.bos_token_id", 259);
        skv_u32(s1, "tokenizer.ggml.eos_token_id", 261);
        skv_u32(s1, "tokenizer.ggml.padding_token_id", 274);
        skv_u32(s1, "kimi-k3.vocab_size", 275);
        ssplit(s1, 0, 2, 1);
        shard_finish(s1);
    } else if (variant == FIX_TOKEN_NUL) {
        /* tokens[0] = byte-level of byte 0 = "\xc4\x80"; corrupt its second
         * byte to NUL. The value walk of the OPEN pass skips strings by length
         * (no content check), so only the tokenizer extraction must fail. */
        for (size_t j = 0; j + 2 <= s1->b.n; j++)
            if (s1->b.b[j] == 0xC4 && s1->b.b[j + 1] == 0x80 &&
                !memcmp(s1->b.b + j, "\xc4\x80", 2)) {
                s1->b.b[j + 1] = 0x00;
                break;
            }
    } else if (variant == FIX_MISALIGN) {
        /* The value-type byte is stomped to 0xFF, so this exercises the OPEN's
         * unknown-type rejection, not the tokenizer walk's boundary check: the
         * open gates the same bytes first, so the second walk's boundary_ok is
         * defense-in-depth and cannot be reached independently (crit-d-1 #5). */
        size_t nlen = strlen("tokenizer.ggml.tokens");
        for (size_t j = 0; j + 8 + nlen + 4 <= s1->b.n; j++)
            if (!memcmp(s1->b.b + j + 8, "tokenizer.ggml.tokens", nlen)) {
                s1->b.b[j + 8 + nlen] = 0xFF;      /* value type 255 */
                break;
            }
    }
    ssplit(s1, 0, 2, 1);
    shard_finish(s1);

    shard_init(s2, 1);
    ssplit(s2, 1, 2, 1);
    stensor(s2, "x.weight", 1, (const uint64_t[]){ 8 }, 0, 0);
    shard_finish(s2);
    unsigned char data[32];
    memset(data, 7, sizeof data);
    bput(&s2->b, data, sizeof data);
}

/* base64 for tiktoken.model lines (k3_tok.h's k3_b64 decodes this alphabet). */
static const char *k3_b64enc(const unsigned char *in, int n)
{
    static char out[512];
    static const char *A =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int o = 0;
    unsigned int acc = 0;
    int bits = 0;
    for (int i = 0; i < n; i++) {
        acc = (acc << 8) | in[i];
        bits += 8;
        while (bits >= 6) { bits -= 6; out[o++] = A[(acc >> bits) & 63]; }
    }
    if (bits) out[o++] = A[(acc << (6 - bits)) & 63];
    while (o % 4) out[o++] = '=';
    out[o] = 0;
    return out;
}

/* The 13 ranks of the merged tokens as raw bytes for tiktoken.model. */
static void hf_files(const char *tokdir)
{
    char path[512];
    snprintf(path, sizeof path, "%s/tiktoken.model", tokdir);
    FILE *f = fopen(path, "wb");
    if (!f) { printf("  FAIL  cannot write tiktoken.model\n"); fails++; return; }
    for (int b = 0; b < G_NBASE; b++) {
        unsigned char raw[1] = { (unsigned char)b };
        fprintf(f, "%s %d\n", k3_b64enc(raw, 1), b);
    }
    for (int i = 0; i < G_NMERGED; i++) {
        const unsigned char *raw = G_MERGED_RAW[i];
        int n = (int)strlen((const char *)raw);
        fprintf(f, "%s %d\n", k3_b64enc(raw, n), G_NBASE + i);
    }
    fclose(f);
    snprintf(path, sizeof path, "%s/tokenizer_config.json", tokdir);
    f = fopen(path, "wb");
    if (!f) { printf("  FAIL  cannot write tokenizer_config.json\n"); fails++; return; }
    fprintf(f, "{\"added_tokens_decoder\": {");
    for (int i = 0; i < G_NADDED; i++) {
        fprintf(f, "%s\"%d\": {\"content\": \"%s\", \"special\": %s}",
                i ? ", " : "", G_NBASE + G_NMERGED + i, G_ADDED[i],
                G_ADDED_SPECIAL[i] ? "true" : "false");
    }
    fprintf(f, "}}\n");
    fclose(f);
}

/* ------------------------------------------------------------ enc helpers -- */

static void enc_line(const Tok *T, const char *text, int *ids, int *n)
{
    *n = tok_encode((Tok *)T, text, (int)strlen(text), ids, MAXIDS);
}

static int same_ids(const int *a, int na, const int *b, int nb)
{
    if (na != nb) return 0;
    for (int i = 0; i < na; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

static void show_ids(const int *a, int n)
{
    for (int i = 0; i < n && i < 32; i++) printf(" %d", a[i]);
    if (n > 32) printf(" ...");
    printf("\n");
}

/* ------------------------------------------------------------ real corpus -- */

/* The same separation cases tok_parity.py uses, plus the special-token shapes
 * the K3 control range exercises, plus longer multi-line text. Non-ASCII is
 * written as explicit UTF-8 escapes so the source stays ASCII. */
static const char *const CORPUS[] = {
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    " leading",
    "trailing ",
    "two  spaces",
    "a\tb",
    "line1\nline2\n\nline4",
    "dos\r\nline",
    "a" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "b",
    "1234567890 42 007",
    "abc123def456",
    "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    "don't can't it's I'm we'll they've he'd",
    "HTTP HTML JSON XML API",
    "camelCaseIdentifierName",
    "snake_case_identifier_name",
    "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c",                    /* han only */
    "\xe4\xbd\xa0\xe5\xa5\xbd world",                       /* han + ascii */
    "hello \xe4\xbd\xa0\xe5\xa5\xbd",                       /* ascii + han */
    "\xe6\x88\x91\xe4\xbb\xac\xe5\x9c\xa8\xe6\xb5\x8b\xe8\xaf\x95"
    "\xe5\x88\x86\xe8\xaf\x8d\xe5\x99\xa8\xe3\x80\x82",
    "\xe7\xac\xac" "123" "\xe9\xa1\xb5",                    /* han + digits */
    "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf"
    "\xe4\xb8\x96\xe7\x95\x8c",
    "\xec\x95\x88\xeb\x85\x95\xed\x95\x98\xec\x84\xb8\xec\x9a\x94",
    "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xd0\xbc\xd0\xb8\xd1\x80",
    "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd8\xa7"
    "\xd9\x84\xd8\xb9\xd8\xa7\xd9\x84\xd9\x85",
    "\xce\x93\xce\xb5\xce\xb9\xce\xb1 \xcf\x83\xce\xbf\xcf\x85",
    "caf\xc3\xa9 na\xc3\xafve r\xc3\xa9sum\xc3\xa9 \xc3\xbc" "ber",
    "\xf0\x9f\x98\x80\xf0\x9f\x9a\x80\xf0\x9f\xa7\xa0",                     /* emoji */
    "ship it \xf0\x9f\x9a\x80 now",
    "\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x92\xbb",       /* emoji zwj */
    "int main(void){ return 0; }",
    "def f(x):\n    return x ** 2\n",
    "{\"key\": [1, 2, {\"n\": null}], \"b\": true}",
    "https://example.com/a/b?c=d&e=f#g",
    "C:\\Users\\dev\\project\\file.c",
    "# Title\n\n- item **bold** `code`\n",
    "aaaaaaaaaaaaaaaaaaaa",
    "supercalifragilisticexpialidocious",
    "hello \xe4\xbd\xa0\xe5\xa5\xbd \xd0\x9c\xd0\xb8\xd1\x80 \xf0\x9f\x98\x80 123",
    " ",
    "x",
    "\xc3\xbf\xc3\xbe\xc3\xbd",
    "\xe2\x88\x91 \xe2\x88\x9e \xe2\x89\xa0 \xe2\x89\xa4 \xcf\x80 \xc3\x97 \xc3\xb7",
    "\xe2\x80\x9cquoted\xe2\x80\x9d and \xe2\x80\x98single\xe2\x80\x99",
    "em\xe2\x80\x94" "dash en\xe2\x80\x93" "dash hyphen-minus",
    /* special-token shapes: the engine's added range, both atomic and reserved */
    "[BOS]",
    "[EOS]",
    "hi <|end_of_msg|> bye",
    "<|open|> and <|close|> and <|sep|>",
    "[start_header_id]user[end_header_id]",
    "[EOT]",
    "<|media_begin|>caption<|media_end|>",
    "<osagent_mode>on",
    "[UNK] unknown [PAD]",
    "reserved <|reserved_token_163597|> inline",
    "<|reserved_token_163838|>",
    "[BOS]hello[EOS]",
    /* longer, structured text */
    "The K3 report (4.1.4) keeps exactly these tensors in higher precision on "
    "purpose: norms, biases, the router, SSM parameters, and the res-scores.\n"
    "Measured consequence: at a fixed 128 GB budget, trunk-first runs 1.69x "
    "faster than cache-first.\n",
    "SELECT id, name FROM users WHERE id = 42 AND name LIKE '%k3%';\n"
    "INSERT INTO logs (ts, msg) VALUES (now(), 'hello \xe4\xb8\x96\xe7\x95\x8c');\n",
    NULL,
};

static int corpus_count(void)
{
    int n = 0;
    while (CORPUS[n]) n++;
    return n;
}

/* ------------------------------------------------------------ real gate ---- */

static void run_real_gate(const char *gguf_path, const char *hf_dir)
{
    printf("== gguf tokenizer (real-file gate) ==\n");
    Tok hf, gr, gm;
    int have = 0;
    char probe[512];
    snprintf(probe, sizeof probe, "%s/tiktoken.model", hf_dir);
    if (access(gguf_path, R_OK) == 0 && access(probe, R_OK) == 0) {
        k3_tok_load(&hf, hf_dir);
        k3_tok_load_gguf_mode(&gr, gguf_path, 0);      /* rankbpe */
        k3_tok_load_gguf_mode(&gm, gguf_path, 1);      /* merges */
        have = 1;
    } else {
        printf("SKIPPED: no real GGUF at %s or no tiktoken.model at %s\n",
               gguf_path, hf_dir);
        return;
    }

    static int ids_hf[MAXIDS], ids_gr[MAXIDS], ids_gm[MAXIDS];
    int ncase = corpus_count();
    long tok_total = 0, tok_match_r = 0, tok_match_m = 0;
    int case_match_r = 0, case_match_m = 0, case_diff_r = 0, case_diff_m = 0;

    for (int i = 0; i < ncase; i++) {
        const char *text = CORPUS[i];
        int nhf, ngr, ngm;
        enc_line(&hf, text, ids_hf, &nhf);
        enc_line(&gr, text, ids_gr, &ngr);
        enc_line(&gm, text, ids_gm, &ngm);
        tok_total += nhf;
        if (same_ids(ids_hf, nhf, ids_gr, ngr)) case_match_r++;
        else { case_diff_r++; }
        if (same_ids(ids_hf, nhf, ids_gm, ngm)) case_match_m++;
        else { case_diff_m++; }
        /* per-token exact-match on the shared prefix */
        int lim = nhf < ngr ? nhf : ngr;
        for (int j = 0; j < lim; j++)
            if (ids_hf[j] == ids_gr[j]) tok_match_r++;
        lim = nhf < ngm ? nhf : ngm;
        for (int j = 0; j < lim; j++)
            if (ids_hf[j] == ids_gm[j]) tok_match_m++;
        if (!same_ids(ids_hf, nhf, ids_gr, ngr)) {
            printf("  DIFF rankbpe vs hf  case %-28s hf=%d gguf=%d\n",
                   text, nhf, ngr);
            printf("    hf  :"); show_ids(ids_hf, nhf);
            printf("    gguf:"); show_ids(ids_gr, ngr);
        }
        if (!same_ids(ids_hf, nhf, ids_gm, ngm)) {
            printf("  DIFF merges   vs hf  case %-28s hf=%d gguf=%d\n",
                   text, nhf, ngm);
            printf("    hf  :"); show_ids(ids_hf, nhf);
            printf("    gguf:"); show_ids(ids_gm, ngm);
        }
    }

    /* an extra leg over the repo README, when readable */
    char rd[512];
    snprintf(rd, sizeof rd, "%s/README.md", ".");
    FILE *f = fopen(rd, "rb");
    if (f) {
        static char big[MAXTEXT];
        size_t n = fread(big, 1, sizeof big - 1, f);
        fclose(f);
        big[n] = 0;
        int nhf, ngr, ngm;
        enc_line(&hf, big, ids_hf, &nhf);
        enc_line(&gr, big, ids_gr, &ngr);
        enc_line(&gm, big, ids_gm, &ngm);
        tok_total += nhf;
        if (same_ids(ids_hf, nhf, ids_gr, ngr)) case_match_r++;
        else case_diff_r++;
        if (same_ids(ids_hf, nhf, ids_gm, ngm)) case_match_m++;
        else case_diff_m++;
        int lim = nhf < ngr ? nhf : ngr;
        for (int j = 0; j < lim; j++)
            if (ids_hf[j] == ids_gr[j]) tok_match_r++;
        lim = nhf < ngm ? nhf : ngm;
        for (int j = 0; j < lim; j++)
            if (ids_hf[j] == ids_gm[j]) tok_match_m++;
        printf("  README.md: %zu bytes -> %d ids (hf)\n", n, nhf);
    }

    printf("\n  rankbpe path: %d/%d cases, %ld/%ld tokens exact\n",
           case_match_r, ncase + (f ? 1 : 0), tok_match_r, tok_total);
    printf("  merges  path: %d/%d cases, %ld/%ld tokens exact\n",
           case_match_m, ncase + (f ? 1 : 0), tok_match_m, tok_total);
    if (case_diff_r) {
        printf("  FAIL  rankbpe path diverges from the HF loader\n");
        fails++;
    }
    if (case_diff_m) {
        printf("  FAIL  merges path diverges from the HF loader\n");
        fails++;
    }

    /* byte-exact roundtrip through the gguf default path */
    for (int i = 0; i < ncase; i++) {
        const char *text = CORPUS[i];
        int n;
        enc_line(&gr, text, ids_gr, &n);
        static char back[MAXTEXT];
        int m = tok_decode(&gr, ids_gr, n, back, MAXTEXT - 1);
        if (m != (int)strlen(text) || memcmp(back, text, (size_t)m) != 0) {
            printf("  FAIL  roundtrip on case %d (%s)\n", i, text);
            fails++;
        }
    }
    printf("  roundtrip: all %d corpus cases byte-exact\n", ncase);
    if (have) printf("\nREAL-FILE GATE: %s\n", fails ? "FAIL" : "PASS");
}

/* --------------------------------------------------------------- synthetic -- */

static void run_synthetic(void)
{
    printf("== gguf tokenizer (synthetic fixtures) ==\n");

    /* ---- (a) happy fixture: both loadable paths agree with the HF loader ---- */
    {
        Shard s1, s2;
        fix_gguf(&s1, &s2, FIX_OK);
        GFile f[2] = { { "s1.gguf", s1.b.b, s1.b.n },
                       { "s2.gguf", s2.b.b, s2.b.n } };
        if (write_fixture(f, 2) != 0) {
            printf("  FAIL  cannot write happy fixture\n");
            fails++;
        } else {
            /* write the HF pair next to the gguf files */
            hf_files(g_dir);

            Tok thf, tg0, tg1;
            k3_tok_load(&thf, g_dir);
            k3_tok_load_gguf_mode(&tg0, g_dir, 0);
            k3_tok_load_gguf_mode(&tg1, g_dir, 1);
            ck(thf.nsp == 16 && tg0.nsp == 16 && tg1.nsp == 16,
               "16 added tokens in all loaders");
            ck(tg0.n_ids == G_NTOK && tg0.rankbpe == 1, "gguf n_ids/rankbpe");
            ck(tg1.rankbpe == 0, "merges path clears rankbpe");
            ck(thf.kimi == 1 && tg0.kimi == 1, "kimi flag from metadata");
            static const char *const cases[] = {
                "hello", "hello world", "he", "ll", "llo", "world", "end of", "of",
                " hello", "\xc4\xa0hello", "hellohello", "world world", "a b c",
                "hello <|open|> world", "[BOS]hi[EOS]", "hi <|end_of_msg|>",
                "x<|reserved_token_285|>y", "[start_header_id]u[end_header_id]",
                "media <|media_begin|>pic<|media_end|>", "llllll", "wo rld",
                "\xc4\xa0\xc4\xa0he", NULL,
            };
            static int a[MAXIDS], b[MAXIDS], c[MAXIDS];
            int ncase = 0;
            for (int i = 0; cases[i]; i++) {
                int na, nb, nc;
                enc_line(&thf, cases[i], a, &na);
                enc_line(&tg0, cases[i], b, &nb);
                enc_line(&tg1, cases[i], c, &nc);
                if (!same_ids(a, na, b, nb)) {
                    printf("  FAIL  rankbpe vs hf on '%s'\n", cases[i]);
                    printf("    hf  :"); show_ids(a, na);
                    printf("    gguf:"); show_ids(b, nb);
                    fails++;
                }
                if (!same_ids(a, na, c, nc)) {
                    printf("  FAIL  merges vs hf on '%s'\n", cases[i]);
                    printf("    hf  :"); show_ids(a, na);
                    printf("    gguf:"); show_ids(c, nc);
                    fails++;
                }
                /* byte-exact roundtrip through the default (rankbpe) path */
                static char back[MAXTEXT];
                int m = tok_decode(&tg0, b, nb, back, MAXTEXT - 1);
                if (m != (int)strlen(cases[i]) ||
                    memcmp(back, cases[i], (size_t)m) != 0) {
                    printf("  FAIL  roundtrip on '%s'\n", cases[i]);
                    fails++;
                }
                ncase++;
            }
            printf("  ok    %d fixture cases: rankbpe == merges == hf, "
                   "roundtrips exact\n", ncase);
        }
        cleanup_fixture(f, 2);
        free(s1.b.b);
        free(s2.b.b);
    }

    /* ---- (b) divergence fixture: rankbpe and merges are genuinely distinct ---- */
    {
        /* base 256 bytes + "ab"(256) "bc"(257) "abc"(258); the canonical split of
         * "abc" recorded as "a bc" (not "ab c"). Text "abcx" (not a token itself):
         * rankbpe merges (a,b) first (id 256 < 257) then (ab,c) -> "abc" (258); the
         * merges path cannot apply "ab c" (only "a bc" is listed) and stops at
         * [256, c]. Plus the full 16-name control set (259..274) the loader
         * requires. */
        Shard s1, s2;
        shard_init(&s1, 0);
        skv_str(&s1, "tokenizer.ggml.model", "gpt2");
        skv_str(&s1, "tokenizer.ggml.pre", "kimi-k2");
        static const char *tokens[275];
        static int32_t ttype[275];
        for (int b = 0; b < 256; b++) tokens[b] = g_byte[b];
        tokens[256] = "ab";
        tokens[257] = "bc";
        tokens[258] = "abc";
        for (int i = 0; i < G_NADDED; i++) tokens[259 + i] = G_ADDED[i];
        for (int i = 0; i < 259; i++) ttype[i] = 1;
        for (int i = 259; i < 275; i++) ttype[i] = 3;
        static const char *merges[3] = { "a b", "b c", "a bc" };
        skv_arr_str(&s1, "tokenizer.ggml.tokens", tokens, 275);
        skv_arr_i32(&s1, "tokenizer.ggml.token_type", ttype, 275);
        skv_arr_str(&s1, "tokenizer.ggml.merges", merges, 3);
        skv_u32(&s1, "tokenizer.ggml.bos_token_id", 259);
        skv_u32(&s1, "tokenizer.ggml.eos_token_id", 261);
        skv_u32(&s1, "tokenizer.ggml.padding_token_id", 274);
        skv_u32(&s1, "kimi-k3.vocab_size", 275);
        ssplit(&s1, 0, 2, 1);
        shard_finish(&s1);
        shard_init(&s2, 1);
        ssplit(&s2, 1, 2, 1);
        stensor(&s2, "x.weight", 1, (const uint64_t[]){ 8 }, 0, 0);
        shard_finish(&s2);
        unsigned char data[32];
        memset(data, 7, sizeof data);
        bput(&s2.b, data, sizeof data);
        GFile f[2] = { { "s1.gguf", s1.b.b, s1.b.n },
                       { "s2.gguf", s2.b.b, s2.b.n } };
        if (write_fixture(f, 2) != 0) {
            printf("  FAIL  cannot write divergence fixture\n");
            fails++;
        } else {
            Tok ta, tb;
            k3_tok_load_gguf_mode(&ta, g_dir, 0);
            k3_tok_load_gguf_mode(&tb, g_dir, 1);
            static int a[MAXIDS], b[MAXIDS];
            int na, nb;
            enc_line(&ta, "abcx", a, &na);
            enc_line(&tb, "abcx", b, &nb);
            /* 'c' and 'x' are raw bytes 0x63/0x78, so their ids are 99/120 in a
             * byte-ordered fixture vocab. */
            ck(na == 2 && a[0] == 258 && a[1] == 120, "rankbpe: abcx -> [258, 120]");
            ck(nb == 3 && b[0] == 256 && b[1] == 99 && b[2] == 120,
               "merges: abcx -> [256, 99, 120]");
            printf("  ok    paths diverge as designed: [258, 120] vs [256, 99, 120]\n");
        }
        cleanup_fixture(f, 2);
        free(s1.b.b);
        free(s2.b.b);
    }

    /* ---- (d) single-file open: a standalone .gguf has no split keys, and
     * k3_gguf_probe must route it. ---- */
    {
        Shard s1, s2;
        fix_gguf(&s1, &s2, FIX_OK);
        /* strip s1's split keys: rebuild without them */
        free(s1.b.b);
        shard_init(&s1, 0);
        skv_str(&s1, "general.architecture", "kimi-k3");
        skv_str(&s1, "tokenizer.ggml.model", "gpt2");
        skv_str(&s1, "tokenizer.ggml.pre", "kimi-k2");
        static const char *tokens[G_NTOK];
        static int32_t ttype[G_NTOK];
        static const char *merges[G_NMERGED];
        std_vocab(tokens, ttype, merges);
        skv_arr_str(&s1, "tokenizer.ggml.tokens", tokens, G_NTOK);
        skv_arr_i32(&s1, "tokenizer.ggml.token_type", ttype, G_NTOK);
        skv_arr_str(&s1, "tokenizer.ggml.merges", merges, G_NMERGED);
        skv_u32(&s1, "tokenizer.ggml.bos_token_id", G_NBASE + G_NMERGED);
        skv_u32(&s1, "tokenizer.ggml.eos_token_id", G_NBASE + G_NMERGED + 2);
        skv_u32(&s1, "tokenizer.ggml.padding_token_id",
                G_NBASE + G_NMERGED + G_NADDED - 1);
        skv_u32(&s1, "kimi-k3.vocab_size", G_NTOK);
        shard_finish(&s1);
        GFile f[1] = { { "single.gguf", s1.b.b, s1.b.n } };
        if (write_fixture(f, 1) != 0) {
            printf("  FAIL  cannot write single-file fixture\n");
            fails++;
        } else {
            ck(k3_gguf_probe(g_dir) == 1, "probe: dir with .gguf -> 1");
            char fp[512];
            snprintf(fp, sizeof fp, "%s/single.gguf", g_dir);
            ck(k3_gguf_probe(fp) == 1, "probe: .gguf file -> 1");
            Tok T;
            k3_tok_load_gguf(&T, fp);   /* no split keys: tolerated for one file */
            ck(T.n_ids == G_NTOK && T.nsp == 16, "single-file load");
            static int ids[MAXIDS];
            int n = tok_encode(&T, "hello world", 11, ids, MAXIDS);
            ck(n == 2 && ids[0] == 259 && ids[1] == 265, "single-file encode");
        }
        cleanup_fixture(f, 1);
        free(s1.b.b);
        free(s2.b.b);
    }

    /* ---- (c) adversarial: every case must fail loud ---- */
    printf("== gguf tokenizer (adversarial fixtures) ==\n");
    for (int v = FIX_NO_TOKENS; v <= FIX_ADDED_COUNT; v++) {
        Shard s1, s2;
        fix_gguf(&s1, &s2, v);
        GFile f[2] = { { "s1.gguf", s1.b.b, s1.b.n },
                       { "s2.gguf", s2.b.b, s2.b.n } };
        if (write_fixture(f, 2) != 0) {
            printf("  FAIL  %-34s cannot write fixture\n", FIX_NAMES[v]);
            fails++;
        } else {
            /* fork so the loader's exit(1) cannot kill the harness. Flush first:
             * the child inherits this stdout buffer and exit() would flush it,
             * duplicating every line the parent has printed so far. */
            fflush(stdout);
            int pid = fork();
            if (pid == 0) {
                Tok T;
                k3_tok_load_gguf(&T, g_dir);
                _exit(0);
            }
            int st = 0;
            waitpid(pid, &st, 0);
            int rejected = WIFEXITED(st) && WEXITSTATUS(st) != 0;
            ck(rejected, FIX_NAMES[v]);
        }
        cleanup_fixture(f, 2);
        free(s1.b.b);
        free(s2.b.b);
    }

    printf("\n%s\n", fails ? "GGUF TOKENIZER SYNTHETIC: FAIL"
                            : "GGUF TOKENIZER SYNTHETIC: PASS");
}

/* ------------------------------------------------------------------- modes -- */

int main(int argc, char **argv)
{
    vm_init();

    /* Standalone encode/roundtrip modes for the python oracle driver. */
    if (argc >= 5 && (!strcmp(argv[3], "encode") || !strcmp(argv[3], "encode-m") ||
                      !strcmp(argv[3], "roundtrip"))) {
        const char *gguf_path = argv[1];
        const char *hf_dir = argv[2];
        (void)hf_dir;
        const int use_merges = !strcmp(argv[3], "encode-m");
        Tok T;
        k3_tok_load_gguf_mode(&T, gguf_path, use_merges);
        static char text[MAXTEXT];
        FILE *f = fopen(argv[4], "rb");
        if (!f) { perror(argv[4]); return 1; }
        size_t n = fread(text, 1, MAXTEXT - 1, f);
        fclose(f);
        text[n] = 0;
        static int ids[MAXIDS];
        int m = tok_encode(&T, text, (int)n, ids, MAXIDS);
        if (!strcmp(argv[3], "roundtrip")) {
            static char back[MAXTEXT];
            int nb = tok_decode(&T, ids, m, back, MAXTEXT - 1);
            int ok = nb == (int)n && !memcmp(back, text, n);
            printf("roundtrip: %zu bytes -> %d ids -> %d bytes : %s\n",
                   n, m, nb, ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }
        for (int i = 0; i < m; i++) printf("%s%d", i ? "," : "", ids[i]);
        printf("\n");
        return 0;
    }

    run_synthetic();

    const char *gguf_path = argc > 1 ? argv[1] : NULL;
    const char *hf_dir = argc > 2 ? argv[2] : NULL;
    char def_gguf[512], def_hf[512];
    if (!gguf_path) {
        gguf_path = getenv("K3_GGUF_REAL");
        if (!gguf_path || !gguf_path[0])
            gguf_path = "/workspace/unsloth/Kimi-K3-GGUF/UD-IQ1_S";
    }
    if (!hf_dir) {
        hf_dir = getenv("K3_TOK_FILES");
        if (!hf_dir || !hf_dir[0]) hf_dir = "/workspace/unsloth/Kimi-K3-GGUF";
    }
    (void)def_gguf;
    (void)def_hf;
    run_real_gate(gguf_path, hf_dir);

    if (fails) {
        printf("\nGGUF TOKENIZER: FAIL\n");
        return 1;
    }
    printf("\nGGUF TOKENIZER: PASS\n");
    return 0;
}
