/*
 * k3_tok.h, construct a tokenizer directly from the released Kimi K3 files.
 *
 * PURPOSE
 *   The vendored BPE implementation (third_party/tok.h) is driven by a HuggingFace
 *   `tokenizer.json`. Kimi K3 does not ship one. It ships:
 *
 *     tiktoken.model          163,584 lines of "base64(token_bytes) rank"
 *     tokenizer_config.json   the special tokens, under added_tokens_decoder
 *
 *   The common workaround is to synthesise a tokenizer.json with a Python script at
 *   build time. This header populates the tokenizer structures from the released files
 *   directly instead, which keeps the engine text-in/text-out with no external step.
 *
 *   Scope is deliberately narrow: this is a loader. Encoding and decoding remain
 *   entirely in tok.h, and tok_encode/tok_decode/tok_id_of work unchanged once
 *   k3_tok_load() returns.
 *
 * THREE INVARIANTS, EACH SILENT WHEN VIOLATED
 *   Every one of these produces a tokenizer that runs, emits ids, and is wrong. There
 *   is no crash and no diagnostic; the first symptom is degraded output.
 *
 *   1. The vocabulary is keyed by the GPT-2 BYTE-LEVEL string, each byte mapped to a
 *      printable codepoint via byte2str, not by raw token bytes. tiktoken.model
 *      supplies RAW bytes. Omitting the conversion yields a hash table that never hits,
 *      so every piece degrades to single bytes and the model runs on garbage ids.
 *      tk_build_bytemap() must therefore run before any key is constructed.
 *
 *   2. rankbpe must be 1. There is no merges list in this format; tiktoken merges the
 *      adjacent pair whose CONCATENATION has the lowest id. With rankbpe at 0 the
 *      encoder consults an empty merges map and emits one token per codepoint.
 *
 *   3. kimi must be 1 (which makes the o200k flag irrelevant). The K3 pre-tokenizer
 *      adds a leading \p{Han}-run rule and excludes Han from its letter classes. tok.h
 *      normally infers the family by inspecting the pre-tokenizer regex inside
 *      tokenizer.json; with no such file the flag must be set explicitly.
 *
 * VERIFICATION
 *   tools/tok_parity.py compares this loader against the reference tokenizer
 *   token-for-token across CJK, emoji, ZWJ sequences, accents, contractions and
 *   whitespace runs. `make tok` runs it.
 */
#ifndef K3_TOK_H
#define K3_TOK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "json.h"
#include "k3_gguf.h"

/* tok.h is a header-only library, vendored unmodified (see NOTICE). Every function in it
 * has file scope, so a translation unit that uses only part of the API draws
 * -Wunused-function for the remainder.
 *
 * The suppression is scoped to this include rather than added to CFLAGS. A global
 * suppression would also silence unused-function warnings in first-party code, where
 * they indicate genuine dead code and should still fail the build. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "tok.h"
#pragma GCC diagnostic pop

/* Vocabulary ceiling from config.json text_config.vocab_size. Ranks occupy
 * [0, 163584) and the added tokens land above that, so one array of this size covers
 * both. Asserted against the files rather than assumed. */
#define K3_TOK_VOCAB 163840

/* ---------------------------------------------------------------- base64 ---- */
/* Standard alphabet, '=' padded. Returns decoded length, or -1 on a malformed group.
 * out must hold at least 3*((n+3)/4) bytes. */
static inline int k3_b64(const char *in, int n, unsigned char *out)
{
    static signed char t[256];
    static int init = 0;
    if (!init) {
        for (int i = 0; i < 256; i++) t[i] = -1;
        const char *A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) t[(unsigned char)A[i]] = (signed char)i;
        init = 1;
    }
    int o = 0;
    unsigned int acc = 0;
    int bits = 0;
    for (int i = 0; i < n; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '=') break;
        signed char v = t[c];
        if (v < 0) return -1;
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) { bits -= 8; out[o++] = (unsigned char)((acc >> bits) & 0xFF); }
    }
    return o;
}

/* Raw token bytes -> the byte-level string tok.h hashes on. Worst case two output
 * bytes per input byte (the substituted codepoints are 256..511, two UTF-8 bytes),
 * so `out` must hold 2*n+1. Returns the string length; NUL-terminates. */
static inline int k3_bytelevel(const Tok *T, const unsigned char *b, int n, char *out)
{
    int o = 0;
    for (int i = 0; i < n; i++) {
        memcpy(out + o, T->byte2str[b[i]], (size_t)T->byte2cp_len[b[i]]);
        o += T->byte2cp_len[b[i]];
    }
    out[o] = 0;
    return o;
}

/* ------------------------------------------------------------------ load ---- */
/* files_dir is the directory holding tiktoken.model and tokenizer_config.json
 * (kimi_k3_hf/files). Exits on any malformed input: a tokenizer that loads
 * half-correctly is worse than one that refuses, because the failure surfaces as
 * subtly wrong text hundreds of tokens later. */
static inline void k3_tok_load(Tok *T, const char *files_dir)
{
    memset(T, 0, sizeof *T);

    /* Order matters: byte2str must exist before any vocab key is built (trap #1). */
    tk_build_bytemap(T);
    T->kimi    = 1;   /* trap #3 */
    T->o200k   = 1;   /* K3 builds on the o200k rules; kimi refines them */
    T->rankbpe = 1;   /* trap #2 */

    T->n_ids     = K3_TOK_VOCAB;
    T->id2str    = (char **)calloc((size_t)T->n_ids, sizeof(char *));
    T->id_added  = (int *)calloc((size_t)T->n_ids, sizeof(int));
    T->id_special= (int *)calloc((size_t)T->n_ids, sizeof(int));
    if (!T->id2str || !T->id_added || !T->id_special) {
        fprintf(stderr, "k3_tok: OOM sizing %d ids\n", T->n_ids); exit(1);
    }

    /* ---- ranks ---- */
    char path[4096];
    snprintf(path, sizeof path, "%s/tiktoken.model", files_dir);
    long nbuf = 0;
    char *buf = tk_read_file(path, &nbuf);      /* tok.h helper; exits if absent */

    /* Power-of-two capacity, ~2x load factor, as tok_load does. */
    int vc = 1; while (vc < 163584 * 2) vc <<= 1;
    hm_init(&T->vocab, vc);

    int nrank = 0, maxrank = -1;
    long i = 0;
    while (i < nbuf) {
        long ls = i;
        while (i < nbuf && buf[i] != '\n') i++;
        long le = i;
        if (i < nbuf) i++;                       /* step over '\n' */
        if (le > ls && buf[le - 1] == '\r') le--;
        if (le <= ls) continue;                  /* blank line */

        /* split on the single space: "<base64> <rank>" */
        long sp = ls;
        while (sp < le && buf[sp] != ' ') sp++;
        if (sp >= le) {
            fprintf(stderr, "k3_tok: tiktoken.model line %d has no rank field\n", nrank);
            exit(1);
        }
        int blen = (int)(sp - ls);
        int rank = atoi(buf + sp + 1);
        if (rank < 0 || rank >= T->n_ids) {
            fprintf(stderr, "k3_tok: rank %d out of range at line %d\n", rank, nrank);
            exit(1);
        }

        unsigned char raw[512];
        if (blen > (int)((sizeof raw) * 4 / 3)) {
            fprintf(stderr, "k3_tok: implausibly long token at rank %d\n", rank); exit(1);
        }
        int rn = k3_b64(buf + ls, blen, raw);
        if (rn < 0) { fprintf(stderr, "k3_tok: bad base64 at rank %d\n", rank); exit(1); }

        char *key = (char *)malloc((size_t)2 * rn + 1);
        if (!key) { fprintf(stderr, "k3_tok: OOM on token %d\n", rank); exit(1); }
        int kl = k3_bytelevel(T, raw, rn, key);

        hm_put(&T->vocab, key, kl, rank);
        T->id2str[rank] = key;                   /* decode reverses this via cp2byte */
        if (rank > maxrank) maxrank = rank;
        nrank++;
    }
    free(buf);
    if (nrank == 0) { fprintf(stderr, "k3_tok: tiktoken.model is empty\n"); exit(1); }

    /* ---- specials ---- */
    snprintf(path, sizeof path, "%s/tokenizer_config.json", files_dir);
    long ncfg = 0;
    char *cfg = tk_read_file(path, &ncfg);
    char *arena = NULL;
    jval *root = json_parse(cfg, &arena);
    jval *adt  = json_get(root, "added_tokens_decoder");
    if (!adt) {
        fprintf(stderr, "k3_tok: tokenizer_config.json has no added_tokens_decoder\n");
        exit(1);
    }

    T->nsp = adt->len;
    T->sp  = (Special *)calloc((size_t)(T->nsp ? T->nsp : 1), sizeof(Special));
    if (!T->sp) { fprintf(stderr, "k3_tok: OOM on %d specials\n", T->nsp); exit(1); }

    for (int k = 0; k < adt->len; k++) {
        /* keys are the ids, as strings: {"163584": {"content": "[BOS]", ...}} */
        int id = atoi(adt->keys[k]);
        jval *e  = adt->kids[k];
        jval *jc = json_get(e, "content");
        if (!jc || jc->t != J_STR || !jc->str) {
            fprintf(stderr, "k3_tok: added token %s has no content\n", adt->keys[k]);
            exit(1);
        }
        if (id < 0 || id >= T->n_ids) {
            fprintf(stderr, "k3_tok: added token id %d out of range\n", id); exit(1);
        }
        T->sp[k].str = jc->str;                  /* json_parse strings are independent */
        T->sp[k].len = (int)strlen(jc->str);
        T->sp[k].id  = id;
        T->id2str[id]   = jc->str;               /* added tokens decode literally */
        T->id_added[id] = 1;
        jval *sf = json_get(e, "special");
        if (sf && sf->t == J_BOOL && sf->boolean) T->id_special[id] = 1;
    }
    /* longest match first, so "<|end_of_msg|>" wins over any prefix of it */
    qsort(T->sp, (size_t)T->nsp, sizeof(Special), cmp_sp_len);

    fprintf(stderr, "[TOK] %d ranks (max id %d) + %d added tokens | kimi=%d rankbpe=%d\n",
            nrank, maxrank, T->nsp, T->kimi, T->rankbpe);
}

/* ------------------------------------------------------- load (gguf, D6) ---- */
/* The tokenizer from GGUF shard-1 metadata, mirroring k3_tok_load's shape and
 * invariants. `path` is a shard directory (holding *-00001-of-*.gguf) or a single
 * .gguf file; the metadata is read through the wave-1 reader (k3_gguf.h), whose
 * quirks handling this path inherits wholesale.
 *
 * VERIFIED SOURCE KEYS (recon-tok-cli 4b, byte-verified):
 *   tokenizer.ggml.model  "gpt2"        -> byte-level vocab keying (trap #1)
 *   tokenizer.ggml.pre    "kimi-k2"     -> kimi=1, o200k=1 (trap #3)
 *   tokenizer.ggml.tokens               -> ALREADY the byte-level strings the
 *       Tok vocab hashes on; no base64, no k3_bytelevel conversion. Index == id.
 *   tokenizer.ggml.merges "l r" pairs   -> byte-level, gpt2-style recovered
 *   tokenizer.ggml.token_type            -> 1 = NORMAL, 3 = CONTROL (verified:
 *       ids >= 163584 are CONTROL; the recon's "all NORMAL" was wrong)
 *   tokenizer.ggml.{bos,eos,padding}_token_id, kimi-k3.vocab_size
 *
 * CONTROL TOKENS (the verified writer's shape): the 16 added-token names from
 * tokenizer_config.json's added_tokens_decoder are atomic (sp[], id_added);
 * the remaining controls are <|reserved_token_<id>|> tokens (the id inside the
 * name equals the token's own id) which stay in the vocab -- decodable, never
 * atomic. This mirrors how the HF config maps special:true today: encode parity
 * with the existing loader is exact, because a pretokenizer piece can never be
 * a whole reserved name. Anything else marked CONTROL fails loud.
 *
 * TRAP #2 (merges vs rankbpe): the gguf ships a recovered merges list; the
 * engine's tiktoken rule is rankbpe (lowest concatenated id, no merges map).
 * BOTH paths stay loadable here; which one is canonical was decided by the
 * Slice-0 parity gate (see K3_TOK_GGUF_USE_MERGES below). */

/* Verified writer facts (tokenizer_config.json added_tokens_decoder): exactly
 * these 16 names are the model's added tokens. */
static const char *const K3_GGUF_ADDED[16] = {
    "[BOS]", "[EOS]", "<|end_of_msg|>", "<|open|>", "<|close|>", "<|sep|>",
    "[start_header_id]", "[end_header_id]", "[EOT]",
    "<|media_begin|>", "<|media_content|>", "<|media_end|>", "<|media_pad|>",
    "<osagent_mode>", "[UNK]", "[PAD]",
};

/* The special:true subset of the same config: the 13 control tokens that are
 * never legitimate response content. Purely informational today (nothing in
 * tok.h's encode/decode consults id_special), kept for exact parity with the
 * config-driven loader. */
static const char *const K3_GGUF_SPECIAL[13] = {
    "[BOS]", "[EOS]", "<|end_of_msg|>", "[start_header_id]", "[end_header_id]",
    "[EOT]", "<|media_begin|>", "<|media_content|>", "<|media_end|>",
    "<|media_pad|>", "<osagent_mode>", "[UNK]", "[PAD]",
};

/* The remaining controls are reserved tokens whose name embeds their own id:
 * <|reserved_token_163592|> lives at id 163592. Verified for all 240 of them. */
static int k3_is_reserved(const char *s, int id)
{
    char want[64];
    snprintf(want, sizeof want, "<|reserved_token_%d|>", id);
    return strcmp(s, want) == 0;
}

static int k3_name_in(const char *const *tab, int n, const char *s)
{
    for (int i = 0; i < n; i++)
        if (!strcmp(tab[i], s)) return 1;
    return 0;
}

/* use_merges: 0 = tiktoken rankbpe (no merges map, trap #2 path A); 1 = the
 * gguf merges list (path B). The parity verdict decides the public default. */
static inline void k3_tok_load_gguf_from_mode(Tok *T, const K3Gguf *g, int use_merges)
{
    memset(T, 0, sizeof *T);
    tk_build_bytemap(T);
    T->kimi    = 1;   /* verified: tokenizer.ggml.pre == "kimi-k2" below */
    T->o200k   = 1;
    T->rankbpe = use_merges ? 0 : 1;

    /* Populate from an ALREADY-OPEN shard set (the CLI's config gate opens one
     * for the shard-1 KV; re-opening just for the tokenizer would re-walk all 14
     * shard headers). The caller keeps ownership of g. */
    K3GgufTok tk;
    if (k3_gguf_tok(g, &tk) != 0) {
        fprintf(stderr, "k3_tok: GGUF tokenizer extraction failed\n");
        exit(1);
    }
    /* tk is ADOPTED from here on: id2str/vocab keys reference its strings for
     * the life of the process, exactly as the tiktoken.model loader's keys do. */

    if (strcmp(tk.model, "gpt2") != 0) {
        fprintf(stderr, "k3_tok: tokenizer.ggml.model is '%s', expected 'gpt2'\n",
                tk.model);
        exit(1);
    }
    if (strcmp(tk.pre, "kimi-k2") != 0) {
        fprintf(stderr, "k3_tok: tokenizer.ggml.pre is '%s', expected 'kimi-k2'\n",
                tk.pre);
        exit(1);
    }
    if (tk.ntok < 257 || tk.ntok > (1 << 21)) {
        fprintf(stderr, "k3_tok: implausible vocabulary of %d tokens\n", tk.ntok);
        exit(1);
    }
    T->n_ids = tk.ntok;
    T->id2str     = (char **)calloc((size_t)T->n_ids, sizeof(char *));
    T->id_added   = (int *)calloc((size_t)T->n_ids, sizeof(int));
    T->id_special = (int *)calloc((size_t)T->n_ids, sizeof(int));
    if (!T->id2str || !T->id_added || !T->id_special) {
        fprintf(stderr, "k3_tok: OOM sizing %d ids\n", T->n_ids);
        exit(1);
    }

    /* ---- vocab: index == id, keyed by the byte-level string (trap #1) ---- */
    int vc = 1;
    while (vc < T->n_ids * 2) vc <<= 1;
    hm_init(&T->vocab, vc);
    for (int i = 0; i < T->n_ids; i++) {
        const char *s = tk.tokens[i];
        /* Fail loud on duplicate byte-level strings: two ids sharing one string
         * would silently collapse in the hash map (last id wins) and encode
         * differently than the ids suggest (crit-d-2 #7). */
        if (hm_get(&T->vocab, s, (int)strlen(s)) >= 0) {
            fprintf(stderr, "k3_tok: tokens %d and earlier share the string '%s'\n",
                    i, s);
            exit(1);
        }
        hm_put(&T->vocab, s, (int)strlen(s), i);
        T->id2str[i] = (char *)s;
    }

    /* ---- control tokens: the verified-writer shape ---- */
    int n_control = 0, n_added = 0;
    for (int i = 0; i < T->n_ids; i++) {
        if (tk.ttype[i] == 1) continue;                 /* NORMAL */
        if (tk.ttype[i] != 3) {
            fprintf(stderr, "k3_tok: token %d has token_type %d, expected 1 or 3\n",
                    i, tk.ttype[i]);
            exit(1);
        }
        n_control++;
        const char *s = tk.tokens[i];
        if (k3_name_in(K3_GGUF_ADDED, 16, s)) {
            n_added++;                                  /* atomic, like the config */
        } else if (!k3_is_reserved(s, i)) {
            fprintf(stderr, "k3_tok: control token %d is '%s', not a known added "
                            "token or <|reserved_token_%d|>\n", i, s, i);
            exit(1);
        }
    }
    /* The verified writer marks EXACTLY the 16 config names as CONTROL and
     * nothing else: a file that marks e.g. [BOS] as NORMAL would silently load
     * with a non-atomic sp[] and diverge from the HF loader (crit-d-1 #3). */
    if (n_added != (int)(sizeof K3_GGUF_ADDED / sizeof *K3_GGUF_ADDED)) {
        fprintf(stderr, "k3_tok: %d control tokens match the 16 added-token names, "
                        "expected exactly %d\n", n_added,
                (int)(sizeof K3_GGUF_ADDED / sizeof *K3_GGUF_ADDED));
        exit(1);
    }
    /* gpt2-family shape: 256 byte-level base symbols + one merge per remaining
     * rank; control tokens have no merges. Verified: 163328 == 163840-512. */
    if (tk.nmerges != T->n_ids - 256 - n_control) {
        fprintf(stderr, "k3_tok: %d merges for %d tokens with %d controls, expected "
                        "%d\n", tk.nmerges, T->n_ids, n_control,
                T->n_ids - 256 - n_control);
        exit(1);
    }

    /* ---- merges (validated in BOTH modes: a malformed merges array is a
     * corrupt file and must fail loud even when the map is not consulted; the
     * map itself is only built for the merges path) ---- */
    for (int i = 0; i < tk.nmerges; i++) {
        /* "left right" (byte-level, single space; a raw space byte is never
         * part of a byte-level string, it is '\xc4\xa0'). */
        const char *s = tk.merges[i];
        const char *sp = strchr(s, ' ');
        if (!sp || sp == s || sp[1] == '\0' || strchr(sp + 1, ' ')) {
            fprintf(stderr, "k3_tok: merge %d '%s' is not a 'left right' pair\n",
                    i, s);
            exit(1);
        }
    }
    if (use_merges) {
        int mc = 1;
        while (mc < tk.nmerges * 2) mc <<= 1;
        hm_init(&T->merges, mc);
        for (int i = 0; i < tk.nmerges; i++) {
            const char *s = tk.merges[i];
            const char *sp = strchr(s, ' ');
            const size_t ll = (size_t)(sp - s);
            const size_t rl = strlen(sp + 1);
            char *key = (char *)malloc(ll + 1 + rl);
            if (!key) { fprintf(stderr, "k3_tok: OOM on merge %d\n", i); exit(1); }
            memcpy(key, s, ll);
            key[ll] = '\0';
            memcpy(key + ll + 1, sp + 1, rl);
            hm_put(&T->merges, key, (int)(ll + 1 + rl), i);
        }
    }

    /* ---- added tokens: sp[] longest-first, exactly the config's 16 ---- */
    T->nsp = n_added;
    T->sp  = (Special *)calloc((size_t)(n_added ? n_added : 1), sizeof(Special));
    if (!T->sp) { fprintf(stderr, "k3_tok: OOM on %d specials\n", n_added); exit(1); }
    int ns = 0;
    for (int i = 0; i < T->n_ids; i++) {
        if (tk.ttype[i] != 3) continue;
        const char *s = tk.tokens[i];
        if (!k3_name_in(K3_GGUF_ADDED, 16, s)) continue;
        T->sp[ns].str = (char *)s;
        T->sp[ns].len = (int)strlen(s);
        T->sp[ns].id  = i;
        T->id_added[i]   = 1;
        T->id_special[i] = k3_name_in(K3_GGUF_SPECIAL, 13, s);
        ns++;
    }
    if (ns != n_added) {
        fprintf(stderr, "k3_tok: added-token pass lost %d\n", n_added);
        exit(1);
    }
    qsort(T->sp, (size_t)T->nsp, sizeof(Special), cmp_sp_len);

    /* ---- bos/eos/pad: verified ids and names ---- */
    if (!tk.have_bos || !tk.have_eos || !tk.have_pad) {
        fprintf(stderr, "k3_tok: GGUF tokenizer is missing a bos/eos/padding id\n");
        exit(1);
    }
    if (tk.bos_id >= (uint32_t)T->n_ids || strcmp(tk.tokens[tk.bos_id], "[BOS]")) {
        fprintf(stderr, "k3_tok: bos_token_id %u is not [BOS]\n", tk.bos_id);
        exit(1);
    }
    if (tk.eos_id >= (uint32_t)T->n_ids ||
        strcmp(tk.tokens[tk.eos_id], "<|end_of_msg|>")) {
        fprintf(stderr, "k3_tok: eos_token_id %u is not <|end_of_msg|>\n", tk.eos_id);
        exit(1);
    }
    if (tk.pad_id >= (uint32_t)T->n_ids || strcmp(tk.tokens[tk.pad_id], "[PAD]")) {
        fprintf(stderr, "k3_tok: padding_token_id %u is not [PAD]\n", tk.pad_id);
        exit(1);
    }

    fprintf(stderr, "[TOK] gguf: %d tokens (%d control, %d added) | kimi=%d "
                    "rankbpe=%d%s\n",
            T->n_ids, n_control, T->nsp, T->kimi, T->rankbpe,
            use_merges ? " (merges)" : "");
}

/* Path-based entry: opens the shard set (directory or single .gguf), loads, and
 * closes it again. */
static inline void k3_tok_load_gguf_mode(Tok *T, const char *path, int use_merges)
{
    struct stat st;
    const int isdir = stat(path, &st) == 0 && S_ISDIR(st.st_mode);
    K3Gguf g;
    if ((isdir ? k3_gguf_open(&g, path) : k3_gguf_open_file(&g, path)) != 0) {
        fprintf(stderr, "k3_tok: cannot open GGUF tokenizer metadata from %s\n", path);
        exit(1);
    }
    k3_tok_load_gguf_from_mode(T, &g, use_merges);
    k3_gguf_close(&g);
}

/* Slice-0 parity verdict (see .agents/reports/dev-d.md): which loadable path is
 * the default. 0 = tiktoken rankbpe (trap #2 path A), 1 = the gguf merges list
 * (path B). */
#define K3_TOK_GGUF_USE_MERGES 0

static inline void k3_tok_load_gguf(Tok *T, const char *path)
{
    k3_tok_load_gguf_mode(T, path, K3_TOK_GGUF_USE_MERGES);
}

/* Default-verdict variant over an already-open shard set. */
static inline void k3_tok_load_gguf_from(Tok *T, const K3Gguf *g)
{
    k3_tok_load_gguf_from_mode(T, g, K3_TOK_GGUF_USE_MERGES);
}

#endif /* K3_TOK_H */
