/* k3_gguf.c - GGUF shard-set reader (P2 Wave 1). See k3_gguf.h for the format
 * digest, the writer quirks and the D1/D5/D7 decisions this file implements.
 *
 * TWO PASSES OVER THE HEADERS, NO DATA
 *   Pass 1 reads each file's 24-byte header and walks the metadata KV section:
 *   split keys, the shard-1 config table and general.alignment all come out of
 *   here. Pass 2 repositions each fd at the recorded end of the KV section and
 *   walks the tensor infos. Bounded by construction: only the 6.9 MB shard-1
 *   metadata and ~14 KB of tensor infos per shard are ever read, and every read is
 *   length-checked against the file size.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "k3_gguf.h"

/* ------------------------------------------------------------------ errors */

/* The one fail-loud helper. Every hard error in this file goes through here, so
 * each one carries the file, the byte offset and what was expected. */
static int gerr(const char *path, uint64_t off, const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "k3_gguf: %s", path ? path : "?");
    if (off) fprintf(stderr, " @+%llu", (unsigned long long)off);
    fprintf(stderr, ": ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    return -1;
}

/* ------------------------------------------------------------------ reader */

#define RDR_CHUNK (1u << 20) /* 1 MiB; shard-1 metadata is 6.9 MB -> ~7 preads */

typedef struct {
    const char    *path;
    int            fd;
    uint64_t       fsize;
    size_t         cap;        /* buffer capacity; grows when the file is larger */
    unsigned char *buf;
    size_t         len, pos;   /* bytes loaded, cursor                       */
} Rdr;

static int rdr_init(Rdr *r, const char *path, int fd, uint64_t fsize)
{
    r->path = path;
    r->fd = fd;
    r->fsize = fsize;
    r->cap = RDR_CHUNK < fsize ? RDR_CHUNK : (size_t)fsize;
    r->buf = malloc(r->cap ? r->cap : 1);
    if (!r->buf) return -1;
    r->len = 0;
    r->pos = 0;
    return 0;
}

static void rdr_free(Rdr *r) { free(r->buf); }

/* Ensure pos + n <= len, preading more as needed. n must be sane (callers cap the
 * length fields before calling); the file size is the hard bound. When the buffer
 * is full it grows. No caller holds a pointer into the buffer across a read: value
 * pointers (Val.s / Val.d) are taken only after their final rdr_need, and consumed
 * by kv_push before the next read. */
static int rdr_need(Rdr *r, size_t n)
{
    while (r->pos + n > r->len) {
        if (r->len >= r->fsize) return -1;
        if (r->len == r->cap) {
            size_t nc = r->cap * 2;
            if (nc < r->cap) return -1;              /* size_t overflow */
            if (nc > r->fsize) nc = (size_t)r->fsize;
            unsigned char *nb = (unsigned char *)realloc(r->buf, nc);
            if (!nb) return -1;
            r->buf = nb;
            r->cap = nc;
        }
        size_t want = r->cap - r->len;               /* room left in the buffer */
        if (want > r->fsize - r->len) want = (size_t)(r->fsize - r->len);
        ssize_t got = pread(r->fd, r->buf + r->len, want, (off_t)r->len);
        if (got <= 0) return -1;
        r->len += (size_t)got;
    }
    return 0;
}

static int rdr_skip(Rdr *r, uint64_t n)
{
    if (n > r->fsize || n < r->pos) return -1;
    if (rdr_need(r, (size_t)(n - r->pos))) return -1;
    r->pos = (size_t)n;
    return 0;
}

static int rdr_u8(Rdr *r, uint8_t *v)
{
    if (rdr_need(r, 1)) return -1;
    *v = r->buf[r->pos++];
    return 0;
}

/* Little-endian, hand-decoded like k3_st.c. */
static int rdr_u16(Rdr *r, uint16_t *v)
{
    if (rdr_need(r, 2)) return -1;
    *v = (uint16_t)r->buf[r->pos] | (uint16_t)(r->buf[r->pos + 1] << 8);
    r->pos += 2;
    return 0;
}

static int rdr_u32(Rdr *r, uint32_t *v)
{
    if (rdr_need(r, 4)) return -1;
    *v = (uint32_t)r->buf[r->pos] | ((uint32_t)r->buf[r->pos + 1] << 8) |
         ((uint32_t)r->buf[r->pos + 2] << 16) | ((uint32_t)r->buf[r->pos + 3] << 24);
    r->pos += 4;
    return 0;
}

static int rdr_u64(Rdr *r, uint64_t *v)
{
    if (rdr_need(r, 8)) return -1;
    uint64_t x = 0;
    for (int i = 7; i >= 0; i--) x = (x << 8) | r->buf[r->pos + (size_t)i];
    r->pos += 8;
    *v = x;
    return 0;
}

static int rdr_f32(Rdr *r, float *v)
{
    uint32_t u;
    if (rdr_u32(r, &u)) return -1;
    memcpy(v, &u, sizeof u);
    return 0;
}

static int rdr_f64(Rdr *r, double *v)
{
    uint64_t u;
    if (rdr_u64(r, &u)) return -1;
    memcpy(v, &u, sizeof u);
    return 0;
}

/* A length-prefixed string. `ascii` additionally requires printable ASCII, which
 * holds for every key and tensor name in the spec and in this file. len must be at
 * least 1: a zero-length key would be a misaligned walk, not a name. */
static int rdr_str(Rdr *r, char *out, size_t cap, size_t *len, int ascii)
{
    uint64_t n;
    if (rdr_u64(r, &n)) return -1;
    if (n == 0 || n > cap - 1 || n > r->fsize || n > (1u << 24)) return -1;
    if (rdr_need(r, (size_t)n)) return -1;
    if (ascii)
        for (uint64_t i = 0; i < n; i++)
            if (r->buf[r->pos + i] < 0x20 || r->buf[r->pos + i] > 0x7E) return -1;
    memcpy(out, r->buf + r->pos, (size_t)n);
    r->pos += (size_t)n;
    out[n] = '\0';
    if (len) *len = (size_t)n;
    return 0;
}

/* ------------------------------------------------------------------ walkers */

/* Wire value types, current spec 0..12. The observed file matches this exactly
 * (5 = int32, 6 = float32, 7 = bool, 8 = string, 9 = array, 10 = uint64). The
 * historical enum disagreed only at type 5, which this file uses as int32. */
enum { G_T_U8 = 0, G_T_I8, G_T_U16, G_T_I16, G_T_U32, G_T_I32, G_T_F32,
       G_T_BOOL, G_T_STRING, G_T_ARRAY, G_T_U64, G_T_I64, G_T_F64 };

static uint64_t g_elemsz(uint32_t t)
{
    switch (t) {
    case G_T_U8:
    case G_T_I8:
    case G_T_BOOL:      return 1;
    case G_T_U16:
    case G_T_I16:       return 2;
    case G_T_U32:
    case G_T_I32:
    case G_T_F32:       return 4;
    case G_T_U64:
    case G_T_I64:
    case G_T_F64:       return 8;
    default:            return 0;   /* string/array: variable */
    }
}

static int mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a && b > UINT64_MAX / a) return -1;
    *out = a * b;
    return 0;
}

/* One parsed metadata value. Strings and fixed-element arrays point into the
 * reader buffer, which stays valid until the walk of this file ends. */
typedef struct {
    uint64_t    u;               /* numeric types, bool as 0/1   */
    double      f;               /* f32, f64                     */
    const char *s;               /* string                       */
    size_t      sn;
    uint32_t    et;              /* array element type           */
    uint64_t    n;               /* array element count          */
    const void *d;               /* fixed-element array bytes    */
    size_t      dn;
} Val;

static int walk_value(Rdr *r, uint32_t type, Val *v, int depth)
{
    memset(v, 0, sizeof *v);
    if (depth > 4) return -1;                /* nested-array sanity bound */
    switch (type) {
    case G_T_U8: { uint8_t x;  if (rdr_u8(r, &x))  return -1; v->u = x; break; }
    case G_T_I8: { uint8_t x;  if (rdr_u8(r, &x)) return -1;
                   v->u = (uint64_t)(int64_t)(int8_t)x; break; }
    case G_T_U16: { uint16_t x; if (rdr_u16(r, &x)) return -1; v->u = x; break; }
    case G_T_I16: { uint16_t x; if (rdr_u16(r, &x)) return -1;
                    v->u = (uint64_t)(int64_t)(int16_t)x; break; }
    case G_T_U32: { uint32_t x; if (rdr_u32(r, &x)) return -1; v->u = x; break; }
    case G_T_I32: { uint32_t x; if (rdr_u32(r, &x)) return -1;
                    v->u = (uint64_t)(int64_t)(int32_t)x; break; }
    case G_T_F32: { float x;    if (rdr_f32(r, &x)) return -1; v->f = x; break; }
    case G_T_BOOL: { uint8_t x; if (rdr_u8(r, &x)) return -1;
                    if (x > 1) return -1;      /* a bool is 0 or 1, nothing else */
                    v->u = x; break; }
    case G_T_STRING: {
        uint64_t n;
        if (rdr_u64(r, &n)) return -1;
        /* n == 0 is a legal EMPTY string value (spec); keys and tensor names go
         * through rdr_str, which keeps the n == 0 ban. */
        if (n > r->fsize || n > (1u << 24)) return -1;
        if (rdr_need(r, (size_t)n)) return -1;
        v->s = (const char *)r->buf + r->pos;
        v->sn = (size_t)n;
        r->pos += (size_t)n;
        break;
    }
    case G_T_ARRAY: {
        uint32_t et;
        uint64_t n;
        if (rdr_u32(r, &et)) return -1;
        if (et > G_T_F64) return -1;           /* unknown element type */
        if (rdr_u64(r, &n)) return -1;
        /* Sane element-count bound: the largest real arrays hold 163,840
         * elements (tokenizer.ggml.token_type), so 2^22 caps the worst-case
         * fixed-element copy at 32 MB on a lying header. */
        if (n > (1u << 22)) return -1;
        v->et = et;
        v->n = n;
        if (et == G_T_STRING) {
            for (uint64_t i = 0; i < n; i++) { /* walk, do not materialize */
                uint64_t sl;
                if (rdr_u64(r, &sl)) return -1;
                if (sl > r->fsize || sl > (1u << 24)) return -1;
                if (rdr_need(r, (size_t)sl)) return -1;
                r->pos += (size_t)sl;
            }
        } else if (et == G_T_ARRAY) {
            for (uint64_t i = 0; i < n; i++) {
                Val sub;
                if (walk_value(r, G_T_ARRAY, &sub, depth + 1)) return -1;
            }
        } else {
            uint64_t esz = g_elemsz(et);
            if (esz == 0) return -1;
            if (mul_u64(n, esz, &v->dn)) return -1;
            if (v->dn > r->fsize) return -1;
            if (rdr_need(r, (size_t)v->dn)) return -1;
            v->d = r->buf + r->pos;
            r->pos += (size_t)v->dn;
        }
        break;
    }
    case G_T_U64: { uint64_t x; if (rdr_u64(r, &x)) return -1; v->u = x; break; }
    case G_T_I64: { uint64_t x; if (rdr_u64(r, &x)) return -1;
                    v->u = (uint64_t)(int64_t)x; break; }
    case G_T_F64: { double x;   if (rdr_f64(r, &x)) return -1; v->f = x; break; }
    default:      return -1;
    }
    return 0;
}

/* D7 alignment verification: after a value, the next bytes must look like the next
 * key (a sane length-prefixed ASCII string), or everything left must be zero
 * padding (a metadata-only shard ends with < alignment zero bytes). A misaligned
 * walk lands mid-data and fails one of the two. */
static int boundary_ok(Rdr *r)
{
    uint64_t rem = r->fsize - r->pos;
    if (rem == 0) return 1;
    if (rem >= 8 && !rdr_need(r, 8)) {
        uint64_t n = 0;
        for (int i = 7; i >= 0; i--) n = (n << 8) | r->buf[r->pos + (size_t)i];
        if (n >= 1 && n <= 4096 && 8 + n <= rem && !rdr_need(r, (size_t)(8 + n))) {
            int printable = 1;
            for (uint64_t i = 0; i < n; i++)
                if (r->buf[r->pos + 8 + i] < 0x20 || r->buf[r->pos + 8 + i] > 0x7E)
                    printable = 0;
            if (printable) return 1;
        }
    }
    /* Not a plausible next key: the remainder must be zero padding. Check only the
     * first < alignment bytes; beyond that would be tensor data. */
    uint64_t lim = rem < 32 ? rem : 32;
    if (!rdr_need(r, (size_t)lim))
        for (uint64_t i = 0; i < lim; i++)
            if (r->buf[r->pos + i]) return 0;
    return 1;
}

/* ------------------------------------------------------------------ arena */

static void *arena_get(K3Gguf *g, size_t n)
{
    /* Keep values 8-byte aligned: the config reader dereferences array elements as
     * uint32_t, and the arena grows past arbitrary-length strings first. */
    size_t off = (g->alen + 7) & ~(size_t)7;
    if (off + n > g->acap) {
        size_t nc = g->acap ? g->acap : (1 << 20);
        while (off + n > nc) nc *= 2;
        char *np = (char *)realloc(g->arena, nc);
        if (!np) return NULL;
        g->arena = np;
        g->acap = nc;
    }
    void *p = g->arena + off;
    g->alen = off + n;
    return p;
}

static int kv_push(K3Gguf *g, const char *key, size_t klen, uint32_t type, const Val *v)
{
    K3GgufKV *nk = (K3GgufKV *)realloc(g->kv, (size_t)(g->nkv + 1) * sizeof *nk);
    if (!nk) return -1;
    g->kv = nk;
    K3GgufKV *k = &g->kv[g->nkv++];
    memcpy(k->key, key, klen + 1);
    k->type = type;
    if (type == G_T_STRING) {
        char *p = (char *)arena_get(g, v->sn);
        if (!p) return -1;
        memcpy(p, v->s, v->sn);
        k->v.s.p = p;
        k->v.s.n = v->sn;
    } else if (type == G_T_ARRAY) {
        k->v.a.et = v->et;
        k->v.a.n = (uint32_t)v->n;
        if (v->d) {                            /* fixed-size elements: copied */
            void *p = arena_get(g, v->dn);
            if (!p) return -1;
            memcpy(p, v->d, v->dn);
            k->v.a.d = p;
            k->v.a.nbytes = v->dn;
        } else {                               /* string/nested: walked only */
            k->v.a.d = NULL;
            k->v.a.nbytes = 0;
        }
    } else if (type == G_T_F32 || type == G_T_F64) {
        k->v.f = v->f;
    } else {
        k->v.u = v->u;
    }
    return 0;
}

/* ------------------------------------------------------------------ shard */

typedef struct {
    char     *path;
    int       fd;
    uint64_t  fsize;
    uint64_t  tensor_count;
    uint64_t  kv_count;
    uint64_t  kv_end;          /* byte offset where the tensor infos begin */
    uint16_t  split_no;
    uint16_t  split_count;
    int32_t   split_tensors;
    int       have_split;      /* all three split keys seen in this shard's walk */
    int       split_synth;     /* keys synthesized for a single file (no splits) */
} ShardMeta;

/* Pass 1: header + metadata walk. Stores the shard-1 KV table in g (all shards are
 * walked into it; the non-zero split.no shards discard theirs afterwards). The
 * split keys are captured from the CURRENT shard's own walk, never from the merged
 * table: a lookup in g->kv would find shard 1's values for every shard.
 *
 * split_optional: a single-file gguf legitimately carries no split keys (the
 * spec's sharding is a llama.cpp convention); k3_gguf_open_file passes 1 and a
 * missing set is synthesized from the file itself. In a multi-shard directory
 * open (0) a missing or partial set is a hard error. */
static int parse_metadata(K3Gguf *g, Rdr *r, uint64_t nkv, ShardMeta *m,
                          int split_optional)
{
    /* Indices [nkv_before, g->nkv) hold THIS shard's own records (the earlier
     * shards' tables were discarded), so a duplicate key within one shard is a
     * hard error: a lying shard-1 carrying kimi-k3.block_count twice would
     * otherwise silently use the first occurrence. */
    const int nkv_before = g->nkv;
    for (uint64_t i = 0; i < nkv; i++) {
        char key[64];
        size_t klen = 0;
        if (rdr_str(r, key, sizeof key, &klen, 1))
            return gerr(r->path, r->pos, "metadata key %llu of %llu is not a sane string",
                        (unsigned long long)i, (unsigned long long)nkv);
        for (int j = nkv_before; j < g->nkv; j++)
            if (!strcmp(g->kv[j].key, key))
                return gerr(r->path, r->pos, "duplicate metadata key '%s'", key);
        uint32_t vt;
        if (rdr_u32(r, &vt))
            return gerr(r->path, r->pos, "truncated value type for key '%s'", key);
        if (vt > G_T_F64)
            return gerr(r->path, r->pos, "unknown metadata value type %u on key '%s'",
                        vt, key);
        Val v;
        if (walk_value(r, vt, &v, 0))
            return gerr(r->path, r->pos, "bad value of type %u on key '%s'", vt, key);
        if (!strcmp(key, "split.no")) {
            if (vt != G_T_U16)
                return gerr(r->path, r->pos, "split.no has wire type %u, expected u16",
                            vt);
            m->split_no = (uint16_t)v.u;
            m->have_split++;
        } else if (!strcmp(key, "split.count")) {
            if (vt != G_T_U16)
                return gerr(r->path, r->pos, "split.count has wire type %u, expected u16",
                            vt);
            m->split_count = (uint16_t)v.u;
            m->have_split++;
        } else if (!strcmp(key, "split.tensors.count")) {
            if (vt != G_T_I32)
                return gerr(r->path, r->pos,
                            "split.tensors.count has wire type %u, expected i32", vt);
            m->split_tensors = (int32_t)v.u;
            m->have_split++;
        }
        if (kv_push(g, key, klen, vt, &v))
            return gerr(r->path, r->pos, "out of memory recording '%s'", key);
        if (!boundary_ok(r))
            return gerr(r->path, r->pos,
                        "metadata walk misaligned after key '%s' (next bytes are not a "
                        "key or zero padding)", key);
    }
    m->kv_end = r->pos;

    if (m->have_split == 0 && split_optional) {
        /* Single file with no split keys: synthesize the trivial set. */
        m->split_no = 0;
        m->split_count = 1;
        m->split_tensors = (int32_t)m->tensor_count;
        m->split_synth = 1;
    } else if (m->have_split != 3) {
        return gerr(r->path, r->pos,
                    "missing or mistyped split key (need split.no:u16, split.count:u16, "
                    "split.tensors.count:i32)");
    }
    return 0;
}

static int ggml_block(uint32_t gt, uint64_t *qk, uint64_t *bps)
{
    switch (gt) {
    case 0:  *qk = 1;   *bps = 4;  return 0;   /* F32   */
    case 8:  *qk = 32;  *bps = 34; return 0;   /* Q8_0  */
    case 19: *qk = 256; *bps = 50; return 0;   /* IQ1_S */
    default: return -1;
    }
}

typedef struct {
    K3Tensor *t;
    uint16_t *gt;
    size_t   *noff;
    size_t    n, cap;
} Build;

static int build_push(Build *b, K3Gguf *g, const char *name, size_t nlen,
                      const K3Tensor *src, uint16_t gt)
{
    if (b->n == b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 4096;
        K3Tensor *nt = (K3Tensor *)realloc(b->t, nc * sizeof *nt);
        uint16_t *ng = (uint16_t *)realloc(b->gt, nc * sizeof *ng);
        size_t   *no = (size_t   *)realloc(b->noff, nc * sizeof *no);
        if (!nt || !ng || !no) {
            if (nt) b->t = nt;
            if (ng) b->gt = ng;
            if (no) b->noff = no;
            return -1;
        }
        b->t = nt; b->gt = ng; b->noff = no; b->cap = nc;
    }
    if (g->strlen_ + nlen + 1 > g->strcap) {
        size_t nc = g->strcap ? g->strcap : (1 << 20);
        while (g->strlen_ + nlen + 1 > nc) nc *= 2;
        char *np = (char *)realloc(g->strpool, nc);
        if (!np) return -1;
        g->strpool = np;
        g->strcap = nc;
    }
    b->noff[b->n] = g->strlen_;
    memcpy(g->strpool + g->strlen_, name, nlen + 1);
    g->strlen_ += nlen + 1;
    b->t[b->n] = *src;
    b->gt[b->n] = gt;
    b->n++;
    return 0;
}

/* Pass 2: tensor infos. Offsets are validated relative to the (still unknown) data
 * section start and fixed up to absolute once the section start is known. */
static int parse_tensor_infos(K3Gguf *g, Rdr *r, const ShardMeta *m, int shard,
                              Build *b, int alignment)
{
    if (rdr_skip(r, m->kv_end))
        return gerr(r->path, m->kv_end, "cannot position at tensor infos");

    const size_t bstart = b->n;
    uint64_t prev_off = 0, prev_nbytes = 0;
    for (uint64_t i = 0; i < m->tensor_count; i++) {
        char name[512];
        size_t nlen = 0;
        if (rdr_str(r, name, sizeof name, &nlen, 1))
            return gerr(r->path, r->pos, "tensor %llu name is not a sane string",
                        (unsigned long long)i);
        uint32_t ndim;
        if (rdr_u32(r, &ndim))
            return gerr(r->path, r->pos, "truncated n_dims for tensor '%s'", name);
        if (ndim > 4)
            return gerr(r->path, r->pos, "tensor '%s' has %u dims, engine supports 4",
                        name, ndim);
        uint64_t dims[4] = { 1, 1, 1, 1 };
        for (uint32_t d = 0; d < ndim; d++) {
            if (rdr_u64(r, &dims[d]))
                return gerr(r->path, r->pos, "truncated dim %u of tensor '%s'", d, name);
            if (dims[d] < 1)
                return gerr(r->path, r->pos, "tensor '%s' has a zero dimension", name);
        }
        uint32_t gt;
        if (rdr_u32(r, &gt))
            return gerr(r->path, r->pos, "truncated ggml type of tensor '%s'", name);
        uint64_t qk, bps;
        if (ggml_block(gt, &qk, &bps))
            return gerr(r->path, r->pos,
                        "tensor '%s' has unsupported ggml type %u (this reader accepts "
                        "only F32=0, Q8_0=8, IQ1_S=19)", name, gt);
        uint64_t off;
        if (rdr_u64(r, &off))
            return gerr(r->path, r->pos, "truncated offset of tensor '%s'", name);

        uint64_t ne0 = dims[0];
        if (ne0 > UINT64_MAX - (qk - 1))
            return gerr(r->path, r->pos, "tensor '%s' ne0 overflows", name);
        uint64_t nbytes = (ne0 + qk - 1) / qk;
        if (mul_u64(nbytes, bps, &nbytes))
            return gerr(r->path, r->pos, "tensor '%s' size overflows", name);
        for (uint32_t d = 1; d < 4; d++)
            if (mul_u64(nbytes, dims[d], &nbytes))
                return gerr(r->path, r->pos, "tensor '%s' size overflows", name);

        if (off > r->fsize)
            return gerr(r->path, r->pos, "tensor '%s' offset %llu is past EOF",
                        name, (unsigned long long)off);
        if (off % (uint64_t)alignment)
            return gerr(r->path, r->pos,
                        "tensor '%s' offset %llu is not %d-aligned", name,
                        (unsigned long long)off, alignment);
        if (nbytes > r->fsize - off)
            return gerr(r->path, r->pos,
                        "tensor '%s' spans %llu bytes at offset %llu, past EOF",
                        name, (unsigned long long)nbytes, (unsigned long long)off);
        if (i > 0 && off < prev_off + prev_nbytes)
            return gerr(r->path, r->pos,
                        "tensor '%s' at offset %llu overlaps the previous tensor",
                        name, (unsigned long long)off);

        K3Tensor t;
        memset(&t, 0, sizeof t);
        t.shard = shard;
        t.dtype = gt == 0 ? K3_DT_F32 : K3_DT_U8;
        t.ndim = (int)ndim;
        for (uint32_t d = 0; d < ndim; d++) t.shape[d] = (int64_t)dims[d];
        t.off = (int64_t)off;                  /* relative; fixed up below */
        t.nbytes = (int64_t)nbytes;
        if (build_push(b, g, name, nlen, &t, (uint16_t)gt))
            return gerr(r->path, r->pos, "out of memory indexing tensor '%s'", name);
        prev_off = off;
        prev_nbytes = nbytes;
    }

    /* Padding between the last tensor info and the data section must be zero. */
    uint64_t dstart = (r->pos + (uint64_t)alignment - 1) & ~(uint64_t)(alignment - 1);
    if (dstart > r->fsize)
        return gerr(r->path, r->pos,
                    "data section start %llu is past EOF", (unsigned long long)dstart);
    uint64_t pad = dstart - r->pos;
    if (pad && rdr_need(r, (size_t)pad))
        return gerr(r->path, r->pos, "truncated before the data section");
    for (uint64_t i = 0; i < pad; i++)
        if (r->buf[r->pos + i])
            return gerr(r->path, r->pos + i,
                        "nonzero padding byte between tensor infos and data section");

    /* Resolve absolute offsets and check the data spans against the file size.
     * Names are not resolved into the strpool yet, so errors cite the shard-local
     * tensor index instead. */
    for (size_t i = bstart; i < b->n; i++) {
        K3Tensor *t = &b->t[i];
        uint64_t absoff = (uint64_t)t->off + dstart;
        uint64_t nb = (uint64_t)t->nbytes;
        if (absoff > r->fsize || nb > r->fsize - absoff)
            return gerr(r->path, r->pos,
                        "tensor %zu of shard %d at absolute offset %llu ends past "
                        "EOF", i - bstart, shard, (unsigned long long)(absoff + nb));
        t->off = (int64_t)absoff;
    }
    return 0;
}

/* ------------------------------------------------------------------ open */

static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static uint64_t fnv1a(const char *s)
{
    uint64_t h = 1469598103934665603ull;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ull; }
    return h;
}

/* Everything after file discovery. Both k3_gguf_open (directory of shards) and
 * k3_gguf_open_file (one file) land here with the file list already built and
 * sorted (single-file lists trivially so). split_optional relaxes the split-key
 * requirement for a one-file set. */
static int gguf_open_indexed(K3Gguf *g, char **files, int nf, int split_optional)
{
    g->path = files;
    g->nshard = nf;
    g->fd = (int *)malloc((size_t)nf * sizeof(int));
    g->dfd = (int *)malloc((size_t)nf * sizeof(int));
    g->shard_kv = (int *)malloc((size_t)nf * sizeof(int));
    /* -1/0 per array right after ITS malloc: k3_gguf_close on a partial-malloc
     * failure would otherwise close garbage fd values (crit-d-2 #6). */
    if (g->fd) for (int i = 0; i < nf; i++) g->fd[i] = -1;
    if (g->dfd) for (int i = 0; i < nf; i++) g->dfd[i] = -1;
    if (g->shard_kv) for (int i = 0; i < nf; i++) g->shard_kv[i] = 0;
    if (!g->fd || !g->dfd || !g->shard_kv) { k3_gguf_close(g); return -1; }

    ShardMeta *meta = (ShardMeta *)calloc((size_t)nf, sizeof *meta);
    if (!meta) { k3_gguf_close(g); return -1; }

    /* ---- pass 1: headers + metadata ---- */
    for (int i = 0; i < nf; i++) {
        meta[i].path = files[i];
        meta[i].fd = open(files[i], O_RDONLY);
        if (meta[i].fd < 0) { gerr(files[i], 0, "cannot open"); goto fail; }
        g->fd[i] = meta[i].fd;

        struct stat st;
        if (fstat(meta[i].fd, &st) != 0 || st.st_size < 24) {
            gerr(files[i], 0, "file is too small for a GGUF header");
            goto fail;
        }
        meta[i].fsize = (uint64_t)st.st_size;

        Rdr r;
        if (rdr_init(&r, files[i], meta[i].fd, meta[i].fsize)) {
            gerr(files[i], 0, "out of memory");
            goto fail;
        }
        if (rdr_need(&r, 4) || memcmp(r.buf, "GGUF", 4) != 0) {
            gerr(files[i], 0, "bad magic (not a GGUF file)");
            rdr_free(&r);
            goto fail;
        }
        r.pos = 4;
        uint32_t version;
        if (rdr_u32(&r, &version)) {
            gerr(files[i], 0, "truncated version");
            rdr_free(&r);
            goto fail;
        }
        if (version != 3) {
            gerr(files[i], 0, "GGUF version %u, expected 3", version);
            rdr_free(&r);
            goto fail;
        }
        if (rdr_u64(&r, &meta[i].tensor_count) || rdr_u64(&r, &meta[i].kv_count)) {
            gerr(files[i], 0, "truncated tensor/metadata counts");
            rdr_free(&r);
            goto fail;
        }
        if (meta[i].tensor_count > (1u << 24) || meta[i].kv_count > (1u << 24)) {
            gerr(files[i], 0, "absurd counts: %llu tensors, %llu metadata pairs",
                  (unsigned long long)meta[i].tensor_count,
                  (unsigned long long)meta[i].kv_count);
            rdr_free(&r);
            goto fail;
        }
        g->shard_kv[i] = (int)meta[i].kv_count;
        size_t wm = g->alen;                 /* discard point for non-shard-0 tables */
        int nkv_before = g->nkv;
        if (parse_metadata(g, &r, meta[i].kv_count, &meta[i], split_optional)) {
            rdr_free(&r);
            goto fail;
        }
        rdr_free(&r);
        if (meta[i].split_no != 0) {
            g->nkv = nkv_before;
            g->alen = wm;
        }
    }

    /* ---- cross-shard split validation ---- */
    int32_t split_tensors = -1;
    for (int i = 0; i < nf; i++) {
        if (meta[i].split_count != (uint16_t)nf) {
            gerr(files[i], 0, "declares split.count %u but the directory holds %d shards",
                 meta[i].split_count, nf);
            goto fail;
        }
        if (meta[i].split_no != (uint16_t)i) {
            gerr(files[i], 0, "declares split.no %u but its sorted position is %d",
                 meta[i].split_no, i);
            goto fail;
        }
        /* A synthesized count comes from this file's own (already capped)
         * tensor_count and may legitimately be 0: a vocab-only single file. */
        if (!meta[i].split_synth &&
            (meta[i].split_tensors <= 0 || meta[i].split_tensors > (1 << 20))) {
            gerr(files[i], 0, "declares an implausible split.tensors.count %d",
                 meta[i].split_tensors);
            goto fail;
        }
        if (i == 0) split_tensors = meta[i].split_tensors;
        else if (meta[i].split_tensors != split_tensors) {
            gerr(files[i], 0, "declares split.tensors.count %d, shard 1 says %d",
                 meta[i].split_tensors, split_tensors);
            goto fail;
        }
    }

    /* Data-section alignment: default 32; general.alignment, when present, must be
     * a sane power of two. */
    g->alignment = 32;
    const K3GgufKV *al = k3_gguf_kv(g, "general.alignment");
    if (al) {
        if (al->type != G_T_U32 || al->v.u == 0 || al->v.u > 4096 ||
            (al->v.u & (al->v.u - 1)) != 0) {
            gerr(files[0], 0, "general.alignment is not a power of two in 1..4096");
            goto fail;
        }
        g->alignment = (int)al->v.u;
    }

    /* ---- pass 2: tensor infos ---- */
    Build b;
    memset(&b, 0, sizeof b);
    for (int i = 0; i < nf; i++) {
        Rdr r;
        if (rdr_init(&r, files[i], meta[i].fd, meta[i].fsize)) {
            gerr(files[i], 0, "out of memory");
            free(b.t); free(b.gt); free(b.noff);
            goto fail;
        }
        if (parse_tensor_infos(g, &r, &meta[i], i, &b, g->alignment)) {
            rdr_free(&r);
            free(b.t); free(b.gt); free(b.noff);
            goto fail;
        }
        rdr_free(&r);
    }

    uint64_t total = 0;
    for (int i = 0; i < nf; i++) total += meta[i].tensor_count;
    if ((int64_t)total != (int64_t)split_tensors) {
        gerr(files[0], 0, "shards hold %llu tensors but split.tensors.count is %d",
             (unsigned long long)total, split_tensors);
        free(b.t); free(b.gt); free(b.noff);
        goto fail;
    }

    if (b.n > INT_MAX) {
        gerr(files[0], 0, "too many tensors (%zu) for the int32 index", b.n);
        free(b.t); free(b.gt); free(b.noff);
        goto fail;
    }
    g->t = b.t;
    g->gtype = b.gt;
    g->nt = (int)b.n;
    for (size_t i = 0; i < b.n; i++) g->t[i].name = g->strpool + b.noff[i];
    free(b.noff);

    /* size_t arithmetic: nt can legitimately be a few million, and int32 would
     * wrap before the table reached the required 2x load-factor headroom. */
    size_t nb = 1024;
    while (nb < (size_t)g->nt * 2) nb <<= 1;
    g->nbucket = (int)nb;
    g->bucket = (int32_t *)malloc(nb * sizeof(int32_t));
    if (!g->bucket) { free(meta); k3_gguf_close(g); return -1; }
    memset(g->bucket, 0xFF, nb * sizeof(int32_t));

    for (int i = 0; i < g->nt; i++) {
        uint64_t h = fnv1a(g->t[i].name);
        int j = (int)(h & (uint64_t)(nb - 1));
        while (g->bucket[j] >= 0) {
            if (!strcmp(g->t[g->bucket[j]].name, g->t[i].name)) {
                gerr(g->path[g->t[i].shard], (uint64_t)g->t[i].off,
                     "duplicate tensor name %s (also in shard %d)",
                     g->t[i].name, g->t[g->bucket[j]].shard);
                free(meta);
                k3_gguf_close(g);
                return -1;
            }
            j = (j + 1) & (nb - 1);
        }
        g->bucket[j] = i;
    }

    free(meta);
    return 0;

fail:
    /* k3_gguf_close closes every fd via g->fd[] and frees every allocation; meta
     * is just the per-shard parse state, with fds aliased into g->fd[]. */
    free(meta);
    k3_gguf_close(g);
    return -1;
}

int k3_gguf_open(K3Gguf *g, const char *dir)
{
    memset(g, 0, sizeof *g);

    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "k3_gguf: cannot open directory %s\n", dir); return -1; }

    char **files = NULL;
    int nf = 0, cf = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n < 5 || strcmp(e->d_name + n - 5, ".gguf")) continue;
        if (nf == cf) {
            cf = cf ? cf * 2 : 16;
            files = (char **)realloc(files, (size_t)cf * sizeof *files);
        }
        size_t len = strlen(dir) + 1 + n + 1;
        files[nf] = (char *)malloc(len);
        if (!files[nf]) { fprintf(stderr, "k3_gguf: out of memory\n"); goto bad_files; }
        snprintf(files[nf], len, "%s/%s", dir, e->d_name);
        nf++;
    }
    closedir(d);

    if (nf == 0) {
        fprintf(stderr, "k3_gguf: no .gguf files in %s\n", dir);
        goto bad_files;
    }
    /* Sort so shard indices are stable: -NNNNN-of-MMMMM names sort in shard order. */
    qsort(files, nf, sizeof *files, cmp_str);
    return gguf_open_indexed(g, files, nf, 0);

bad_files:
    if (files) { for (int i = 0; i < nf; i++) free(files[i]); free(files); }
    return -1;
}

int k3_gguf_open_file(K3Gguf *g, const char *file)
{
    memset(g, 0, sizeof *g);
    char **files = (char **)malloc(sizeof *files);
    if (!files) { fprintf(stderr, "k3_gguf: out of memory\n"); return -1; }
    files[0] = (char *)malloc(strlen(file) + 1);
    if (!files[0]) {
        free(files);
        fprintf(stderr, "k3_gguf: out of memory\n");
        return -1;
    }
    strcpy(files[0], file);
    return gguf_open_indexed(g, files, 1, 1);
}

int k3_gguf_probe(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    if (S_ISDIR(st.st_mode)) {
        DIR *d = opendir(path);
        if (!d) return 0;
        int found = 0;
        struct dirent *e;
        while ((e = readdir(d))) {
            size_t n = strlen(e->d_name);
            if (n >= 5 && !strcmp(e->d_name + n - 5, ".gguf")) { found = 1; break; }
        }
        closedir(d);
        return found;
    }
    if (!S_ISREG(st.st_mode)) return 0;
    size_t n = strlen(path);
    return n >= 5 && !strcmp(path + n - 5, ".gguf");
}

void k3_gguf_close(K3Gguf *g)
{
    if (g->fd)
        for (int i = 0; i < g->nshard; i++)
            if (g->fd[i] >= 0) close(g->fd[i]);
    free(g->fd);
    if (g->dfd)
        for (int i = 0; i < g->nshard; i++)
            if (g->dfd[i] >= 0) close(g->dfd[i]);
    free(g->dfd);
    if (g->path) {
        for (int i = 0; i < g->nshard; i++) free(g->path[i]);
        free(g->path);
    }
    free(g->t);
    free(g->bucket);
    free(g->strpool);
    free(g->gtype);
    free(g->shard_kv);
    free(g->kv);
    free(g->arena);
    memset(g, 0, sizeof *g);
}

const K3Tensor *k3_gguf_find(const K3Gguf *g, const char *name)
{
    if (!g->bucket) return NULL;
    uint64_t h = fnv1a(name);
    int j = (int)(h & (uint64_t)(g->nbucket - 1));
    while (g->bucket[j] >= 0) {
        const K3Tensor *t = &g->t[g->bucket[j]];
        if (!strcmp(t->name, name)) return t;
        j = (j + 1) & (g->nbucket - 1);
    }
    return NULL;
}

int64_t k3_gguf_read(const K3Gguf *g, const K3Tensor *t, void *buf)
{
    int64_t got = 0;
    while (got < t->nbytes) {
        ssize_t r = pread(g->fd[t->shard], (char *)buf + got,
                          (size_t)(t->nbytes - got), (off_t)(t->off + got));
        if (r <= 0) {
            fprintf(stderr, "k3_gguf: short read on %s at +%lld\n",
                    t->name, (long long)got);
            return got;
        }
        got += r;
    }
    return got;
}

const K3GgufKV *k3_gguf_kv(const K3Gguf *g, const char *key)
{
    for (int i = 0; i < g->nkv; i++)
        if (!strcmp(g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}

/* ------------------------------------------------------------------ tokenizer */

/* Copy a string array's OFFSETS into one growing blob. The caller resolves the
 * offsets into pointers only after EVERY array is collected: the blob reallocs
 * as it grows, which MOVES it, so a pointer taken mid-walk dangles after the
 * next realloc (this bit the first version: the merges collection moved the
 * blob under the already-resolved tokens pointers). Every string must be
 * NUL-free (an embedded NUL would truncate a vocab key in the hash table and
 * silently change the tokenizer). Mirrors walk_value's bounds discipline:
 * length-capped, file-size-checked, misalignment fails. Returns a malloc'd
 * offset array (caller frees), or NULL on any violation. */
static size_t *tok_str_offsets(Rdr *r, size_t *nout, char **blob,
                               size_t *bn, size_t *bcap)
{
    uint32_t et;
    uint64_t n;
    if (rdr_u32(r, &et) || rdr_u64(r, &n)) return NULL;
    if (et != G_T_STRING) return NULL;
    if (n == 0 || n > (1u << 22)) return NULL;
    size_t *offs = (size_t *)malloc((size_t)n * sizeof(size_t));
    if (!offs) return NULL;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t sl;
        if (rdr_u64(r, &sl)) { free(offs); return NULL; }
        if (sl > r->fsize || sl > (1u << 24) || sl == 0) { free(offs); return NULL; }
        if (rdr_need(r, (size_t)sl)) { free(offs); return NULL; }
        if (memchr(r->buf + r->pos, 0, (size_t)sl)) { free(offs); return NULL; }
        size_t off = *bn;
        if (sl > SIZE_MAX - off - 1) { free(offs); return NULL; }
        if (off + sl + 1 > *bcap) {
            size_t nc = *bcap ? *bcap : (1 << 20);
            while (off + sl + 1 > nc) nc *= 2;
            char *nb = (char *)realloc(*blob, nc);
            if (!nb) { free(offs); return NULL; }
            *blob = nb;
            *bcap = nc;
        }
        memcpy(*blob + off, r->buf + r->pos, (size_t)sl);
        (*blob)[off + sl] = '\0';
        offs[i] = off;
        *bn = off + sl + 1;
        r->pos += (size_t)sl;
    }
    *nout = (size_t)n;
    return offs;
}

static char **tok_resolve(const size_t *offs, size_t n, const char *blob)
{
    char **arr = (char **)malloc(n * sizeof(char *));
    if (!arr) return NULL;
    for (size_t i = 0; i < n; i++) arr[i] = (char *)blob + offs[i];
    return arr;
}

int k3_gguf_tok(const K3Gguf *g, K3GgufTok *t)
{
    memset(t, 0, sizeof *t);
    if (g->nshard < 1 || !g->fd || g->fd[0] < 0 || !g->path) {
        fprintf(stderr, "k3_gguf: tokenizer extraction needs an open shard set\n");
        return -1;   /* r is not initialized yet: cannot reach tok_fail */
    }
    struct stat st;
    if (fstat(g->fd[0], &st) != 0 || st.st_size < 24) {
        fprintf(stderr, "k3_gguf: cannot size shard 1 for the tokenizer walk\n");
        return -1;
    }
    Rdr r;
    /* Declared before the first goto tok_fail: the rdr_init failure path below
     * jumps past later initializers, and tok_fail frees these unconditionally. */
    char *blob = NULL;
    size_t bn = 0, bcap = 0;
    size_t *offs_tokens = NULL, *offs_merges = NULL;
    size_t n_tokens = 0, n_merges = 0;
    if (rdr_init(&r, g->path[0], g->fd[0], (uint64_t)st.st_size)) {
        fprintf(stderr, "k3_gguf: out of memory on the tokenizer walk\n");
        goto tok_fail;
    }
    /* One cleanup for every failure after this point: k3_gguf_tok_free is
     * NULL-safe and releases ttype/tokens/merges/strblob (whichever were
     * allocated); blob is freed directly until it is adopted as t->strblob. */
    /* The open already validated magic and version; re-checking is one pread and
     * keeps this walk self-contained. */
    if (rdr_need(&r, 24) || memcmp(r.buf, "GGUF", 4) != 0) {
        fprintf(stderr, "k3_gguf: %s: bad magic on the tokenizer walk\n", g->path[0]);
        goto tok_fail;
    }
    r.pos = 24;   /* magic, version, tensor_count, kv_count */

    const uint64_t nkv = (uint64_t)g->shard_kv[0];
    for (uint64_t i = 0; i < nkv; i++) {
        char key[64];
        size_t klen = 0;
        if (rdr_str(&r, key, sizeof key, &klen, 1)) {
            fprintf(stderr, "k3_gguf: %s @+%llu: tokenizer walk: bad key %llu of %llu\n",
                    g->path[0], (unsigned long long)r.pos,
                    (unsigned long long)i, (unsigned long long)nkv);
            goto tok_fail;
        }
        uint32_t vt;
        if (rdr_u32(&r, &vt)) {
            fprintf(stderr, "k3_gguf: %s: truncated value type on key '%s'\n",
                    g->path[0], key);
            goto tok_fail;
        }
        if (vt > G_T_F64) {
            fprintf(stderr, "k3_gguf: %s: unknown value type %u on key '%s'\n",
                    g->path[0], vt, key);
            goto tok_fail;
        }
        int bad = 0;
        if (!strcmp(key, "tokenizer.ggml.tokens")) {
            if (offs_tokens) bad = 1;
            else offs_tokens = tok_str_offsets(&r, &n_tokens, &blob, &bn, &bcap);
            if (!offs_tokens) bad = 1;
        } else if (!strcmp(key, "tokenizer.ggml.merges")) {
            if (offs_merges) bad = 1;
            else offs_merges = tok_str_offsets(&r, &n_merges, &blob, &bn, &bcap);
            if (!offs_merges) bad = 1;
        } else if (!strcmp(key, "tokenizer.ggml.token_type")) {
            /* The D7 quirk: elements are declared as type 5 (int32) and are 4
             * bytes each on the wire. Verified byte-level on the real file. */
            uint32_t et;
            uint64_t n;
            if (t->ttype || rdr_u32(&r, &et) || rdr_u64(&r, &n) || et != G_T_I32 ||
                n == 0 || n > (1u << 22) || n > r.fsize / 4)
                bad = 1;
            else {
                if (rdr_need(&r, (size_t)n * 4)) { bad = 1; }
                else {
                    t->ttype = (int32_t *)malloc((size_t)n * sizeof(int32_t));
                    if (!t->ttype) { bad = 1; }
                    else {
                        memcpy(t->ttype, r.buf + r.pos, (size_t)n * 4);
                        t->nttype = (int)n;
                        r.pos += (size_t)n * 4;
                    }
                }
            }
        } else {
            Val v;
            if (walk_value(&r, vt, &v, 0)) bad = 1;
        }
        if (bad) {
            fprintf(stderr, "k3_gguf: %s @+%llu: bad value of type %u on key '%s'\n",
                    g->path[0], (unsigned long long)r.pos, vt, key);
            goto tok_fail;
        }
        if (!boundary_ok(&r)) {
            fprintf(stderr, "k3_gguf: %s @+%llu: tokenizer walk misaligned after key "
                            "'%s'\n", g->path[0], (unsigned long long)r.pos, key);
            goto tok_fail;
        }
    }
    rdr_free(&r);
    r.buf = NULL;   /* the cleanup's rdr_free must not see the freed buffer */
    if (!offs_tokens || !offs_merges || !t->ttype) {
        fprintf(stderr, "k3_gguf: %s: tokenizer section is missing tokens, merges or "
                        "token_type\n", g->path[0]);
        goto tok_fail;
    }
    t->ntok = (int)n_tokens;
    t->nmerges = (int)n_merges;
    if (t->nttype != t->ntok) {
        fprintf(stderr, "k3_gguf: %s: token_type has %d entries but tokens has %d\n",
                g->path[0], t->nttype, t->ntok);
        goto tok_fail;
    }
    /* Resolve AFTER every collection: the blob is stable now, so these pointers
     * survive (the blob moves during collection, which is why the offsets were
     * kept instead). */
    t->tokens = tok_resolve(offs_tokens, n_tokens, blob);
    t->merges = tok_resolve(offs_merges, n_merges, blob);
    free(offs_tokens);
    free(offs_merges);
    offs_tokens = NULL;
    offs_merges = NULL;   /* the cleanup must not free these again */
    if (!t->tokens || !t->merges) {
        fprintf(stderr, "k3_gguf: %s: out of memory resolving tokenizer arrays\n",
                g->path[0]);
        goto tok_fail;
    }
    t->strblob = blob;   /* owned by the caller from here on */

    /* Scalar keys come from the already-validated open-time table. */
    const K3GgufKV *k;
    k = k3_gguf_kv(g, "tokenizer.ggml.model");
    if (!k || k->type != G_T_STRING || k->v.s.n == 0 || k->v.s.n >= sizeof t->model) {
        fprintf(stderr, "k3_gguf: %s: tokenizer.ggml.model missing or mistyped\n",
                g->path[0]);
        goto tok_fail;
    }
    memcpy(t->model, k->v.s.p, k->v.s.n);
    t->model[k->v.s.n] = '\0';
    k = k3_gguf_kv(g, "tokenizer.ggml.pre");
    if (!k || k->type != G_T_STRING || k->v.s.n == 0 || k->v.s.n >= sizeof t->pre) {
        fprintf(stderr, "k3_gguf: %s: tokenizer.ggml.pre missing or mistyped\n",
                g->path[0]);
        goto tok_fail;
    }
    memcpy(t->pre, k->v.s.p, k->v.s.n);
    t->pre[k->v.s.n] = '\0';
    k = k3_gguf_kv(g, "kimi-k3.vocab_size");
    if (!k || k->type != G_T_U32) {
        fprintf(stderr, "k3_gguf: %s: kimi-k3.vocab_size missing or mistyped\n",
                g->path[0]);
        goto tok_fail;
    }
    t->vocab_size = (uint32_t)k->v.u;
    if (t->vocab_size != (uint32_t)t->ntok) {
        fprintf(stderr, "k3_gguf: %s: kimi-k3.vocab_size %u != %d tokens\n",
                g->path[0], t->vocab_size, t->ntok);
        goto tok_fail;
    }
    k = k3_gguf_kv(g, "tokenizer.ggml.bos_token_id");
    if (k && k->type == G_T_U32) { t->bos_id = (uint32_t)k->v.u; t->have_bos = 1; }
    k = k3_gguf_kv(g, "tokenizer.ggml.eos_token_id");
    if (k && k->type == G_T_U32) { t->eos_id = (uint32_t)k->v.u; t->have_eos = 1; }
    k = k3_gguf_kv(g, "tokenizer.ggml.padding_token_id");
    if (k && k->type == G_T_U32) { t->pad_id = (uint32_t)k->v.u; t->have_pad = 1; }

    /* The blob is owned by the caller (the tokenizer bootstrap adopts it; the
     * strings it references live for the life of the process). */
    return 0;

tok_fail:
    free(offs_tokens);
    free(offs_merges);
    if (t->strblob == NULL) free(blob);   /* not yet adopted: release it raw */
    rdr_free(&r);
    k3_gguf_tok_free(t);   /* NULL-safe; frees ttype/tokens/merges/strblob */
    return -1;
}

void k3_gguf_tok_free(K3GgufTok *t)
{
    free(t->tokens);
    free(t->merges);
    free(t->ttype);
    free(t->strblob);
    memset(t, 0, sizeof *t);
}

/* ------------------------------------------------------------------ config */

/* The complete verified kimi-k3.* inventory (recon-gguf-spec.md §5). Every key the
 * file carries is checked against this: an unknown one fails the config load, so a
 * renamed or retyped field surfaces loudly instead of silently changing the model. */
static const char *const k3_gguf_known_keys[] = {
    "kimi-k3.block_count",                  "kimi-k3.context_length",
    "kimi-k3.embedding_length",             "kimi-k3.feed_forward_length",
    "kimi-k3.attention.head_count",         "kimi-k3.attention.head_count_kv",
    "kimi-k3.rope.freq_base",               "kimi-k3.attention.layer_norm_rms_epsilon",
    "kimi-k3.expert_count",                 "kimi-k3.expert_used_count",
    "kimi-k3.expert_group_used_count",      "kimi-k3.expert_gating_func",
    "kimi-k3.attention.key_length",         "kimi-k3.attention.value_length",
    "kimi-k3.vocab_size",                   "kimi-k3.ssm.conv_kernel",
    "kimi-k3.kda.head_dim",                 "kimi-k3.kda.gate_lower_bound",
    "kimi-k3.attention.q_lora_rank",        "kimi-k3.attention.kv_lora_rank",
    "kimi-k3.rope.dimension_count",         "kimi-k3.attention.key_length_mla",
    "kimi-k3.attention.value_length_mla",   "kimi-k3.expert_feed_forward_length",
    "kimi-k3.expert_shared_count",          "kimi-k3.leading_dense_block_count",
    "kimi-k3.expert_weights_scale",         "kimi-k3.expert_weights_norm",
    "kimi-k3.expert_latent_length",         "kimi-k3.activation.situ_beta",
    "kimi-k3.activation.situ_linear_beta",  "kimi-k3.attn_res.block_size",
};

typedef struct {
    const K3Gguf  *g;
    const char    *missing[32];
    int            nmissing;
    const char    *unknown[8];
    int            nunknown;
} CfgSrc;

static int cfg_u32(CfgSrc *s, const char *key, uint32_t *out)
{
    const K3GgufKV *k = k3_gguf_kv(s->g, key);
    /* A value beyond INT32_MAX cannot live in a K3Cfg int field; a lying header
     * must fail loud instead of wrapping negative. */
    if (!k || k->type != G_T_U32 || k->v.u > INT32_MAX) {
        if (s->nmissing < (int)(sizeof s->missing / sizeof s->missing[0]))
            s->missing[s->nmissing] = key;
        s->nmissing++;
        return 0;
    }
    if (out) *out = (uint32_t)k->v.u;
    return 1;
}

static int cfg_f32(CfgSrc *s, const char *key, float *out)
{
    const K3GgufKV *k = k3_gguf_kv(s->g, key);
    if (!k || k->type != G_T_F32) {
        if (s->nmissing < (int)(sizeof s->missing / sizeof s->missing[0]))
            s->missing[s->nmissing] = key;
        s->nmissing++;
        return 0;
    }
    if (out) *out = (float)k->v.f;
    return 1;
}

static int cfg_bool(CfgSrc *s, const char *key, int *out)
{
    const K3GgufKV *k = k3_gguf_kv(s->g, key);
    if (!k || k->type != G_T_BOOL) {
        if (s->nmissing < (int)(sizeof s->missing / sizeof s->missing[0]))
            s->missing[s->nmissing] = key;
        s->nmissing++;
        return 0;
    }
    if (out) *out = (int)k->v.u;
    return 1;
}

int k3_gguf_cfg(const K3Gguf *g, K3Cfg *c, int *fa, int fa_max)
{
    memset(c, 0, sizeof *c);
    if (!g->kv)
        return gerr(g->path ? g->path[0] : "?", 0,
                    "no shard-1 metadata table; cannot derive a config");

    CfgSrc s;
    memset(&s, 0, sizeof s);
    s.g = g;

    /* ---- required, mapped to K3Cfg (D5) ----
     * One local per key: a missing key leaves ITS local zero, never a stale value
     * from a neighbouring key. */
    uint32_t u_hidden = 0, u_layers = 0, u_vocab = 0, u_heads = 0, u_hdim = 0;
    uint32_t u_conv = 0, u_qlora = 0, u_kvlora = 0, u_rope = 0, u_kmla = 0;
    uint32_t u_vmla = 0, u_experts = 0, u_topk = 0, u_shared = 0, u_latent = 0;
    uint32_t u_moeinter = 0, u_dense = 0, u_dinter = 0, u_attnres = 0;
    float f_eps = 0.0f, f_lb = 0.0f, f_rscale = 0.0f, f_b1 = 0.0f, f_b2 = 0.0f;
    int b_renorm = 0;

    cfg_u32(&s, "kimi-k3.embedding_length", &u_hidden);      c->hidden = (int)u_hidden;
    cfg_u32(&s, "kimi-k3.block_count", &u_layers);           c->n_layers = (int)u_layers;
    cfg_u32(&s, "kimi-k3.vocab_size", &u_vocab);             c->vocab = (int)u_vocab;
    cfg_f32(&s, "kimi-k3.attention.layer_norm_rms_epsilon", &f_eps);
    c->rms_eps = f_eps;

    cfg_u32(&s, "kimi-k3.attention.head_count", &u_heads);
    c->kda_heads = (int)u_heads;
    cfg_u32(&s, "kimi-k3.kda.head_dim", &u_hdim);
    c->kda_head_dim = (int)u_hdim;
    cfg_u32(&s, "kimi-k3.ssm.conv_kernel", &u_conv);
    c->conv_k = (int)u_conv;
    cfg_f32(&s, "kimi-k3.kda.gate_lower_bound", &f_lb);
    c->gate_lb = f_lb;

    c->n_heads = c->kda_heads;      /* one head count in this architecture */
    cfg_u32(&s, "kimi-k3.attention.q_lora_rank", &u_qlora);  c->q_lora = (int)u_qlora;
    cfg_u32(&s, "kimi-k3.attention.kv_lora_rank", &u_kvlora); c->kv_lora = (int)u_kvlora;
    cfg_u32(&s, "kimi-k3.rope.dimension_count", &u_rope);    c->qk_rope = (int)u_rope;
    cfg_u32(&s, "kimi-k3.attention.key_length_mla", &u_kmla);
    /* Derived, deliberately: key_length_mla is the FULL per-head MLA width
     * (192 = qk_nope 128 + qk_rope 64), so qk_nope = key_length_mla - rope width. */
    c->qk_nope = (int)u_kmla - c->qk_rope;
    cfg_u32(&s, "kimi-k3.attention.value_length_mla", &u_vmla);
    c->v_head = (int)u_vmla;
    /* No GGUF key exists for the MLA output gate. Constant from the released
     * config.json (mla_use_output_gate=true); the file's attn_gate tensors are
     * consistent with it but nothing here verifies that. */
    c->mla_out_gate = 1;

    cfg_u32(&s, "kimi-k3.expert_count", &u_experts);
    c->n_experts = (int)u_experts;
    cfg_u32(&s, "kimi-k3.expert_used_count", &u_topk);
    c->topk = (int)u_topk;
    cfg_u32(&s, "kimi-k3.expert_shared_count", &u_shared);
    c->n_shared = (int)u_shared;
    cfg_u32(&s, "kimi-k3.expert_latent_length", &u_latent);  c->latent = (int)u_latent;
    cfg_u32(&s, "kimi-k3.expert_feed_forward_length", &u_moeinter);
    c->moe_inter = (int)u_moeinter;
    cfg_f32(&s, "kimi-k3.expert_weights_scale", &f_rscale);  c->routed_scale = f_rscale;
    cfg_bool(&s, "kimi-k3.expert_weights_norm", &b_renorm);  c->moe_renorm = b_renorm;
    /* No GGUF key exists for latent-norm. Constant from the released config.json
     * (latent_moe_use_norm=true); nothing in the file verifies it. */
    c->latent_norm = 1;

    cfg_u32(&s, "kimi-k3.leading_dense_block_count", &u_dense);
    c->first_dense = (int)u_dense;
    cfg_u32(&s, "kimi-k3.feed_forward_length", &u_dinter);
    c->dense_inter = (int)u_dinter;
    cfg_u32(&s, "kimi-k3.attn_res.block_size", &u_attnres);
    c->attn_res_block = (int)u_attnres;
    cfg_f32(&s, "kimi-k3.activation.situ_beta", &f_b1);
    c->situ_b1 = f_b1;
    cfg_f32(&s, "kimi-k3.activation.situ_linear_beta", &f_b2);
    c->situ_b2 = f_b2;

    /* ---- present in the verified inventory, validated, not consumed ---- */
    cfg_u32(&s, "kimi-k3.context_length", NULL);
    cfg_u32(&s, "kimi-k3.attention.key_length", NULL);
    cfg_u32(&s, "kimi-k3.attention.value_length", NULL);
    cfg_u32(&s, "kimi-k3.expert_gating_func", NULL);
    cfg_u32(&s, "kimi-k3.expert_group_used_count", NULL);
    cfg_f32(&s, "kimi-k3.rope.freq_base", NULL);

    /* ---- every kimi-k3.* key in the file must be in the verified inventory ---- */
    for (int i = 0; i < g->nkv; i++) {
        if (strncmp(g->kv[i].key, "kimi-k3.", 8)) continue;
        int known = 0;
        for (size_t j = 0;
             j < sizeof k3_gguf_known_keys / sizeof *k3_gguf_known_keys; j++)
            if (!strcmp(g->kv[i].key, k3_gguf_known_keys[j])) {
                known = 1;
                break;
            }
        if (!known && s.nunknown < (int)(sizeof s.unknown / sizeof s.unknown[0]))
            s.unknown[s.nunknown++] = g->kv[i].key;
    }

    if (s.nmissing || s.nunknown) {
        if (s.nmissing) {
            fprintf(stderr, "k3_gguf: shard-1 metadata is missing %d required key(s):\n",
                    s.nmissing);
            int shown = s.nmissing;
            int cap = (int)(sizeof s.missing / sizeof s.missing[0]);
            if (shown > cap) shown = cap;
            for (int i = 0; i < shown; i++) fprintf(stderr, "    %s\n", s.missing[i]);
            if (s.nmissing > shown)
                fprintf(stderr, "    ... and %d more\n", s.nmissing - shown);
        }
        if (s.nunknown) {
            fprintf(stderr,
                    "k3_gguf: shard-1 metadata carries %d unknown kimi-k3.* "
                    "key(s):\n",
                    s.nunknown);
            int shown = s.nunknown;
            int cap = (int)(sizeof s.unknown / sizeof s.unknown[0]);
            if (shown > cap) shown = cap;
            for (int i = 0; i < shown; i++)
                fprintf(stderr, "    %s\n", s.unknown[i]);
            if (s.nunknown > shown)
                fprintf(stderr, "    ... and %d more\n", s.nunknown - shown);
        }
        fprintf(stderr, "  refusing to substitute defaults: a config this reader cannot\n"
                        "  fully understand would silently produce a DIFFERENT model.\n");
        return 0;
    }

    /* ---- layer map from the per-layer KV array (VERIFIED quirk: type-5 array
     * elements are 4 bytes each) ---- */
    const K3GgufKV *hck = k3_gguf_kv(g, "kimi-k3.attention.head_count_kv");
    if (!hck || hck->type != G_T_ARRAY || hck->v.a.et != G_T_I32 || !hck->v.a.d ||
        hck->v.a.n != (uint32_t)c->n_layers) {
        fprintf(stderr, "k3_gguf: kimi-k3.attention.head_count_kv must be an array of "
                        "%d int32 values (one per layer)\n", c->n_layers);
        return 0;
    }
    const uint32_t *hc = (const uint32_t *)hck->v.a.d;
    for (uint32_t i = 0; i < hck->v.a.n; i++) {
        if (hc[i] > 1) {
            fprintf(stderr, "k3_gguf: kimi-k3.attention.head_count_kv[%u] = %u, "
                            "expected 0 or 1\n", i, hc[i]);
            return 0;
        }
        if (hc[i]) {
            if (c->n_full_attn >= fa_max) {
                fprintf(stderr, "k3_gguf: more MLA layers than the %d-entry buffer\n",
                        fa_max);
                return 0;
            }
            fa[c->n_full_attn++] = (int)i + 1;      /* ONE-based, as k3_cfg_load */
        }
    }
    /* THE pointer k3_is_mla() dereferences. k3_cfg_load assigns it the same way;
     * leaving it NULL makes every layer-classification call crash. */
    c->full_attn = fa;

    /* ---- structural checks, mirroring k3_cfg_load ---- */
    if (c->n_layers <= 0 || c->hidden <= 0 || c->vocab <= 0) {
        fprintf(stderr, "k3_gguf: non-positive layers/hidden/vocab\n");
        return 0;
    }
    if (c->n_full_attn == 0) {
        fprintf(stderr, "k3_gguf: no MLA layers in head_count_kv; every layer would "
                        "run KDA\n");
        return 0;
    }
    if (c->n_full_attn >= c->n_layers) {
        fprintf(stderr, "k3_gguf: %d of %d layers are MLA, leaving no KDA layers\n",
                c->n_full_attn, c->n_layers);
        return 0;
    }
    if (c->topk <= 0 || c->topk > K3_MAX_TOPK) {
        fprintf(stderr, "k3_gguf: selects top-%d, this build supports 1..%d "
                        "(K3_MAX_TOPK)\n", c->topk, K3_MAX_TOPK);
        return 0;
    }
    if (c->topk > c->n_experts) {
        fprintf(stderr, "k3_gguf: selects %d of %d experts\n", c->topk, c->n_experts);
        return 0;
    }
    if (c->n_experts <= 0) {
        fprintf(stderr, "k3_gguf: expert_count %d\n", c->n_experts);
        return 0;
    }
    if (c->attn_res_block <= 0) {
        fprintf(stderr, "k3_gguf: attn_res_block_size %d; layer_idx %% 0 would divide "
                        "by zero\n", c->attn_res_block);
        return 0;
    }
    if (c->conv_k < 1) {
        fprintf(stderr, "k3_gguf: short_conv_kernel_size %d\n", c->conv_k);
        return 0;
    }
    if (c->qk_nope <= 0) {
        fprintf(stderr, "k3_gguf: derived qk_nope %d (key_length_mla %d - rope "
                        "dimension_count %d) is not positive\n",
                c->qk_nope, c->qk_nope + c->qk_rope, c->qk_rope);
        return 0;
    }
    for (int i = 0; i < c->n_full_attn; i++)
        if (fa[i] < 1 || fa[i] > c->n_layers) {
            fprintf(stderr, "k3_gguf: full_attn_layers[%d] = %d is outside 1..%d\n",
                    i, fa[i], c->n_layers);
            return 0;
        }

    printf("config: %s (shard-1 GGUF kv) | hidden=%d layers=%d vocab=%d | %d MLA + "
           "%d KDA | experts %d top%d shared%d | latent=%d\n",
           g->path ? g->path[0] : "?", c->hidden, c->n_layers, c->vocab,
           c->n_full_attn, c->n_layers - c->n_full_attn,
           c->n_experts, c->topk, c->n_shared, c->latent);
    return 1;
}
