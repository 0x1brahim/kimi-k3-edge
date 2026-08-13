/* k3_gguf_expert.c - see k3_gguf_expert.h. */
#define _POSIX_C_SOURCE 200809L

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3_gguf_expert.h"
#include "k3_gguf_dequant.h"
#include "k3_gguf_map.h"
#include "k3_mxfp4_quant.h"
#include "k3_st.h"

#define K3_GGUF_SLOT_EMPTY (-1)
#define K3_GGUF_SLOT_INFLIGHT (-2)

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* A true K3St view of the GGUF index: the D1a seam, identical to k3_gguf_map.c. */
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

static int xfail(const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "k3_gguf_expert: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    return -1;
}

/* ------------------------------------------------------------------ geometry */

/* Resolve the three merged tensors of the first MoE layer and fill the geometry
 * (which is identical for every expert of every layer). Header-only: every MoE
 * layer's three tensors are validated, so a malformed one fails the run before
 * any token, never mid-decode. */
static int resolve_geometry(K3GgufExpertSrc *s)
{
    const K3Cfg *c = s->c;

    /* rows/cols per matrix, engine layout: w1/w3 [moe_inter][latent],
     * w2 [latent][moe_inter]. */
    s->rows[0] = s->rows[2] = c->moe_inter;
    s->cols[0] = s->cols[2] = c->latent;
    s->rows[1]              = c->latent;
    s->cols[1]              = c->moe_inter;

    int probe = -1;
    for (int L = 0; L < c->n_layers; L++) {
        if (k3_is_dense(c, L)) continue;
        probe = L;
        break;
    }
    if (probe < 0) return xfail("no MoE layer in the config");

    for (int L = 0; L < c->n_layers; L++) {
        if (k3_is_dense(c, L)) continue;
        char gate[64], up[64], down[64];
        k3_gguf_expert_names(L, gate, up, down);
        const char *name[3] = {gate, down, up}; /* w1, w2, w3 */
        for (int m = 0; m < 3; m++) {
            const K3Tensor *t = k3_gguf_find(s->g, name[m]);
            if (!t) return xfail("layer %d needs %s, absent from the index", L, name[m]);
            const int64_t want[4] = {s->cols[m], s->rows[m], c->n_experts, 1};
            if (t->ndim != 3 || t->shape[0] != want[0] || t->shape[1] != want[1] ||
                t->shape[2] != want[2])
                return xfail("%s: shape [%lld %lld %lld], contract says "
                             "[%lld %lld %lld]",
                             name[m], (long long)t->shape[0], (long long)t->shape[1],
                             (long long)t->shape[2], (long long)want[0],
                             (long long)want[1], (long long)want[2]);
            if (s->g->gtype[t - s->g->t] != K3_GGUF_TYPE_IQ1_S)
                return xfail("%s: ggml type %d, contract says IQ1_S", name[m],
                             s->g->gtype[t - s->g->t]);
            const int64_t nb = k3_gguf_dequant_nbytes(K3_GGUF_TYPE_IQ1_S, 3, want);
            if (nb != t->nbytes)
                return xfail("%s: %lld bytes on disk, the shape formula gives %lld",
                             name[m], (long long)t->nbytes, (long long)nb);
            if (t->nbytes % t->shape[2] != 0)
                return xfail("%s: %lld bytes does not divide evenly into %lld "
                             "experts",
                             name[m], (long long)t->nbytes, (long long)t->shape[2]);
        }
    }
    for (int m = 0; m < 3; m++) {
        /* Per-expert IQ1_S window: ne2 is the slowest dim, so expert e owns the
         * contiguous span [e*per, (e+1)*per). The 2-D sub-tensor dequant below
         * re-validates the formula on every admit. */
        s->p_bytes[m] = (int64_t)s->rows[m] * (s->cols[m] / 2);
        s->s_bytes[m] = (int64_t)s->rows[m] * (s->cols[m] / K3_MXFP4_GROUP);
        if (s->cols[m] % K3_MXFP4_GROUP != 0 || s->cols[m] % 2 != 0)
            return xfail("expert width %d is not MXFP4-compatible (group 32)",
                         s->cols[m]);
    }
    /* Canonical run order inside a slot: w1.p w1.s w2.p w2.s w3.p w3.s
     * (k3_load.h's measured layout; K3ExpertQ maps p1/s1=w1, p2/s2=w2, p3/s3=w3). */
    int64_t o = 0;
    for (int m = 0; m < 3; m++) {
        s->p_off[m] = o;
        o += s->p_bytes[m];
        s->s_off[m] = o;
        o += s->s_bytes[m];
    }
    s->slot_bytes = o;
    return 0;
}

/* ------------------------------------------------------------------ admit */

static void fill_q(const K3GgufExpertSrc *s, int slot, K3ExpertQ *q)
{
    const unsigned char *b = s->arena + (size_t)slot * s->slot_bytes;
    q->p1                  = b + s->p_off[0];
    q->s1                  = b + s->s_off[0];
    q->p2                  = b + s->p_off[1];
    q->s2                  = b + s->s_off[1];
    q->p3                  = b + s->p_off[2];
    q->s3                  = b + s->s_off[2];
}

static int pick_victim(K3GgufExpertSrc *s)
{
    int best        = -1;
    uint64_t oldest = (uint64_t)-1;
    for (int i = 0; i < s->nslot; i++) {
        if (s->key_of[i] == K3_GGUF_SLOT_INFLIGHT) continue;
        if (s->key_of[i] == K3_GGUF_SLOT_EMPTY) return i;
        if (s->used_at[i] < oldest) {
            oldest = s->used_at[i];
            best   = i;
        }
    }
    return best;
}

/* The whole admit: dequant one expert's three IQ1_S windows to fp32 and requant
 * to MXFP4 into the reserved slot. The layer's OWN tensors are resolved per
 * admit (the geometry is validated for every layer at init; a dense layer has
 * no expert tensors and fails here, loudly). Returns 0 on success. */
static int fill_slot(K3GgufExpertSrc *s, int layer, int expert, int slot)
{
    if (k3_is_dense(s->c, layer))
        return xfail("layer %d is a dense layer; it has no routed experts", layer);
    char gate[64], up[64], down[64];
    k3_gguf_expert_names(layer, gate, up, down);
    const char *name[3] = {gate, down, up}; /* w1, w2, w3 */
    const K3Tensor *t[3];
    for (int m = 0; m < 3; m++) {
        t[m] = k3_gguf_find(s->g, name[m]);
        if (!t[m])
            return xfail("layer %d needs %s, absent from the index", layer, name[m]);
    }

    const K3St sv = st_view(s->g);
    for (int m = 0; m < 3; m++) {
        /* The window: expert e of the merged tensor. The full-tensor contract
         * was validated at init; the 2-D sub-view re-checks the per-window
         * nbytes against the block formula on every admit (fail loud on a
         * corrupt shard, never a silent garbage expert). */
        const int64_t pe = t[m]->nbytes / t[m]->shape[2];
        if ((size_t)pe > s->raw_cap) {
            size_t nc         = (size_t)pe;
            unsigned char *np = (unsigned char *)realloc(s->raw, nc);
            if (!np) return xfail("out of memory for the %zu-byte expert window", nc);
            s->raw     = np;
            s->raw_cap = nc;
        }
        K3Tensor win = *t[m];
        win.name     = (char *)"expert window";
        win.off      = t[m]->off + expert * pe;
        win.nbytes   = pe;
        win.ndim     = 2;
        win.shape[0] = s->cols[m];
        win.shape[1] = s->rows[m];
        win.shape[2] = win.shape[3] = 1;
        const int64_t got           = k3_st_read(&sv, &win, s->raw);
        if (got != pe)
            return xfail("short read of the %s window of L%d expert %d (%lld of "
                         "%lld bytes)",
                         t[m]->name, layer, expert, (long long)got, (long long)pe);

        const int64_t ne[4] = {s->cols[m], s->rows[m], 1, 1};
        K3GgufTensor gt;
        memset(&gt, 0, sizeof gt);
        gt.ggml_type = K3_GGUF_TYPE_IQ1_S;
        gt.ndim      = 2;
        memcpy(gt.ne, ne, sizeof ne);
        gt.nbytes = pe;
        gt.data   = s->raw;
        K3GgufExpect ex;
        memset(&ex, 0, sizeof ex);
        ex.ndim = 2;
        memcpy(ex.ne, ne, sizeof ne);
        /* The dequant PADS every row to the 256-block: ceil(cols/256)*256 values
         * per row, not cols. Dequant into the padded scratch, then compact each
         * row so the requant sees a contiguous [rows][cols] matrix. (The real
         * file's widths are exact block multiples, where the two coincide; the
         * compact step is what keeps non-multiple widths correct instead of
         * silently reading the padding as weights.) */
        const int64_t prow  = (s->cols[m] / K3_GGUF_IQ1_S_QK + 1) * K3_GGUF_IQ1_S_QK;
        const size_t padded = (size_t)prow * s->rows[m];
        if (padded * sizeof(float) > s->f32_cap) {
            size_t nc = padded * sizeof(float);
            float *np = (float *)realloc(s->f32, nc);
            if (!np)
                return xfail("out of memory for the %zu-float padded scratch",
                             (size_t)padded);
            s->f32     = np;
            s->f32_cap = nc;
        }
        if (k3_gguf_dequant(&gt, K3_GGUF_DEQ_F32, s->f32, &ex) != 0) return -1;
        if (prow != s->cols[m]) {
            /* Every move is DOWNWARD (dest = r*cols < src = r*prow), so the
             * copies must run ASCENDING: r=4's destination lands exactly on
             * r=1's source (4*cols == prow), and a descending loop would
             * destroy it before it is read. */
            const size_t rowb = (size_t)s->cols[m] * sizeof(float);
            for (int r = 0; r < s->rows[m]; r++)
                memmove((unsigned char *)s->f32 + (size_t)r * rowb,
                        (unsigned char *)s->f32 + (size_t)r * prow * sizeof(float), rowb);
        }

        unsigned char *pk = s->arena + (size_t)slot * s->slot_bytes + s->p_off[m];
        unsigned char *sc = s->arena + (size_t)slot * s->slot_bytes + s->s_off[m];
        if (k3_mxfp4_quant(pk, sc, s->f32, s->rows[m], s->cols[m]) != 0) return -1;
    }
    s->bytes_read +=
        (uint64_t)(t[0]->nbytes / t[0]->shape[2] + t[1]->nbytes / t[1]->shape[2] +
                   t[2]->nbytes / t[2]->shape[2]);
    return 0;
}

/* Bring (layer, expert) resident and return its slot, or -1. */
static int admit(K3GgufExpertSrc *s, int layer, int expert)
{
    const int32_t key = layer * s->c->n_experts + expert;
    int slot          = s->slot_of[key];
    if (slot >= 0) {
        s->hits++;
        s->used_at[slot] = ++s->clock;
        return slot;
    }
    s->misses++;

    slot = pick_victim(s);
    if (slot < 0) {
        fprintf(stderr,
                "k3_gguf_expert: every slot is busy, cannot admit L%d "
                "expert %d\n",
                layer, expert);
        return -1;
    }
    if (s->key_of[slot] >= 0) {
        s->slot_of[s->key_of[slot]] = -1;
        s->evictions++;
    }
    /* INFLIGHT: the slot is reserved but does not yet claim its key, so a
     * concurrent getmany can neither hand it out nor evict it. */
    s->key_of[slot] = K3_GGUF_SLOT_INFLIGHT;

    const double t0 = now_s();
    const int rc    = fill_slot(s, layer, expert, slot);
    s->load_seconds += now_s() - t0;
    if (rc != 0) {
        s->key_of[slot] = K3_GGUF_SLOT_EMPTY;
        return -1;
    }
    s->key_of[slot]  = key;
    s->slot_of[key]  = slot;
    s->used_at[slot] = ++s->clock;
    return slot;
}

/* ------------------------------------------------------------------ vtable */

static int gsrc_get(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3GgufExpertSrc *s = (K3GgufExpertSrc *)self; /* src is the first member */
    if (layer < 0 || layer >= s->c->n_layers || expert < 0 || expert >= s->c->n_experts) {
        fprintf(stderr, "k3_gguf_expert: out of range L%d expert %d\n", layer, expert);
        return -1;
    }
    const int slot = admit(s, layer, expert);
    if (slot < 0) return -1;
    fill_q(s, slot, out);
    return 0;
}

static int gsrc_getmany(K3ExpertSrc *self, int layer, const int *experts, int n)
{
    K3GgufExpertSrc *s = (K3GgufExpertSrc *)self;
    if (n <= 0) return 0;
    /* Same boundary as gsrc_get: an out-of-range layer would index slot_of[]
     * out of bounds in phase 1. k3_moe only ever asks for config-validated
     * layers, but the API boundary must fail loud, not corrupt. */
    if (layer < 0 || layer >= s->c->n_layers) {
        fprintf(stderr, "k3_gguf_expert: getmany out of range layer %d\n", layer);
        return -1;
    }
    if (n > K3_MAX_TOPK) n = K3_MAX_TOPK; /* the batch array is that big */

    /* Phase 1 (serial): resolve the misses and reserve each a slot. The whole
     * batch is reserved before any fill, so the fills can never collide. */
    typedef struct {
        int slot;
        int expert;
    } Work;
    Work w[K3_MAX_TOPK];
    int nw = 0;
    for (int i = 0; i < n && nw < (int)(sizeof w / sizeof *w); i++) {
        const int e = experts[i];
        if (e < 0 || e >= s->c->n_experts) continue;
        const int32_t key = layer * s->c->n_experts + e;
        if (s->slot_of[key] >= 0) continue; /* already resident */
        int dup = 0;
        for (int j = 0; j < nw; j++)
            if (w[j].expert == e) {
                dup = 1;
                break;
            }
        if (dup) continue;
        const int slot = pick_victim(s);
        if (slot < 0) break;
        if (s->key_of[slot] >= 0) {
            s->slot_of[s->key_of[slot]] = -1;
            s->evictions++;
        }
        s->key_of[slot]  = K3_GGUF_SLOT_INFLIGHT;
        s->used_at[slot] = ++s->clock;
        w[nw].slot       = slot;
        w[nw].expert     = e;
        nw++;
    }
    /* Phase 2+3 (serial fills, publish after each): the dequant+requant is
     * CPU-bound and the reads are page-cache hits, so parallel fills would only
     * multiply the fp32 transient; the INFLIGHT discipline above is still the
     * real state machine. */
    int ok = 0;
    for (int i = 0; i < nw; i++) {
        const int32_t key = layer * s->c->n_experts + w[i].expert;
        const double t0   = now_s();
        const int rc      = fill_slot(s, layer, w[i].expert, w[i].slot);
        s->load_seconds += now_s() - t0;
        if (rc != 0) {
            s->key_of[w[i].slot] = K3_GGUF_SLOT_EMPTY; /* release the reservation */
            continue;
        }
        s->key_of[w[i].slot]  = key;
        s->slot_of[key]       = w[i].slot;
        s->used_at[w[i].slot] = ++s->clock;
        /* Same semantics as k3_cache's prefetch_reads: this expert was pulled
         * off disk microseconds ago, so the get() that follows will record a
         * hit that is not a RAM retention. The report subtracts this counter
         * from hits for the TRUE resident rate. */
        s->prefetch_reads++;
        ok++;
    }
    return ok;
}

static int gsrc_resident(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3GgufExpertSrc *s = (K3GgufExpertSrc *)self;
    if (layer < 0 || layer >= s->c->n_layers || expert < 0 || expert >= s->c->n_experts)
        return 0;
    const int32_t key = layer * s->c->n_experts + expert;
    const int slot    = s->slot_of[key];
    if (slot < 0) return 0;
    if (out) fill_q(s, slot, out);
    return 1;
}

/* ------------------------------------------------------------------ init */

int k3_gguf_expert_src_init(K3GgufExpertSrc *s, const K3Gguf *g, const K3Cfg *c,
                            int64_t budget_bytes)
{
    memset(s, 0, sizeof *s);
    s->src.get      = gsrc_get;
    s->src.getmany  = gsrc_getmany;
    s->src.resident = gsrc_resident;
    s->src.ctx      = s;
    s->g            = g;
    s->c            = c;

    if (resolve_geometry(s) != 0) {
        k3_gguf_expert_src_free(s);
        return -1;
    }

    s->nslot = (int)(budget_bytes / s->slot_bytes);
    if (s->nslot < c->topk + 1) {
        fprintf(stderr,
                "k3_gguf_expert: budget %.2f GB gives %d slots of %.2f MB, but "
                "top-%d needs at least %d. A cache smaller than one token's "
                "working set would evict an expert that is still being multiplied.\n",
                (double)budget_bytes / 1e9, s->nslot, (double)s->slot_bytes / 1e6,
                c->topk, c->topk + 1);
        k3_gguf_expert_src_free(s);
        return -1;
    }

    const size_t want = (size_t)s->nslot * (size_t)s->slot_bytes;
    if (posix_memalign((void **)&s->arena, 4096, want) != 0) {
        fprintf(stderr, "k3_gguf_expert: cannot allocate %.2f GB arena\n",
                (double)want / 1e9);
        k3_gguf_expert_src_free(s);
        return -1;
    }

    const size_t nkey = (size_t)c->n_layers * c->n_experts;
    s->slot_of        = (int32_t *)malloc(nkey * sizeof(int32_t));
    s->key_of         = (int32_t *)malloc((size_t)s->nslot * sizeof(int32_t));
    s->used_at        = (uint64_t *)calloc((size_t)s->nslot, sizeof(uint64_t));
    if (!s->slot_of || !s->key_of || !s->used_at) {
        k3_gguf_expert_src_free(s);
        return -1;
    }
    for (size_t i = 0; i < nkey; i++) s->slot_of[i] = -1;
    for (int i = 0; i < s->nslot; i++) s->key_of[i] = K3_GGUF_SLOT_EMPTY;

    /* The fp32 transient: the largest per-expert matrix (rows*cols floats). */
    {
        int64_t maxv = 0;
        for (int m = 0; m < 3; m++)
            if ((int64_t)s->rows[m] * s->cols[m] > maxv)
                maxv = (int64_t)s->rows[m] * s->cols[m];
        s->f32_cap = (size_t)maxv * sizeof(float);
        s->f32     = (float *)malloc(s->f32_cap);
        if (!s->f32) {
            k3_gguf_expert_src_free(s);
            return -1;
        }
    }
    return 0;
}

void k3_gguf_expert_src_free(K3GgufExpertSrc *s)
{
    free(s->arena);
    free(s->slot_of);
    free(s->key_of);
    free(s->used_at);
    free(s->raw);
    free(s->f32);
    memset(s, 0, sizeof *s);
}

void k3_gguf_expert_src_reset_stats(K3GgufExpertSrc *s)
{
    s->hits = s->misses = s->evictions = s->bytes_read = 0;
    /* prefetch_reads belongs to the same window as hits (k3_cache's rule): the
     * report derives the true resident rate as hits - prefetch_reads, so both
     * must cover the same interval. */
    s->prefetch_reads = 0;
    s->load_seconds   = 0.0;
}

void k3_gguf_expert_src_report(const K3GgufExpertSrc *s, const char *label)
{
    const uint64_t n = s->hits + s->misses;
    int resident     = 0;
    for (int i = 0; i < s->nslot; i++)
        if (s->key_of[i] >= 0) resident++;
    printf("gguf expert cache [%s]\n", label ? label : "");
    printf("  slots        : %d of %.2f MB = %.2f GB arena (%d resident)\n", s->nslot,
           (double)s->slot_bytes / 1e6, (double)s->nslot * s->slot_bytes / 1e9, resident);
    printf("  requests     : %llu  hits %llu (%.2f%%)  misses %llu  evictions %llu\n",
           (unsigned long long)n, (unsigned long long)s->hits,
           n ? 100.0 * s->hits / n : 0.0, (unsigned long long)s->misses,
           (unsigned long long)s->evictions);
    if (s->prefetch_reads) {
        const unsigned long long served =
            (s->hits > s->prefetch_reads) ? s->hits - s->prefetch_reads : 0;
        printf(
            "  of those hits : %llu came from the batch prefetch, i.e. read from disk\n"
            "                  this token; TRUE resident hit rate %.2f%%\n",
            (unsigned long long)s->prefetch_reads, n ? 100.0 * served / n : 0.0);
    }
    printf("  read from disk: %.2f GB in %.2f s (%.0f MB/s while loading)\n",
           (double)s->bytes_read / 1e9, s->load_seconds,
           s->load_seconds > 0 ? (double)s->bytes_read / 1e6 / s->load_seconds : 0.0);
}
