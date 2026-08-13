/* k3_gguf.h - GGUF shard-set reader for the Kimi K3 UD-IQ1_S checkpoint.
 *
 * FORMAT, verified byte-level against the real 14-shard set:
 *   Each shard is a self-contained GGUF v3 file: [magic "GGUF"][version u32][tensor
 *   count u64][metadata kv count u64][metadata kv...][tensor infos...][pad to 32]
 *   [tensor data]. Shard 1 is metadata-only (tensor count 0, 64 KV pairs including
 *   the full tokenizer); shards 2-14 carry 2,573 tensors and only the three split
 *   keys. Every value is little-endian; tensor offsets are relative to the start of
 *   the data section, which begins at the 32-aligned end of the tensor infos.
 *   The data section alignment comes from `general.alignment` (absent here: 32).
 *
 * WRITER QUIRKS (D7, verified byte-level, see recon-gguf-spec.md §6):
 *   The value-type enum in this file matches the current spec's 0..12 (5 = int32,
 *   6 = float32, 7 = bool, 8 = string, 9 = array, 10 = uint64), with one trap:
 *   ARRAY elements declared as type 5 are 4 bytes each (the observed
 *   kimi-k3.attention.head_count_kv and tokenizer.ggml.token_type arrays both do
 *   this). A naive reader that expects 8-byte elements walks off the rail silently.
 *   The reader therefore hardcodes 4-byte type-5 elements, and after EVERY metadata
 *   value verifies that the next bytes form a sane key/type boundary. Misalignment,
 *   an unknown type enum, a truncated section, or a nonzero padding byte is a hard
 *   error with file+offset context. Silent wrong-model is the failure class this
 *   reader exists to kill.
 *
 * WHY pread AND NOT mmap (D1): same reasoning as k3_st.h. Pages read into buffers
 * the engine owns never become file-backed mappings counted against the process, so
 * peak RSS tracks what is actually resident. O_DIRECT is deliberately not used here:
 * the buffered path through the page cache is the architect's D1(a) decision, and
 * `dfd[]` exists only so the struct shape matches K3St.
 *
 * WHY A SHARED HASH: 2,573 tensors across 14 files, looked up by exact name from
 * the bind layer. Same open-addressed FNV-1a index and strpool as k3_st.c, so the
 * find is O(1) and a duplicate name across shards is a hard error.
 *
 * SCOPE LOCK: exactly three ggml types occur in the file (F32=0, Q8_0=8, IQ1_S=19)
 * and exactly those three are accepted. Anything else fails loud. Quantized tensors
 * record K3_DT_U8 (raw codes) with the ggml type kept in gtype[] for the dequant
 * wave; F32 records K3_DT_F32.
 *
 * The struct mirrors K3St's field names and layout for its core fields, so a
 * later wave can drive the k3_st_read* helpers with it (or copy the tensors into a
 * K3St) without touching k3_st.c/h.
 */
#ifndef K3_GGUF_H
#define K3_GGUF_H

#include <stddef.h>
#include <stdint.h>

#include "k3.h"
#include "k3_st.h"

/* One shard-1 metadata entry. String and fixed-element-array values are copied into
 * the reader's arena; string arrays (tokenizer.ggml.tokens/merges, general.tags) are
 * walked with full bounds checks but recorded with d == NULL: this unit has no
 * consumer for them, and wave 2's tokenizer bootstrap adds a dedicated reader. */
typedef struct {
    char     key[64];          /* all verified keys are shorter than this        */
    uint32_t type;             /* wire value type, 0..12                         */
    union {
        uint64_t u;            /* u8/i8/u16/i16/u32/i32/u64/i64; bool as 0/1     */
        double   f;            /* f32, f64                                        */
        struct { const char *p; size_t n; } s;                       /* string   */
        struct { uint32_t et, n; const void *d; size_t nbytes; } a;  /* array    */
    } v;
} K3GgufKV;

typedef struct {
    /* K3St-compatible core: same names, same order, so the k3_st_read* helpers see
     * a familiar shape. dfd[] is all -1: buffered pread only (D1). */
    int       *fd;             /* one open descriptor per shard                  */
    int       *dfd;            /* always -1 here; shape-compat with K3St         */
    char     **path;
    int        nshard;

    K3Tensor  *t;              /* every tensor, in discovery order               */
    int        nt;

    int32_t   *bucket;         /* open-addressed hash, -1 empty                  */
    int        nbucket;

    char      *strpool;        /* all names, one allocation                      */
    size_t     strcap, strlen_;

    /* GGUF-specific */
    uint16_t  *gtype;          /* ggml type enum per tensor, parallel to t[]     */
    int       *shard_kv;       /* metadata kv count per shard                    */
    int        alignment;      /* data-section alignment (default 32)            */
    K3GgufKV  *kv;             /* shard-1 (split.no == 0) metadata table         */
    int        nkv;
    char      *arena;          /* string/array value storage for kv              */
    size_t     acap, alen;
} K3Gguf;

/* Open every *.gguf in dir and index every tensor across all shards. Validates
 * magic, version == 3, the split keys, the metadata walk (quirks included) and
 * every tensor info. Returns 0 on success, -1 with a context-rich message on any
 * violation. */
int  k3_gguf_open(K3Gguf *g, const char *dir);
void k3_gguf_close(K3Gguf *g);

/* O(1) lookup; NULL when absent (callers must treat that as fatal). */
const K3Tensor *k3_gguf_find(const K3Gguf *g, const char *name);

/* Raw bytes, exactly as stored; buf must hold t->nbytes. Returns bytes read. */
int64_t k3_gguf_read(const K3Gguf *g, const K3Tensor *t, void *buf);

/* Shard-1 metadata lookup, or NULL. */
const K3GgufKV *k3_gguf_kv(const K3Gguf *g, const char *key);

/* Derive K3Cfg from the shard-1 kimi-k3.* keys (D5). fa receives the ONE-BASED MLA
 * layer list, as k3_cfg_load does; k3_is_mla() compares against layer+1. Returns 1
 * on success. On failure it prints every missing/unknown key and returns 0: an
 * absent field is an error, never a default (k3_cfg.h's own rule). */
int k3_gguf_cfg(const K3Gguf *g, K3Cfg *c, int *fa, int fa_max);

#endif /* K3_GGUF_H */
