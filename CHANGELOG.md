# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **GGUF checkpoints, read directly**: the engine now runs from the unsloth UD-IQ1_S
  shard set (or a single `.gguf` file) with no conversion and no pack step. A
  shard-set reader (`src/io/k3_gguf.c`) indexes all 2,573 tensors of the 14-shard
  594 GB file and derives the config from shard-1 metadata; the tokenizer loads from
  shard-1 metadata too, so no `config.json`, no `tiktoken.model` and no `--tok` are
  needed. (Greedy-only remains the engine's only decode mode; the earlier claim
  that repetitive loops on a 1-bit checkpoint are expected behavior is corrected in
  `### Fixed` below — the loops were the requant bug, now fixed.)
- **GGUF weight path**: the trunk is streamed per layer with Q8_0/F32 dequantised to
  bf16 into one reusable layer buffer; the routed experts are cached as
  backend-native IQ1_S blocks and dotted by a native IQ1_S matmul
  (`k3_matmul_iq1_s`), so the safetensors path keeps its MXFP4 experts and kernels
  unchanged. The dequant/trunk hot loops are OpenMP-parallel with bit-identical
  output at any thread count.
- **Head-to-head benchmark vs llama.cpp** (PR #26185, the reference runner for the
  GGUF): templated 40/40 exact token match (greedy both, model's own chat template,
  byte-identical 99-token prompt), decode 60.0 vs 306.6 s/token (5.1×), peak RSS
  18.2 vs 39.9 GB (2.2×) — box-specific (39-41 GB RAM, warm host cache), a
  faithfulness benchmark, not a portability claim.
- **GGUF parity gates**: weightless tiny-fixture gates for the new path — argmax
  113/113 within the repo budget on single and multi-shard fixtures (single vs
  multi bit-identical), a bit-exact trunk sub-gate against the safetensors bytes,
  and a 96/96 per-expert gate against a raw fp32 dequant reference (`test_gguf_gate`,
  `make parity-tiny`, `make tok-gguf`).

### Fixed

- **GGUF 1-bit experts: the degenerate loop was a bug, not "expected 1-bit
  behavior"** (`be394b7`). A bake-off against an independent oracle — llama.cpp
  PR #26185 (kimi-k3), unsloth's own runner for this file — on the same 14-shard
  GGUF, bare "Hi", greedy, diverged at the first generated token (',' vs '.') and
  continued coherently (", I'm a") where the engine looped (". 1. 7. 0. 7. 7. 1.").
  The cause: the IQ1_S→MXFP4 (E2M1) expert requant at cache admit, measured at 13.3%
  mean per-element weight error on the real bytes (structural group-range
  clipping), flattening the logits and flipping near-tie top tokens. The fix: a
  native IQ1_S matmul (`k3_matmul_iq1_s`, math matched to llama.cpp's reference, no
  transform), backend-native cache blocks tagged at the dispatch point (6.45 MB vs
  17.55 MB per expert triple; ~1,550 vs 569 slots at `--cache-gb 10`), and GATE 2
  de-mirrored to a raw fp32 dequant reference (rel-L2 ~5e-7 vs 1e-4 budget). Rerun:
  20/20 tokens EXACT match vs the oracle, token-1 ','; 68.3 s/token, peak RSS
  17.84 GB. The safetensors/MXFP4 path and its fixtures are byte-unchanged.
- **A_log convention deviation fixed for correctness** (`be394b7`): the GGUF ships
  `ssm_a = -exp(A_log)`; the finder now unfolds `A_log = ln(-ssm_a)` at bind with
  fail-loud on non-negative values. Measured behaviorally insensitive at top-10, but
  a real convention deviation — fixed with a unit so it cannot regress silently.

## [1.0.0] - 2026-08-07

Verified end to end on the full released checkpoint, and made substantially faster, with
byte-identical output preserved at every step. The first-run experience, which was broken
on a clean clone, now works.

### Added

- **`--preset auto`**: sizes the trunk and expert-cache budgets from the machine's own free
  RAM, trunk-first, so a user need not pick a preset by hand. A gigabyte given to the trunk
  is worth far more than a gigabyte of expert cache, and auto pins accordingly, capping the
  pin below the RAM ceiling after a heavy-pin regression was measured.
- **Chunk-union prefill**: a batched-prefill MoE that fetches each unique routed expert once
  per chunk instead of once per token, measured to read about half the expert bytes on a
  prompt, with the generated token bit-identical to the per-token path.
- **Conversation resume** (`--save-state` / `--load-state`): carries the recurrent state and
  KV cache to disk so a second turn resumes instead of re-reading the whole prompt, measured
  3.9x faster on turn two with identical output. Refuses to restore state from a different
  architecture.
- **`--spec N`**: speculative decode by n-gram drafting with batched greedy verification;
  output is exactly the serial greedy decode by construction.
- `--tf-check`, teacher-forced agreement over an id sequence in one sweep, for measuring
  draft quality; `tools/qdq_trunk.py` and `tools/int8_trunk.py` for deriving quantized
  trunks.

### Changed

- **Fused matmul kernels** (fp32, bf16, MXFP4): sixteen partitioned accumulators with
  explicitly fused products, taking the trunk matmul to its memory floor (about eight times
  less per-token compute) while keeping the scalar and AVX2 paths bitwise identical.
- **KDA recurrence parallelised over heads**, bit-identical to the serial form.
- `scripts/k3-doctor.sh` per-preset speed expectations refreshed to the v1.0.0 numbers, with
  the streaming presets noted as disk-bound and the resident tier as compute-bound.

### Fixed

- All shell scripts are committed executable; the first documented command no longer fails
  with Permission denied on a clean clone.
- `scripts/download-model.sh` uses the current `hf` CLI and pins an immutable revision with
  checksum verification; it no longer attempts a pip install that cannot succeed on the
  target OS, and refuses to start without free space for the checkpoint.
- `scripts/k3-doctor.sh` no longer fails a machine that can build and test the engine; the
  memory floor is a warning about running the checkpoint, not a hard stop.
- The config-refusal fixtures the docs describe now exist and are gated in `make test`,
  ctest and CI; the tokenizer leg reports NOT RUN rather than passing silently; CI runs
  `make test` rather than a hand-picked subset.
- A silent-corruption path in the MLA KV overflow and one in the single-slot trunk reader,
  both of which could emit a plausible wrong token, now abort or are prevented.
- The MXFP4 packer alignment and the tiny-checkpoint scale rule.

### Research notes, not shipped as features

- Lossless trunk compression and a quantized-self-draft hybrid were both built and measured,
  and both turned out to help only narrow regimes. The findings and prototypes are kept in
  [`docs/notes/`](docs/notes/).

## [0.1.0] - 2026-07-31

First public release.

### Added

- Full 93-layer Kimi K3 inference: 69 KDA + 24 Gated MLA layers, 896 routed experts with
  top-16 selection, SiTU-GLU, Attention Residuals, native MXFP4 expert weights.
- **Trunk streaming**, which turns the memory budget into a dial rather than a floor. The
  model runs in 8 GB and in 224 GB and produces byte-identical output at every budget
  measured in between.
- MXFP4 matmul that consumes packed nibbles directly, never materialising a dequantised
  expert.
- BPE tokenizer in C, reading the released `tiktoken.model` directly, text in, text out
  with no external step.
- Config reader that loads the checkpoint's own `config.json` and **refuses** a config it
  cannot fully understand rather than defaulting missing fields.
- Incremental decode with a KV cache and carried recurrent state, verified to produce the
  same tokens as full recompute.
- Named memory presets (`--preset laptop|desktop|workstation|server|max`) derived from
  the measured memory ladder.
- `scripts/k3-doctor.sh`, reports whether a machine can run the model, which preset
  fits, and how fast its storage is.
- `scripts/download-model.sh`, fetches the checkpoint and verifies it byte-exactly
  against the published total, because a partial download produces wrong output silently.
- Test suite that runs entirely without model weights: op fixtures, expert cache,
  safetensors reader, config reader, and end-to-end oracle gates (teacher forcing,
  greedy decode, and incremental decode).
- CI: build matrix across GCC and Clang, warnings-as-errors, ASan and UBSan, Python and
  shell lint. Tokenizer parity is built and reported but CANNOT gate on a clean
  checkout, because it needs the vocabulary that ships with the model weights; run
  `make tok` locally against a downloaded checkpoint.

### Known limitations

- No chunked prefill, so long prompts are impractical despite a 32k context ceiling.
- Greedy decoding only; no chat template; no serving layer; no vision; CPU only.

See [docs/ROADMAP.md](docs/ROADMAP.md).

[Unreleased]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/releases/tag/v0.1.0
