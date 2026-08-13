#!/usr/bin/env python
"""tok_parity_gguf.py - prove the GGUF-metadata tokenizer matches the oracle.

WHY THIS EXISTS
    dev-d's D6 loader (k3_tok_load_gguf in src/tokenizer/k3_tok.h) populates the
    Tok struct from GGUF shard-1 metadata: byte-level tokens, a recovered merges
    list, token_type control flags, and the kimi-k2 pre-tokenizer family. The
    existing gate (tok_parity.py) proves the tiktoken.model loader against
    tiktoken itself; this gate proves the GGUF loader against the SAME oracle,
    through both loadable paths: rankbpe (tiktoken's lowest-concatenated-id rule,
    the engine default) and the gguf merges list.

    The Slice-0 verdict (dev-d report) is derived from this script plus the
    in-C comparison test_tok_gguf runs against the HF loader: on the real
    shard-1 metadata both paths are token-for-token identical to the oracle.

usage:  tok_parity_gguf.py <path-to-test_tok_gguf> [--verbose]
env:    K3_GGUF_REAL   shard directory or single .gguf (default
                       /workspace/unsloth/Kimi-K3-GGUF/UD-IQ1_S)
        K3_TOK_FILES   the released HF tokenizer files (default
                       /workspace/unsloth/Kimi-K3-GGUF)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

sys.path.insert(0, HERE)
import tok as oracle  # noqa: E402  the reference Python tokenizer

GGUF = os.environ.get("K3_GGUF_REAL", "/workspace/unsloth/Kimi-K3-GGUF/UD-IQ1_S")
# The HF tokenizer files come from the same place tok.py finds them; the env
# override lets this script run where the oracle search does not reach.
FILES = os.environ.get("K3_TOK_FILES") or oracle.find_files()

# The same separation cases tok_parity.py uses, plus the special-token shapes
# the K3 control range exercises (atomic added tokens and reserved tokens, which
# are NOT atomic in the released config).
CASES = [
    ("ascii plain", "Hello world"),
    ("ascii sentence", "The quick brown fox jumps over the lazy dog."),
    ("leading space", " leading"),
    ("trailing space", "trailing "),
    ("double space", "two  spaces"),
    ("tab", "a\tb"),
    ("newlines", "line1\nline2\n\nline4"),
    ("crlf", "dos\r\nline"),
    ("digits run", "1234567890 42 007"),
    ("mixed alnum", "abc123def456"),
    ("punctuation", "!@#$%^&*()_+-=[]{}|;':\",./<>?"),
    ("contractions", "don't can't it's I'm we'll they've he'd"),
    ("caps run", "HTTP HTML JSON XML API"),
    ("camelCase", "camelCaseIdentifierName"),
    ("snake_case", "snake_case_identifier_name"),
    ("han only", "\u4f60\u597d\u4e16\u754c"),
    ("han + ascii", "\u4f60\u597d world"),
    ("ascii + han", "hello \u4f60\u597d"),
    ("han sentence", "\u6211\u4eec\u5728\u6d4b\u8bd5\u5206\u8bcd\u5668\u3002"),
    ("han + digits", "\u7b2c123\u9875"),
    ("japanese", "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c"),
    ("korean", "\uc548\ub155\ud558\uc138\uc694"),
    ("cyrillic", "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"),
    ("arabic", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"),
    ("greek", "\u0393\u03b5\u03b9\u03b1 \u03c3\u03bf\u03c5"),
    ("accents", "caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u00fcber"),
    ("emoji", "\U0001f600\U0001f680\U0001f9e0"),
    ("emoji + text", "ship it \U0001f680 now"),
    ("emoji zwj", "\U0001f469\u200d\U0001f4bb"),
    ("code c", "int main(void){ return 0; }"),
    ("code python", "def f(x):\n    return x ** 2\n"),
    ("json", '{"key": [1, 2, {"n": null}], "b": true}'),
    ("url", "https://example.com/a/b?c=d&e=f#g"),
    ("markdown", "# Title\n\n- item **bold** `code`\n"),
    ("repeated char", "aaaaaaaaaaaaaaaaaaaa"),
    ("long word", "supercalifragilisticexpialidocious"),
    ("mixed script", "hello \u4f60\u597d \u041c\u0438\u0440 \U0001f600 123"),
    ("high bytes", "\u00ff\u00fe\u00fd"),
    ("math symbols", "\u2211 \u221e \u2260 \u2264 \u03c0 \u00d7 \u00f7"),
    ("quotes typographic", "\u201cquoted\u201d and \u2018single\u2019"),
    ("dashes", "em\u2014dash en\u2013dash hyphen-minus"),
    # special-token shapes
    ("bos", "[BOS]"),
    ("eos", "[EOS]"),
    ("end_of_msg", "hi <|end_of_msg|> bye"),
    ("open close sep", "<|open|> and <|close|> and <|sep|>"),
    ("header ids", "[start_header_id]user[end_header_id]"),
    ("eot", "[EOT]"),
    ("media", "<|media_begin|>caption<|media_end|>"),
    ("osagent", "<osagent_mode>on"),
    ("unk pad", "[UNK] unknown [PAD]"),
    ("reserved inline", "reserved <|reserved_token_163597|> inline"),
    ("reserved tail", "<|reserved_token_163838|>"),
    ("bos hello eos", "[BOS]hello[EOS]"),
]


def gguf_encode(exe: str, text: str, mode: str) -> list[int]:
    """Always go through a file: argv is re-encoded at the process boundary, and
    a non-ASCII case sent as an argument reaches C as different bytes than the
    oracle saw."""
    fd, tmp = tempfile.mkstemp(prefix="k3_gguf_parity_", suffix=".txt")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        r = subprocess.run([exe, GGUF, FILES, mode, tmp],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"test_tok_gguf failed: {r.stderr.strip()}")
        line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        return [int(x) for x in line.split(",")] if line else []
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    exe = sys.argv[1]
    verbose = "--verbose" in sys.argv
    if not os.path.isdir(GGUF) and not os.path.isfile(GGUF):
        print(f"SKIPPED: no real GGUF at {GGUF} (set K3_GGUF_REAL)")
        return 0

    enc, _cfg, special = oracle.load()
    allowed = set(special)
    ok = True
    for mode, label in (("encode", "rankbpe"), ("encode-m", "merges ")):
        npass = nfail = 0
        for name, text in CASES:
            want = enc.encode(text, allowed_special=allowed)
            got = gguf_encode(exe, text, mode)
            if want == got:
                npass += 1
                if verbose:
                    print(f"  PASS  [{label}] {name:22s} {len(got):3d} ids")
            else:
                nfail += 1
                ok = False
                print(f"  FAIL  [{label}] {name:22s} oracle={len(want)} c={len(got)}")
                print(f"        oracle: {want}")
                print(f"        c     : {got}")
        print(f"gguf tokenizer parity [{label}]: {npass}/{npass + nfail} cases match")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
