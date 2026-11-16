"""chat/assets/tokenizer.bin (PTOK), and a comparison of one with a GGUF's token list.

The app's tokenizer file (pc/export_chat.py: export_tokenizer) holds, for every token id, the bytes it encodes
(byte-level BPE, so those bytes are what the token stands for in UTF-8) and whether it is a special token, plus the
merge table in rank order. A GGUF holds the same vocabulary as strings in Qwen's byte-to-unicode alphabet, so the
two can be compared token by token - which is how we know that a model built on Qwen3.5 can keep the app's
tokenizer.bin and its lm_head keep list (pc/vocab_order.npy) instead of rebuilding them.
"""

import struct
from dataclasses import dataclass


@dataclass
class Tok:
    n_vocab: int
    n_merges: int
    raw: list          # bytes of each token
    special: list      # 1 for special / added tokens
    merges: list       # (a, b, merged) in rank order


def read(path):
    d = open(path, "rb").read()
    if not d.startswith(b"PTOK"):
        raise ValueError(f"{path}: not a PTOK file")
    n_vocab, n_merges = struct.unpack_from("<II", d, 4)
    p = 12 + 1024
    raw, special = [], []
    for _ in range(n_vocab):
        special.append(d[p])
        n = d[p + 1] | d[p + 2] << 8
        raw.append(d[p + 3:p + 3 + n])
        p += 3 + n
    merges = [struct.unpack_from("<III", d, p + 12 * r) for r in range(n_merges)]
    return Tok(n_vocab, n_merges, raw, special, merges)


def bytes_to_unicode():
    """Qwen's / GPT-2's byte -> unicode alphabet (tokenizer.json's vocabulary is written in it)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs))), {v: k for k, v in zip(bs, map(chr, cs))}


class Report:
    def __init__(self, lines, ok):
        self.lines, self.ok = lines, ok

    def text(self):
        return "\n".join(self.lines)


def compare(tok, gguf_tokens, gguf_merges, gguf_types):
    """Every token's bytes and the merge order, against the app's file."""
    enc, dec = bytes_to_unicode()
    lines, ok = [], True
    if not gguf_tokens:
        return Report(["GGUF: no token list in the metadata"], False)
    lines.append(f"GGUF: {len(gguf_tokens)} tokens, {len(gguf_merges)} merges")
    if len(gguf_tokens) < tok.n_vocab:
        lines.append(f"  the GGUF has fewer tokens than the app's file: {len(gguf_tokens)} < {tok.n_vocab}")
        ok = False
    elif len(gguf_tokens) > tok.n_vocab:
        lines.append(f"  the GGUF has {len(gguf_tokens) - tok.n_vocab} ids past the app's vocabulary (unused: the "
                     f"keep list stops at {tok.n_vocab})")

    def as_bytes(s):
        """A GGUF token string -> the bytes it stands for (byte-to-unicode, or UTF-8 for the special ones)."""
        try:
            return bytes(dec[c] for c in s)
        except KeyError:
            return s.encode("utf-8")

    n = min(len(gguf_tokens), tok.n_vocab)
    bad = [i for i in range(n) if as_bytes(gguf_tokens[i]) != tok.raw[i]]
    if bad:
        ok = False
        lines.append(f"  {len(bad)} of {n} tokens differ, e.g. " +
                     ", ".join(f"{i}: {as_bytes(gguf_tokens[i])!r} != {tok.raw[i]!r}" for i in bad[:8]))
    else:
        lines.append(f"  all {n} tokens hold the same bytes as the app's tokenizer.bin")
    if gguf_merges:
        want = [b" ".join([tok.raw[a], tok.raw[b]]) for a, b, _m in tok.merges]
        got = [m.encode("utf-8", "replace") if isinstance(m, str) else m for m in gguf_merges]
        same = [i for i in range(min(len(want), len(got))) if want[i] == got[i]]
        if len(got) != len(want) or len(same) != len(want):
            ok = False
            lines.append(f"  merges differ: {len(got)} in the GGUF, {len(want)} in the app's file, "
                         f"{len(same)} in the same rank")
        else:
            lines.append(f"  all {len(got)} merges in the same rank order")
    # the ids the app's C code names
    named = {}
    for i, t in enumerate(gguf_tokens):
        s = t if isinstance(t, str) else t.decode("utf-8", "replace")
        if s in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<think>", "</think>",
                 "<|vision_start|>", "<|image_pad|>"):
            named.setdefault(s, i)
    lines.append("  special ids: " + ", ".join(f"{k}={v}" for k, v in sorted(named.items())))
    want_ids = {"<|im_start|>": 248045, "<|im_end|>": 248046, "<|endoftext|>": 248044,
                "<think>": 248068, "</think>": 248069}
    for k, v in want_ids.items():
        if k in named and named[k] != v:
            ok = False
            lines.append(f"  MISSING BINDING: the app's C code has {k} as {v}, the GGUF has it as {named[k]}")
    if gguf_types:
        n_special = sum(1 for t in gguf_types if t in (3, 4))   # CONTROL / USER_DEFINED
        n_app = sum(tok.special)
        if n_special != n_app:
            lines.append(f"  NOTE: {n_special} control/user-defined tokens in the GGUF, {n_app} special in the app's")
    lines.append("  => " + ("the app's tokenizer.bin and keep list can be used as they are" if ok else
                            "the tokenizer differs: tokenizer.bin and the keep list must be rebuilt"))
    return Report(lines, ok)
