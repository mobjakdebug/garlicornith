"""Self-test of the pieces of pc/ornith_convert.py that do not need the model: the q4s and q3s layouts, and the
undoing of llama.cpp's value-head order.

Run: python tests/ornith_layout_test.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pc"))
import ornith_convert as oc  # noqa: E402


def from_q4s(wb, sb, rows, cols):
    """pc/convert_9b_q4s.py's reader: the inverse of to_q4s, used here to check the writer."""
    T, nb = rows // 128, cols // 32
    words = np.frombuffer(wb, np.uint32).reshape(T, nb, 2, 64, 4).transpose(0, 3, 1, 2, 4).reshape(T, 64, nb, 8)
    q = np.zeros((T, 2, 64, nb, 8, 4), np.uint8)
    for h in range(2):
        for i in range(4):
            q[:, h, :, :, :, i] = (words >> (16 * h + 4 * i)) & 15
    sc = np.frombuffer(sb, np.uint32).reshape(T, nb, 64).transpose(0, 2, 1)
    d = np.stack([(sc & 0xFFFF).astype(np.uint16), (sc >> 16).astype(np.uint16)], axis=1).view(np.float16)
    return q.reshape(rows, cols), d.reshape(rows, nb)


def main():
    rng = np.random.default_rng(0)
    # q4s: 16 nibbles a word-pair, tiles of 128 rows
    q = rng.integers(0, 16, (256, 128)).astype(np.uint8)
    d = rng.standard_normal((256, 4)).astype(np.float16)
    q2, d2 = from_q4s(*oc.to_q4s(q, d), 256, 128)
    assert (q == q2).all() and np.array_equal(d, d2), "q4s does not round-trip"
    # the byte counts the layouts imply: 4 bits and a scale per 32 for q4s; 3 bits and a scale for q3s
    w = rng.standard_normal((200, 96)).astype(np.float32)          # 200 rows -> 256 after padding
    wb, sb, rows = oc.q4s(w)
    assert rows == 256 and len(wb) == 256 * 96 // 2 and len(sb) == 256 * 96 // 32 * 2, (len(wb), len(sb))
    w3, s3, rows3 = oc.q3s(w)
    assert rows3 == 256 and len(w3) == 2 * (96 // 32) * 384 * 4 and len(s3) == 256 * 96 // 32 * 2, \
        (len(w3), len(s3))
    # the value-head order: llama.cpp's head i holds Hugging Face's perm[i]; undoing it must give the input back
    m = oc.Model.__new__(oc.Model)
    m.n_vh = 32
    perm = np.arange(32).reshape(16, 2).T.ravel()
    blocks = (perm[:, None] * 128 + np.arange(128)).ravel()
    x = np.arange(4096, dtype=np.float32)
    assert np.array_equal(m.hf_heads(x[blocks], 0, 128), x), "value heads (rows, 128-blocks)"
    mat = np.tile(x, (4, 1))
    assert np.array_equal(m.hf_heads(mat[:, blocks], 1, 128), mat), "value heads (columns)"
    assert np.array_equal(m.hf_heads(x[:32][perm], 0), x[:32]), "value heads (one value each)"
    print("ornith_layout_test: ok")


if __name__ == "__main__":
    main()
