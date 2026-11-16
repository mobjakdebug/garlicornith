"""Self-test of pc/gguf.py on a synthetic file: metadata of every kind, a BF16 and an F32 tensor, alignment.

Run: python tests/gguf_test.py
"""

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pc"))
from gguf import GGUF  # noqa: E402


def s(text):
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def write(path):
    kv = []
    kv.append(s("general.architecture") + struct.pack("<I", 8) + s("qwen35"))
    kv.append(s("qwen35.block_count") + struct.pack("<Ii", 5, 32))
    kv.append(s("general.alignment") + struct.pack("<II", 4, 32))
    tokens = ["<|im_start|>", "<|im_end|>", "hello", "world"]
    arr = struct.pack("<I", 8) + struct.pack("<Q", len(tokens)) + b"".join(s(t) for t in tokens)
    kv.append(s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + arr)
    kv.append(s("qwen35.rope.freq_base") + struct.pack("<If", 6, 10000000.0))
    kv.append(s("some.bool") + struct.pack("<IB", 7, 1))
    meta = b"".join(kv)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 40)).astype(np.float32)
    bf = (a.view(np.uint32) >> 16).astype(np.uint16)          # the same weights as bf16
    b = rng.standard_normal(9).astype(np.float32)
    blobs = [bf.tobytes(), b.tobytes()]
    infos = [("blk.0.ffn_up.weight", (40, 6), 30), ("some.vector", (9,), 0)]
    tensor_info = b""
    offsets = []
    off = 0
    for (name, dims, tid), blob in zip(infos, blobs):
        offsets.append(off)
        off += (-off) % 32 + len(blob)                         # every tensor starts 32-byte aligned
    for (name, dims, tid), o in zip(infos, offsets):
        tensor_info += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        tensor_info += struct.pack("<I", tid) + struct.pack("<Q", o)

    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), 6) + meta + tensor_info
    data = b""
    for blob, o in zip(blobs, offsets):
        data += b"\0" * (o - len(data)) + blob
    body = head + b"\0" * ((-len(head)) % 32) + data
    Path(path).write_bytes(body)
    return a, bf, b


def main():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.gguf"
        a, bf, b = write(p)
        g = GGUF.open(p)
        assert g.meta["general.architecture"] == "qwen35"
        assert g.meta["qwen35.block_count"] == 32
        assert g.meta["tokenizer.ggml.tokens"][:2] == ["<|im_start|>", "<|im_end|>"]
        assert abs(g.meta["qwen35.rope.freq_base"] - 1e7) < 1
        assert g.meta["some.bool"] == 1
        t = g.tensors["blk.0.ffn_up.weight"]
        assert t.shape == (6, 40), t.shape
        assert t.type_name == "BF16"
        got = g.tensor("blk.0.ffn_up.weight")
        ref = np.frombuffer(bf.tobytes(), np.uint16).astype(np.uint32) << 16
        assert np.array_equal(got.reshape(-1), ref.view(np.float32)), "bf16 values"
        assert np.allclose(got, (a.astype(np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32))
        vec = g.tensor("some.vector")
        assert np.array_equal(vec, b), "f32 vector"
        assert g.header_bytes % 32 == 0
        print("gguf_test: ok:", len(g.tensors), "tensors, header", g.header_bytes, "bytes")


if __name__ == "__main__":
    main()
