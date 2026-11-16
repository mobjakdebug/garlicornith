"""GGUF reader for the Ornith-1.5-9B conversion: metadata, tensor table and dequantization.

Only what the conversion needs, with no dependencies beyond numpy:

  g = GGUF.open("models/Ornith-1.5-9B-BF16.gguf")     # a local file, or an https URL (HTTP range requests)
  g.meta["general.architecture"]                       # the metadata, by key
  for t in g.tensors.values(): t.name, t.shape, t.type_name
  w = g.tensor("blk.0.ffn_gate.weight")                # -> float32 numpy array, shape = the file's (rows, cols)

A tensor's shape is given as the file stores it (GGUF dim 0 is the fastest-changing one, so a weight of
[out, in] is ne = [in, out]); `t.shape` returns (rows, cols) = (out, in) for 2-D tensors and (n,) for 1-D ones.
`g.header_bytes` after open() is how much of the file had to be read (metadata + tensor table, not the weights).

Dequantization covers what llama.cpp writes for these models: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K.
"""

import json
import struct
import urllib.request

import numpy as np

GGUF_MAGIC = b"GGUF"

# ggml type id -> (name, elements per block, bytes per block)
TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}

# metadata value types
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STRING, _ARRAY, _U64, _I64, _F64 = range(13)
_SCALARS = {_U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i", _F32: "<f", _BOOL: "<B",
            _U64: "<Q", _I64: "<q", _F64: "<d"}


class Tensor:
    def __init__(self, name, dims, type_id, offset):
        self.name = name
        self.dims = dims                     # GGUF order: fastest first
        self.type_id = type_id
        self.offset = offset                 # from the start of the data section
        self.shape = tuple(reversed(dims))   # (rows, cols) for a 2-D weight, (n,) for 1-D
        name_, self.block_elems, self.block_bytes = TYPES.get(type_id, (f"type{type_id}", 0, 0))
        self.type_name = name_

    @property
    def n_elements(self):
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self):
        if not self.block_elems:
            raise ValueError(f"{self.name}: unsupported ggml type {self.type_id}")
        return self.n_elements // self.block_elems * self.block_bytes

    def __repr__(self):
        return f"<Tensor {self.name} {self.type_name} {self.shape}>"


class _Reader:
    """Bytes of a file, local or over HTTP with range requests."""

    def __init__(self, path):
        self.path = str(path)
        self.remote = self.path.startswith(("http://", "https://"))
        self.f = None if self.remote else open(self.path, "rb")
        self.size = None if self.remote else self.f.seek(0, 2)
        self.n_read = 0

    def read(self, off, n):
        if n <= 0:
            return b""
        if not self.remote:
            self.f.seek(off)
            b = self.f.read(n)               # short only at the end of the file (the header reader grows its slice)
            self.n_read += len(b)
            return b
        out = []
        got = 0
        while got < n:                       # a long range may be cut short: ask again for the rest
            last = None
            for attempt in range(6):
                try:
                    req = urllib.request.Request(self.path, headers={"Range": f"bytes={off + got}-{off + n - 1}"})
                    with urllib.request.urlopen(req, timeout=600) as r:
                        b = r.read()
                    if b:
                        out.append(b)
                        got += len(b)
                        break
                    raise IOError("empty range")
                except Exception as ex:      # noqa: BLE001  (retry: the connection may drop)
                    last = ex
            else:
                raise IOError(f"{self.path}: cannot read bytes {off + got}..{off + n} ({last})")
        self.n_read += got
        return b"".join(out)[:n]


class GGUF:
    def __init__(self, reader):
        self.r = reader
        self.meta = {}
        self.tensors = {}
        self.header_bytes = 0
        self._parse()

    @classmethod
    def open(cls, path):
        return cls(_Reader(path))

    # ---- header ---------------------------------------------------------------------------------------------
    def _parse(self):
        head = self.r.read(0, 4)
        if head != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {head!r}")
        # The metadata holds the whole vocabulary (tens of MiB), so read in growing slices rather than guessing.
        size = 1 << 22
        while True:
            buf = self.r.read(0, size)
            try:
                return self._parse_from(buf)
            except (IndexError, struct.error):   # not enough bytes yet
                if len(buf) < size:
                    raise ValueError("truncated GGUF header") from None
                size *= 2

    def _parse_from(self, buf):
        off = 0
        assert buf[:4] == GGUF_MAGIC
        version, = struct.unpack_from("<I", buf, 4)
        if version not in (2, 3):
            raise ValueError(f"GGUF version {version} is not supported")
        n_tensors, n_kv = struct.unpack_from("<QQ", buf, 8)
        off = 24

        def rd_u64():
            nonlocal off
            v, = struct.unpack_from("<Q", buf, off)
            off += 8
            return v

        def rd_string():
            nonlocal off
            n = rd_u64()
            s = buf[off:off + n]
            off += n
            return s.decode("utf-8", "replace")

        def rd_value(t):
            nonlocal off
            if t == _STRING:
                return rd_string()
            if t == _ARRAY:
                et, = struct.unpack_from("<I", buf, off)
                off += 4
                n = rd_u64()
                if et == _STRING:            # the token table: keep it as the list it is
                    return [rd_string() for _ in range(n)]
                f = _SCALARS[et]
                size = struct.calcsize(f)
                vals = np.frombuffer(buf, dtype=np.dtype(f), count=n, offset=off)
                off += size * n
                return vals.tolist()
            f = _SCALARS[t]
            v, = struct.unpack_from(f, buf, off)
            off += struct.calcsize(f)
            return v

        for _ in range(n_kv):
            key = rd_string()
            t, = struct.unpack_from("<I", buf, off)
            off += 4
            self.meta[key] = rd_value(t)

        for _ in range(n_tensors):
            name = rd_string()
            nd, = struct.unpack_from("<I", buf, off)
            off += 4
            dims = struct.unpack_from("<" + "Q" * nd, buf, off)
            off += 8 * nd
            tid, = struct.unpack_from("<I", buf, off)
            off += 4
            o, = struct.unpack_from("<Q", buf, off)
            off += 8
            self.tensors[name] = Tensor(name, tuple(dims), tid, o)

        # the data section starts at the next multiple of the alignment (32 unless the file says otherwise)
        align = self.meta.get("general.alignment", 32)
        self.data_offset = (off + align - 1) // align * align
        self.header_bytes = self.data_offset
        return self

    # ---- tensors --------------------------------------------------------------------------------------------
    def raw(self, name):
        t = self.tensors[name]
        return self.r.read(self.data_offset + t.offset, t.nbytes)

    def tensor(self, name):
        """The tensor as float32, in the file's (rows, cols) shape."""
        t = self.tensors[name]
        raw = self.raw(name)
        w = dequantize(raw, t.type_id, t.n_elements)
        return w.reshape(t.shape) if len(t.shape) > 1 else w

    def rows(self, name, r0, r1):
        """Rows r0..r1 of a 2-D tensor as float32, for the big matrices (read only what is needed)."""
        t = self.tensors[name]
        if len(t.shape) != 2:
            raise ValueError(f"{name}: not 2-D")
        rows, cols = t.shape
        if t.type_id in (0, 1, 30):                       # F32, F16, BF16: a row range is a byte range
            per = cols * t.block_bytes
            raw = self.r.read(self.data_offset + t.offset + r0 * per, (r1 - r0) * per)
            return dequantize(raw, t.type_id, (r1 - r0) * cols).reshape(r1 - r0, cols)
        return self.tensor(name)[r0:r1]                   # quantized blocks mix rows: take the whole tensor

    def token_list(self, key="tokenizer.ggml.tokens"):
        return self.meta.get(key)

    def describe(self):
        out = {"architecture": self.meta.get("general.architecture"),
               "quant": self.meta.get("general.file_type"),
               "header_bytes": self.header_bytes,
               "metadata_keys": sorted(self.meta),
               "tensors": [{"name": t.name, "type": t.type_name, "shape": list(t.shape), "bytes": t.nbytes}
                           for t in self.tensors.values()]}
        return json.dumps(out, indent=1)


def _fp16(b):
    return np.frombuffer(b, dtype=np.float16)


def dequantize(raw, type_id, n):
    """Bytes of one tensor's blocks -> float32 values (flat, n of them)."""
    name = TYPES[type_id][0]
    if type_id == 0:                                        # F32
        return np.frombuffer(raw, dtype=np.float32, count=n).astype(np.float32)
    if type_id == 1:                                        # F16
        return np.frombuffer(raw, dtype=np.float16, count=n).astype(np.float32)
    if type_id == 30:                                       # BF16
        u = np.frombuffer(raw, dtype=np.uint16, count=n).astype(np.uint32) << 16
        return u.view(np.float32)
    if type_id == 8:                                        # Q8_0: 32 per block, fp16 d then 32 int8
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        q = b[:, 2:].copy().view(np.int8).astype(np.float32)
        return (q * d).reshape(-1)
    if type_id == 2:                                        # Q4_0: fp16 d, 16 bytes of nibbles
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        q = b[:, 2:]
        lo = (q & 15).astype(np.float32) - 8.0
        hi = (q >> 4).astype(np.float32) - 8.0
        return np.concatenate([lo, hi], axis=1).reshape(-1) * np.repeat(d, 32, axis=1).reshape(-1)
    if type_id in (12, 13):                                 # Q4_K / Q5_K: 256 per block
        return _dequant_k(raw, q5=type_id == 13).reshape(-1)
    if type_id == 14:                                       # Q6_K
        return _dequant_q6k(raw).reshape(-1)
    raise ValueError(f"dequantizing {name} is not implemented")


def _scale_min_k4(j, q):
    """ggml's get_scale_min_k4: the 6-bit scale and min of sub-block j (of 8) in a Q4_K/Q5_K block."""
    if j < 4:
        return (q[j] & 63).astype(np.int32), (q[j + 4] & 63).astype(np.int32)
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return d.astype(np.int32), m.astype(np.int32)


def _dequant_k(raw, q5):
    """Q4_K (144 bytes) / Q5_K (176 bytes) blocks of 256 -> float32 [blocks, 256]."""
    bs = 176 if q5 else 144
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, bs)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    scales = b[:, 4:16]
    if q5:
        qh = b[:, 16:48]
        qs = b[:, 48:176]
    else:
        qs = b[:, 16:144]
    out = np.empty((b.shape[0], 256), np.float32)
    for j in range(8):                                      # sub-blocks of 32 values, two per 32-byte group
        sc, mn = _scale_min_k4(j, scales)
        d1 = d * sc.reshape(-1, 1).astype(np.float32)
        m1 = dmin * mn.reshape(-1, 1).astype(np.float32)
        grp = j // 2
        cur = qs[:, grp * 32:(grp + 1) * 32]
        q = (cur & 0xF) if j % 2 == 0 else (cur >> 4)
        if q5:
            bit = 1 << j
            q = q + np.where(qh & bit, 16, 0)
        out[:, j * 32:(j + 1) * 32] = d1 * q.astype(np.float32) - m1
    return out


def _dequant_q6k(raw):
    """Q6_K blocks of 256 (210 bytes) -> float32 [blocks, 256]."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 210)
    ql = b[:, 0:128]
    qh = b[:, 128:192]
    scales = b[:, 192:208].copy().view(np.int8).astype(np.float32)      # [blocks, 16]
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    out = np.empty((b.shape[0], 256), np.float32)
    lo = ql[:, :64]
    hi = ql[:, 64:128]
    l = np.arange(32)
    for half in range(2):                                   # 128 values at a time: 64 bytes of ql, 32 of qh
        s = scales[:, half * 8:(half + 1) * 8]
        ql0 = ql[:, half * 64:(half + 1) * 64]
        qh0 = qh[:, half * 32:(half + 1) * 32]
        q1 = ((ql0[:, :32] & 0xF) | ((qh0 >> 0) & 3) << 4).astype(np.int16) - 32
        q2 = ((ql0[:, 32:64] & 0xF) | ((qh0 >> 2) & 3) << 4).astype(np.int16) - 32
        q3 = ((ql0[:, :32] >> 4) | ((qh0 >> 4) & 3) << 4).astype(np.int16) - 32
        q4 = ((ql0[:, 32:64] >> 4) | ((qh0 >> 6) & 3) << 4).astype(np.int16) - 32
        base = half * 128
        for k, qq in enumerate((q1, q2, q3, q4)):
            s0 = s[:, k * 2].reshape(-1, 1)
            s1 = s[:, k * 2 + 1].reshape(-1, 1)
            sc = np.where((l < 16).reshape(1, 32), s0, s1)
            out[:, base + k * 32:base + (k + 1) * 32] = d * sc * qq.astype(np.float32)
    return out
