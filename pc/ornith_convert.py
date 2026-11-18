"""Ornith-1.5-9B for the PS4: GGUF -> the weight files and indexes Garlic LLM's 9B app reads.

Garlic LLM (github.com/leonardo-alegria/garlic-llm) runs Qwen3.5-9B on a PS4's GPU with two custom weight formats,
a trimmed output head and a speculative-decoding draft head. Ornith-1.5-9B (ornith-ai/Ornith-1.5-9B-GGUF) has the
same architecture - llama.cpp's qwen35: 32 layers, 4096 hidden, head_dim 256, three Gated DeltaNet layers per
full-attention one, a 1-layer MTP head - so the app's kernels run it as they are: what changes is the weights, the
names in the indexes, and a few strings.

This writes, exactly in the layout of the shipped 9B (pc/convert_9b_q4s.py):

  models/ornith-1.5-9b-q4s.bin   GPU part (final_norm, layers, trimmed lm_head) + the input-embedding rows
                                 (q4row) + the token id of every lm_head row ("ids")
  models/ornith-1.5-9b-mtp.bin   the MTP head in q4s + the embedding rows of the tokens it drafts
  chat/assets_9b/model.idx       PMQ5: the model's shape, and where every tensor sits in the file
  chat/assets_9b/mtp.idx         PMTP: the same for the draft head

Formats (shaders/spec.h and shaders/qwen35.cl):
  MLP gate|up             q3s  3.5 bits per weight: the biggest matrices, in the format that fits the memory
  every other matrix      q4s  4.5 bits per weight, SWAR products
  input embedding         q4row  the CPU reads one row per token; every token keeps its row
  lm_head                 4-bit, only the rows of pc/vocab_order.npy (token ids by frequency)
  norms, conv1d, A, dt_bias, gnorm   f32 (norms as 1 + w, as the kernels want them)

  python ornith_convert.py inspect [--gguf PATH|URL] [--quant bf16|q8_0|q6_k|q5_k_m|q4_k_m]
  python ornith_convert.py convert [...]              -> the four files above

The weights come from the BF16 GGUF by default: the console's own formats then lose the least. Dequantization of
Q8_0/Q4_K/Q5_K/Q6_K sources is in pc/gguf.py, so any file of that repository can be the source.
"""

import argparse
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from gguf import GGUF  # noqa: E402
from q3 import quantize_q3  # noqa: E402
from q4 import quantize_q4  # noqa: E402

REPO = "ornith-ai/Ornith-1.5-9B-GGUF"
HF = f"https://huggingface.co/{REPO}/resolve/main/"
BASE_REPO = "ornith-ai/Ornith-1.5-9B"        # where the MTP head is, if the GGUF does not carry it
BASE_HF = f"https://huggingface.co/{BASE_REPO}/resolve/main/"
QUANTS = {"bf16": "Ornith-1.5-9B-BF16.gguf", "q8_0": "Ornith-1.5-9B-Q8_0.gguf",
          "q6_k": "Ornith-1.5-9B-Q6_K.gguf", "q5_k_m": "Ornith-1.5-9B-Q5_K_M.gguf",
          "q4_k_m": "Ornith-1.5-9B-Q4_K_M.gguf"}
NAME = "ornith-1.5-9b-q4s"
MTP_NAME = "ornith-1.5-9b-mtp"
MODELS = ROOT / "models"
IDX_OUT = ROOT / "chat" / "assets_9b"
KEEP = HERE / "vocab_order.npy"              # every token id, most frequent first (Qwen3.5's list: same ids)
ND = 32768                                   # draft rows: the first ND rows of the trimmed lm_head
FMT = {"q4": 2, "q8": 1, "q4i": 3, "q4row": 4, "q3i": 5, "q4s": 6, "ids": 7, "q3s": 8}

# The MTP head's tensors in mtp.idx's order (pc/convert_9b_q4s.py: MTP_ORDER), from Hugging Face's names
MTP_ORDER = ["pre_e", "pre_h", "fc", "attn_norm", "mlp_norm", "q_norm", "k_norm", "wqkv", "wo", "wgu", "wdown", "norm"]
MTP_HF = {"pre_e": "pre_fc_norm_embedding.weight", "pre_h": "pre_fc_norm_hidden.weight", "fc": "fc.weight",
          "attn_norm": "layers.0.input_layernorm.weight", "mlp_norm": "layers.0.post_attention_layernorm.weight",
          "q_norm": "layers.0.self_attn.q_norm.weight", "k_norm": "layers.0.self_attn.k_norm.weight",
          "norm": "norm.weight"}


# ---- the app's layouts (pc/convert_9b_q4s.py, which the shipped 9B files were written with) -------------------------

def to_q4s(q, d):
    """q [rows, cols] nibbles 0..15, d [rows, nb] fp16 -> (q4s words bytes, scale bytes); rows % 128 == 0."""
    rows, cols = q.shape
    T, nb = rows // 128, cols // 32
    q = q.astype(np.uint32).reshape(T, 2, 64, nb, 8, 4)
    half = q[..., 0] | (q[..., 1] << 4) | (q[..., 2] << 8) | (q[..., 3] << 12)
    words = half[:, 0] | (half[:, 1] << 16)
    words = words.reshape(T, 64, nb, 2, 4).transpose(0, 2, 3, 1, 4)
    d = d.astype(np.float16).view(np.uint16).astype(np.uint32).reshape(T, 2, 64, nb)
    sc = (d[:, 0] | (d[:, 1] << 16)).transpose(0, 2, 1)
    return np.ascontiguousarray(words).tobytes(), np.ascontiguousarray(sc).tobytes()


def to_q3s(q, d):
    """q [rows, cols] in 0..7, d [rows, nb] fp16 -> q3s words and scales (shaders/spec.h SWAR3_MVS)."""
    rows, cols = q.shape
    T, nb = rows // 128, cols // 32
    q = q.astype(np.uint32).reshape(T, 2, 64, nb, 32)
    w = np.zeros((T, 64, nb, 6), np.uint32)
    for h in range(2):
        for k in range(6):
            for i in range(5):
                w[..., k] |= q[:, h, :, :, 5 * k + i] << (16 * h + 3 * i)
        for j in range(3):   # columns 30 and 31: bit j in the top bit of each half of words j and 3 + j
            w[..., j] |= ((q[:, h, :, :, 30] >> j) & 1) << (16 * h + 15)
            w[..., 3 + j] |= ((q[:, h, :, :, 31] >> j) & 1) << (16 * h + 15)
    run0 = w[..., :4].transpose(0, 2, 1, 3)
    run1 = w[..., 4:].transpose(0, 2, 1, 3)
    words = np.concatenate([run0.reshape(T, nb, 256), run1.reshape(T, nb, 128)], axis=2)
    dd = d.astype(np.float16).view(np.uint16).astype(np.uint32).reshape(T, 2, 64, nb)
    sc = (dd[:, 0] | (dd[:, 1] << 16)).transpose(0, 2, 1)
    return np.ascontiguousarray(words).tobytes(), np.ascontiguousarray(sc).tobytes()


def pad128(q, d):
    """Rows padded (q = 8: a zero weight) to a multiple of 128, which the q4s and q3s layouts need."""
    rows = q.shape[0]
    p = (-rows) % 128
    if p:
        q = np.concatenate([q, np.full((p, q.shape[1]), 8, np.uint8)])
        d = np.concatenate([d, np.full((p, d.shape[1]), 6.2e-5, np.float16)])
    return q, d


def q4_of(w):
    """[rows, cols] float -> (q4 nibbles 0..15 [rows, cols], fp16 scales)."""
    packed, d16 = quantize_q4(np.ascontiguousarray(w, np.float32))
    rows = packed.shape[0]
    p = packed.reshape(rows, -1, 16)
    return np.concatenate([p & 15, p >> 4], axis=2).reshape(rows, -1), d16


def q4s(w):
    """[rows, cols] float -> (words, scales, rows after padding) in q4s."""
    q, d = pad128(*q4_of(w))
    wb, sb = to_q4s(q, d)
    return wb, sb, q.shape[0]


def q3s(w):
    q, d = pad128(*quantize_q3(np.ascontiguousarray(w, np.float32)))
    wb, sb = to_q3s(q, d)
    return wb, sb, q.shape[0]


# ---- the GGUF's tensors -> the roles the app knows --------------------------------------------------------------

# llama.cpp's qwen35 names first, then the variants of other llama.cpp architectures (qwen3next) and of Hugging
# Face's checkpoint, so that a rename in a future GGUF shows up as a clear message instead of a wrong weight.
ROLE_NAMES = {
    "attn_norm": ["attn_norm.weight", "input_layernorm.weight", "attention_norm.weight"],
    "mlp_norm": ["ffn_norm.weight", "post_attention_norm.weight", "post_attention_layernorm.weight"],
    "q_norm": ["attn_q_norm.weight", "self_attn.q_norm.weight"],
    "k_norm": ["attn_k_norm.weight", "self_attn.k_norm.weight"],
    "q": ["attn_q.weight", "self_attn.q_proj.weight"],
    "k": ["attn_k.weight", "self_attn.k_proj.weight"],
    "v": ["attn_v.weight", "self_attn.v_proj.weight"],
    "o": ["attn_output.weight", "self_attn.o_proj.weight"],
    "gate": ["attn_gate.weight", "attn_q_gate.weight", "self_attn.q_gate.weight"],
    "in_qkv": ["attn_qkv.weight", "linear_attn.in_proj_qkv.weight", "ssm_qkv.weight"],
    "in_z": ["attn_gate.weight", "linear_attn.in_proj_z.weight", "ssm_z.weight"],
    "in_b": ["linear_attn.in_proj_b.weight", "ssm_beta.weight"],
    "in_a": ["linear_attn.in_proj_a.weight", "ssm_alpha.weight"],
    "conv": ["linear_attn.conv1d.weight", "ssm_conv1d.weight"],
    "A": ["linear_attn.A_log", "ssm_a"],
    "dt": ["linear_attn.dt_bias", "ssm_dt.bias"],
    "gnorm": ["linear_attn.norm.weight", "ssm_norm.weight"],
    "wout": ["linear_attn.out_proj.weight", "ssm_out.weight"],
    "gu_gate": ["ffn_gate.weight", "mlp.gate_proj.weight"],
    "gu_up": ["ffn_up.weight", "mlp.up_proj.weight"],
    "down": ["ffn_down.weight", "mlp.down_proj.weight"],
}
FULL_ROLES = ("q_norm", "k_norm", "q", "k", "v", "o")
LINEAR_ROLES = ("in_qkv", "in_z", "in_b", "in_a", "conv", "A", "dt", "gnorm", "wout")
TOP_NAMES = {"embed": ["token_embd.weight", "model.embed_tokens.weight"],
             "final_norm": ["output_norm.weight", "model.norm.weight"],
             "lm_head": ["output.weight", "lm_head.weight"]}


class Model:
    """The GGUF read as the app's model: the shape, the layer kinds, and the tensor behind every role."""

    def __init__(self, src):
        self.src = src
        self.g = GGUF.open(src)
        m = self.g.meta
        self.vocab = len(m.get("tokenizer.ggml.tokens") or [])
        self.n_blocks = int(m.get("qwen35.block_count", 0))
        self.hidden = int(m.get("qwen35.embedding_length", 0))
        self.inter = int(m.get("qwen35.feed_forward_length", 0))
        self.n_heads = int(m.get("qwen35.attention.head_count", 0))
        self.n_kv = int(m.get("qwen35.attention.head_count_kv", 0))
        self.head_dim = int(m.get("qwen35.attention.key_length", 0))
        self.eps = float(m.get("qwen35.attention.layer_norm_rms_epsilon", 1e-6))
        self.rope_theta = float(m.get("qwen35.rope.freq_base", 10000000.0))
        self.n_kh = int(m.get("qwen35.ssm.group_count") or 0)        # linear attention: key heads
        self.n_vh = int(m.get("qwen35.ssm.time_step_rank") or 0)     # value heads
        self.dk = int(m.get("qwen35.ssm.state_size") or 0)
        self.dv = self.dk
        self.conv_k = int(m.get("qwen35.ssm.conv_kernel") or 0)
        # llama.cpp writes the MTP head's layer after the model's own: the last nextn_predict_layers blocks are its
        self.nextn = int(m.get("qwen35.nextn_predict_layers") or 0)
        self.n_layers = self.n_blocks - self.nextn
        for k, v in (("layers", self.n_layers), ("hidden", self.hidden), ("heads", self.n_heads),
                     ("head_dim", self.head_dim)):
            if not v:
                raise SystemExit(f"{self.src}: the metadata has no {k}: keys are {sorted(m)}")
        self.roles, self.layers, self.extra = self._classify()
        self.mtp_block = self.n_blocks - 1 if self.nextn else None

    def _classify(self, strict=True):
        roles = {}
        for key, cands in TOP_NAMES.items():
            roles[key] = self._pick(cands, key, strict)
        layers, extra, problems = {}, {}, []
        for i in range(self.n_layers):
            pre = f"blk.{i}."
            sub = {n[len(pre):]: n for n in self.g.tensors if n.startswith(pre)}
            found = {}
            linear = any(k in sub for k in ("ssm_out.weight", "ssm_alpha.weight", "ssm_conv1d.weight",
                                            "ssm_a", "ssm_norm.weight"))
            keys = LINEAR_ROLES if linear else FULL_ROLES
            for key in ("attn_norm", "mlp_norm", *keys, "gu_gate", "gu_up", "down"):
                for c in ROLE_NAMES[key]:
                    if c in sub:
                        found[key] = sub[c]
                        break
            if not linear:      # full attention: q | k | v | gate, fused in one tensor or not
                found.pop("q", None), found.pop("k", None), found.pop("v", None)
                for c in ("attn_qkv.weight", "attn_qkvz.weight"):
                    if c in sub:
                        found["qkv"] = sub[c]
                        break
                else:
                    for key in ("q", "k", "v"):
                        for c in ROLE_NAMES[key]:
                            if c in sub:
                                found[key] = sub[c]
            missing = [k for k in ("attn_norm", "mlp_norm", *keys, "gu_gate", "gu_up", "down")
                       if k not in found and not (k in ("q", "k", "v") and "qkv" in found)]
            if missing:
                problems.append(f"layer {i}: no tensor for {', '.join(missing)}; it has:\n    "
                                + "\n    ".join(sorted(sub)))
            roles[i] = found
            layers[i] = "linear_attention" if linear else "full_attention"
            unused = [n for s2, n in sub.items() if n not in found.values()]
            if unused:
                extra[i] = unused
        if problems and strict:
            raise SystemExit(f"{self.src}: " + "\n  ".join(problems))
        # the delta-rule geometry, from the projections when the metadata does not give it
        lin = next((i for i, k in layers.items() if k == "linear_attention"), None)
        if lin is not None and "in_qkv" in roles[lin]:
            r = roles[lin]
            qkv, z = self.shape(r["in_qkv"])[0], self.shape(r["in_z"])[0]
            if not self.n_vh:
                self.n_vh = self.shape(r["in_b"])[0]
            if not self.n_kh:
                self.n_kh = max(1, (qkv - z) // 2 // max(1, z // max(1, self.n_vh)))
            if not self.dk:
                self.dk = z // max(1, self.n_vh)
            self.dv = self.dv or self.dk
            self.conv_k = self.conv_k or self.shape(r["conv"])[1]
        return roles, layers, extra

    def _pick(self, cands, what, strict=True):
        for c in cands:
            if c in self.g.tensors:
                return c
        if strict:
            raise SystemExit(f"{self.src}: no {what} tensor (tried {cands}); "
                             f"the file has e.g. {sorted(self.g.tensors)[:8]}")
        return None

    def shape(self, name):
        return self.g.tensors[name].shape

    def config(self):
        return {"num_hidden_layers": self.n_layers, "hidden_size": self.hidden,
                "intermediate_size": self.inter, "num_attention_heads": self.n_heads,
                "num_key_value_heads": self.n_kv, "head_dim": self.head_dim, "vocab_size": self.vocab,
                "rms_norm_eps": self.eps, "layer_types": [self.layers[i] for i in range(self.n_layers)],
                "rope_parameters": {"rope_theta": self.rope_theta, "partial_rotary_factor": 0.25},
                "linear_num_key_heads": self.n_kh, "linear_num_value_heads": self.n_vh,
                "linear_key_head_dim": self.dk, "linear_value_head_dim": self.dv}

    def head_order(self):
        """Hugging Face's index of each of the GGUF's value heads.

        llama.cpp groups the delta rule's value heads the other way round: it reads them as (n_kh key heads, 2)
        transposed, so its head i is Hugging Face's i // 2 + (i % 2) * n_kh (depthwise: the app's kernels, like
        Hugging Face's, keep a head's 128 values together). pc/ornith_check_layout.py shows it for one layer.
        """
        n = self.n_vh
        return np.array([j // 2 + (j % 2) * (n // 2) for j in range(n)])

    def hf_heads(self, x, axis, block=1):
        """A GGUF delta-rule tensor's value-head axis -> Hugging Face's order (block = the head's size)."""
        inv = self.head_order()
        idx = inv if block == 1 else (inv[:, None] * block + np.arange(block)).ravel()
        return np.take(x, idx, axis=axis)

    def mat(self, name):
        """A whole tensor as float32: [rows, cols], or [n] for a vector."""
        t = self.g.tensors[name]
        if len(t.shape) != 2:
            return self.g.tensor(name)
        return self.g.rows(name, 0, t.shape[0])

    def qkv(self, i):
        """q | k | v | gate rows: the app's wqkv (Qwen3.5's query projection also makes the output gate)."""
        r, nh, hd, nkv = self.roles[i], self.n_heads, self.head_dim, self.n_kv
        qd, kd = nh * hd, nkv * hd
        if "qkv" in r:                                   # llama.cpp fuses them: q | k | v (| gate)
            w = self.mat(r["qkv"])
            rows = w.shape[0]
            if rows == 2 * qd + 2 * kd:                  # q | k | v | gate, one after the other
                return w
            if rows == 2 * qd + 2 * kd + qd:             # ... and the gate again at the end
                return np.concatenate([w[:2 * qd + 2 * kd], w[2 * qd + 2 * kd:]])
            if rows == qd + 2 * kd:                      # q | k | v: the gate is its own tensor
                gate = self.mat(r["gate"]) if "gate" in r else None
                return np.concatenate([w, gate]) if gate is not None else w
            if rows == 2 * nh * hd + 2 * kd:             # per head [query | gate], then k | v
                qg = w[:2 * qd].reshape(nh, 2, hd, -1)
                return np.concatenate([qg[:, 0].reshape(qd, -1), w[2 * qd:],
                                       qg[:, 1].reshape(qd, -1)])
            raise SystemExit(f"wqkv: {r['qkv']} has {rows} rows: not q|k|v (| gate) for "
                             f"{nh} heads of {hd} and {nkv} kv heads")
        q = self.mat(r["q"])
        if q.shape[0] == 2 * qd:                         # one tensor holding, per head, [query | gate]
            qg = q.reshape(nh, 2, hd, -1)
            q, gate = qg[:, 0].reshape(qd, -1), qg[:, 1].reshape(qd, -1)
        else:
            gate = self.mat(r["gate"])
        return np.concatenate([q, self.mat(r["k"]), self.mat(r["v"]), gate])

    def w_in(self, i):
        """q | k | v | z | b | a in Hugging Face's order (the delta rule's projections, value heads unpermuted)."""
        r, vd = self.roles[i], self.n_vh * self.dv
        qkv = self.mat(r["in_qkv"])
        qkv = np.concatenate([qkv[:-vd], self.hf_heads(qkv[-vd:], 0, self.dv)])   # the v rows
        z = self.hf_heads(self.mat(r["in_z"]), 0, self.dv)
        b = self.hf_heads(self.mat(r["in_b"]), 0)
        a = self.hf_heads(self.mat(r["in_a"]), 0)
        return np.concatenate([qkv, z, b, a])

    def wout(self, i):
        return self.hf_heads(self.mat(self.roles[i]["wout"]), 1, self.dv)         # the value columns

    def a_matrix(self, i):
        """The delta rule's decay: llama.cpp already stores -exp(A_log) (checked against A_log), in its head order.

        A_log (not this) would be positive for a decay near 1: if a future GGUF gives one, say so rather than
        quietly decaying the wrong way.
        """
        w = self.hf_heads(self.mat(self.roles[i]["A"]).reshape(-1).astype(np.float32), 0)
        if np.any(w > 0):
            print(f"  NOTE: blk.{i}.ssm_a looks like A_log, not -exp(A_log): taking -exp", flush=True)
            return -np.exp(w)
        return w

    def conv(self, i):
        w, vd = self.mat(self.roles[i]["conv"]), self.n_vh * self.dv
        w = np.concatenate([w[:-vd], self.hf_heads(w[-vd:], 0, self.dv)])         # the v channels
        k = self.conv_k or w.shape[1]
        return w.reshape(-1, k)


# ---- the MTP head, from the base repository's safetensors ---------------------------------------------------------

class Safetensors:
    """One tensor of a Hugging Face repository by byte range (its index says which shard, the shard's header where)."""

    def __init__(self, base=BASE_HF):
        self.base = base
        self.index = json.loads(self._get("model.safetensors.index.json"))["weight_map"]
        self.headers = {}

    def _get(self, path, rng=None):
        req = urllib.request.Request(self.base + path, headers={"Range": rng} if rng else {})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return r.read()
            except Exception as ex:  # noqa: BLE001  (a long download may drop: try again)
                last = ex
        raise IOError(f"{self.base + path}: {last}")

    def has(self, name):
        return name in self.index

    def rows(self, name, r0=0, r1=4):
        """Rows r0..r1 of a 2-D tensor (whole vectors), straight from its byte range."""
        shard = self.index[name]
        if shard not in self.headers:
            n = struct.unpack("<Q", self._get(shard, "bytes=0-7"))[0]
            self.headers[shard] = (8 + n, json.loads(self._get(shard, f"bytes=8-{8 + n - 1}")))
        start, h = self.headers[shard]
        e = h[name]
        size = {"BF16": 2, "F16": 2, "F32": 4}.get(e["dtype"])
        if size is None:
            raise SystemExit(f"{name}: dtype {e['dtype']} is not supported")
        a, b = e["data_offsets"]
        shape = e["shape"]
        if len(shape) == 1:
            r0, r1 = 0, shape[0]
        r0, r1 = max(0, r0), min(r1, shape[0])
        off = a + r0 * shape[1] * size if len(shape) == 2 else a
        raw = self._get(shard, f"bytes={start + off}-{start + off + (r1 - r0) * (shape[1] if len(shape) == 2 else 1) * size - 1}")
        if e["dtype"] == "BF16":
            w = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif e["dtype"] == "F16":
            w = np.frombuffer(raw, np.float16).astype(np.float32)
        else:
            w = np.frombuffer(raw, np.float32)
        return w.reshape(r1 - r0, -1) if len(shape) == 2 else w

    def tensor(self, name):
        shard = self.index[name]
        if shard not in self.headers:
            n = struct.unpack("<Q", self._get(shard, "bytes=0-7"))[0]
            self.headers[shard] = (8 + n, json.loads(self._get(shard, f"bytes=8-{8 + n - 1}")))
        start, h = self.headers[shard]
        e = h[name]
        a, b = e["data_offsets"]
        raw = self._get(shard, f"bytes={start + a}-{start + b - 1}")
        if e["dtype"] == "BF16":
            return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])
        if e["dtype"] == "F16":
            return np.frombuffer(raw, np.float16).astype(np.float32).reshape(e["shape"])
        if e["dtype"] == "F32":
            return np.frombuffer(raw, np.float32).reshape(e["shape"])
        raise SystemExit(f"{name}: dtype {e['dtype']} is not supported")


# ---- convert ------------------------------------------------------------------------------------------------------

def convert(src, outdir, idxdir, q3_spec, with_mtp=True):
    t0 = time.time()
    m = Model(src)
    c = m.config()
    g = m.g
    keep = np.load(KEEP).astype(np.int64)
    q3_layers = set() if q3_spec == "none" else set(range(int(q3_spec.split("-")[0]), int(q3_spec.split("-")[1]) + 1))
    outdir.mkdir(parents=True, exist_ok=True)
    out = open(outdir / f"{NAME}.bin", "wb")
    tensors = {}

    def put(key, wb, sb=b"", **meta):
        off = out.tell()
        out.write(wb)
        out.write(sb)
        tensors[key] = dict(offset=off, size=len(wb) + len(sb), **meta)
        out.write(b"\0" * ((-out.tell()) % 256))

    def f32(key, w):
        put(key, np.ascontiguousarray(w, np.float32).tobytes(), fmt="f32")

    def put_q4s(key, w, **meta):
        wb, sb, rows = q4s(w)
        put(key, wb, sb, rows=rows, cols=w.shape[1], scales=len(wb), fmt="q4s", **meta)

    def put_q3s(key, w, **meta):
        wb, sb, rows = q3s(w)
        put(key, wb, sb, rows=rows, cols=w.shape[1], scales=len(wb), fmt="q3s", **meta)

    print(f"{NAME}: {m.n_layers} layers ({c['layer_types'].count('linear_attention')} delta-net), "
          f"hidden {m.hidden}, vocab {m.vocab}, gate/up in q3s: {sorted(q3_layers)}", flush=True)
    f32("final_norm", m.mat(m.roles["final_norm"]).reshape(-1))
    for i in range(m.n_layers):
        r, kind = m.roles[i], m.layers[i]
        # the norms: the GGUF already holds them as 1 + w, which is what the app's kernels add, except the delta
        # rule's ssm_norm, which is a plain weight (and is plain in the GGUF too)
        f32(f"{i}.attn_norm", m.mat(r["attn_norm"]).reshape(-1))
        f32(f"{i}.mlp_norm", m.mat(r["mlp_norm"]).reshape(-1))
        if kind == "full_attention":
            f32(f"{i}.q_norm", m.mat(r["q_norm"]).reshape(-1))
            f32(f"{i}.k_norm", m.mat(r["k_norm"]).reshape(-1))
            put_q4s(f"{i}.wqkv", m.qkv(i))
            put_q4s(f"{i}.wo", m.mat(r["o"]))
        else:
            put_q4s(f"{i}.w_in", m.w_in(i))
            f32(f"{i}.conv", m.conv(i))
            f32(f"{i}.A", m.a_matrix(i))
            f32(f"{i}.dt_bias", m.hf_heads(m.mat(r["dt"]).reshape(-1), 0))
            f32(f"{i}.gnorm", m.mat(r["gnorm"]).reshape(-1))
            put_q4s(f"{i}.wout", m.wout(i))
        gu = np.concatenate([m.mat(r["gu_gate"]), m.mat(r["gu_up"])])
        (put_q3s if i in q3_layers else put_q4s)(f"{i}.wgu", gu)
        put_q4s(f"{i}.wdown", m.mat(r["down"]))
        print(f"  layer {i} ({kind}) done, {out.tell() / 2**20:.0f} MiB, {time.time() - t0:.0f} s", flush=True)

    # ---- lm_head: only the keep list's rows, in its order
    head = m.roles["lm_head"]
    hrows, cols = m.shape(head)
    packed = np.empty((len(keep), cols // 2), np.uint8)
    scales = np.empty((len(keep), cols // 32), np.float16)
    pos = np.full(hrows, -1, np.int64)
    pos[keep] = np.arange(len(keep))
    step = 8192
    for r0 in range(0, hrows, step):
        r1 = min(r0 + step, hrows)
        p = pos[r0:r1]
        if not (p >= 0).any():
            continue
        w = g.rows(head, r0, r1)
        pk, d16 = quantize_q4(w[p >= 0])
        packed[p[p >= 0]] = pk
        scales[p[p >= 0]] = d16
    del pos
    q = np.concatenate([packed.reshape(len(keep), -1, 16) & 15, packed.reshape(len(keep), -1, 16) >> 4],
                       axis=2).reshape(len(keep), cols)
    del packed
    q, d = pad128(q, scales)
    del scales
    parts = [to_q4s(q[r:r + step], d[r:r + step]) for r in range(0, q.shape[0], step)]
    wb = b"".join(p[0] for p in parts)
    sb = b"".join(p[1] for p in parts)
    del q, d, parts
    put("lm_head", wb, sb, rows=len(keep) + (-len(keep)) % 128, cols=cols, scales=len(wb), fmt="q4s",
        vocab=len(keep))
    del wb, sb
    gpu_size = out.tell()
    print(f"  lm_head {len(keep)} rows done, GPU part {gpu_size / 2**20:.0f} MiB, {time.time() - t0:.0f} s", flush=True)

    # ---- the input embedding: one record per row (nibbles then fp16 scales), every token keeps its row
    emb = m.roles["embed"]
    erows, ecols = m.shape(emb)
    rowbytes = ecols // 2 + ecols // 32 * 2
    start = out.tell()
    for r0 in range(0, erows, step):
        r1 = min(r0 + step, erows)
        pk, d16 = quantize_q4(g.rows(emb, r0, r1))
        out.write(np.concatenate([pk, d16.view(np.uint8)], axis=1).tobytes())
    tensors["embed"] = dict(offset=start, size=erows * rowbytes, rows=erows, cols=ecols, scales=ecols // 2,
                            rowbytes=rowbytes, fmt="q4row")
    out.write(b"\0" * ((-out.tell()) % 256))
    ids = keep.astype(np.uint32).tobytes()
    tensors["ids"] = dict(offset=out.tell(), size=len(ids), rows=len(keep), cols=1, scales=0, fmt="ids")
    out.write(ids)
    size = out.tell()
    out.close()
    print(f"  embedding {erows} rows + {len(keep)} ids, {size / 2**20:.0f} MiB, {time.time() - t0:.0f} s", flush=True)

    meta = {"config": dict(c, vocab_size=len(keep)), "size": size, "gpu_size": gpu_size, "tensors": tensors,
            "source": src if not src.startswith("http") else src.rsplit("/", 1)[-1], "vocab_kept": len(keep),
            "keep": KEEP.name, "q3_layers": sorted(q3_layers)}
    (outdir / f"{NAME}.json").write_text(json.dumps(meta, indent=0))
    write_index(meta, idxdir)

    if with_mtp:
        mtp(outdir, idxdir, m, keep, size)
    return 0


def write_index(meta, idxdir):
    """model.idx: as pc/export_chat.py's export_model_index_qwen35, + flags (lm_head, embed rows, ids)."""
    c, t = meta["config"], meta["tensors"]
    order, mask = ["embed", "final_norm"], 0
    for i, kind in enumerate(c["layer_types"]):
        full = kind == "full_attention"
        mask |= full << i
        mixer = ("q_norm", "k_norm", "wqkv", "wo") if full else ("w_in", "conv", "A", "dt_bias", "gnorm", "wout")
        order += [f"{i}.{k}" for k in ("attn_norm", "mlp_norm", *mixer, "wgu", "wdown")]
    order += ["lm_head", "ids"]
    flags = 1 | 2 | 4
    rp = c["rope_parameters"]
    rope_dim = int(c["head_dim"] * rp.get("partial_rotary_factor", 1.0))
    path = f"/data/{NAME}.bin".encode()
    head = struct.pack("<4s10I2f128sQI", b"PMQ5", c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"],
                       c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"],
                       8192, 248046, 248044, c["rms_norm_eps"], rp["rope_theta"], path, meta["gpu_size"],
                       len(order))
    head += struct.pack("<8I", c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"],
                        c["linear_value_head_dim"], rope_dim, mask, 248045, flags)
    recs = []
    for k in order:
        e = t[k]
        recs.append(struct.pack("<QQIIQI", e["offset"], e["size"], e.get("rows", 0), e.get("cols", 0),
                                e.get("scales", 0), FMT.get(e.get("fmt"), 0)))
    idxdir.mkdir(parents=True, exist_ok=True)
    (idxdir / "model.idx").write_bytes(head + b"".join(recs))
    print(f"  model.idx: {len(order)} records, flags {flags}, weights at {path.decode()}")


def mtp(outdir, idxdir, m, keep, size):
    """mtp.idx/mtp.bin: the MTP (draft) head in q4s, and the embedding rows of the tokens it drafts.

    llama.cpp writes Qwen3.5's MTP head as one more block after the model's layers (blk.<n>.*) plus its three
    nextn.* tensors in that block. Hugging Face calls the same weights mtp.pre_fc_norm_embedding, mtp.pre_fc_norm_hidden,
    mtp.fc, mtp.layers.0.* and mtp.norm; the app drafts with it and the full model verifies: engine_spec.inc.
    """
    t0 = time.time()
    if m.mtp_block is None:
        print("  MTP: the GGUF has no draft head: the app runs without speculative decoding", flush=True)
        return
    B = f"blk.{m.mtp_block}."
    g = m.g
    names = {n[len(B):] for n in g.tensors if n.startswith(B)}
    need = ["nextn.enorm.weight", "nextn.hnorm.weight", "nextn.eh_proj.weight", "nextn.shared_head_norm.weight",
            "attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate.weight",
            "ffn_up.weight", "ffn_down.weight"]
    missing = [n for n in need if n not in names]
    if missing:
        print(f"  MTP: the draft block {B[:-1]} has no {missing}: the app runs without speculative decoding",
              flush=True)
        return
    mat = lambda n: m.mat(B + n)              # noqa: E731
    nh, hd = m.n_heads, m.head_dim
    out = open(outdir / f"{MTP_NAME}.bin", "wb")
    recs = {}

    def put(key, data, **meta):
        off = out.tell()
        out.write(data)
        recs[key] = (off, len(data), meta.get("rows", 0), meta.get("cols", 0), meta.get("scales", 0),
                     FMT.get(meta.get("fmt"), 0))
        out.write(b"\0" * ((-out.tell()) % 256))

    def f32(key, w):
        put(key, np.ascontiguousarray(w, np.float32).tobytes(), cols=w.size)

    def q4s_(key, w):
        wb, sb, rows = q4s(w)
        put(key, wb + sb, rows=rows, cols=w.shape[1], scales=len(wb), fmt="q4s")

    f32("pre_e", mat("nextn.enorm.weight"))        # (llama.cpp stores norms as 1 + w: see convert)
    f32("pre_h", mat("nextn.hnorm.weight"))
    q4s_("fc", mat("nextn.eh_proj.weight"))
    f32("attn_norm", mat("attn_norm.weight"))
    f32("mlp_norm", mat("post_attention_norm.weight"))
    f32("q_norm", mat("attn_q_norm.weight"))
    f32("k_norm", mat("attn_k_norm.weight"))
    q = mat("attn_q.weight")
    if q.shape[0] == 2 * nh * hd:             # per head [query | gate], as in the model's own attention
        qg = q.reshape(nh, 2, hd, -1)
        q, gate = qg[:, 0].reshape(nh * hd, -1), qg[:, 1].reshape(nh * hd, -1)
    else:
        gate = mat("attn_q_gate.weight")
    q4s_("wqkv", np.concatenate([q, mat("attn_k.weight"), mat("attn_v.weight"), gate]))
    q4s_("wo", mat("attn_output.weight"))
    q4s_("wgu", np.concatenate([mat("ffn_gate.weight"), mat("ffn_up.weight")]))
    q4s_("wdown", mat("ffn_down.weight"))
    f32("norm", mat("nextn.shared_head_norm.weight"))
    wsize = out.tell()
    # the draft's own tokens: the input-embedding rows of the first ND kept tokens, in lm_head row order
    ecols = m.shape(m.roles["embed"])[1]
    rowbytes = ecols // 2 + ecols // 32 * 2
    eoff = json.loads((outdir / f"{NAME}.json").read_text())["tensors"]["embed"]["offset"]
    with open(outdir / f"{NAME}.bin", "rb") as f:
        for tokid in keep[:ND]:
            f.seek(eoff + int(tokid) * rowbytes)
            out.write(f.read(rowbytes))
    tsize = out.tell() - wsize
    out.close()
    path = f"/data/{MTP_NAME}.bin".encode()
    head = struct.pack("<4sI128sQQI", b"PMTP", ND, path, wsize, tsize, len(MTP_ORDER))
    body = b"".join(struct.pack("<QQIIQI", *recs[k]) for k in MTP_ORDER)
    idxdir.mkdir(parents=True, exist_ok=True)
    (idxdir / "mtp.idx").write_bytes(head + body)
    print(f"  {MTP_NAME}.bin: tensors {wsize / 2**20:.1f} MiB + draft table {tsize / 2**20:.1f} MiB; "
          f"mtp.idx {len(head + body)} B, {time.time() - t0:.0f} s", flush=True)


# ---- inspect -------------------------------------------------------------------------------------------------------

def inspect(src, probe_mtp=True):
    t0 = time.time()
    try:
        m = Model(src)
    except SystemExit as ex:
        print(f"the reader could not map the file: {ex}")
        return names_report(src)
    g = m.g
    c = m.config()
    print(f"source: {src}")
    print(f"read: {g.r.n_read / 2**20:.1f} MiB (the header, not the weights) in {time.time() - t0:.1f} s")
    print(f"architecture={g.meta.get('general.architecture')} name={g.meta.get('general.name')} "
          f"quant={g.meta.get('general.file_type')} tensors={len(g.tensors)}")
    print(f"vocab={m.vocab} layers={m.n_layers} hidden={m.hidden} inter={m.inter} heads={m.n_heads} kv={m.n_kv} "
          f"head_dim={m.head_dim} eps={m.eps:g} rope_theta={m.rope_theta:g}")
    print(f"delta net: key heads={m.n_kh} value heads={m.n_vh} key dim={m.dk} value dim={m.dv} conv={m.conv_k}")
    kinds = {k: c["layer_types"].count(k) for k in sorted(set(c["layer_types"]))}
    print(f"layer kinds: {kinds}")
    for k in ("qwen35.block_count", "qwen35.nextn_predict_layers", "qwen35.full_attention_interval",
              "qwen35.context_length", "qwen35.rope.dimension_count", "qwen35.rope.dimension_sections",
              "qwen35.ssm.inner_size", "qwen35.ssm.group_count", "qwen35.ssm.time_step_rank",
              "qwen35.attention.value_length", "tokenizer.ggml.pre", "tokenizer.ggml.eos_token_id",
              "tokenizer.ggml.padding_token_id", "general.file_type", "general.size_label"):
        if k in g.meta:
            v = g.meta[k]
            print(f"  {k} = {v if not isinstance(v, list) or len(v) < 16 else str(v[:16]) + ' ...'}")
    print("metadata keys: " + ", ".join(sorted(g.meta)))
    # the tensors of one layer of each kind, by role, with the shapes the app will see
    for kind in ("linear_attention", "full_attention"):
        i = next((i for i, k in m.layers.items() if k == kind), None)
        if i is None:
            continue
        print(f"\nlayer {i} ({kind}):")
        for k, v in m.roles[i].items():
            print(f"    {k:9s} {v:42s} {m.shape(v)}")
        try:
            print(f"    -> wqkv {m.qkv(i).shape}" if kind == "full_attention" else
                  f"    -> w_in {m.w_in(i).shape}, conv {m.conv(i).shape}, A {m.a_matrix(i).shape}")
        except (Exception, SystemExit) as ex:  # noqa: BLE001
            print(f"    -> NOT CONVERTIBLE: {ex}")
    if m.extra:
        print("\ntensors the app does not use (ignored):")
        for i, names in sorted(m.extra.items()):
            for n in names:
                print(f"    {n} {m.shape(n)}")
    want = {"layers": 32, "hidden": 4096, "inter": 12288, "heads": 16, "kv": 4, "head_dim": 256,
            "key heads": 16, "value heads": 32, "key dim": 128, "value dim": 128}
    got = {"layers": m.n_layers, "hidden": m.hidden, "inter": m.inter, "heads": m.n_heads, "kv": m.n_kv,
           "head_dim": m.head_dim, "key heads": m.n_kh, "value heads": m.n_vh, "key dim": m.dk, "value dim": m.dv}
    print("\nthe app's kernel shapes: " + ", ".join(
        f"{k} {got[k]}" + ("" if got[k] == want[k] else f" (WANTS {want[k]})") for k in want))
    types = {}
    for t in g.tensors.values():
        types.setdefault(t.type_name, [0, 0])
        types[t.type_name][0] += 1
        types[t.type_name][1] += t.nbytes
    print("tensors by type: " + ", ".join(f"{k} {n} ({b / 2**30:.2f} GiB)" for k, (n, b) in sorted(types.items())))
    # ---- the tokenizer, against the one the app has
    import tokenizer_bin
    tok = tokenizer_bin.read(ROOT / "chat" / "assets" / "tokenizer.bin")
    print(f"\ntokenizer.bin: {tok.n_vocab} tokens, {tok.n_merges} merges")
    rep = tokenizer_bin.compare(tok, g.meta.get("tokenizer.ggml.tokens") or [],
                                g.meta.get("tokenizer.ggml.merges") or [],
                                g.meta.get("tokenizer.ggml.token_type") or [])
    print(rep.text())
    keep = np.load(KEEP)
    print(f"keep list: {len(keep)} ids, {int(keep.min())}..{int(keep.max())}, {len(np.unique(keep))} unique")
    # ---- the MTP head: llama.cpp keeps it after the model's layers, as blk.<last>.* plus its nextn.* tensors
    in_gguf = [n for n in g.tensors if "nextn" in n]
    print(f"\nMTP head in the GGUF: {len(in_gguf)} nextn tensors" if in_gguf else "\nMTP head in the GGUF: no")
    if m.mtp_block is not None:
        print(f"  its block, blk.{m.mtp_block}: " + ", ".join(sorted(
            n.split(".", 2)[2] for n in g.tensors if n.startswith(f"blk.{m.mtp_block}."))))
    else:
        print("  no nextn_predict_layers: no speculative decoding")
    # ---- what the converter reads, against Hugging Face's checkpoint (the order of the delta rule's heads, the
    # norms' 1 + w, and a row of each kind of matrix): what catches a mistake before a 40-minute conversion
    ok = True
    if probe_mtp:
        try:
            import ornith_check_layout
            ok = ornith_check_layout.report(g, 0)
        except (Exception, SystemExit) as ex:  # noqa: BLE001
            print(f"  cannot compare with {BASE_REPO}: {ex}")
    print(f"\ndone in {time.time() - t0:.0f} s; {g.r.n_read / 2**20:.1f} MiB read")
    return 0 if ok else 1


def names_report(src):
    """The file's tensors, by name, when the roles cannot be resolved: one layer of each kind, and the top level."""
    g = GGUF.open(src)
    print(f"\nmetadata keys: {', '.join(sorted(g.meta))}")
    for k in sorted(g.meta):
        if k != "tokenizer.ggml.tokens" and not k.startswith("tokenizer.ggml.merges"):
            v = g.meta[k]
            print(f"  {k} = {v if not isinstance(v, list) or len(v) < 12 else str(v[:12]) + ' ...'}")
    print("\ntensors:")
    seen = set()
    for n, t in g.tensors.items():
        base = n.split(".", 2)[2] if n.startswith("blk.") else n
        if n.startswith("blk.") and base in seen:
            continue
        seen.add(base)
        print(f"  {n:44s} {t.type_name:5s} {t.shape}")
    import tokenizer_bin
    tok = tokenizer_bin.read(ROOT / "chat" / "assets" / "tokenizer.bin")
    rep = tokenizer_bin.compare(tok, g.meta.get("tokenizer.ggml.tokens") or [],
                                g.meta.get("tokenizer.ggml.merges") or [],
                                g.meta.get("tokenizer.ggml.token_type") or [])
    print("\n" + rep.text())
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["inspect", "convert"], nargs="?", default="convert")
    ap.add_argument("--gguf", default=None, help="the GGUF to read (a path or an https URL)")
    ap.add_argument("--quant", default="bf16", help="which file of the GGUF repository: " + ", ".join(QUANTS))
    ap.add_argument("--out", default=str(MODELS), help="where the weight files go (default models/)")
    ap.add_argument("--idx", default=str(IDX_OUT), help="where model.idx and mtp.idx go")
    ap.add_argument("--q3", default="0-31", help="layers whose gate/up stays q3s (default all: the memory)")
    ap.add_argument("--no-mtp", action="store_true", help="no draft head: no speculative decoding")
    ap.add_argument("--no-probe", action="store_true", help="inspect: do not look for the MTP head online")
    args = ap.parse_args()
    src = args.gguf or (HF + QUANTS[args.quant])
    if args.what == "inspect":
        return inspect(src, not args.no_probe)
    return convert(src, Path(args.out), Path(args.idx), args.q3, not args.no_mtp)


if __name__ == "__main__":
    sys.exit(main() or 0)
