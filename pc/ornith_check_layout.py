"""Which order does the GGUF keep the delta rule's value heads in? (a probe for pc/ornith_convert.py)

llama.cpp reorders Qwen3.5's Gated DeltaNet tensors so that its own kernels find the value heads grouped by key
head; Hugging Face's checkpoint (which the app's kernels were converted from) has another order. This compares the
small per-head tensors of one layer - the 32 decay values, dt_bias, and the beta / alpha projections - and reports
the permutation, then checks the same permutation on the big ones: `ssm_out`'s columns, `attn_gate`'s rows,
`attn_qkv`'s value rows and `ssm_conv1d`'s channels.

  python pc/ornith_check_layout.py [--gguf PATH|URL] [--layer 0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ornith_convert import HF, QUANTS, Safetensors  # noqa: E402
from gguf import GGUF  # noqa: E402

L = "model.language_model.layers."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=HF + QUANTS["bf16"])
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()
    g = GGUF.open(args.gguf)
    st = Safetensors()
    i = args.layer
    head = f"blk.{i}."

    a_gg = g.tensor(head + "ssm_a").astype(np.float64)
    a_hf = st.tensor(L + f"{i}.linear_attn.A_log").astype(np.float64)
    dt_gg = g.tensor(head + "ssm_dt.bias").astype(np.float64)
    dt_hf = st.tensor(L + f"{i}.linear_attn.dt_bias").astype(np.float64)

    # llama.cpp seems to split the 32 value heads into (16 key heads, 2) and read them the other way round:
    # gguf = hf.reshape(n_kh, 2, ...).transpose(1, 0, 2), i.e. all the first value heads, then all the second
    n_kh, n_vh = 16, 32
    perm = np.arange(n_vh).reshape(n_kh, 2).T.reshape(-1)          # hf index of each gguf head
    print(f"gguf ssm_a   {a_gg[:6]}")
    print(f"hf -exp(Alog){(-np.exp(a_hf))[:6]}")
    print(f"permutation  {perm.tolist()}")
    print(f"  heads {n_kh} x 2 read the other way: {perm.tolist()}")
    print(f"  ssm_a == -exp(A_log) permuted: {np.allclose(a_gg, (-np.exp(a_hf))[perm], atol=1e-6)}")
    print(f"  dt_bias the same permutation:  {np.allclose(dt_gg, dt_hf[perm], atol=1e-6)}")

    for name, hf_name in (("ssm_beta.weight", "in_proj_b.weight"), ("ssm_alpha.weight", "in_proj_a.weight"),
                          ("ssm_out.weight", "out_proj.weight"), ("attn_gate.weight", "in_proj_z.weight")):
        gg = g.tensor(head + name).astype(np.float64)
        hf = st.tensor(L + f"{i}.linear_attn.{hf_name}").astype(np.float64)
        print(f"\n{name}: {gg.shape} gguf, {hf.shape} hf")
        if gg.shape == hf.shape:
            same = np.isclose(gg, hf, atol=2e-2).mean()
            print(f"  as it is: {100 * same:.1f}% of the values equal")
        # rows permuted (a projection over the value heads) or columns permuted (its output)
        if gg.shape[0] == perm.size and gg.shape[1] != perm.size:
            print(f"  rows permuted by perm: {np.allclose(gg, hf[perm], atol=2e-2)}")
        if gg.shape[1] == perm.size and gg.shape[0] != perm.size:
            print(f"  cols permuted by perm: {np.allclose(gg, hf[:, perm], atol=2e-2)}")
        if gg.shape[0] == hf.shape[0] and gg.shape[0] % 128 == 0 and gg.shape[1] == hf.shape[1]:
            for block in (128,):
                blocks = gg.shape[0] // block
                if blocks == perm.size:
                    idx = (np.arange(blocks)[:, None] * block + np.arange(block)).ravel()
                    src = (perm[:, None] * block + np.arange(block)).ravel()
                    print(f"  rows permuted in {block}-blocks ({blocks} of them): "
                          f"{np.allclose(gg[idx], hf[src], atol=2e-2)}")
        if gg.shape[1] == hf.shape[1] and gg.shape[1] % 128 == 0 and gg.shape[0] == hf.shape[0]:
            blocks = gg.shape[1] // 128
            if blocks == perm.size:
                idx = (np.arange(blocks)[None, :] * 128 + np.arange(128)[:, None]).ravel()
                src = (perm[None, :] * 128 + np.arange(128)[:, None]).ravel()
                print(f"  cols permuted in 128-blocks ({blocks} of them): "
                      f"{np.allclose(gg[:, idx], hf[:, src], atol=2e-2)}")

    # the fused input projection: q | k | v, with the value rows (the last 4096) permuted?
    qkv_gg = g.tensor(head + "attn_qkv.weight").astype(np.float64)
    qkv_hf = st.tensor(L + f"{i}.linear_attn.in_proj_qkv.weight").astype(np.float64)
    nk = qkv_hf.shape[0] // 2 - 4096 // 2           # key rows: (out - value) / 2
    val = qkv_hf.shape[0] - (qkv_hf.shape[0] - 4096) // 2 - 4096
    qk = (qkv_hf.shape[0] - 4096)
    print(f"\nattn_qkv: {qkv_gg.shape} gguf, {qkv_hf.shape} hf; key rows {qk} (q and k), value rows 4096")
    print(f"  q|k part as it is: {np.allclose(qkv_gg[:qk], qkv_hf[:qk], atol=2e-2)}")
    blocks = 4096 // 128
    idx = (np.arange(blocks)[:, None] * 128 + np.arange(128)).ravel()
    src = (perm[:, None] * 128 + np.arange(128)).ravel()
    print(f"  v part with 128-blocks permuted: {np.allclose(qkv_gg[qk:], qkv_hf[qk:][src], atol=2e-2)}")
    print(f"  v part as it is: {np.allclose(qkv_gg[qk:], qkv_hf[qk:], atol=2e-2)}")

    conv_gg = g.tensor(head + "ssm_conv1d.weight").astype(np.float64)
    conv_hf = st.tensor(L + f"{i}.linear_attn.conv1d.weight").astype(np.float64).reshape(conv_gg.shape[0], -1)
    print(f"\nssm_conv1d: {conv_gg.shape} gguf, {conv_hf.shape} hf")
    print(f"  as it is: {np.allclose(conv_gg, conv_hf, atol=2e-2)}")
    qk2 = conv_hf.shape[0] - 4096
    src = (perm[:, None] * 128 + np.arange(128)).ravel()
    print(f"  channels {qk2}.. permuted in 128-blocks: "
          f"{np.allclose(conv_gg[qk2:], conv_hf[qk2:][src], atol=2e-2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
