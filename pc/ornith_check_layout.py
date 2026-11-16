"""Is the GGUF's weight layout the one the app's kernels expect? (a probe for pc/ornith_convert.py)

llama.cpp reorders Qwen3.5's Gated DeltaNet tensors so that its own kernels find the value heads grouped by key
head, and it writes every RMSNorm as 1 + w. Hugging Face's checkpoint - which the app's original 9B was converted
from - has the heads the other way round and stores the norms as w. This compares one layer's tensors with
ornith-ai/Ornith-1.5-9B and says, for each, what the GGUF holds:

  - the 32 decay values, dt_bias, beta and alpha: the head permutation (n_kh key heads, 2, transposed)
  - ssm_out's columns, attn_gate's rows, attn_qkv's value rows, ssm_conv1d's value channels: the same, in 128-blocks
  - the norms: 1 + w, except the delta rule's own norm, which is plain w
  - the matrices the conversion copies as they are (embeddings, output head, attention, MLP): equal

  python pc/ornith_check_layout.py [--gguf PATH|URL] [--layer 0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ornith_convert import BASE_REPO, HF, QUANTS, Safetensors  # noqa: E402
from gguf import GGUF  # noqa: E402

L = "model.language_model.layers."


def report(g, layer=0):
    """Print the comparison for one layer; returns True when every check holds."""
    st = Safetensors()
    i = layer
    head = f"blk.{i}."
    ok = True

    def say(label, good, detail=""):
        nonlocal ok
        ok = ok and bool(good)
        print(f"  {label:56s} {'yes' if good else 'NO'}{detail}")

    # the order of the value heads, from the 32 decay values
    a_gg = g.tensor(head + "ssm_a").astype(np.float64)
    a_hf = st.tensor(L + f"{i}.linear_attn.A_log").astype(np.float64)
    n_vh = a_gg.size
    # the GGUF's head i holds Hugging Face's head perm[i]: all the first value heads, then all the second
    perm = np.arange(n_vh).reshape(n_vh // 2, 2).T.ravel()
    blocks = (perm[:, None] * 128 + np.arange(128)).ravel()          # the same, 128 values a head
    print(f"delta rule of layer {i}: {n_vh} value heads; the GGUF's head i holds Hugging Face's "
          f"{perm[:4].tolist()}... (all the first, then all the second of each pair)")
    say("ssm_a == -exp(A_log) in that order", np.allclose(a_gg, (-np.exp(a_hf))[perm], atol=1e-5))
    say("ssm_dt.bias in that order",
        np.allclose(g.tensor(head + "ssm_dt.bias"), st.tensor(L + f"{i}.linear_attn.dt_bias")[perm], atol=1e-5))
    for name, hf_name, axis, idx in (("ssm_beta.weight", "in_proj_b.weight", 0, perm),
                                     ("ssm_alpha.weight", "in_proj_a.weight", 0, perm),
                                     ("attn_gate.weight", "in_proj_z.weight", 0, blocks),
                                     ("ssm_out.weight", "out_proj.weight", 1, blocks)):
        gg = g.tensor(head + name).astype(np.float64)
        hf = st.tensor(L + f"{i}.linear_attn.{hf_name}").astype(np.float64)
        got = np.take(hf, idx, axis=axis) if gg.shape == hf.shape else None
        say(f"{name} in that order", got is not None and np.allclose(gg, got, atol=2e-2))

    qkv_gg = g.tensor(head + "attn_qkv.weight").astype(np.float64)
    qkv_hf = st.tensor(L + f"{i}.linear_attn.in_proj_qkv.weight").astype(np.float64)
    qk = qkv_hf.shape[0] - 4096
    say("attn_qkv q|k rows as they are", np.allclose(qkv_gg[:qk], qkv_hf[:qk], atol=2e-2))
    say("attn_qkv v rows in that order", np.allclose(qkv_gg[qk:], qkv_hf[qk:][blocks], atol=2e-2))
    conv_gg = g.tensor(head + "ssm_conv1d.weight").astype(np.float64)
    conv_hf = st.tensor(L + f"{i}.linear_attn.conv1d.weight").astype(np.float64).reshape(conv_gg.shape[0], -1)
    say("ssm_conv1d v channels in that order", np.allclose(conv_gg[qk:], conv_hf[qk:][blocks], atol=2e-2))

    # the norms: 1 + w in the GGUF, except the delta rule's
    for gg_name, hf_name, plus_one in (
            ("output_norm.weight", "model.language_model.norm.weight", True),
            (head + "attn_norm.weight", L + f"{i}.input_layernorm.weight", True),
            (head + "post_attention_norm.weight", L + f"{i}.post_attention_layernorm.weight", True),
            (head + "ssm_norm.weight", L + f"{i}.linear_attn.norm.weight", False)):
        gg = g.tensor(gg_name).astype(np.float64)
        hf = st.tensor(hf_name).astype(np.float64)
        want = hf + 1.0 if plus_one else hf
        say(f"{gg_name.split('.', 2)[-1] if gg_name.startswith('blk') else gg_name} is "
            f"{'1 + w' if plus_one else 'w'} as the app wants it", np.allclose(gg, want, atol=2e-3))
    # the tensors the conversion copies untouched
    for gg_name, hf_name in (("token_embd.weight", "model.language_model.embed_tokens.weight"),
                             ("output.weight", "lm_head.weight"),
                             (f"blk.0.ffn_gate.weight", L + "0.mlp.gate_proj.weight"),
                             (f"blk.3.attn_q.weight", L + "3.self_attn.q_proj.weight"),
                             (f"blk.3.attn_output.weight", L + "3.self_attn.o_proj.weight")):
        a = g.rows(gg_name, 0, 4)
        b = st.rows(hf_name, 0, 4)
        say(f"{gg_name} as it is", a.shape == b.shape and np.allclose(a, b, atol=2e-2))
    print("  => " + ("the layout is the one the converter assumes" if ok else "SOMETHING IS NOT AS EXPECTED"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=HF + QUANTS["bf16"])
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()
    return 0 if report(GGUF.open(args.gguf), args.layer) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
