# Garlic LLM · Ornith-1.5-9B

[Garlic LLM](https://github.com/leonardo-alegria/garlic-llm) runs a 9-billion-parameter language model on a
PlayStation 4's own GPU, offline. Upstream it runs Qwen3.5-9B; **this repository runs
[Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF) instead**, converted from the model's GGUF into
the formats the console's kernels read, and built as a PS4 package by GitHub Actions.

Ornith-1.5-9B is the same architecture as Qwen3.5-9B (llama.cpp's `qwen35`: 32 layers, 4096 hidden, 256-wide
attention heads, three Gated DeltaNet layers per full-attention one, a 1-layer MTP head), so the app's compute
kernels, its weight formats, its 4-bit attention cache and its speculative decoding all carry over unchanged. What
changes is the weights, the layout inside them, the names in the indexes and a few strings.

The model is a far stronger one: Ornith-1.5-9B scores 70.6 on SWE-bench Verified, 86.4 on GPQA Diamond and 46.2 on
Terminal-Bench 2.1, against 53.2 / 81.7 / 21.3 for Qwen3.5-9B on the same table in its model card.

## What is in here

| Path | What it is |
| --- | --- |
| `pc/ornith_convert.py` | **the port**: Ornith-1.5-9B's GGUF → the four files the app reads, in the layout of the shipped 9B |
| `pc/gguf.py` | a GGUF reader with no dependencies beyond numpy (metadata, tensor table, dequantization of F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K, by byte range) |
| `pc/ornith_check_layout.py` | compares the GGUF's tensors with Hugging Face's checkpoint: the order of the delta rule's value heads, the norms, a row of each matrix |
| `pc/vocab_order.npy`, `chat/assets/tokenizer.bin` | the app's vocabulary and the lm_head keep list: Ornith's tokens and merges are identical to Qwen3.5's, so both are used unchanged |
| `pc/ci_toolchain.py` | fetches the OpenOrbis PS4 toolchain and Zig for the build |
| `.github/workflows/build.yml` | convert + compile + package, on GitHub Actions; publishes the package as an artifact |
| `.github/workflows/probe.yml` | cheap checks: what the GGUF holds, what the toolchain release holds |
| `chat/`, `shaders/`, `build_chat.py`, … | Garlic LLM itself, unchanged except for the model's name and the package's files |

## The conversion

`python pc/ornith_convert.py convert --gguf models/Ornith-1.5-9B-BF16.gguf` writes:

```
models/ornith-1.5-9b-q4s.bin    [GPU part: final_norm, 32 layers, trimmed lm_head][input-embedding rows][token ids]
models/ornith-1.5-9b-mtp.bin    the MTP head in q4s + the embedding rows of the 32 768 tokens it drafts
chat/assets_9b/model.idx        PMQ5: the shape, and where every tensor sits in the file
chat/assets_9b/mtp.idx          PMTP: the same for the draft head
```

The formats are the ones the shipped 9B uses, chosen for what the console has (`shaders/spec.h`, `shaders/qwen35.cl`):

| Tensors | Format | Why |
| --- | --- | --- |
| MLP gate and up (32 × 12 288 × 4096) | `q3s` | 3.5 bits a weight: the two biggest matrices, in the format that fits the console's memory |
| every other matrix | `q4s` | 4.5 bits a weight, with the SWAR dot products (`swar_mv1..4`) |
| input embedding | `q4row` | kept out of GPU memory: the CPU reads the one row of each token from the file |
| `lm_head` | 4 bits, 248 070 of 248 320 rows | the rows of `pc/vocab_order.npy`, the token ids by frequency, plus the id of each row |
| norms, `conv1d`, `A`, `dt_bias`, `gnorm` | f32 | as the kernels read them |

GPU memory: about 3.8 GiB of weights, plus the caches — the same budget the Qwen3.5 build lives in.

### Three things the GGUF does differently, and how they are undone

1. **The norms are stored as `1 + w`.** llama.cpp writes Qwen3.5's RMSNorm weights already offset. The app's kernels
   add the same 1, so they are written out as they come — *except* the delta rule's `ssm_norm`, which is a plain
   weight in both and is kept plain. (`pc/ornith_check_layout.py` shows the ±1 for every norm.)
2. **The delta rule's value heads are in llama.cpp's order.** llama.cpp reads the 32 value heads as
   (16 key heads × 2) transposed, so its head *i* is Hugging Face's `i // 2 + (i % 2) * 16`. That one permutation
   reaches `ssm_a`, `ssm_dt.bias`, `ssm_beta`, `ssm_alpha`, `ssm_out`'s columns, `attn_gate`'s rows, `attn_qkv`'s
   value rows and `ssm_conv1d`'s value channels. The converter puts them back in Hugging Face's order, which is what
   the app's kernels were written against; `pc/ornith_check_layout.py` checks it tensor by tensor.
3. **`ssm_a` is already `-exp(A_log)`.** The converter keeps it (and says so if a future GGUF ever ships the log).

The **MTP head** (speculative decoding) is in the GGUF too: llama.cpp writes it as one more block after the model's
layers, `blk.32.*`, with `nextn.enorm`, `nextn.hnorm`, `nextn.eh_proj` and `nextn.shared_head_norm` — Hugging Face's
`mtp.pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `fc`, the MTP layer and `mtp.norm`. The converter writes it to
`mtp.bin` (about 540 MiB) so the app drafts up to two tokens ahead and the full model verifies them: that is what
takes generation from ~20 to ~31–39 tokens/s on a base PS4.

**The vocabulary is the same one.** All 248 070 tokens and all 247 587 merges match the app's `tokenizer.bin` byte for
byte and in rank order (checked by `pc/tokenizer_bin.py`), and `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<think>`
and `</think>` are at the ids the C code has (248045, 248046, 248044, 248068, 248069). So the tokenizer, the keep
list and the chat template in `chat/main.c` (which already writes
`<|im_start|>assistant\n<think>\n\n</think>\n\n` for thinking off — Ornith's own template) need no change.

## Building it

```bash
python pc/ci_toolchain.py                     # OpenOrbis PS4 toolchain + Zig into toolchain/
python pc/ornith_convert.py inspect           # optional: the metadata, the tensor map, the checks
python pc/ornith_convert.py convert           # downloads the BF16 GGUF and writes the four files
python build_chat.py --full                   # build/chat/IV0000-PSAI00003_00-PS4LOCALAICHAT03.pkg
```

On GitHub Actions: push a change to `ci/build` (or run **build-ornith-ps4** by hand). The workflow installs the
toolchain and Zig, downloads the GGUF, converts it, compiles the app, builds the package and keeps:

* `garlic-ornith-ps4-pkg` — the all-in-one package (~5.5 GB, the model inside)
* `garlic-ornith-ps4-model` — the weight files and the two indexes, for an app-only package with the weights in `/data`
* `logs` — the conversion, build and inspection output

Each run also writes its logs into `ci/last-run/<utc>-<job>/` on the branch, because artifacts are not always
reachable from everywhere.

`--quant` / the workflow's `quant` input picks another file of the GGUF repository (`bf16`, `q8_0`, `q6_k`, `q5_k_m`,
`q4_k_m`); BF16 is the default because the console's own formats then lose the least. A `q4_k_m` run needs a fifth of
the bandwidth and about a third of the disk.

## Installing

1. Download `IV0000-PSAI00003_00-PS4LOCALAICHAT03.pkg` from the run's artifacts.
2. Copy it to the console and install it (GoldHEN: *Settings → Debug Settings → Game → Package Installer*).
   It installs as **Garlic LLM Ornith** (title id `PSAI00003`), next to the original Garlic LLM if that one is there.
3. About 6 GB free, or 12 GB if you copy the package to the console's own disk first. Same controls as upstream
   (✕ type, □ dictate, ○ stop, △ new chat, L1 voice, R1 run the Python, R3/touchpad think mode, L3 API,
   L2/R2 temperature).

## What this build does not have

* **Voice and dictation.** Chatterbox (speech) and Parakeet (dictation) need their own weights —
  `chatterbox-t3.bin`, `chatterbox-s3gen-mf.bin`, `parakeet-v3-q4i.bin` — which `pc/convert_parakeet.py` and
  `pc/convert_chatterbox_*.py` produce from their own checkpoints; they are not part of this conversion. Without them
  the app starts, chats and runs Python, and says voice and dictation are not available. `build_chat.py` packs them
  automatically if they are in `models/`.
* **Sampling beyond temperature.** The app samples on the GPU with temperature and min-p 0.05; Ornith's card
  suggests `top_p 0.95`, `top_k 20` and `presence_penalty 1.5` for general chat, which would need different kernels.
  Temperature is L2/R2, and 0.6–0.7 suits the model's coding recommendation.
* **Think mode is off by default**, as upstream: R3 turns it on, and a thought is capped at 1024 tokens.

## Credits and licences

Garlic LLM is © Leonardo Alegría, under the licence in [`LICENSE`](LICENSE) (see [`CREDITS.md`](CREDITS.md) for the
fonts, Twemoji, MicroPython and the models it ships). Ornith-1.5-9B is MIT-licensed. This repository adds the GGUF
converter, the layout checks and the CI; the app, the kernels and the build scripts are upstream's.
