---
license: mit
language:
  - en
  - es
  - pt
  - ru
tags:
  - ps4
  - homebrew
  - playstation
  - quantized
  - 4-bit
  - qwen3
base_model: ornith-ai/Ornith-1.5-9B
pipeline_tag: text-generation
---

# Station 9B

**Ornith-1.5-9B, re-cut for a PlayStation 4's GPU.** The four files here are what
[Station AI](https://github.com/mobjakdebug/garlicornith) — a PS4 homebrew app — loads from `/data/models`: 4.32 GiB of
4-bit weights, a draft head, and the two index files that say where every tensor sits.

This is **not** a GGUF and **not** a `transformers` checkpoint. It is the console's own layout, written by the
converter in the app's repository for the kernels that read it. If you want the model on a PC, take
[the base model](#where-it-comes-from) or any GGUF of it; if you want it on a PS4, you are in the right place.

<img src="screenshots/chat.png" alt="Station AI answering on a PlayStation 4" width="820">

## The files

| File | Size | What it is |
| --- | --- | --- |
| `ornith-1.5-9b-q4s.bin` | 4 641 752 088 B (4.32 GiB) | the weights: `[final norm, 32 layers, trimmed lm_head]` for the GPU, then the input-embedding rows, then the token ids |
| `ornith-1.5-9b-mtp.bin` | 212 420 608 B (203 MiB) | the multi-token-prediction head, so the app drafts two tokens ahead and the full model verifies them |
| `model.idx` | 11 312 B | the layout: a `PMQ5` header with the shapes, and the offset and length of every tensor in the big file |
| `mtp.idx` | 588 B | the same for the draft head, as `PMTP` |

## Running it on the console

1. Install **Station AI** from its [releases](https://github.com/mobjakdebug/garlicornith/releases): the small package,
   the one without weights in it. A PS4 with [GoldHEN](https://github.com/GoldHEN/GoldHEN) and ~5 GiB free in `/data`.
2. On first start the app asks for a repository. Type `officialmobai/station-9b` (or the whole
   `https://huggingface.co/officialmobai/station-9b`).
3. It reads this repository's file list, fetches the `.idx` first and then the two `.bin` files into
   `/data/models/station-9b/`, resuming with `Range:` if the connection drops, and starts the model when it is done.

No downloader? Copy the four files into `/data/models/station-9b/` by FTP and point the app at them, or put the two
`.bin` files in `/data` — that is where `model.idx` says to look when the app has no weights of its own.

## What it costs

The app runs the whole model on the console's GPU: **about 3.8 GiB of weights plus the KV cache**, which is what a
base PS4's 5.5 GiB of usable GDDR5 has room for, alongside the UI and the video buffers.

| | base PS4 |
| --- | --- |
| generation, no draft head | ~20 tokens/s |
| generation, with `-mtp.bin` | ~31–39 tokens/s |

Those are the numbers the app reports as it generates, on a base console; a Pro is faster, and a long context is
slower because there is more of it to read.

## What is inside, format by format

| Tensors | Format | Why |
| --- | --- | --- |
| MLP gate and up (32 × 12 288 × 4096) | `q3s`, 3.5 bits | the two biggest matrices, in the format that fits the console's memory |
| every other matrix | `q4s`, 4.5 bits | 4-bit groups of 32 with the SWAR dot products the GPU kernels use |
| input embedding | `q4row` | kept out of GPU memory: the CPU reads the one row of each token from the file |
| `lm_head` | 4 bits, 248 070 of 248 320 rows | the rows ordered by frequency, plus the id of each row, so the frequent ones are cached |
| norms, `conv1d`, `A`, `dt_bias`, `gnorm` | f32 | read, not approximated: they are small and the kernels want them that way |

The 32 delta-rule value heads are put back in Hugging Face's order (llama.cpp stores them interleaved as
`(16 key heads × 2)` transposed), the RMSNorm weights are written as llama.cpp stores them (already offset), and
`ssm_a` is kept as `-exp(A_log)`. Each of those is checked tensor by tensor by `pc/ornith_check_layout.py`, and the
converter says nothing about them at run time because there is nothing left to say: the layout was verified against
Hugging Face's own index, all eighteen lines.

**The vocabulary is Qwen3.5's**, all 248 070 tokens and 247 587 merges, byte for byte and in rank order with the
tokenizer the app carries — `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<think>` and `</think>` at 248045,
248046, 248044, 248068 and 248069. Chat template, keep list and stop tokens live in the app, not in these files.

## Where it comes from

* Base model: [`ornith-ai/Ornith-1.5-9B`](https://huggingface.co/ornith-ai/Ornith-1.5-9B) (MIT), converted from its
  BF16 GGUF by [`pc/ornith_convert.py`](https://github.com/mobjakdebug/garlicornith/blob/main/pc/ornith_convert.py) in
  the Station AI repository. The conversion is a pure function of the GGUF, so these bytes are reproducible: the same
  sha256 comes out of every run.
* The app it runs in is a fork of [Garlic LLM](https://github.com/leonardo-alegria/garlic-llm) by Leonardo Alegría,
  built with the [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain) and
  [Zig](https://ziglang.org)'s clang and lld, and run through [GoldHEN](https://github.com/GoldHEN/GoldHEN).
* The mark in the app is Station AI's own; its wordmark is set in [Inter](https://rsms.me/inter/) by Rasmus Andersson
  (SIL OFL 1.1).

## Limitations

* **A phone-sized model, not a frontier one.** Nine billion parameters at 4.5 bits is a capable assistant for chat,
  summaries, small code and translation; it is not a substitute for a large model on hard reasoning, and it will make
  things up with confidence.
* **Quantised, so slightly dumber than the base.** The two biggest matrices are at 3.5 bits; expect small
  regressions in long arithmetic and unusual languages.
* **Four languages are what the app tests** (English, Spanish, Portuguese, Russian). Others work — the vocabulary is
  Qwen3.5's — but they have not been checked on hardware.
* **It only runs in that one app**, on that one console. There is no `config.json`, no `tokenizer.json` and no
  `safetensors` here on purpose: those would be a different, slower thing.

---

<p align="center"><img src="logo.png" alt="Station AI" width="120"></p>
<p align="center">Made for the couch. No internet needed once it is installed.</p>
