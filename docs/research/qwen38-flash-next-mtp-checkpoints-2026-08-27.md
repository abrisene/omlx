# Qwen3.8 Flash-Next MTP checkpoints and oMLX patch path

Research date: 2026-08-27

## Request Type

Current best-practice research with source-backed checkpoint and runtime verification.

## Direct Answer

Yes. There are real Qwen3.8 Flash-Next checkpoints and derivatives that carry native MTP weights, but the important split is:

1. The official Qwen repo has native MTP in the model config and serves as the source of truth for the embedded head.
2. Some community conversions preserve those tensors in the same checkpoint.
3. Other community repos extract the MTP head into a separate drafter artifact.
4. If a conversion dropped `mtp.*` tensors, you cannot “turn MTP on” with config alone. You need a checkpoint that already contains the head, or a separate drafter path.

For oMLX specifically, first-class `qwen4_exp` plus Lightning MTP support landed in upstream oMLX 0.6.3, with initial support in PR #3169 and production hardening in PR #3174. Older builds without explicit `qwen4_exp` native-MTP support are called out as incompatible with the checkpoint.

## What I Found

### Official source checkpoint

- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
  - Multimodal `image-text-to-text` model.
  - `model_type: qwen4_exp`.
  - `architectures: Qwen4ExpForConditionalGeneration`.
  - `mtp.num_hidden_layers: 1`, `mtp.hybrid: true`.
  - BF16, 180B params on the Hub.

Interpretation: this is the upstream source model with native MTP described in the config. I did not find evidence here of a separate official “MTP-only” checkpoint; the MTP appears embedded in the base model.

### MTP-preserving community conversions

- [Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP)
  - MLX safetensors, 4-bit quantization, multimodal `image-text-to-text`.
  - The repo card says it was rebuilt from the official BF16 checkpoint with the native MTP draft block preserved.
  - It explicitly reports 3,747 total indexed tensors, including 76 MTP entries.
  - Weight shards: 22.
  - Weight size: 113.209 GB / 105.434 GiB.
  - The repo warns that runtimes without explicit `qwen4_exp` native-MTP support may reject the MTP tensors.

This is the strongest evidence I found for an MTP-bearing MLX checkpoint that is meant to be loaded as a full model.

### Standalone MTP drafter artifacts

- [sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16)
  - MLX safetensors, BF16, 3B params.
  - The repo card says it extracts the native multi-token-prediction head from the official BF16 weights into a standalone speculative drafter.
  - It says public MLX conversions of the base model drop these tensors, and this repo restores them.
  - It reports 31 `mtp.*` tensors in the provenance section.
  - Size: about 4.9 GB.

- [agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF)
  - GGUF Q8_0 draft head, size 4.14 GB.
  - The repo card describes it as the Qwen3.8 Flash-Next MTP draft head.
  - It is framed as a draft artifact to pair with any Qwen3.8-Flash-Next GGUF target.

These are useful if the model you actually have does not preserve native MTP and you want to attach a drafter, but they are not full oMLX serving checkpoints by themselves.

## oMLX Status

- [jundot/omlx releases](https://github.com/jundot/omlx/releases)
  - oMLX 0.6.3, released 2026-08-27, says it now serves `qwen4_exp` for text and image requests, including structured tool calls, continuous batching, prefix-cache state preservation, oQ conversion, and Lightning MTP.
  - The release notes attribute initial support to PR [#3169](https://github.com/jundot/omlx/pull/3169) and production hardening to PR [#3174](https://github.com/jundot/omlx/pull/3174).
  - The same release note says older builds without explicit `qwen4_exp` native-MTP support are not compatible with the checkpoint.

### Open adjacent PRs in upstream oMLX

These are not the core enablement patches, but they show active qwen4_exp / MTP follow-up work in upstream:

- [#3214](https://github.com/jundot/omlx/pull/3214) - cache/scheduler fix for QSA batching.
- [#3219](https://github.com/jundot/omlx/pull/3219) - prefix-cache reconstruction for mixed Qwen4 QSA text/MRoPE positions.
- [#3221](https://github.com/jundot/omlx/pull/3221) - oQ quantization registration for `qwen4_exp`.
- [#3225](https://github.com/jundot/omlx/pull/3225) - keeps the dense Qwen3.5 MTP patch off qwen4 hidden capture.

The practical reading is that the MTP path is real and upstreamed, but qwen4_exp edge cases are still being hardened.

## Version Note

- This is time-sensitive as of 2026-08-27.
- The oMLX answer hinges on 0.6.3 or newer with explicit `qwen4_exp` native-MTP support.
- Community conversion behavior varies: some repos preserve native MTP, others extract it into a separate drafter, and some drop it entirely.

## Caveats

- A config block alone does not prove usable MTP tensors are present in a converted checkpoint.
- If a model page says “public conversions drop these tensors,” that means a runtime patch cannot recreate them from nothing.
- I did not verify the hidden tensor keys inside every shard of the largest repos; I relied on the repo cards and the model file manifests that explicitly name MTP entries.

## Reusable Takeaway

If you want MTP in oMLX, start from a checkpoint that already contains native `mtp.*` weights, or from a separate drafter artifact built from the official BF16 release. Then run it on oMLX 0.6.3 or newer, because that is the upstream release line that explicitly advertises `qwen4_exp` and Lightning MTP support.

## Sources

- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP)
- [sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16)
- [agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF)
- [jundot/omlx releases](https://github.com/jundot/omlx/releases)
- [jundot/omlx PR #3169](https://github.com/jundot/omlx/pull/3169)
- [jundot/omlx PR #3174](https://github.com/jundot/omlx/pull/3174)
- [jundot/omlx PR #3214](https://github.com/jundot/omlx/pull/3214)
- [jundot/omlx PR #3219](https://github.com/jundot/omlx/pull/3219)
- [jundot/omlx PR #3221](https://github.com/jundot/omlx/pull/3221)
- [jundot/omlx PR #3225](https://github.com/jundot/omlx/pull/3225)
