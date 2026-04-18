# Qwen3.6 oMLX Tuning Notes

## Scope

These notes capture local tuning work for:

- model: `Qwen3.6-35B-A3B-8bit`
- draft model: `Qwen3.6-35B-A3B-DFlash`
- runtime: local oMLX app build on Apple Silicon

The focus was maximizing practical inference speed for the real workload:

- preserve-thinking enabled
- agentic / multi-turn usage
- mixed short-turn and longer-context requests

## Key findings

### 1. Preserve thinking is effectively required

For this model, preserving reasoning across turns is necessary for stable behavior.
Disabling preserve-thinking led to poor continuity and looping behavior.

Implication:

- benchmark and tune with preserve-thinking on
- do not optimize by removing reasoning replay for this model

### 2. DFlash helps, but only when it falls back early enough

With the default `DFLASH_MAX_CTX=4096`, DFlash remained active for medium-length
prompts and caused severe slowdowns.

After reducing the cutoff to `2000`, behavior improved dramatically:

- short prompt / decode-heavy requests stayed on DFlash and became much faster
- longer prompt / prefill-heavy requests fell back to the VLM engine sooner

This was the biggest tuning win.

## Persistent settings added to oMLX

Local code changes added:

- `ModelSettings.dflash_max_ctx`
- `SchedulerSettings.prefill_batch_size`
- `SchedulerSettings.prefill_step_size`

These are now persisted through settings and available in the rebuilt app.

## Benchmarks

### DFlash threshold tuning

#### Before: DFlash active too long

Rough observed results:

- short/decode-heavy request: ~20 tok/s
- medium prompt (~3778 prompt tokens): ~3 tok/s

#### After: `dflash_max_ctx = 2000`

Observed:

- short/decode-heavy request: ~160 tok/s
- medium prompt (~3778 prompt tokens): ~44 tok/s

Conclusion:

- `dflash_max_ctx = 2000` is clearly better than the default 4096 for this model/workload

### Prefill step size sweep

Tested with:

- `max_concurrent_requests = 8`
- `prefill_batch_size = 1`
- preserve-thinking style long prompt path

Results:

| prefill_step_size | avg elapsed (s) | avg tok/s |
|---|---:|---:|
| 2048 | 1.821 | 52.72 |
| 1024 | 1.942 | 49.43 |
| 512  | 2.118 | 45.32 |
| 256  | 2.365 | 40.59 |
| 128  | 2.888 | 33.24 |

Conclusion:

- larger step size was better
- the default `2048` was the best tested value
- small step sizes do **not** help here

### Prefill batch size sweep (4 concurrent long requests)

Tested with:

- `max_concurrent_requests = 8`
- `prefill_step_size = 2048`
- 4 simultaneous long-context requests

Results:

| prefill_batch_size | wall time (s) | aggregate tok/s |
|---|---:|---:|
| 1 | 3.971 | 96.71 |
| 2 | 3.759 | 102.15 |
| 4 | 3.691 | 104.04 |
| 8 | 3.683 | 104.28 |

Conclusion:

- prompt-side batching helps
- `8` was best, but only slightly better than `4`

### Higher prompt batch sweep (16 concurrent long requests)

To make values above 8 meaningful, the benchmark was rerun with:

- temporary `max_concurrent_requests = 25`
- 16 simultaneous long-context requests

Results:

| prefill_batch_size | wall time (s) | aggregate tok/s |
|---|---:|---:|
| 8  | 11.126 | 138.06 |
| 12 | 10.853 | 141.53 |
| 16 | 11.404 | 134.70 |
| 25 | 11.761 | 130.60 |

Conclusion:

- `12` was best in the higher-load test
- `25` was worse than 8 and 12
- the “25” idea did **not** win on this machine/workload

## Interpretation

### oMLX knobs are not identical to llama.cpp batch knobs

In current oMLX:

- `max_concurrent_requests` controls decode-side concurrency / active sequences
- `prefill_batch_size` is the closest equivalent to prompt-side batching
- `prefill_step_size` controls prompt chunk size

This is not exactly the same as llama.cpp `--batch-size` / `--ubatch-size`, but
`prefill_batch_size` is the nearest conceptual match for prompt-side batching.

### Why the twitter / llama.cpp “25” result didn’t transfer directly

Likely reasons:

- different runtime architecture
- different batching semantics
- preserve-thinking workload growth
- DFlash + fallback interactions
- model and bitness differences

## Recommended live config

Current recommended live settings:

```json
{
  "scheduler": {
    "max_concurrent_requests": 8,
    "prefill_batch_size": 8,
    "prefill_step_size": 2048
  }
}
```

Per-model:

```json
{
  "is_default": true,
  "max_context_window": 200000,
  "dflash_enabled": true,
  "dflash_draft_model": "/Users/dr/Models/Text/Qwen3.6-35B-A3B-DFlash",
  "dflash_max_ctx": 2000
}
```

Global cache:

```json
{
  "hot_cache_max_size": "70GB"
}
```

## Validated

- **Preserve-thinking is required** for `Qwen3.6-35B-A3B-8bit` in the intended agentic workflow.
- **PR #814 behavior works end-to-end** after local integration:
  - `reasoning_content` / Anthropic `thinking` can be reconstructed back into `<think>` blocks
  - preserved reasoning can be recalled on later turns
- **DFlash is useful only when it falls back early enough**:
  - `dflash_max_ctx = 2000` was clearly better than the default 4096
- **The best tested `prefill_step_size` was 2048**
- **Prompt-side batching helps**, and **`prefill_batch_size = 8`** is a strong practical live default
- **Hot cache at 70GB** is working and persisted
- **The app now persists the important local tuning knobs**:
  - `dflash_max_ctx`
  - `prefill_batch_size`
  - `prefill_step_size`

## Invalidated

- **“25” is not a winning live value in oMLX for this workload**
  - `max_concurrent_requests = 25` was not a good practical setting
  - `prefill_batch_size = 25` under higher concurrent load was worse than 12 and 8
- **The current oMLX `max_concurrent_requests` knob is not equivalent to llama.cpp batch sizing**
  - it behaves more like concurrency / decode-side batching
  - it is not the same thing as llama.cpp prompt-eval batching
- **Tiny prompt chunks do not help**
  - shrinking `prefill_step_size` from 2048 toward 128 consistently hurt throughput

## Operational note

There is also a local helper script for maintenance and rebuilds:

- `scripts/update-local-build.sh`

It is currently uncommitted unless explicitly added later.

## Summary

Best practical optimizations found:

1. Keep preserve-thinking on
2. Enable DFlash
3. Set `dflash_max_ctx = 2000`
4. Keep `prefill_step_size = 2048`
5. Use `prefill_batch_size = 8`
6. Keep `max_concurrent_requests = 8`

Biggest win:

- early DFlash fallback

Second-order win:

- modest prompt-side batching

Not helpful:

- tiny prefill step sizes
- pushing prompt batch size to 25 on this workload
