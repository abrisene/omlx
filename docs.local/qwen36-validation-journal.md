# Qwen3.6 Validation Journal

## Snapshot

- Repo branch: `integration/local-patches`
- Model: `Qwen3.6-35B-A3B-8bit`
- Draft model: `Qwen3.6-35B-A3B-DFlash`

## Final live settings

```json
{
  "scheduler": {
    "max_concurrent_requests": 8,
    "prefill_batch_size": 8,
    "prefill_step_size": 2048
  },
  "cache": {
    "hot_cache_max_size": "70GB"
  },
  "qwen": {
    "is_default": true,
    "max_context_window": 200000,
    "dflash_enabled": true,
    "dflash_draft_model": "/Users/dr/Models/Text/Qwen3.6-35B-A3B-DFlash",
    "dflash_max_ctx": 2000
  }
}
```

## Validated hypotheses

### Preserve-thinking support is needed and working

- The model behaves better when prior reasoning is preserved across turns.
- Local integration of PR #814 was successful.
- End-to-end validation showed that replaying `reasoning_content` allowed the model to recover hidden prior reasoning on a later turn, while the control path without replay did not.

### DFlash can be a big win for short turns

- With DFlash active and `dflash_max_ctx = 2000`, short decode-heavy requests became dramatically faster.
- This was the single most important speed win discovered.

### DFlash must fall back early for longer prompt-heavy turns

- Letting DFlash stay active too long was harmful for preserve-thinking style workloads.
- For the tested long-context path, `dflash_max_ctx = 2000` was much better than the default 4096.

### Prefill step size should stay large

- Tested `prefill_step_size` values:
  - 2048
  - 1024
  - 512
  - 256
  - 128
- Best result was `2048`.
- Smaller chunks consistently degraded performance.

### Prefill batch size does help

- Tested `prefill_batch_size` values under concurrent long-prompt load:
  - 1
  - 2
  - 4
  - 8
- `8` was the best practical live value in the first sweep.

### Higher prompt-side batching above 8 can help somewhat, but not to 25

- Tested under heavier concurrent load:
  - 8
  - 12
  - 16
  - 25
- `12` was the best in the high-load test.
- `25` was worse than 12 and worse than 8.

## Invalidated hypotheses

### “25 is the sweet spot” did not hold here

- It may be valid for another runtime / model / benchmark methodology.
- It was not the best value for this oMLX + Qwen3.6 + preserve-thinking workload.

### oMLX `max_concurrent_requests` is not the same as llama.cpp batch size

- The tested oMLX concurrency knob behaved as decode/concurrency control, not prompt-eval batching.
- The closest local equivalent to llama.cpp-style prompt-side tuning is `prefill_batch_size`.

### Tiny prefill chunks are not beneficial

- The expectation that smaller prompt chunks might improve throughput was not supported by testing.

## Code changes validated

The following local capabilities were added and validated:

- persistent `dflash_max_ctx`
- persistent `prefill_batch_size`
- persistent `prefill_step_size`
- Qwen3.6 reasoning-aware requests now fall back off DFlash onto the standard VLM path
- unterminated `<think>` blocks are now treated as reasoning content instead of plain visible text

## Additional validation: reasoning extraction fix

### Problem observed

Qwen3.6 with `enable_thinking=true` often produced visible reasoning text in
`content` while leaving `reasoning_content=null`, especially when the DFlash
path was active or when the `<think>` block was not cleanly closed before the
output budget ran out.

### Fixes applied

- DFlash now falls back to the standard VLM path for reasoning-aware requests
- non-streaming thinking extraction now handles open `<think>` blocks with no
  closing `</think>`
- scheduler finalization preserves the synthetic open-think prefix when needed

### Evidence

#### Good case

With:

- `enable_thinking=true`
- `preserve_thinking=true`
- `thinking_budget=64`
- `max_tokens=256`

Observed:

- `content = "4"`
- `reasoning_content` contained the model's thinking text

#### Tight-budget case

With:

- `thinking_budget=32`
- `max_tokens=128`

Observed:

- `reasoning_content` was present
- visible output still leaked some partial reasoning text

Conclusion:

- the extraction/pathing bug is fixed
- but too-small `thinking_budget` / `max_tokens` can still produce awkward
  partial outputs, so Qwen3.6 needs enough room to both think and answer cleanly

These are now available through the rebuilt app and persisted in settings.

## Practical takeaway

For this machine and workload:

1. preserve-thinking on
2. DFlash enabled
3. `dflash_max_ctx = 2000`
4. `prefill_step_size = 2048`
5. `prefill_batch_size = 8`
6. `max_concurrent_requests = 8`

This is the current known-good baseline.
