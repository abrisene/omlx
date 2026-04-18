# New Samplers Roadmap

## Branch

- `feature/new-samplers`

## Goal

Add a stronger sampler stack to oMLX for reasoning-heavy and loop-prone models.

Planned wave:

- DRY
- top-nσ
- min-k
- dynamic temperature
- sampler priority
- `temperature_last`

## Why these

### DRY

Practical anti-loop and anti-repetition control. High utility for models that
can drift into repetitive trajectories.

### Top-nσ

Recent logit-space truncation method. Promising because it filters in logit
space rather than probability space, making token selection less sensitive to
temperature changes.

Reference:
- ACL 2025: Top-nσ: Eliminating Noise in Logit Space for Robust Token Sampling of LLM

### Min-k

Very recent local logit-shape truncation method. Uses the structure of the top
sorted logits to detect a semantic cliff rather than relying on global
probability-space or global-variance filtering.

Reference:
- ACL 2026 / arXiv: Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics

### Dynamic temperature

Useful adaptive control that complements logit-space truncation and can help
preserve quality while allowing diversity.

### Sampler priority / temperature_last

Needed infrastructure so the sampler stack can be composed explicitly rather
than hardcoded in a fixed order.

## Current oMLX sampler support

Already present:

- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `xtc_probability`
- `xtc_threshold`
- `repetition_penalty`
- `presence_penalty`
- `frequency_penalty`

Current `mlx_lm.make_sampler(...)` supports only:

- temperature
- top-p
- min-p
- top-k
- XTC

Anything else will require custom oMLX-side logic.

## Implementation strategy

### Phase A — ordering infrastructure

Add:

- `sampler_priority`
- `temperature_last`

Need:

- a stable internal nickname scheme for sampler stages
- deterministic ordering of truncation and penalty stages
- default priority list

### Phase B — request/settings plumbing

Add request + per-model settings for:

- `dry_multiplier`
- `dry_base`
- `dry_allowed_length`
- `dry_sequence_breakers`

- `top_n_sigma`

- `min_k`

- `dynamic_temperature`
- `dynatemp_low`
- `dynatemp_high`
- `dynatemp_exponent`

- `temperature_last`
- `sampler_priority`

Potentially also global defaults later.

### Phase C — sampler implementation

Implement custom sampler / logits processing stages:

- DRY logits processor
- top-nσ logit filter
- min-k logit filter
- dynamic temperature warper

Then combine them into an ordered sampler chain.

## Recommended implementation order

1. sampler priority + `temperature_last`
2. dynamic temperature
3. DRY
4. top-nσ
5. min-k

Reason:

- ordering infra is required first
- dynamic temperature is self-contained
- DRY is practical and testable
- top-nσ and min-k are the more experimental logit-space filters

## Naming

Planned names:

- `dry_multiplier`
- `dry_base`
- `dry_allowed_length`
- `dry_sequence_breakers`

- `top_n_sigma`

- `min_k`

- `dynamic_temperature`
- `dynatemp_low`
- `dynatemp_high`
- `dynatemp_exponent`

- `temperature_last`
- `sampler_priority`

## Validation goals

We want to prove:

- no behavior change when the new samplers are unset
- ordering is deterministic
- dynamic temperature can be enabled/disabled cleanly
- DRY reduces pathological repetition
- top-nσ and min-k behave sensibly under varying temperatures

## Notes

- Contrastive decoding is intentionally excluded from this wave because it is a
  much larger architectural feature, not a simple sampler addition.
- Typical / TFS / Top-A are intentionally lower priority than the current wave.
