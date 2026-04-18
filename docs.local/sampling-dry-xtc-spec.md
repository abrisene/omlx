# DRY + XTC Sampling Controls

## Goal

Add first-class support for:

- DRY decoding controls
- persisted/admin-exposed XTC controls
- a small cleanup of sampling-setting consistency

while preserving existing behavior when the new settings are unset.

## Scope

### In scope

1. DRY support
   - request-level
   - per-model settings
   - optional global defaults
2. XTC settings exposure
   - persisted in model/global settings
   - surfaced in admin API/UI
3. Sampling settings cleanup
   - fix the per-model `frequency_penalty` persistence gap

### Out of scope

- new decoding algorithms beyond DRY/XTC
- major sampler architecture refactors
- backward-incompatible API changes

## Current state

### Already present

- Request-level sampling supports:
  - `temperature`
  - `top_p`
  - `min_p`
  - `presence_penalty`
  - `frequency_penalty`
  - `max_tokens`
  - `seed`
  - `thinking_budget`
  - `xtc_probability`
  - `xtc_threshold`
- Global/per-model settings already support:
  - `temperature`
  - `top_p`
  - `top_k`
  - `repetition_penalty`
  - `min_p`
  - `presence_penalty`
  - `max_tokens`
  - `force_sampling`
- XTC is already wired through:
  - request schema
  - request object
  - server sampling resolution
  - scheduler sampler construction

### Gaps

- XTC is not a first-class persisted/admin setting.
- DRY is not implemented.
- `frequency_penalty` is used in request/runtime code but is not cleanly persisted as a per-model setting.
- `top_k` exists in global/per-model resolution but is not exposed in the OpenAI request schema.

## Functional requirements

### General

- Default behavior must remain unchanged when DRY/XTC are unset.
- Request-level overrides must take precedence over model/global settings.
- DRY must be implemented as a logits processor.
- XTC should continue to use the existing sampler path.

### Precedence

For each supported sampling parameter, resolution order should be:

1. request
2. per-model settings
3. global defaults
4. hardcoded fallback

## API and settings changes

### Request-level fields

Add to OpenAI-compatible request models:

- `dry_multiplier: float | None`
- `dry_base: float | None`
- `dry_allowed_length: int | None`
- `dry_sequence_breakers: list[str] | None`

Keep existing XTC request fields:

- `xtc_probability`
- `xtc_threshold`

### Per-model settings

Add to `omlx/model_settings.py`:

- `frequency_penalty: Optional[float] = None`
- `xtc_probability: Optional[float] = None`
- `xtc_threshold: Optional[float] = None`
- `dry_multiplier: Optional[float] = None`
- `dry_base: Optional[float] = None`
- `dry_allowed_length: Optional[int] = None`
- `dry_sequence_breakers: Optional[list[str]] = None`

### Optional global defaults

Add to global sampling settings:

- `sampling_frequency_penalty`
- `sampling_xtc_probability`
- `sampling_xtc_threshold`
- `sampling_dry_multiplier`
- `sampling_dry_base`
- `sampling_dry_allowed_length`
- `sampling_dry_sequence_breakers`

## Behavior

### XTC

Keep the current sampler integration, but extend resolution so XTC can come from:

- request
- model settings
- global settings
- fallback defaults (`0.0`, `0.1`)

### DRY

Implement DRY as a logits processor that:

- detects repeated suffix continuation patterns
- penalizes candidate tokens that would continue an over-repeated sequence
- respects:
  - `dry_multiplier`
  - `dry_base`
  - `dry_allowed_length`
  - `dry_sequence_breakers`

### Defaults

- DRY is disabled if `dry_multiplier` is unset or `0`
- XTC is disabled if `xtc_probability == 0`

## Implementation plan

### Files likely to change

- `omlx/request.py`
- `omlx/api/openai_models.py`
- `omlx/server.py`
- `omlx/model_settings.py`
- `omlx/admin/routes.py`
- `omlx/scheduler.py`
- admin UI templates/JS if these controls are surfaced visually

### Core work items

1. Extend `SamplingParams`
2. Extend request schemas
3. Extend settings dataclasses + serialization
4. Update sampling parameter resolution logic
5. Add a DRY logits processor
6. Wire DRY into existing logits processor construction
7. Expose XTC/DRY in admin API/UI

## Validation rules

### XTC

- `xtc_probability`: `0.0..1.0`
- `xtc_threshold`: `0.0..1.0`

### DRY

- `dry_multiplier >= 0`
- `dry_base >= 1.0`
- `dry_allowed_length >= 1`
- `dry_sequence_breakers`: optional list of strings

Invalid values should return `400` in API/admin update paths.

## Testing

### Unit tests

- request parsing accepts new fields
- settings round-trip persists new fields
- sampling precedence works correctly
- DRY disabled path is a no-op
- DRY processor penalizes repeated continuations
- XTC model/global fallback works

### Integration tests

- baseline output unchanged when unset
- request-level DRY/XTC overrides beat model/global settings
- admin update persists and survives restart

## Risks

- DRY behavior may be tokenizer-sensitive
- poor defaults may over-penalize and hurt output quality
- admin UI may become cluttered with too many expert knobs

## Recommended rollout

### Phase 1

- fix `frequency_penalty` persistence
- expose XTC in model/global settings
- add DRY request-level support only

### Phase 2

- add DRY model/global persistence
- add DRY admin UI controls

## Suggested success criteria

- Existing users see no output behavior change without opt-in.
- XTC can be configured consistently at request/model/global levels.
- DRY works as an opt-in anti-repetition control.
- Sampling configuration semantics become more internally consistent.
