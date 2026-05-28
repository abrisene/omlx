# Rebase archive — 2026-05-28 (dev onto upstream/main `a1ade10`)

This record documents the local commits **dropped** when rebasing `dev` onto
`upstream/main` on 2026-05-28, so the work can be referenced or re-ported later.

## Safety / recovery

- **Backup branch:** `backup/dev-pre-rebase-20260528` → `e9fa4d5` (the full
  pre-rebase `dev`, every dropped line of code preserved verbatim).
- Pre-rebase base: merge-base `7395ca5` (2026-05-07). Upstream advanced +237
  commits to `a1ade10` (2026-05-28).
- Local `dev` had 22 commits over the old base; 15 were kept, 7 dropped.

To recover any dropped commit:
```bash
git checkout backup/dev-pre-rebase-20260528 -- <path>      # restore a file
git cherry-pick <hash>                                     # replay a commit (expect conflicts)
```

## Decisions

| Area | Commits | Decision | Reason |
|------|---------|----------|--------|
| Dynamic DFlash (per-request switch) | `bde04a0` | **Drop code** | PR #876 closed upstream; strict subset of upstream DFlash stack |
| Dynamic DFlash hook notes | `bd10056` | **Keep doc** | Still-relevant analysis of dflash-mlx hooks |
| Qwen3.6 sampler stack + phase bias + tuning | `2cbe199` `ab88d9b` `2418f10` `6d81b1b` `c97f950` `efdebfe` | **Drop, archived here** | Never upstreamed; conflicts heavily with upstream's rewritten scheduler/server/sampling. Genuinely-lost functionality — see inventory below |
| Qwen3.6 reasoning extraction | `08ec2b5` | **Drop, redundant** | Superseded upstream — see Investigation 3 |
| Reasoning-extraction validation doc | `a8ac9f0` | **Drop** | Documents the now-dropped `08ec2b5` |
| DeepSeek V4 support path | `3ec7437` `e9fa4d5` | **Drop, redundant** | Superseded by upstream `omlx/patches/deepseek_v4/` (same mlx-lm PR #1192 source, more complete) — see Investigation 4 |
| hot_cache_only / SSD cache (PR #895, open) | `b4e64d6` `78e75f4` `77e5d43` `1e9f2c5` `b6a1c0a` `8b1ae5b` `86176f4` `40fa478` | **Keep** | Open upstream PR; still wanted |
| Repo tooling (agent docs, rebase helper) | `591241e` `4f279dc` | **Keep** | Local workflow tooling |

> Note: open PRs **#960 (TokenBuffer seeding)** and **#961 (penalty_window)** live
> on separate `pr-960`/`pr-961` branches, not on `dev` — unaffected by this rebase.

---

## Investigation 1 — Dynamic DFlash (`bde04a0`)

### What our dynamic-DFlash did (mechanism + the exact diff summary)

Commit `bde04a0` touches a single file, `omlx/engine/dflash.py`, in two spots —
the fallback preamble of `generate()` (~`dflash.py:618`) and `stream_generate()`
(~`dflash.py:757`). The diff does only two things:

1. Renames an inline call to a named local: `if self._should_fallback(prompt_tokens):`
   becomes `use_batched = self._should_fallback(prompt_tokens)` then `if use_batched:`.
2. Rewords the log line from "DFlash context fallback: … switching to … engine" to
   "Switching to batched mode: prompt=… tokens (max_dflash_ctx=…)".

The mode-selection mechanism is unchanged and purely context-length based.
`_should_fallback` (`dflash.py:363`) is the whole decision: when the encoded prompt
reaches `dflash_max_ctx`, evict the dflash draft/target models and delegate to the
BatchedEngine/VLMBatchedEngine fallback.

Two claims in the commit message are NOT supported by the code on `dev`:
- There is **no `mode` parameter** on `generate()`/`stream_generate()`. Selection is
  implicit and length-driven only; a caller cannot force a mode per request.
- There is **no 8192 default**. `_max_dflash_ctx` defaults to `None` (unlimited).

So in substance the commit is a cosmetic rename plus a log reword.

### What upstream now provides (the superseding stack)

Upstream's `omlx/engine/dflash.py` (1205 lines vs our 1009) contains the identical
length-based fallback — `_should_fallback` at `upstream/main:…dflash.py:541` is
byte-for-byte identical — plus a full stack our branch never had:

- **#1276 (`393be01`)** — `draft_window_size` / `draft_sink_size` / `verify_mode`
  exposed end-to-end with a "Long-context tuning" UI.
- **#1326 (`f73961d`)** + **#1338 (`c57846b`)** — per-model `dflash_ssd_cache_max_bytes`
  (default 20 GiB) bounding the L2 SSD cache + profile-field classification.
- **#1344 (`d0f60ec`)** — multimodal detection triggering VLM fallback *before*
  chat-template flattening, plus an `asyncio.Lock` on fallback transitions.
- **#1318 (`0a6d016`)** / **#1388 (`915190d`)** — MTP pre-load patch + self-healing
  lifecycle wrap of the global dflash hooks (stops `n_confirmed` leakage / hook crashes).
- Gemma4 family support, `OutputParserSession` wiring, configurable prefix-cache
  `max_entries`, richer load-time logging.

### Gap analysis — capabilities in ours NOT in upstream

**None.** The decision logic is identical; the advertised `mode=` override and 8192
threshold don't exist in the code. The only real deltas are a rename and a log reword.

### Recommendation

Drop `bde04a0`; re-port nothing. Keep `bd10056`
(`docs/experimental/dflash_mlx_hook_details.md`) — it documents dflash-mlx's three
class-level hooks and their fallback-path regression (~25% TG / 30–40% prefill), the
very problem upstream #1388 addresses, so it remains a useful reference.

---

## Investigation 2 — Qwen3.6 experimental stack

Scope: 6 code/doc commits (`2cbe199`, `ab88d9b`, `2418f10`, `6d81b1b`, `c97f950`,
`efdebfe`). All code preserved on `backup/dev-pre-rebase-20260528`.

### Inventory of dropped functionality

**A. Experimental sampler stack — `omlx/sampling.py` (new, 282→317 lines @ `2418f10`/`efdebfe`)**
- **DRY** repetition suppression — `DRYLogitsProcessor` / `make_dry_processor`. Knobs:
  `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`.
- **top-nσ** filtering — `apply_top_n_sigma()`. Knob: `top_n_sigma`.
- **Min-k** heuristic (experimental) — `apply_min_k()`. Knob: `min_k`.
- **Dynamic (entropy) temperature** — `apply_dynamic_temperature()`. Knobs:
  `dynamic_temperature`, `dynatemp_low` (0.7), `dynatemp_high` (1.2), `dynatemp_exponent` (1.0).
- **Ordered/priority sampler composition** — `make_ordered_sampler()` +
  `normalize_sampler_priority()`. Knob: `sampler_priority` (list[str]).
- **temperature-last** handling. Knob: `temperature_last` (bool).

**B. Phase-aware penalty processors (the "penalty_window" feature) @ `efdebfe`**
- Windowed repetition / presence / frequency penalties over a bounded recent-token
  window (`context_size=20` default — the "penalty_window"), via `PhaseAwareSampler` /
  `PhaseAwareLogitsProcessor`.
- Phase-specific sampler switching — `PhaseSamplingProfile`, `PhaseTracker`,
  `normalize_phase_sampler_priority()`. Per-phase variants of every knob via
  `reasoning_*` / `response_*` prefixed fields.

**C. API / plumbing** — the knobs above added to `openai_models.py`, `responses_models.py`,
`request.py` (`SamplingParams`), `model_settings.py` (per-model persistence), `server.py`
(+300), `scheduler.py` (+57), admin `routes.py` / `dashboard.js` / `_modal_model_settings.html`,
and `tests/test_sampling.py`.

**D. Persistent prefill tuning knobs — `2cbe199`**
- `prefill_batch_size` (default 1) made persistent via `SchedulerSettings`/`GlobalSettings`.
- `prefill_step_size` (default 2048) promoted to a persistent, validated setting.
- Admin UI exposure.

**E. Qwen3.6 phase-bias chat template — `c97f950` (+ WIP `efdebfe`)**
- `omlx/chat_templates/qwen3_6_phase_bias.jinja` (185 lines) with `bias_reasoning()` /
  `bias_response()` macros driven by `chat_template_kwargs`: `reasoning_prepend`,
  `reasoning_append`, `response_prepend`, `response_append`.
- `omlx/utils/tokenizer.py` — `is_qwen36_model()`, `get_chat_template_override()`,
  `apply_chat_template_override()`: applies the packaged template to Qwen3.6 tokenizers
  at load time. Wired through `engine/batched.py`, `engine/dflash.py`, `engine/vlm.py`,
  `utils/__init__.py`.

**F. Docs (`docs.local/`)** — `sampling-dry-xtc-spec.md`, `new-samplers-roadmap.md`,
`qwen36-omlx-tuning-notes.md`, `integration-branch-workflow.md`, `pi-qwen36-debug-brief.md`,
`audio-voice-api-consolidation-spec.md`, `summary-agent-architecture-note.md`, plus
`scripts/update-local-build.sh`.

### Upstream coverage check

Upstream has **no** `omlx/sampling.py`. Its only sampler surface is
`omlx/utils/sampling.py`: `make_sampler(temp, top_p, min_p, min_tokens_to_keep, top_k,
xtc_probability, xtc_threshold, xtc_special_tokens)`. Grep of `upstream/main` for
`dry_multiplier|top_n_sigma|min_k|dynatemp|dynamic_temperature|sampler_priority|temperature_last`
returns nothing.

- **Redundant (upstream covers):** `min_p`, `xtc_*`, `top_p`, `top_k`, and plain
  `repetition_penalty` / `presence_penalty` / `frequency_penalty` request fields.
- **Genuinely NOT in upstream:** DRY, top-nσ, Min-k, dynamic temperature, ordered
  `sampler_priority`, `temperature_last`, windowed/phase-aware penalties, per-model
  persistence + admin UI, Qwen3.6 phase-bias template + load-time override,
  `prefill_batch_size` persistence.
- **Superseded by a different design:** `prefill_step_size` — upstream pursued adaptive
  prefill (`chunked_prefill`, `prefill_memory_guard`, `prefill_safe_zone_ratio`,
  `prefill_min_chunk_tokens`; commits `acd0533`, `4cfbc8b`) instead. The local static
  knob is what conflicts with the rewritten scheduler.

### What is genuinely lost (net list)

1. DRY repetition sampler.
2. top-nσ sampler.
3. Min-k experimental sampler.
4. Dynamic (entropy) temperature.
5. Ordered sampler composition (`sampler_priority`) + `temperature_last`.
6. Windowed repetition/presence/frequency penalties (`context_size` "penalty_window").
7. Phase-aware sampler switching (`reasoning_*` / `response_*`, `PhaseTracker`).
8. Per-model persistence + admin UI for all sampler knobs.
9. Qwen3.6 phase-bias chat template + load-time tokenizer override.
10. Persistent `prefill_batch_size` (and static `prefill_step_size`).
11. All `docs.local/` tuning records + `scripts/update-local-build.sh`.

(`min_p` / XTC / top_p / top_k / plain penalties are NOT lost — upstream has them.)

### How to recover later

- Backup branch `backup/dev-pre-rebase-20260528` (tip `e9fa4d5`).
  Each dropped commit is recoverable with `git cherry-pick <hash>` from there.

---

## Investigation 3 — Qwen3.6 reasoning extraction (`08ec2b5`, `a8ac9f0`) — REDUNDANT

Originally slated to keep, but found redundant during conflict resolution and dropped.

`08ec2b5` ("Make Qwen3.6 reasoning extraction survive DFlash and truncated think
blocks") did two things, both now handled upstream:

1. **DFlash thinking fallback** — added `_thinking_requested()` to `omlx/engine/dflash.py`
   and forced an evict-to-batched-engine fallback for *every* thinking-enabled request.
   The commit's own message lists "Rejected: teach the DFlash runtime the full oMLX
   reasoning pipeline first | too invasive". **Upstream took exactly that rejected
   (better) path**: `dflash.py` now handles reasoning natively — it prepends the
   `<think>` tag on the first chunk so the streaming `ThinkingParser` separates
   reasoning from content (`upstream/main:omlx/engine/dflash.py:914-995`), plus
   `_detect_open_think_tag` and `OutputParserSession` wiring. Re-applying our fallback
   would be a **regression**: it would disable DFlash for all reasoning requests that
   upstream now serves in-place.

2. **Truncated `<think>` parsing** — added `_THINKING_OPEN_ONLY_PATTERN` to
   `omlx/api/thinking.py` for an open `<think>` with no closing tag. **Upstream's
   `thinking.py` already handles this** (`upstream/main:omlx/api/thinking.py:81-86`,
   "Malformed: `<think>` opened but never closed. Drop the open tag…").

Both behaviors upstream covers; nothing unique remains. `a8ac9f0` only documented the
validation of this now-removed behavior, so it was dropped with it.

---

## Investigation 4 — DeepSeek V4 support path (`3ec7437`, `e9fa4d5`) — REDUNDANT

Decision: drop ours, adopt upstream's implementation.

Both our local path and upstream's port the **same source — mlx-lm PR #1192**
(`Blaizzy/mlx-lm` `pc/add-deepseekv4flash-model`). Upstream's is the complete,
correctly-structured version:

- `omlx/patches/deepseek_v4/` — a self-contained monkey-patch package:
  `deepseek_v4_model.py` (1171 lines, 1:1 from PR #1192), `tokenizer_patch.py` (268),
  `chat_template_v4.py` (415, full DSML template **with tool-call support**),
  `tool_parser_v4.py` (121), `cache_extras.py` (PoolingCache/BatchPoolingCache),
  `utils_patch.py`, `hyper_connection.py`, `cache_handlers.py`, `__init__.py`.
- `omlx/patches/mlx_lm_mtp/deepseek_v4_model.py` — MTP support.
- Gated on `model_type == "deepseek_v4"`; injects modules into `sys.modules` **without
  modifying the pinned mlx-lm package** (so re-pinning stays clean). Activated from
  `omlx/utils/model_loading.py::load_text_model` and `engine/batched.py::BatchedEngine.start`.
- Deps stay on upstream's tested pins: `transformers>=5.0.0`, `mlx-lm @ ed1fca4 (v0.31.3)`.
  Upstream's `8d65e3d` only blocks deepseek_v4 *oQ-quant* (a known temp limitation),
  not loading/inference.

Our version (`3ec7437`): `omlx/utils/deepseek_v4_encoding.py` (379), the
**tool-call-free** `omlx/chat_templates/deepseek_v4_clean.jinja` (24), cache type
handlers, and **PR-branch dependency pins** (`transformers` PR #45643, `mlx-lm` PR #1192
direct git pins). The README of upstream's patch explicitly calls out why PR-branch
pinning (our approach) is fragile and why the in-repo monkey-patch (their approach) is
preferred. `e9fa4d5` was only a post-rebase import-safety touch-up to our handlers.

Nothing unique is lost:
- Our `deepseek_v4_clean.jinja` is a stripped chat-only template (thinking-mode +
  `reasoning_content` replay, **no tool calls**). Upstream's `chat_template_v4.py` is a
  strict superset (same DeepSeek thinking behavior **plus** DSML tool-call grammar).
- Our `deepseek_v4_encoding.py` duplicates what upstream's `tokenizer_patch.py` +
  `chat_template_v4.py` do, against an older PR HEAD.

Recovery: backup branch as above; cherry-pick `3ec7437` if the custom PR-pin path is
ever needed again (expect heavy conflicts against upstream's deps + patches).

---

## Branch cleanup (2026-05-28)

Deleted stale one-off local branches after the rebase. All content recoverable as noted.

| Branch | Tip | Recoverable from |
|---|---|---|
| `archive/qwen36-phase-template-rebase-upstream-main-20260427` | `3e85f04` | `backup/dev-pre-rebase-20260528` + archive (Qwen3.6 work) |
| `feature/new-samplers` | `26ea996` | `origin/feature/new-samplers` + backup |
| `feature/new-samplers-integrated-snapshot-20260418-082933` | `9dc7286` | backup + archive |
| `feature/qwen36-phase-template` | `7238677` | backup + archive |
| `feature/template-phase-bias` | `616d066` | backup + archive |
| `integration/local-patches` | `616d066` | `origin/integration/local-patches` |
| `pr-814` | `ef7681c` | merged upstream (#814) |
| `pr-876` | `bf59f4f` | backup (`bde04a0`); PR #876 closed upstream |
| `pr-895` | `d2d64fb` | content in `dev` (hot_cache commits); PR #895 (`gh pr checkout 895`) |
| `pr-960` | `77ab99f` | GitHub PR #960 (`gh pr checkout 960`) |
| `pr-961` | `d7625a4` | GitHub PR #961 (`gh pr checkout 961`) |
| `safety/qwen36-phase-template-pre-rebase-20260419-233758` | `7ea0f13` | backup + archive |

Kept: `dev`, `main`, `backup/dev-pre-rebase-20260528`. Git reflog also retains these
tips for ~90 days. Remote branches `origin/feature/new-samplers` and
`origin/integration/local-patches` were left in place (delete with
`git push origin --delete <name>` if desired).
- Cherry-pick targets: sampler core `2418f10` (`omlx/sampling.py` + plumbing);
  phase-aware penalties + per-phase fields `efdebfe` (largest, most conflict-prone);
  phase-bias template `c97f950`; persistent prefill knobs `2cbe199`; docs `ab88d9b`,
  `6d81b1b`, `a8ac9f0`.
- **Cleanest re-port:** `omlx/sampling.py` is mostly self-contained and imports the same
  `apply_top_p/min_p/top_k/xtc` helpers upstream still exposes in `omlx/utils/sampling.py`.
  The DRY / top-nσ / dynamic-temp primitives are generically useful (not Qwen-specific)
  and are the recommended first re-port as a clean feature against upstream's current
  `BatchGenerator` wiring. `efdebfe`/`2cbe199` conflict hardest — re-port by hand against
  the rewritten scheduler/adaptive-prefill, don't straight cherry-pick.
