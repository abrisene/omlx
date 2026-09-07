# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-09-02
- Primary product surfaces: oMLX web chat, response telemetry, generation preset editor.
- Evidence reviewed: `omlx/admin/templates/chat.html`, existing prompt profiles, model settings, streaming usage handling, and the archived phase-aware sampler implementation documented in `docs.local/rebase-2026-05-28-archive.md`.

## Brand
- Personality: Quiet, capable local inference tooling.
- Trust signals: Effective runtime state is visible; unsupported controls fail clearly; saved configuration is reproducible.
- Avoid: Administrative forms in the conversation flow, unexplained badges, and controls that only cosmetically transform output.

## Product goals
- Goals: Make generation behavior selectable, editable, and reusable from chat; expose thinking and cache behavior where the response is experienced; support phase-aware sampling and stateful generation rules.
- Non-goals: Moving model installation, memory tuning, or checkpoint management into chat.
- Success signals: A user can select or edit a preset, generate a response with its resolved policy, and inspect the effective policy and runtime telemetry on that response.

## Personas and jobs
- Primary personas: Local-model writers, agent developers, and inference experimenters.
- User jobs: Reuse a known generation behavior, tune one conversation without mutating a model, constrain terminology, recover pathological output, and understand latency or speculative performance.
- Key contexts of use: Long narrative conversations, reasoning/tool workflows, and model comparisons.

## Information architecture
- Primary navigation: Existing chat model selector and right sidebar.
- Core routes/screens: Chat composer, Generation popover/editor, response performance disclosure.
- Content hierarchy: Active preset first; common overrides second; phase stacks and generation rules in the editor; raw runtime detail on demand.

## Design principles
- Conversation behavior belongs to the conversation, not model administration.
- Show effective behavior, not nullable configuration.
- A rewrite changes model context through rollback and replay; cosmetic display replacement is not offered.
- Advanced controls remain discoverable without occupying permanent chat space.
- Tradeoffs: Preset selection is immediate; changes that require unsupported engine behavior must fail rather than silently degrade.

## Visual language
- Color: Reuse existing chat semantic variables and accent colors.
- Typography: Reuse existing Inter-based chat typography and compact tabular numerics for telemetry.
- Spacing/layout rhythm: Match existing model/profile sidebar controls.
- Shape/radius/elevation: Match existing rounded popovers, fields, and assistant cards.
- Motion: Existing short opacity/position transitions only.
- Imagery/iconography: Existing Lucide icons.

## Components
- Existing components to reuse: Right sidebar tabs, prompt-profile list/editor, compact composer controls, response timeline/status, model settings inputs.
- New/changed components: Generation preset selector, preset editor, phase panels with ordinary-preset inheritance, lexicon/rule editor, response telemetry disclosure.
- Variants and states: Built-in, saved, conversation-modified, response-only override, unsupported, invalid, and actively streaming.
- Token/component ownership: Chat owns preset selection and overrides; server owns policy validation and execution; model management reports capabilities only.

## Accessibility
- Target standard: Match the existing keyboard-accessible web chat.
- Keyboard/focus behavior: Preset editor is reachable from the composer and sidebar; controls use native inputs/buttons; Escape closes transient surfaces.
- Contrast/readability: Use existing semantic colors and text sizes no smaller than current secondary chat controls.
- Screen-reader semantics: Label every phase, rule type, and effective state explicitly.
- Reduced motion and sensory considerations: No essential meaning conveyed through animation.

## Responsive behavior
- Supported breakpoints/devices: Existing desktop and mobile chat layouts.
- Layout adaptations: Compact selector near the composer; full editor uses the existing right sidebar on desktop and overlay sidebar on mobile.
- Touch/hover differences: Runtime detail opens on click/tap; hover is optional enhancement only.

## Interaction states
- Loading: Keep the selected preset visible and mark validation pending.
- Empty: `Model recommended` is the default preset.
- Error: Keep edits and show the exact unsupported or invalid field.
- Success: Active preset name updates and the next response records its revision.
- Disabled: Unsupported phase/rule controls explain the missing runtime capability on invocation.
- Offline/slow network: Saved local presets remain editable; server validation waits until reachable.

## Content voice
- Tone: Short, concrete, and operational.
- Terminology: Generation preset, Reasoning, Response, Prefer, Avoid, Forbid, Rewrite, Recover.
- Microcopy rules: State consequences (`rewinds and replaces model context`) instead of implementation jargon.

## Implementation constraints
- Framework/styling system: Existing Jinja, Alpine.js, Tailwind utilities, and FastAPI request models.
- Design-token constraints: Reuse chat CSS variables; introduce no parallel component library.
- Performance constraints: No per-token browser round trips; generation policies resolve server-side before decode. Transactional regex rules buffer the response, rewind at the first match, and replay from the corrected assistant prefix.
- Compatibility constraints: Standard OpenAI-compatible fields remain valid; oMLX extensions are optional.
- Test/screenshot expectations: Request validation, preset persistence, phase switching, rule behavior, streaming telemetry, desktop, and mobile chat must be exercised.

## Open questions
- [ ] Whether named presets should later be server-synced across browsers; initial ownership is browser-local with resolved policies stored in chat history.
- [ ] Whether external clients should receive a preset registry API; initial API accepts resolved generation policy objects.
