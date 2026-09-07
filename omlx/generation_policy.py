# SPDX-License-Identifier: Apache-2.0
"""Per-request generation policies used by the chat preset editor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import mlx.core as mx

from .utils.sampling import apply_min_p, apply_top_k, apply_top_p


def _as_tokens(tokens: Any) -> list[int]:
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token) for token in tokens]


def _endswith(tokens: Sequence[int], suffix: Sequence[int]) -> bool:
    return bool(suffix) and len(tokens) >= len(suffix) and list(tokens[-len(suffix) :]) == list(suffix)


def _contains(tokens: Sequence[int], needle: Sequence[int]) -> bool:
    if not needle or len(tokens) < len(needle):
        return False
    width = len(needle)
    return any(list(tokens[i : i + width]) == list(needle) for i in range(len(tokens) - width + 1))


def _set_score(scores: mx.array, token_id: int, value: Any) -> None:
    if scores.ndim == 1:
        scores[token_id] = value
    else:
        scores[:, token_id] = value


def _add_score(scores: mx.array, token_id: int, delta: float) -> None:
    if scores.ndim == 1:
        scores[token_id] = scores[token_id] + delta
    else:
        scores[:, token_id] = scores[:, token_id] + delta


@dataclass(frozen=True)
class LexiconEntry:
    token_ids: tuple[int, ...]
    mode: str
    strength: float
    phase: str


class GenerationPolicyProcessor:
    """Apply phase-specific sampling and sequence-aware lexicon steering."""

    def __init__(
        self,
        *,
        policy: dict[str, Any],
        tokenizer: Any,
        prompt_token_count: int,
        starts_in_reasoning: bool,
        think_end_token_ids: Sequence[int],
    ) -> None:
        self.policy = policy
        self.prompt_token_count = prompt_token_count
        self.starts_in_reasoning = starts_in_reasoning
        self.think_end_token_ids = tuple(int(token) for token in think_end_token_ids)
        self.lexicon = self._compile_lexicon(policy.get("lexicon") or [], tokenizer)

    @staticmethod
    def _compile_lexicon(entries: list[dict[str, Any]], tokenizer: Any) -> list[LexiconEntry]:
        compiled: list[LexiconEntry] = []
        for entry in entries:
            text = str(entry.get("text") or "")
            mode = str(entry.get("mode") or "prefer")
            if not text or mode not in {"prefer", "avoid", "forbid"}:
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            compiled.append(
                LexiconEntry(
                    tuple(int(token) for token in token_ids),
                    mode,
                    float(entry.get("strength", 2.0)),
                    str(entry.get("phase") or "both"),
                )
            )
        return compiled

    def _phase(self, generated: list[int]) -> str:
        if not self.starts_in_reasoning:
            return "response"
        if self.think_end_token_ids and _contains(generated, self.think_end_token_ids):
            return "response"
        return "reasoning"

    def _apply_phase_sampling(self, scores: mx.array, phase: str) -> mx.array:
        phase_policy = self.policy.get(phase) or {}
        temperature = phase_policy.get("temperature")
        if temperature is not None and float(temperature) > 0 and float(temperature) != 1:
            scores = scores / float(temperature)
        top_k = int(phase_policy.get("top_k") or 0)
        if top_k > 0 and top_k < scores.shape[-1]:
            scores = apply_top_k(scores, top_k)
        top_p = float(phase_policy.get("top_p") or 0)
        if 0 < top_p < 1:
            scores = apply_top_p(scores, top_p)
        min_p = float(phase_policy.get("min_p") or 0)
        if min_p > 0:
            scores = apply_min_p(scores, min_p)
        if temperature is not None and float(temperature) == 0:
            best = mx.argmax(scores, axis=-1)
            masked = mx.full_like(scores, -float("inf"))
            if scores.ndim == 1:
                masked[int(best.item())] = scores[int(best.item())]
            else:
                for row in range(scores.shape[0]):
                    idx = int(best[row].item())
                    masked[row, idx] = scores[row, idx]
            scores = masked
        return scores

    def _apply_lexicon(self, scores: mx.array, generated: list[int], phase: str) -> mx.array:
        for entry in self.lexicon:
            if entry.phase not in {"both", phase}:
                continue
            # Bias the next token for every matching phrase prefix. This keeps
            # multi-token entries coherent instead of independently biasing
            # each token in the phrase.
            for prefix_len in range(len(entry.token_ids)):
                prefix = entry.token_ids[:prefix_len]
                if prefix_len and not _endswith(generated, prefix):
                    continue
                if not prefix_len and generated and generated[-1] == entry.token_ids[0]:
                    continue
                next_token = entry.token_ids[prefix_len]
                if entry.mode == "forbid" and prefix_len == len(entry.token_ids) - 1:
                    _set_score(scores, next_token, -float("inf"))
                elif entry.mode == "prefer":
                    _add_score(scores, next_token, abs(entry.strength))
                elif entry.mode == "avoid":
                    _add_score(scores, next_token, -abs(entry.strength))
        return scores

    def __call__(self, tokens: Any, logits: mx.array) -> mx.array:
        token_list = _as_tokens(tokens)
        generated = token_list[self.prompt_token_count :]
        phase = self._phase(generated)
        scores = self._apply_phase_sampling(logits, phase)
        return self._apply_lexicon(scores, generated, phase)


def build_generation_policy_processor(
    *,
    policy: dict[str, Any] | None,
    tokenizer: Any,
    prompt_token_count: int,
    starts_in_reasoning: bool,
    think_end_token_ids: Sequence[int],
) -> GenerationPolicyProcessor | None:
    if not policy or not policy.get("enabled", True):
        return None
    if not any(policy.get(key) for key in ("reasoning", "response", "lexicon")):
        return None
    return GenerationPolicyProcessor(
        policy=policy,
        tokenizer=tokenizer,
        prompt_token_count=prompt_token_count,
        starts_in_reasoning=starts_in_reasoning,
        think_end_token_ids=think_end_token_ids,
    )
