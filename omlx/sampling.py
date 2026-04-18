# SPDX-License-Identifier: Apache-2.0
"""Custom sampler helpers for oMLX.

This module extends mlx_lm's native sampler stack with:

- ordered sampler composition
- temperature-last handling
- dynamic temperature
- top-nσ
- an experimental Min-k heuristic
- DRY logits processing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import mlx.core as mx
from mlx_lm.sample_utils import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    apply_xtc,
    categorical_sampling,
)

DEFAULT_SAMPLER_PRIORITY = [
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "dry",
    "top_k",
    "top_p",
    "min_p",
    "top_n_sigma",
    "min_k",
    "dynamic_temperature",
    "temperature",
    "xtc",
]

_SAMPLER_STAGE_NAMES = {
    "top_k",
    "top_p",
    "min_p",
    "top_n_sigma",
    "min_k",
    "dynamic_temperature",
    "temperature",
    "xtc",
}


def _ensure_2d(logprobs: mx.array) -> tuple[mx.array, bool]:
    if logprobs.ndim == 1:
        return mx.expand_dims(logprobs, axis=0), True
    return logprobs, False


def _restore_shape(logprobs: mx.array, squeezed: bool) -> mx.array:
    return logprobs[0] if squeezed else logprobs


def normalize_sampler_priority(
    sampler_priority: Sequence[str] | None,
    temperature_last: bool,
) -> list[str]:
    ordered = []
    source = list(sampler_priority) if sampler_priority else list(DEFAULT_SAMPLER_PRIORITY)
    for name in source:
        if name in _SAMPLER_STAGE_NAMES and name not in ordered:
            ordered.append(name)
    for name in DEFAULT_SAMPLER_PRIORITY:
        if name in _SAMPLER_STAGE_NAMES and name not in ordered:
            ordered.append(name)
    if temperature_last:
        for name in ("temperature", "dynamic_temperature"):
            if name in ordered:
                ordered.remove(name)
                ordered.append(name)
    return ordered


def apply_top_n_sigma(
    logprobs: mx.array,
    top_n_sigma: float,
    min_tokens_to_keep: int = 1,
) -> mx.array:
    """Filter tokens to those within n * sigma of the top logit/logprob."""
    if top_n_sigma <= 0:
        return logprobs
    scores, squeezed = _ensure_2d(logprobs)
    max_scores = mx.max(scores, axis=-1, keepdims=True)
    sigma = mx.std(scores, axis=-1, keepdims=True)
    threshold = max_scores - (sigma * top_n_sigma)
    masked = mx.where(scores >= threshold, scores, -float("inf"))
    if min_tokens_to_keep > 0:
        vocab = scores.shape[-1]
        k = min(max(min_tokens_to_keep, 1), vocab)
        top_idx = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        keep_values = mx.take_along_axis(scores, top_idx, axis=-1)
        masked = mx.put_along_axis(masked, top_idx, keep_values, axis=-1)
    return _restore_shape(masked, squeezed)


def apply_min_k(
    logprobs: mx.array,
    min_k: int,
) -> mx.array:
    """Experimental Min-k heuristic based on local sorted-logit cliffs.

    This is an approximate implementation inspired by the paper's abstract:
    it computes a position-weighted relative decay across sorted logits and
    truncates at the strongest detected cliff while guaranteeing at least
    ``min_k`` tokens survive.
    """
    if min_k <= 0:
        return logprobs

    scores, squeezed = _ensure_2d(logprobs)
    vocab = scores.shape[-1]
    keep_min = min(max(min_k, 1), vocab)
    sorted_scores = mx.sort(scores, axis=-1)[:, ::-1]
    diffs = sorted_scores[:, :-1] - sorted_scores[:, 1:]
    denom = mx.maximum(mx.abs(sorted_scores[:, :-1]), 1e-6)
    rel_decay = diffs / denom
    positions = mx.arange(1, rel_decay.shape[-1] + 1, dtype=scores.dtype)
    weighted = rel_decay * mx.sqrt(positions)
    if keep_min - 1 > 0:
        weighted[:, : keep_min - 1] = -float("inf")
    cliff_idx = mx.argmax(weighted, axis=-1)
    keep_counts = mx.maximum(keep_min, cliff_idx + 1)

    sorted_idx = mx.argsort(scores, axis=-1)[:, ::-1]
    masked = mx.full_like(scores, -float("inf"))
    for row in range(scores.shape[0]):
        k = int(keep_counts[row].item())
        idx = sorted_idx[row, :k]
        vals = scores[row, idx]
        masked[row] = mx.put_along_axis(
            masked[row : row + 1], mx.expand_dims(idx, 0), mx.expand_dims(vals, 0), axis=-1
        )[0]
    return _restore_shape(masked, squeezed)


def apply_dynamic_temperature(
    logprobs: mx.array,
    dynatemp_low: float,
    dynatemp_high: float,
    dynatemp_exponent: float,
) -> mx.array:
    if dynatemp_low <= 0 or dynatemp_high <= 0:
        return logprobs
    scores, squeezed = _ensure_2d(logprobs)
    probs = mx.softmax(scores, axis=-1)
    entropy = -mx.sum(probs * mx.log(mx.maximum(probs, 1e-12)), axis=-1, keepdims=True)
    max_entropy = mx.log(mx.array(scores.shape[-1], dtype=scores.dtype))
    entropy_norm = mx.clip(entropy / mx.maximum(max_entropy, 1e-6), 0.0, 1.0)
    exponent = max(dynatemp_exponent, 1e-6)
    dynatemp = dynatemp_low + (dynatemp_high - dynatemp_low) * mx.power(entropy_norm, exponent)
    adjusted = scores / mx.maximum(dynatemp, 1e-6)
    return _restore_shape(adjusted, squeezed)


def make_ordered_sampler(
    *,
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: List[int] | None = None,
    top_n_sigma: float = 0.0,
    min_k: int = 0,
    dynamic_temperature: bool = False,
    dynatemp_low: float = 0.7,
    dynatemp_high: float = 1.2,
    dynatemp_exponent: float = 1.0,
    temperature_last: bool = False,
    sampler_priority: Sequence[str] | None = None,
) -> Callable[[mx.array], mx.array]:
    """Create an ordered sampler chain with custom oMLX stages."""
    if xtc_special_tokens is None:
        xtc_special_tokens = []

    ordered_names = normalize_sampler_priority(sampler_priority, temperature_last)

    stages: list[Callable[[mx.array], mx.array]] = []
    use_argmax = temp == 0.0 and not dynamic_temperature

    stage_map: dict[str, Callable[[mx.array], mx.array] | None] = {
        "top_k": (lambda x: apply_top_k(x, top_k)) if top_k > 0 else None,
        "top_p": (lambda x: apply_top_p(x, top_p)) if 0 < top_p < 1.0 else None,
        "min_p": (lambda x: apply_min_p(x, min_p, min_tokens_to_keep)) if min_p > 0.0 else None,
        "top_n_sigma": (lambda x: apply_top_n_sigma(x, top_n_sigma, min_tokens_to_keep)) if top_n_sigma > 0.0 else None,
        "min_k": (lambda x: apply_min_k(x, min_k)) if min_k > 0 else None,
        "xtc": (
            lambda x: apply_xtc(x, xtc_probability, xtc_threshold, xtc_special_tokens)
        ) if xtc_probability > 0.0 else None,
        "dynamic_temperature": (
            lambda x: apply_dynamic_temperature(x, dynatemp_low, dynatemp_high, dynatemp_exponent)
        ) if dynamic_temperature else None,
        "temperature": (lambda x: x / temp) if temp not in (0.0, 1.0) else None,
    }

    for name in ordered_names:
        stage = stage_map.get(name)
        if stage is not None:
            stages.append(stage)

    def sampler(logprobs: mx.array):
        scores = logprobs
        for stage in stages:
            scores = stage(scores)
        if use_argmax:
            return mx.argmax(scores, axis=-1)
        return categorical_sampling(scores, 1.0)

    return sampler


@dataclass
class DRYLogitsProcessor:
    multiplier: float
    base: float
    allowed_length: int
    breaker_token_ids: set[int]
    max_pattern_length: int = 32

    def __call__(self, tokens, logits):
        if self.multiplier <= 0.0:
            return logits
        token_list = list(tokens)
        if len(token_list) <= self.allowed_length:
            return logits

        penalties: dict[int, float] = {}
        max_len = min(self.max_pattern_length, len(token_list) - 1)

        for length in range(self.allowed_length + 1, max_len + 1):
            suffix = token_list[-length:]
            if any(tok in self.breaker_token_ids for tok in suffix):
                continue
            for start in range(0, len(token_list) - length):
                if token_list[start : start + length] != suffix:
                    continue
                next_idx = start + length
                if next_idx >= len(token_list):
                    continue
                next_token = token_list[next_idx]
                penalty = self.multiplier * (self.base ** (length - self.allowed_length))
                penalties[next_token] = max(penalties.get(next_token, 0.0), penalty)

        if not penalties:
            return logits

        if logits.ndim == 1:
            for token_id, penalty in penalties.items():
                logits[token_id] = logits[token_id] - penalty
            return logits

        for token_id, penalty in penalties.items():
            logits[:, token_id] = logits[:, token_id] - penalty
        return logits


def make_dry_processor(
    multiplier: float,
    base: float,
    allowed_length: int,
    breaker_token_ids: Iterable[int],
) -> DRYLogitsProcessor:
    return DRYLogitsProcessor(
        multiplier=multiplier,
        base=base,
        allowed_length=max(allowed_length, 1),
        breaker_token_ids=set(breaker_token_ids),
    )
