# SPDX-License-Identifier: Apache-2.0
"""Transactional regular-expression rewrites for generated text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import regex


@dataclass(frozen=True)
class RewriteMatch:
    text: str
    rule_index: int
    start: int
    end: int
    resume_at: int


def _flags(value: str) -> regex.RegexFlag:
    result = regex.RegexFlag(0)
    for flag in value.lower():
        if flag == "i":
            result |= regex.IGNORECASE
        elif flag == "m":
            result |= regex.MULTILINE
        elif flag == "s":
            result |= regex.DOTALL
        elif flag == "g":
            continue
        else:
            raise ValueError(f"Unsupported regex flag: {flag}")
    return result


def _python_replacement(value: str) -> str:
    # The chat editor uses the familiar JavaScript replacement spelling.
    value = value.replace("$&", r"\g<0>")
    return regex.sub(r"\$(\d+)", r"\\g<\1>", value)


def _phase_spans(text: str, phase: str, starts_in_reasoning: bool) -> Iterable[tuple[int, int]]:
    if phase == "both":
        yield 0, len(text)
        return

    close = text.find("</think>")
    open_tag = text.find("<think>")
    reasoning_start = open_tag + len("<think>") if open_tag >= 0 else 0
    if phase == "reasoning":
        if open_tag >= 0 or starts_in_reasoning:
            yield reasoning_start, close if close >= 0 else len(text)
        return

    # Response rules become active only after a reasoning block closes.
    if close >= 0:
        yield close + len("</think>"), len(text)
    elif not starts_in_reasoning and open_tag < 0:
        yield 0, len(text)


def rewrite_once(
    text: str,
    rules: list[dict[str, Any]],
    *,
    starts_in_reasoning: bool = False,
) -> RewriteMatch | None:
    """Apply the earliest matching rule once and return the corrected text."""
    candidates: list[tuple[int, int, regex.Match, str]] = []
    for index, rule in enumerate(rules):
        pattern = str(rule.get("pattern") or "")
        if not pattern:
            continue
        compiled = regex.compile(pattern, _flags(str(rule.get("flags") or "")))
        phase = str(rule.get("phase") or "response")
        for start, end in _phase_spans(text, phase, starts_in_reasoning):
            try:
                match = compiled.search(text, start, end, timeout=0.02)
            except TimeoutError as exc:
                raise ValueError("Generation rewrite regex exceeded 20 ms") from exc
            if match is not None:
                candidates.append(
                    (
                        match.start(),
                        index,
                        match,
                        _python_replacement(str(rule.get("replacement") or "")),
                    )
                )
                break
    if not candidates:
        return None

    _, rule_index, match, replacement = min(candidates, key=lambda item: (item[0], item[1]))
    expanded = match.expand(replacement)
    corrected = text[: match.start()] + expanded + text[match.end() :]
    if corrected == text:
        return None
    return RewriteMatch(
        corrected,
        rule_index,
        match.start(),
        match.end(),
        match.start() + len(expanded),
    )


def validate_rewrite_rules(rules: list[dict[str, Any]]) -> None:
    if len(rules) > 64:
        raise ValueError("generation_policy.rewrites supports at most 64 rules")
    for index, rule in enumerate(rules):
        pattern = str(rule.get("pattern") or "")
        if not pattern:
            raise ValueError(f"Rewrite rule {index + 1} has an empty pattern")
        phase = str(rule.get("phase") or "response")
        if phase not in {"reasoning", "response", "both"}:
            raise ValueError(f"Rewrite rule {index + 1} has invalid phase: {phase}")
        try:
            regex.compile(pattern, _flags(str(rule.get("flags") or "")))
        except regex.error as exc:
            raise ValueError(f"Rewrite rule {index + 1} has invalid regex: {exc}") from exc
