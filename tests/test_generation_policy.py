"""Generation-policy regression tests."""

from pathlib import Path

import mlx.core as mx
import pytest
from pydantic import ValidationError

from omlx.api.openai_models import ChatCompletionRequest
from omlx.engine.base import GenerationOutput
from omlx.generation_policy import GenerationPolicyProcessor
from omlx.generation_rewrite import rewrite_once
from omlx.request import SamplingParams
from omlx.server import _stream_chat_with_transactional_rewrites


class _Tokenizer:
    _tokens = {
        "crown princess": [4, 5],
        "bad loop": [6, 7],
    }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self._tokens[text]


def _processor(policy, *, reasoning=True):
    return GenerationPolicyProcessor(
        policy=policy,
        tokenizer=_Tokenizer(),
        prompt_token_count=2,
        starts_in_reasoning=reasoning,
        think_end_token_ids=[9],
    )


def test_phase_temperature_switches_after_think_end():
    proc = _processor(
        {
            "reasoning": {"temperature": 0.5},
            "response": {"temperature": 2.0},
        }
    )
    logits = mx.array([0.0, 2.0, 4.0])

    reasoning = proc([20, 21, 3], logits)
    response = proc([20, 21, 3, 9], logits)
    mx.eval(reasoning, response)

    assert reasoning.tolist() == [0.0, 4.0, 8.0]
    assert response.tolist() == [0.0, 1.0, 2.0]


def test_lexicon_bias_follows_multi_token_phrase_prefix():
    proc = _processor(
        {
            "lexicon": [
                {
                    "text": "crown princess",
                    "mode": "prefer",
                    "phase": "both",
                    "strength": 3,
                }
            ]
        },
        reasoning=False,
    )

    first = proc([20, 21], mx.zeros((10,)))
    second = proc([20, 21, 4], mx.zeros((10,)))
    mx.eval(first, second)

    assert first[4].item() == 3
    assert second[5].item() == 3


def test_forbid_masks_only_phrase_completion():
    proc = _processor(
        {
            "lexicon": [
                {
                    "text": "bad loop",
                    "mode": "forbid",
                    "phase": "response",
                }
            ]
        },
        reasoning=False,
    )

    before_prefix = proc([20, 21], mx.zeros((10,)))
    at_prefix = proc([20, 21, 6], mx.zeros((10,)))
    mx.eval(before_prefix, at_prefix)

    assert before_prefix[7].item() == 0
    assert at_prefix[7].item() == -float("inf")


def test_policy_survives_api_and_sampling_models():
    policy = {"enabled": True, "response": {"temperature": 0.8}}
    request = ChatCompletionRequest(model="model", messages=[], generation_policy=policy)
    params = SamplingParams(generation_policy=request.generation_policy)

    assert params.generation_policy == policy


def test_chat_owns_editable_generation_presets_and_cache_telemetry():
    html = (
        Path(__file__).parents[1] / "omlx/admin/templates/chat.html"
    ).read_text()

    assert "saveGenerationPresetAs()" in html
    assert "buildGenerationPolicy()" in html
    assert "generation_policy: generationPolicy" not in html
    assert "gen.generation_policy = generationPolicy" in html
    assert "Lexicon steering" in html
    assert "Transactional rewrites" in html
    assert "phasePresetSettings" in html
    assert "cachedPromptTokens(msg.meta.usage)" in html


def test_regex_rewrite_uses_javascript_capture_spelling():
    match = rewrite_once(
        "The Crown Princess Crown Princess waited.",
        [
            {
                "pattern": r"(Crown Princess)(?: \1)+",
                "replacement": "$1",
                "phase": "response",
            }
        ],
    )

    assert match is not None
    assert match.text == "The Crown Princess waited."
    assert match.text[: match.resume_at] == "The Crown Princess"


def test_response_rule_waits_for_reasoning_to_close():
    rules = [{"pattern": "loop", "replacement": "answer", "phase": "response"}]

    assert rewrite_once("loop", rules, starts_in_reasoning=True) is None
    match = rewrite_once("<think>loop</think>loop", rules, starts_in_reasoning=True)

    assert match is not None
    assert match.text == "<think>loop</think>answer"


def test_invalid_regex_is_rejected_at_request_validation():
    with pytest.raises(ValidationError, match="invalid regex"):
        ChatCompletionRequest(
            model="model",
            messages=[],
            generation_policy={
                "rewrites": [{"pattern": "(", "replacement": ""}]
            },
        )


class _RewriteTokenizer:
    def encode(self, text):
        return list(range(len(text)))


class _RewriteEngine:
    tokenizer = _RewriteTokenizer()

    def __init__(self):
        self.partial_prefixes = []

    async def stream_chat(self, messages, **kwargs):
        partial = messages[-1] if messages[-1]["role"] == "assistant" else None
        if partial:
            self.partial_prefixes.append(partial["content"])
            yield GenerationOutput(
                text=" waited.",
                new_text=" waited.",
                completion_tokens=2,
                prompt_tokens=20,
                cached_tokens=8,
                finished=True,
            )
            return
        yield GenerationOutput(
            text="The Crown Princess Crown Princess waited.",
            new_text="The Crown Princess Crown Princess waited.",
            completion_tokens=7,
            prompt_tokens=10,
            cached_tokens=0,
            finished=True,
        )


@pytest.mark.asyncio
async def test_transactional_rewrite_replays_from_corrected_assistant_prefix():
    engine = _RewriteEngine()
    rules = [
        {
            "pattern": r"(Crown Princess)(?: \1)+",
            "replacement": "$1",
            "phase": "response",
        }
    ]

    outputs = [
        output
        async for output in _stream_chat_with_transactional_rewrites(
            engine,
            [{"role": "user", "content": "Continue."}],
            {
                "max_tokens": 32,
                "generation_policy": {"rewrites": rules},
            },
            rules=rules,
            starts_in_reasoning=False,
        )
    ]

    assert engine.partial_prefixes == ["The Crown Princess"]
    assert outputs[0].new_text == "The Crown Princess waited."
    assert outputs[0].completion_tokens == len("The Crown Princess waited.")
    assert outputs[0].cached_tokens == 8
