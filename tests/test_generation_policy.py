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
        Path(__file__).parents[1] / "apps/omlx-web/omlx_web/templates/chat.html"
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
        self.prepared_prompts = []
        self.partial_flags = []

    async def stream_chat(self, messages, **kwargs):
        self.prepared_prompts.append(kwargs.get("_prepared_prompt"))
        self.partial_flags.append(kwargs.get("is_partial"))
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


@pytest.mark.asyncio
async def test_rewrite_replay_renders_its_own_prompt():
    # The server prepares the original request's prompt once for reuse and
    # resolves its partial mode. A replay continues the corrected assistant
    # prefix, so it must render that prefix as a partial assistant turn.
    engine = _RewriteEngine()
    rules = [
        {
            "pattern": r"(Crown Princess)(?: \1)+",
            "replacement": "$1",
            "phase": "response",
        }
    ]
    original_prompt = ("<original>", [1, 2, 3])

    _ = [
        output
        async for output in _stream_chat_with_transactional_rewrites(
            engine,
            [{"role": "user", "content": "Continue."}],
            {
                "max_tokens": 32,
                "generation_policy": {"rewrites": rules},
                "_prepared_prompt": original_prompt,
                "is_partial": False,
            },
            rules=rules,
            starts_in_reasoning=False,
        )
    ]

    assert engine.prepared_prompts == [original_prompt, None]
    assert engine.partial_flags == [False, True]


@pytest.mark.parametrize("starts_in_reasoning", [False, True])
def test_scheduler_policy_uses_request_prompt_reasoning_state(starts_in_reasoning):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from omlx.scheduler import Scheduler

    scheduler = Scheduler(model=MagicMock(), tokenizer=MagicMock())
    scheduler._resolve_think_end_token_ids = lambda: [9]
    _, processors = scheduler._build_sampler_and_processors(
        SamplingParams(generation_policy={"response": {"temperature": 2}}),
        SimpleNamespace(needs_think_prefix=starts_in_reasoning, num_prompt_tokens=2),
    )
    policy = next(p for p in processors if isinstance(p, GenerationPolicyProcessor))
    assert policy.starts_in_reasoning is starts_in_reasoning
    assert policy.prompt_token_count == 2


@pytest.mark.parametrize("preserve_reasoning", [False, True])
def test_decode_capture_respects_reasoning_output_cacheability(preserve_reasoning):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from omlx.scheduler import Scheduler, SchedulerConfig

    scheduler = Scheduler(model=MagicMock(), tokenizer=MagicMock(),
                          config=SchedulerConfig(paged_cache_block_size=4))
    scheduler.block_aware_cache = MagicMock()
    scheduler._boundary_snapshot_required = True
    scheduler._extract_boundary_snapshot = MagicMock(return_value=None)
    request = SimpleNamespace(request_id="capture", num_tokens=4, num_prompt_tokens=2,
                              needs_think_prefix=True, preserve_reasoning=preserve_reasoning)
    scheduler._maybe_capture_boundary_snapshot(request, 123)
    assert scheduler._extract_boundary_snapshot.called is preserve_reasoning
