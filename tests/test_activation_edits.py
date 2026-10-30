"""Request isolation contracts without loading model weights."""

from collections import deque
from types import SimpleNamespace as NS

import mlx.core as mx
import mlx.nn as nn
import pytest
from pydantic import ValidationError

from omlx.activation_edits import (
    ActivationEdit,
    activation_step,
    admission_allowed,
    dimensions,
    residual_edits,
    validate_edits,
)
from omlx.exceptions import InvalidRequestError


class Block(nn.Module):
    def __call__(self, x):
        return x * 2


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [Block(), Block()]
        self.config = NS(text_config=NS(hidden_size=3))


def request(name, edits=()):
    return NS(request_id=name, sampling_params=NS(activation_edits=edits))


@pytest.mark.parametrize(
    "edit",
    [
        dict(layer=True, neuron=0, add=1),
        dict(layer=-1, neuron=0, add=1),
        dict(layer=0, neuron=1.5, add=1),
        dict(layer=0, neuron=0, add=True),
        dict(layer=0, neuron=0, add=float("nan")),
        dict(layer=0, neuron=0, add=float("inf")),
        dict(layer=0, neuron=0, add=1e39),
        dict(layer=0, neuron=0, add="1"),
    ],
)
def test_invalid_schema(edit):
    with pytest.raises(ValidationError):
        ActivationEdit.model_validate(edit)


def test_bounds_and_duplicate_validation():
    model = Model()
    assert dimensions(model) == (2, 3)
    for edits in (
        [dict(layer=2, neuron=0, add=1)],
        [dict(layer=0, neuron=3, add=1)],
        [dict(layer=0, neuron=0, add=1)] * 2,
    ):
        with pytest.raises(InvalidRequestError):
            validate_edits(model, edits)


def test_effect_capture_and_cleanup_on_failure():
    model = Model()
    originals = list(model.layers)
    edits = validate_edits(model, [dict(layer=0, neuron=1, add=5)])
    observed = []
    with pytest.raises(RuntimeError):
        with residual_edits(
            model, edits, observer=lambda i, x: observed.append((i, x.tolist()))
        ):
            assert model.layers[0](mx.array([1.0, 2.0, 3.0])).tolist() == [
                2.0,
                9.0,
                6.0,
            ]
            raise RuntimeError("cancelled")
    assert all(a is b for a, b in zip(model.layers, originals))
    assert model.layers[0](mx.array([1.0, 2.0, 3.0])).tolist() == [2.0, 4.0, 6.0]
    assert observed[0] == (0, [2.0, 9.0, 6.0])
    with residual_edits(
        model, [], observer=lambda i, x: observed.append((i, x.tolist())), layer_ids=[1]
    ):
        assert model.layers[0] is originals[0]
        model.layers[1](mx.array([1.0, 2.0, 3.0]))
    assert observed[-1] == (1, [2.0, 4.0, 6.0])


def test_admission_and_abort_cannot_leak_edits_to_next_request():
    model = Model()
    edits = validate_edits(model, [dict(layer=0, neuron=0, add=1)])
    edited, baseline = request("edit", edits), request("base")
    scheduler = NS(
        model=model, running={}, prefilling=[], waiting=deque([edited, baseline])
    )
    scheduler._num_admitted_requests = lambda: (
        len(scheduler.running) + len(scheduler.prefilling)
    )
    original = model.layers[0]

    @activation_step
    def step(state):
        assert admission_allowed(state, edited)
        assert not admission_allowed(state, baseline)
        assert state.model.layers[0] is not original
        # Simulate pending abort drained before any forward.
        state.waiting.popleft()
        assert not admission_allowed(state, state.waiting[0])
        raise RuntimeError("abort")

    with pytest.raises(RuntimeError):
        step(scheduler)
    assert model.layers[0] is original
    assert scheduler._activation_request_id is None
    assert admission_allowed(scheduler, baseline)


def test_baseline_step_never_wraps_layers():
    model = Model()
    original = model.layers[0]
    scheduler = NS(
        model=model, running={}, prefilling=[], waiting=deque([request("base")])
    )

    @activation_step
    def step(state):
        assert state.model.layers[0] is original
        return "baseline"

    assert step(scheduler) == "baseline"


def test_pipeline_slice_property_updates_real_owning_layers():
    inner = Model()

    class Pipeline:
        def __init__(self):
            self.language_model = NS(model=inner)
            self.config = inner.config

        @property
        def layers(self):
            return self.language_model.model.layers[1:]

    model = Pipeline()
    original = inner.layers[1]
    edits = validate_edits(model, [dict(layer=0, neuron=0, add=7)])
    with residual_edits(model, edits):
        assert inner.layers[1] is not original
        assert inner.layers[1](mx.array([1.0, 2.0, 3.0])).tolist() == [9.0, 4.0, 6.0]
    assert inner.layers[1] is original


def test_edited_prefix_and_boundary_paths_do_not_touch_persistent_cache():
    from omlx.scheduler import Scheduler

    edited = request("edit", (ActivationEdit(layer=0, neuron=0, add=1),))
    edited.prompt_token_ids = [1, 2, 3]
    scheduler = NS(_prefix_cache_prepared=set(), requests={"edit": edited})
    Scheduler._prepare_prefix_cache_for_request(scheduler, edited)
    assert edited.remaining_tokens == [1, 2, 3]
    assert scheduler._prefix_cache_prepared == {"edit"}
    # No model/cache infrastructure exists on this stub: either method would
    # fail immediately if it tried to snapshot or persist edited state.
    Scheduler._maybe_capture_boundary_snapshot(scheduler, edited, 1)
    Scheduler._on_prefill_boundary_snapshot(scheduler, "edit", [], 256)


def test_text_completion_rejects_unsupported_nonempty_edits():
    from omlx.api.openai_models import CompletionRequest
    with pytest.raises(ValidationError, match="chat/completions only"):
        CompletionRequest(model="test", prompt="test", activation_edits=[dict(layer=0, neuron=0, add=1)])
    assert CompletionRequest(model="test", prompt="test", activation_edits=[]).model == "test"
