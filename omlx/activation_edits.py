"""Request-scoped post-block residual edits on the resident generator."""

from contextlib import contextmanager
from functools import wraps
from math import isfinite

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from .exceptions import InvalidRequestError


class ActivationEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layer: StrictInt = Field(ge=0)
    neuron: StrictInt = Field(ge=0)
    add: float

    @field_validator("add", mode="before")
    @classmethod
    def finite_add(cls, value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or abs(value) > 3.4028234663852886e38
        ):
            raise ValueError("add must be a finite number")
        return value


def dimensions(model):
    layers = model.layers
    candidates = [model]
    seen = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        for name in (
            "_language_model",
            "language_model",
            "model",
            "config",
            "args",
            "text_config",
        ):
            child = getattr(candidate, name, None)
            if child is not None and id(child) not in seen:
                candidates.append(child)
    for candidate in candidates:
        for name in ("hidden_size", "dim", "d_model"):
            width = getattr(candidate, name, None)
            if isinstance(width, int) and width > 0:
                return len(layers), width
    raise InvalidRequestError("Cannot determine residual width for activation_edits")


def validate_edits(model, edits):
    if not edits:
        return ()
    count, width = dimensions(model)
    parsed = tuple(ActivationEdit.model_validate(edit) for edit in edits)
    indices = [(edit.layer, edit.neuron) for edit in parsed]
    if len(set(indices)) != len(indices):
        raise InvalidRequestError(
            "activation_edits contains duplicate layer/neuron indices"
        )
    if any(edit.layer >= count or edit.neuron >= width for edit in parsed):
        raise InvalidRequestError(
            f"activation_edits bounds: layers 0..{count - 1}, neurons 0..{width - 1}"
        )
    return parsed


class ResidualEdit(nn.Module):
    def __init__(self, layer, edits, width, index, observer):
        super().__init__()
        self.layer = layer
        self.index = index
        self.observer = observer
        values = [0.0] * width
        for edit in edits:
            values[edit.neuron] = edit.add
        self.delta = mx.array(values, dtype=mx.float32) if edits else None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            layer = self.get("layer")
            if layer is None:
                raise
            return getattr(layer, name)

    def __call__(self, *args, **kwargs):
        output = self.layer(*args, **kwargs)
        residual = output[0] if isinstance(output, tuple) else output
        if self.delta is not None:
            residual = residual + self.delta.astype(residual.dtype)
        if self.observer is not None:
            self.observer(self.index, residual)
        return (residual, *output[1:]) if isinstance(output, tuple) else residual


def mutable_layer_slots(model):
    """Map exposed pipeline layers to their owning lists, never a slice copy."""
    exposed = model.layers
    candidates, seen, owners = [model], set(), []
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        fields = candidate if isinstance(candidate, dict) else vars(candidate)
        for name in ("layers", "blocks"):
            value = fields.get(name)
            if isinstance(value, list):
                owners.append(value)
        for name in ("_language_model", "language_model", "model"):
            child = getattr(candidate, name, None)
            if child is not None and id(child) not in seen:
                candidates.append(child)
    slots = []
    for layer in exposed:
        matches = [
            (owner, index)
            for owner in owners
            for index, item in enumerate(owner)
            if item is layer
        ]
        if not matches:
            raise InvalidRequestError("Cannot locate mutable transformer layer owner")
        slots.append(matches[-1])
    return slots


@contextmanager
def residual_edits(model, edits, *, observer=None, layer_ids=None):
    """Restore exact layer instances even on cancellation or forward failure."""
    if not edits and observer is None:
        yield
        return
    layers = model.layers
    slots = mutable_layer_slots(model)
    _, width = dimensions(model)
    selected = {edit.layer for edit in edits}
    if observer is not None:
        selected.update(range(len(layers)) if layer_ids is None else layer_ids)
    if any(index < 0 or index >= len(layers) for index in selected):
        raise InvalidRequestError("Observer layer is out of bounds")
    originals = {}
    try:
        for index in sorted(selected):
            owner, slot = slots[index]
            originals[index] = owner[slot]
            owner[slot] = ResidualEdit(
                owner[slot],
                [e for e in edits if e.layer == index],
                width,
                index,
                observer,
            )
        yield
    finally:
        for index, layer in originals.items():
            owner, slot = slots[index]
            owner[slot] = layer


def request_edits(request):
    return (
        getattr(getattr(request, "sampling_params", None), "activation_edits", None)
        or ()
    )


def admission_allowed(scheduler, request):
    selected = getattr(scheduler, "_activation_request_id", None)
    if selected is not None:
        return request.request_id == selected and not scheduler._num_admitted_requests()
    # A fresh step must first install an edited request's scoped layers.
    return not request_edits(request)


def activation_step(step):
    @wraps(step)
    def run(scheduler):
        active = list(scheduler.running.values()) + list(scheduler.prefilling)
        selected = next((request for request in active if request_edits(request)), None)
        if selected is None and not active and scheduler.waiting:
            candidate = scheduler.waiting[0]
            if request_edits(candidate):
                selected = candidate
        if selected is None:
            return step(scheduler)
        scheduler._activation_request_id = selected.request_id
        try:
            with residual_edits(scheduler.model, request_edits(selected)):
                return step(scheduler)
        finally:
            scheduler._activation_request_id = None

    return run


def metadata_from_path(model_path):
    """Read authored dimensions without loading weights."""
    import json
    from pathlib import Path

    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
        config = config.get("text_config", config)
        count = config.get("num_hidden_layers", config.get("n_layers"))
        width = config.get("hidden_size", config.get("dim"))
        if (
            isinstance(count, int)
            and isinstance(width, int)
            and count > 0
            and width > 0
        ):
            return {
                "layers": count,
                "neurons": width,
                "position": "post_block_residual",
                "index_base": 0,
            }
    except (OSError, ValueError, TypeError):
        pass
    return None
