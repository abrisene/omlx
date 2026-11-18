"""Storage read-ahead must not silently change a checkpoint's prefill geometry."""
from functools import partial
from types import SimpleNamespace
import importlib

import pytest
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from omlx.scheduler import Scheduler

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")


@pytest.mark.parametrize("first_chunk, expected_narrow", [("wide", False), ("narrow", True)])
def test_scheduler_honors_row_store_first_chunk_policy_without_disabling_prefetch(first_chunk, expected_narrow):
    table = SimpleNamespace(prefetch=lambda ids: None, prefill_first_chunk=first_chunk)
    layer = SimpleNamespace(ple=SimpleNamespace(ple_embedding=SimpleNamespace(ngram_embedding=table)))
    inner = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    probe = partial(language.LanguageModel.ple_gathers_ahead, inner)
    model = SimpleNamespace(prefetch_ple=lambda *args: None, ple_gathers_ahead=probe)
    assert probe() is True
    assert Scheduler._qwen4_ple_gathers_ahead(SimpleNamespace(model=model)) is expected_narrow
