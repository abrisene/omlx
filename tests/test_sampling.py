# SPDX-License-Identifier: Apache-2.0
"""Unit tests for custom sampler helpers."""

from omlx.request import SamplingParams
from omlx.sampling import DEFAULT_SAMPLER_PRIORITY, normalize_sampler_priority


class TestSamplingParamsDefaults:
    def test_sampler_defaults_initialized(self):
        params = SamplingParams()
        assert params.sampler_priority == DEFAULT_SAMPLER_PRIORITY
        assert params.dry_sequence_breakers == ["\n", ":", "\"", "*"]


class TestNormalizeSamplerPriority:
    def test_uses_default_priority(self):
        assert normalize_sampler_priority(None, False) == DEFAULT_SAMPLER_PRIORITY

    def test_dedupes_and_keeps_known_names(self):
        ordered = normalize_sampler_priority(
            ["top_p", "dry", "top_p", "unknown", "temperature"],
            False,
        )
        assert ordered[:3] == ["top_p", "dry", "temperature"]
        assert "unknown" not in ordered

    def test_temperature_last_moves_temperature_stages(self):
        ordered = normalize_sampler_priority(
            ["temperature", "dynamic_temperature", "top_k", "xtc"],
            True,
        )
        assert ordered[-2:] == ["temperature", "dynamic_temperature"] or ordered[-2:] == ["dynamic_temperature", "temperature"]
        assert ordered.index("top_k") < ordered.index("temperature")
