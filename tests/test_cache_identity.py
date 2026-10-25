"""Tests for runtime-specific paged-cache identity."""

from omlx.cache.identity import model_cache_identity


def test_base_model_cache_identity_is_versioned_and_stable():
    identity = model_cache_identity("model-id")
    assert identity == "model-id:omlx-cache-v2"
    assert model_cache_identity("model-id") == identity


def test_adapter_cache_identity_tracks_config_and_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = adapter / "adapter_config.json"
    weights = adapter / "adapters.safetensors"
    config.write_text('{"rank": 4}', encoding="utf-8")
    weights.write_bytes(b"adapter-a")

    first = model_cache_identity("model-id", str(adapter))
    assert first == model_cache_identity("model-id", str(adapter))
    assert first.startswith("model-id:omlx-cache-v2:adapter-sha256:")

    weights.write_bytes(b"adapter-b")
    second = model_cache_identity("model-id", str(adapter))
    assert second != first

    config.write_text('{"rank": 8}', encoding="utf-8")
    assert model_cache_identity("model-id", str(adapter)) != second
