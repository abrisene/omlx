"""Public adapter routes use real auth, model resolution, and settings storage."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import omlx.server as server
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings, ModelSettingsManager


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    state = server.ServerState()
    state.api_key = "test-adapter-key"
    state.settings_manager = ModelSettingsManager(tmp_path)
    state.settings_manager.set_settings(
        "physical", ModelSettings(model_alias="friendly")
    )
    state.engine_pool = EnginePool()
    entry = EngineEntry("physical", str(tmp_path / "model"), "llm", "batched", 0)
    state.engine_pool._entries["physical"] = entry
    monkeypatch.setattr(server, "_server_state", state)
    # TestClient without lifespan exercises the mounted public routes without
    # initializing a second GPU service or loading any user model.
    client = TestClient(server.app)
    client.headers["Authorization"] = "Bearer test-adapter-key"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"lora_layers": 1}')
    (adapter / "adapters.safetensors").write_bytes(b"path-validation-only")
    return client, state, entry, adapter


def test_public_route_requires_bearer_without_admin_cookie(runtime):
    client, _, _, _ = runtime
    assert client.get("/v1/lora-adapters?model=physical").status_code == 200
    client.headers.pop("Authorization")
    assert client.get("/v1/lora-adapters?model=physical").status_code == 401
    assert (
        client.post(
            "/v1/lora-adapters?model=physical", json={"adapter_path": None}
        ).status_code
        == 401
    )


def test_alias_selection_persists_to_physical_model(runtime):
    client, state, _, adapter = runtime
    response = client.post(
        "/v1/lora-adapters?model=friendly", json={"adapter_path": str(adapter)}
    )
    assert response.status_code == 200
    assert response.json()["resolved_model"] == "physical"
    assert response.json()["mode"] == "load_time"
    assert response.json()["runtime_scale_supported"] is False
    assert state.settings_manager.get_settings("physical").adapter_path == str(adapter)
    assert ModelSettingsManager(state.settings_manager.base_path).get_settings(
        "physical"
    ).adapter_path == str(adapter)


def test_loaded_adapter_is_preserved_until_explicit_reload(runtime):
    client, state, entry, adapter = runtime
    loaded_settings = state.settings_manager.get_settings("physical")
    entry.engine = SimpleNamespace(_model_settings=loaded_settings)
    response = client.post(
        "/v1/lora-adapters?model=physical", json={"adapter_path": str(adapter)}
    )
    assert response.status_code == 200
    assert response.json()["reload_required"] is True
    assert response.json()["loaded"] is True
    assert response.json()["active_adapter_path"] is None
    assert entry.engine._model_settings.adapter_path is None
    assert (
        client.post(
            "/v1/lora-adapters?model=physical", json={"adapter_path": None}
        ).json()["reload_required"]
        is False
    )


def test_unsupported_loading_and_invalid_selection_preserve_settings(runtime):
    client, state, entry, _ = runtime
    entry.engine_type = "embedding"
    assert client.get("/v1/lora-adapters?model=physical").json()["supported"] is False
    assert (
        client.post(
            "/v1/lora-adapters?model=physical", json={"adapter_path": None}
        ).status_code
        == 400
    )
    entry.engine_type = "batched"
    entry.is_loading = True
    assert (
        client.post(
            "/v1/lora-adapters?model=physical", json={"adapter_path": None}
        ).status_code
        == 409
    )
    entry.is_loading = False
    assert (
        client.post(
            "/v1/lora-adapters?model=physical", json={"adapter_path": "/not/an/adapter"}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/v1/lora-adapters?model=physical",
            json={"adapter_path": None, "scale": 0.5},
        ).status_code
        == 422
    )
    assert state.settings_manager.get_settings("physical").adapter_path is None


def test_exposed_profile_is_reported_as_unsupported(runtime, monkeypatch):
    client, state, _, _ = runtime
    original = state.settings_manager.get_exposed_profile_source_model_id
    monkeypatch.setattr(
        state.settings_manager,
        "get_exposed_profile_source_model_id",
        lambda model: "physical" if model == "profile-id" else original(model),
    )
    response = client.get("/v1/lora-adapters?model=profile-id")
    assert response.status_code == 200
    assert response.json()["supported"] is False
    assert response.json()["unsupported_reason"] == "profile_override"
    assert (
        client.post(
            "/v1/lora-adapters?model=profile-id", json={"adapter_path": None}
        ).status_code
        == 400
    )


def test_custom_loader_is_reported_as_unsupported(runtime):
    from pathlib import Path

    client, _, entry, adapter = runtime
    model_dir = Path(entry.model_path)
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"quantization_config": {"quant_method": "paroquant"}}'
    )
    response = client.get("/v1/lora-adapters?model=physical")
    assert response.json()["supported"] is False
    assert response.json()["unsupported_reason"] == "custom_loader"
    assert client.post(
        "/v1/lora-adapters?model=physical", json={"adapter_path": str(adapter)}
    ).status_code == 400


def test_invalid_config_and_unknown_model(runtime):
    client, _, _, adapter = runtime
    (adapter / "adapter_config.json").write_text("[]")
    assert client.post(
        "/v1/lora-adapters?model=physical", json={"adapter_path": str(adapter)}
    ).status_code == 400
    assert client.get("/v1/lora-adapters?model=missing").status_code == 404


@pytest.mark.parametrize("model_type", ["deepseek_v41", "mimo_v2", "mimo_v2_flash"])
def test_new_specialized_vlm_loaders_reject_adapters(runtime, monkeypatch, model_type):
    import json
    from pathlib import Path
    from omlx.patches.mimo_v2 import omnimodal

    client, state, entry, adapter = runtime
    entry.engine_type = "vlm"
    root = Path(entry.model_path)
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": model_type}))
    monkeypatch.setattr(omnimodal, "has_vision_sidecar", lambda path: True)
    info = client.get("/v1/lora-adapters?model=physical").json()
    assert info["supported"] is False
    assert info["unsupported_reason"] == "custom_loader"
    response = client.post("/v1/lora-adapters?model=physical",
                           json={"adapter_path": str(adapter)})
    assert response.status_code == 400
    assert state.settings_manager.get_settings("physical").adapter_path is None
