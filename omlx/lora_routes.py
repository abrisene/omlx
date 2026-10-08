"""Load-time MLX adapter configuration."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict


class AdapterSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    adapter_path: str | None


def validate_adapter_path(value: str | None) -> str | None:
    if not value or not value.strip():
        return None
    directory = Path(value.strip()).expanduser()
    required = (directory / "adapter_config.json", directory / "adapters.safetensors")
    if not all(p.is_file() for p in required):
        raise HTTPException(
            400,
            "Invalid MLX adapter directory; requires "
            "adapter_config.json and adapters.safetensors",
        )
    try:
        config = json.loads(required[0].read_text())
        if not isinstance(config, dict):
            raise ValueError("adapter config must be an object")
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"Invalid MLX adapter config: {exc}") from exc
    return str(directory.resolve())


def create_lora_router(get_state):
    router = APIRouter()

    def context(model):
        state = get_state()
        if state.engine_pool is None or state.settings_manager is None:
            raise HTTPException(503, "Server not initialized")
        resolved = state.engine_pool.resolve_model_id(model, state.settings_manager)
        profile_source = None
        get_profile_source = getattr(
            state.settings_manager,
            "get_exposed_profile_source_model_id",
            None,
        )
        if callable(get_profile_source):
            profile_source = get_profile_source(model)
        entry = state.engine_pool.get_entry(resolved)
        if entry is None:
            raise HTTPException(404, f"Model not found: {model}")
        return state, entry, resolved, profile_source is not None

    def describe(
        state,
        entry,
        model,
        resolved_model,
        is_profile_model,
    ):
        settings = state.settings_manager.get_settings(resolved_model)
        configured = settings.adapter_path
        supported = (
            not is_profile_model
            and entry.engine_type in {"batched", "vlm"}
            and getattr(entry, "source_type", "local") != "cluster"
            and not settings.dflash_enabled
        )
        custom_loader = False
        try:
            config = json.loads((Path(entry.model_path) / "config.json").read_text())
            quant = config.get("quantization_config") or {}
            method = quant.get("quant_method", "").lower()
            custom_loader = method == "paroquant" or (
                entry.engine_type == "vlm"
                and config.get("model_type") in {"cohere2_moe", "deepseek_v41"}
            )
            if entry.engine_type == "vlm" and config.get("model_type") in {"mimo_v2", "mimo_v2_flash"}:
                from .patches.mimo_v2.omnimodal import has_vision_sidecar

                custom_loader = custom_loader or has_vision_sidecar(entry.model_path)
            if method == "compressed-tensors":
                from .patches.qwen38_modelopt_mixed import is_supported_config

                custom_loader = is_supported_config(config)
        except (OSError, ValueError):
            pass
        supported = supported and not custom_loader
        loaded = entry.engine is not None
        active_settings = getattr(entry.engine, "_model_settings", None)
        active = getattr(active_settings, "adapter_path", None) if loaded else None
        return {
            "model": model,
            "unsupported_reason": (
                "profile_override"
                if is_profile_model
                else ("custom_loader" if custom_loader else (None if supported else "engine"))
            ),
            "resolved_model": resolved_model,
            "supported": supported,
            "mode": "load_time" if supported else "unsupported",
            "runtime_scale_supported": False,
            "max_active_adapters": 1 if supported else 0,
            "catalog_scope": "configured_adapter",
            "adapters": ([{"path": configured}] if configured else []),
            "configured_adapter_path": configured,
            "active_adapter_path": active,
            "loaded": loaded,
            "reload_required": loaded and configured != active,
        }

    @router.get("/v1/lora-adapters")
    async def list_adapters(model: str):
        state, entry, resolved, is_profile_model = context(model)
        return describe(state, entry, model, resolved, is_profile_model)

    @router.post("/v1/lora-adapters")
    async def select_adapter(selection: AdapterSelection, model: str):
        state, entry, resolved, is_profile_model = context(model)
        current = describe(state, entry, model, resolved, is_profile_model)
        if not current["supported"]:
            raise HTTPException(400, "This engine does not support MLX adapters")
        if entry.is_loading:
            raise HTTPException(409, "Model is loading; retry after loading completes")
        settings = state.settings_manager.get_settings(resolved)
        settings.adapter_path = validate_adapter_path(selection.adapter_path)
        state.settings_manager.set_settings(resolved, settings)
        state.engine_pool._clear_load_failure(entry)
        return describe(state, entry, model, resolved, is_profile_model)

    return router
