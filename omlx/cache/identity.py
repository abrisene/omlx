"""Helpers for separating persistent cache keys by runtime model weights."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CACHE_IDENTITY_VERSION = "omlx-cache-v2"


def model_cache_identity(model_name: str, adapter_path: str | None = None) -> str:
    """Return a stable hash namespace for the model and its loaded adapter.

    Paged-cache metadata continues to use ``model_name``. This identity salts
    block hashes and VLM feature keys so cached activations cannot cross
    adapter weights.
    Versioning every runtime policy invalidates legacy entries that could have
    been written while an adapter shared the base-model namespace. The stable
    versioned identity still permits reuse after subsequent reloads.
    """
    if not adapter_path:
        return f"{model_name}:{_CACHE_IDENTITY_VERSION}"

    adapter_dir = Path(adapter_path)
    digest = hashlib.sha256()
    for filename in ("adapter_config.json", "adapters.safetensors"):
        path = adapter_dir / filename
        digest.update(filename.encode("utf-8") + b"\0")
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)

    return f"{model_name}:{_CACHE_IDENTITY_VERSION}:adapter-sha256:{digest.hexdigest()}"
