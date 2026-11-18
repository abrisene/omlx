"""PLE token hashing and storage selection for the optimized Qwen runtime."""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.cache import ArraysCache

from .config import TextConfig
from . import language as runtime


class Qwen4ExpNGramEmbedding(nn.Module):
    def __init__(
        self,
        config: TextConfig,
        embedding_dim: int,
        layer_idx: int,
        ple_layer_index: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.seed = config.seed
        eos = config.eos_token_id
        self.eos_token_id = eos[0] if isinstance(eos, list) else eos

        head_vocab_sizes = []
        head_offsets = []
        total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = ple_layer_index * self.ngram_heads + head_idx
            size = runtime._find_nth_prime_after(
                config.ngram_vocab_size_base - 1, global_head_idx + 1
            )
            head_vocab_sizes.append(size)
            head_offsets.append(total_vocab_size)
            total_vocab_size += size

        self.layer_multipliers = runtime._build_layer_multipliers(
            self.unigram_vocab_size,
            self.ngram_size,
            ple_layer_index,
            self.seed,
        )
        self.ngram_heads_vocab_sizes = mx.array(head_vocab_sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(head_offsets, dtype=mx.int64)
        # Rows each head's indices fall in (the sharded lookup picks shards
        # per head on the device).
        self._head_row_ranges = tuple(
            (offset, offset + size)
            for offset, size in zip(head_offsets, head_vocab_sizes)
        )
        divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(total_vocab_size / divisor) * divisor
        embedding_args = (
            padded_vocab_size,
            embedding_dim // self.ngram_heads,
            config.split_ngram_parts,
        )
        if config.ple_storage or runtime._PLE_RUNTIME_MODE == "mmap":
            model_path = runtime._PLE_RUNTIME_MODEL_PATH
            if model_path is None:
                manifest = (config.ple_storage or {}).get("manifest")
                if not manifest or not runtime.Path(manifest).is_absolute():
                    raise RuntimeError("SSD-backed PLE has no configured model path")
                model_path = runtime.Path(manifest).parent
            prefix = (
                f"model.language_model.layers.{layer_idx}.ple.ple_embedding."
                "ngram_embedding"
            )
            self.ngram_embedding = runtime.DiskBackedShardedEmbedding(
                model_path, prefix, *embedding_args, storage=config.ple_storage,
            )
        else:
            self.ngram_embedding = runtime.ShardedEmbedding(*embedding_args)

    def _shift_right_ignore_eos(self, token_ids: mx.array, shift: int):
        if shift == 0:
            return token_ids
        batch, seq_len = token_ids.shape
        positions = mx.arange(seq_len, dtype=mx.int64)
        eos_positions = mx.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = mx.cummax(eos_positions, axis=1)
        previous_eos = mx.concatenate(
            [mx.full((batch, 1), -1, dtype=mx.int64), previous_eos_inclusive[:, :-1]],
            axis=1,
        )
        segment_start = previous_eos + 1
        position_in_segment = positions[None] - segment_start
        source_positions = positions - shift
        gather_positions = mx.broadcast_to(
            mx.maximum(source_positions, 0)[None], (batch, seq_len)
        )
        shifted = mx.take_along_axis(token_ids, gather_positions, axis=1)
        valid = (position_in_segment >= shift) & (source_positions[None] >= 0)
        return mx.where(valid, shifted, self.eos_token_id)

    def _previous_context(
        self, input_ids: mx.array, cache: ArraysCache | None
    ) -> mx.array:
        batch = input_ids.shape[0]
        if cache is not None and cache[3] is not None:
            return cache[3]
        return mx.full(
            (batch, self.context_len), self.eos_token_id, dtype=mx.int64
        )

    def _ngram_indices(self, token_history: mx.array, length: int) -> mx.array:
        """Hashed table rows for the last ``length`` tokens of ``token_history``."""
        shifted_tokens = [
            self._shift_right_ignore_eos(token_history, shift)
            for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = mx.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ngram_ids = mixed_ids[..., None] % sizes[None, None]
            blocks.append(ngram_ids + offsets[None, None])

        return mx.concatenate(blocks, axis=-1)[:, -length:]

    def prefetch(self, next_ids: mx.array, previous_context: mx.array) -> None:
        """Gather the rows of an upcoming chunk ahead of time (SSD-backed tables only)."""
        prefetch = getattr(self.ngram_embedding, "prefetch", None)
        if prefetch is None:
            return
        next_ids = next_ids.astype(mx.int64)
        history = mx.concatenate([previous_context.astype(mx.int64), next_ids], axis=-1)
        prefetch(self._ngram_indices(history, next_ids.shape[1]))

    def __call__(self, input_ids: mx.array, cache: Optional[ArraysCache]):
        input_ids = input_ids.astype(mx.int64)
        previous_context = self._previous_context(input_ids, cache)

        token_history = mx.concatenate([previous_context, input_ids], axis=-1)
        if cache is not None:
            cache.update_window(3, token_history, self.context_len)

        ngram_ids = self._ngram_indices(token_history, input_ids.shape[1])
        if type(self.ngram_embedding) is runtime.ShardedEmbedding:
            embeddings = self.ngram_embedding(ngram_ids, self._head_row_ranges)
        else:
            embeddings = self.ngram_embedding(ngram_ids)
        return embeddings.reshape(*embeddings.shape[:-2], -1)

