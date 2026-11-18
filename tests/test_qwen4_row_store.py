"""External PLE checkpoints preserve quantized rows through the shared SSD path."""
import importlib
import json

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
TextConfig = importlib.import_module("mlx_vlm.models.qwen4_exp.config").TextConfig


def row_store(tmp_path):
    rows, width = 36, 32
    original = (mx.arange(rows * width).reshape(rows, width) % 37).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(original, bits=4, group_size=32)
    scales, biases = scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)
    expected = mx.dequantize(weight, scales, biases, bits=4, group_size=32)
    mx.eval(weight, scales, biases, expected)
    with (tmp_path / "table.rows").open("wb") as file:
        for row in range(rows):
            file.write(np.asarray(weight[row]).tobytes())
            file.write(np.asarray(scales[row].view(mx.uint16)).tobytes())
            file.write(np.asarray(biases[row].view(mx.uint16)).tobytes())
    manifest = tmp_path / "ple-store.json"
    manifest.write_text(json.dumps({
        "version": 2, "source_root": ".", "layout": "interleaved_q4",
        "data_file": "table.rows", "row_width": width, "row_count": rows,
        "row_stride": 20, "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "cache_rows": 0,
    }))
    config = TextConfig.from_dict(dict(
        model_type="qwen4_exp_text", hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=3,
        num_experts=4, num_experts_per_tok=2, shared_expert_intermediate_size=16,
        moe_intermediate_size=16, rms_norm_eps=1e-6, vocab_size=64,
        num_key_value_heads=2, max_position_embeddings=128, head_dim=8,
        hc_count=2, hc_lowrank=8, ple_layer_ids=[1], ple_embed_dim=128,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=5,
        make_ngram_vocab_size_divisible_by=4, split_ngram_parts=4, eos_token_id=1,
        ple_storage={"manifest": str(manifest), "cache_rows": 0},
    ))
    return config, expected


def test_external_ple_matches_direct_dequantization_and_keeps_duplicate_order(tmp_path):
    config, expected = row_store(tmp_path)
    embedding = language.Qwen4ExpNGramEmbedding(config, config.ple_embed_dim, 0, 0)
    ids = mx.array([[35, 0, 35, 12]], dtype=mx.int64)
    actual = embedding.ngram_embedding(ids)
    assert mx.array_equal(actual, expected[ids]).item()
    embedding.ngram_embedding.prefetch(ids)
    assert mx.array_equal(embedding.ngram_embedding(ids), expected[ids]).item()
    embedding.ngram_embedding.close()


def test_external_ple_rejects_geometry_mismatch(tmp_path):
    config, _ = row_store(tmp_path)
    config.ple_embed_dim *= 2
    with pytest.raises(ValueError, match="row width"):
        language.Qwen4ExpNGramEmbedding(config, config.ple_embed_dim, 0, 0)


def test_external_nvfp4_rows_match_direct_dequantization_after_prefetch(tmp_path):
    config, _ = row_store(tmp_path)
    original = (mx.arange(36 * 32).reshape(36, 32) % 19).astype(mx.bfloat16)
    weight, scales = mx.quantize(original, bits=4, group_size=16, mode="nvfp4")
    expected = mx.dequantize(weight, scales, group_size=16, bits=4, mode="nvfp4")
    mx.eval(weight, scales, expected)
    with (tmp_path / "table.rows").open("wb") as out:
        for row in range(36):
            out.write(np.asarray(weight[row]).tobytes())
            out.write(np.asarray(scales[row]).tobytes())
    manifest = tmp_path / "ple-store.json"
    data = json.loads(manifest.read_text())
    data.update(row_stride=18, quantization={"bits": 4, "group_size": 16, "mode": "nvfp4"})
    manifest.write_text(json.dumps(data))
    embedding = language.Qwen4ExpNGramEmbedding(config, config.ple_embed_dim, 0, 0)
    ids = mx.array([[35, 0, 35, 12]], dtype=mx.int64)
    try:
        embedding.ngram_embedding.prefetch(ids)
        assert mx.array_equal(embedding.ngram_embedding(ids), expected[ids]).item()
        assert embedding.ngram_embedding.last_prefetch_hit
    finally:
        embedding.ngram_embedding.close()
