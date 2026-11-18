"""Normalize published PLE storage into the existing tensor-reader interface."""
from pathlib import Path
import json


_DTYPES = {"U32": 4, "BF16": 2, "U8": 1}


def external_layout(model_path, storage, prefix, rows, dims):
    if storage.get("prefill_first_chunk", "wide") not in {"wide", "narrow"}:
        raise ValueError("external PLE prefill_first_chunk must be wide or narrow")
    manifest = storage.get("manifest")
    if not manifest:
        raise ValueError("ple_storage requires a manifest path")
    manifest = Path(manifest)
    if not manifest.is_absolute():
        manifest = Path(model_path) / manifest
    data = json.loads(manifest.read_text())
    if data.get("version") != 2:
        raise ValueError("unsupported external PLE manifest version")
    if data.get("row_width") != dims:
        raise ValueError("external PLE row width does not match model configuration")
    cache_rows = storage.get("cache_rows", data.get("cache_rows", 0))
    if cache_rows is not None and (not isinstance(cache_rows, int) or cache_rows < 0):
        raise ValueError("external PLE cache_rows must be non-negative")
    quant = data.get("quantization", {})
    kind = (quant.get("bits"), quant.get("group_size"), quant.get("mode"))
    if kind == (4, 32, "affine"):
        families = (("weight", "U32", dims // 8), ("scales", "BF16", dims // 32), ("biases", "BF16", dims // 32))
    elif kind == (4, 16, "nvfp4"):
        families = (("weight", "U32", dims // 8), ("scales", "U8", dims // 16))
    else:
        raise ValueError(f"unsupported external PLE quantization: {kind}")
    if dims <= 0 or dims % quant["group_size"]:
        raise ValueError("external PLE row width is not divisible by group size")
    weight_map, layouts, sizes = {}, {}, []

    def filename(root, name):
        path = Path(name)
        if path.is_absolute() or path.name != str(path):
            raise ValueError("PLE storage files must be filenames beside their declared root")
        return (root / path).resolve()

    def add(index, family, path, dtype, shape, offset, stride=None):
        if offset < 0 or len(shape) != 2 or shape[0] <= 0:
            raise ValueError("invalid external PLE tensor layout")
        width = shape[1] * _DTYPES[dtype]
        step = width if stride is None else stride
        if step < width or path.stat().st_size < offset + (shape[0] - 1) * step + width:
            raise ValueError("external PLE tensor exceeds file size")
        key = f"{prefix}.shards.{index}.{family}"
        weight_map[key] = str(path)
        entry = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + shape[0] * width]}
        if stride is not None:
            entry["row_stride"] = stride
        layouts.setdefault(str(path), {})[key] = entry

    layout = data.get("layout", "safetensors_ranges")
    if layout == "interleaved_q4":
        if data.get("row_count") != rows:
            raise ValueError("external PLE row count does not match model configuration")
        stride = sum(width * _DTYPES[dtype] for _, dtype, width in families)
        if data.get("row_stride") != stride:
            raise ValueError("external PLE row stride does not match quantization")
        path = filename(manifest.parent, data["data_file"])
        if path.stat().st_size != rows * stride:
            raise ValueError("external PLE file size does not match manifest")
        offset = 0
        for family, dtype, width in families:
            add(0, family, path, dtype, [rows, width], offset, stride)
            offset += width * _DTYPES[dtype]
        sizes.append(rows)
    elif layout == "safetensors_ranges":
        root = Path(data["source_root"])
        if not root.is_absolute():
            root = manifest.parent / root
        total = 0
        for index, shard in enumerate(data["shards"]):
            count = shard["row_count"]
            if shard["row_start"] != total or count <= 0:
                raise ValueError("external PLE shards must be contiguous and ordered")
            for family, dtype, width in families:
                tensor = shard[family]
                if tensor["dtype"] != dtype or tensor["shape"] != [count, width]:
                    raise ValueError("external PLE tensor geometry does not match quantization")
                add(index, family, filename(root, tensor["file"]), dtype, tensor["shape"], tensor["offset"])
            sizes.append(count)
            total += count
        if total != rows:
            raise ValueError("external PLE row count does not match model configuration")
    else:
        raise ValueError(f"unsupported external PLE layout: {layout}")
    return tuple(sizes), weight_map, layouts


def shard_spec(base, size, dims, weight_map, reader):
    weight, scales, biases = (f"{base}.{part}" for part in ("weight", "scales", "biases"))
    source = reader(weight)
    shape, dtype = source.tensor_shape(weight), source.tensor_dtype(weight)
    if len(shape) != 2 or shape[0] != size:
        raise ValueError(f"Unexpected shape for {weight}: {shape}")
    if scales not in weight_map and biases not in weight_map:
        if shape[1] != dims:
            raise ValueError(f"Unexpected dense PLE width for {weight}: expected {dims}, got {shape[1]}")
        if dtype not in {"BF16", "F16", "F32", "F8_E4M3"}:
            raise TypeError(f"SSD-backed dense Qwen4 PLE does not support {dtype}")
        return weight, None, None, None, None
    if scales not in weight_map:
        raise ValueError(f"Incomplete affine PLE tensors for {base}: both scales and biases are required")
    scale_reader = reader(scales)
    scale_shape = scale_reader.tensor_shape(scales)
    if biases not in weight_map:
        if dtype != "U32" or shape[1] != dims // 8 or dims % 16 or scale_reader.tensor_dtype(scales) != "U8" or scale_shape != (size, dims // 16):
            raise ValueError(f"Unsupported nvfp4 PLE layout for {base}")
        return weight, scales, None, 4, 16
    if dtype != "U32":
        raise TypeError(f"Affine Qwen4 PLE weight must be U32, got {dtype} for {weight}")
    bias_shape = reader(biases).tensor_shape(biases)
    if scale_shape != bias_shape or len(scale_shape) != 2:
        raise ValueError(f"Invalid affine PLE parameter shapes for {base}: scales={scale_shape}, biases={bias_shape}")
    if scale_shape[0] != size or scale_shape[1] <= 0:
        raise ValueError(f"Unexpected affine PLE scale shape for {base}: {scale_shape}")
    if dims % scale_shape[1]:
        raise ValueError(f"Cannot infer affine PLE group size for {base}: dims={dims}, scales={scale_shape}")
    group = dims // scale_shape[1]
    if shape[1] * 32 % dims:
        raise ValueError(f"Cannot infer affine PLE bits for {base}: dims={dims}, packed_shape={shape}")
    bits = shape[1] * 32 // dims
    if bits not in {2, 3, 4, 5, 6, 8} or group not in {32, 64, 128}:
        raise ValueError(f"Unsupported affine PLE layout for {base}: bits={bits}, group_size={group}")
    if dims % group or shape[1] != dims * bits // 32:
        raise ValueError(f"Inconsistent affine PLE layout for {base}: weight={shape}, scales={scale_shape}, dims={dims}")
    return weight, scales, biases, bits, group
