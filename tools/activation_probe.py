"""Extract a contrastive residual direction, then compare request-scoped edits.

Run from the repository with ``python -m tools.activation_probe --help``.
The extractor reuses the runtime's layer hook. Generation uses the HTTP API.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.request


def read_prompts(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, data):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def extract(args):
    import mlx.core as mx
    from omlx.utils.model_loading import lm_load_compat, maybe_apply_pre_load_patches
    from omlx.activation_edits import dimensions, residual_edits

    if args.responses and (args.target or args.contrast):
        raise ValueError("--responses cannot be combined with --target or --contrast")
    if args.responses:
        pairs = [json.loads(line) for line in Path(args.responses).read_text().splitlines() if line.strip()]
        for pair in pairs:
            if not all(isinstance(pair.get(key), str) and pair[key] for key in ("prompt", "positive", "negative")):
                raise ValueError("Response pairs require nonempty prompt, positive and negative strings")
    else:
        if not args.target or not args.contrast:
            raise ValueError("Provide --responses or both --target and --contrast")
        target, contrast = read_prompts(args.target), read_prompts(args.contrast)
        if not target or len(target) != len(contrast):
            raise ValueError("Target and contrast must have equal, nonzero counts")
        pairs = [{"positive": p, "negative": n} for p, n in zip(target, contrast)]
    pairs = pairs[:args.pairs] if args.pairs else pairs
    if not pairs:
        raise ValueError("At least one pair is required")
    maybe_apply_pre_load_patches(args.model)
    model, tokenizer = lm_load_compat(args.model)
    count, width = dimensions(model)
    layers = [int(value) for value in args.layer.split(",")]
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= layer < count for layer in layers):
        raise ValueError(f"Layers must be unique and in 0..{count - 1}")

    def capture(pair, side):
        captured = {}
        prompt = pair.get("prompt", pair[side])
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        start = len(tokens)
        if args.responses:
            tokens = tokens + tokenizer.encode(pair[side], add_special_tokens=False)

        def observe(layer, residual):
            values = residual[0, start:] if args.responses else residual[0, -1:]
            captured[layer] = mx.mean(values.astype(mx.float32), axis=0)

        with residual_edits(model, (), observer=observe, layer_ids=layers):
            logits = model(mx.array([tokens]))
            mx.eval(logits, *captured.values())
        if set(captured) != set(layers) or any(value.shape != (width,) for value in captured.values()):
            raise RuntimeError("Selected layers did not produce one residual capture each")
        return captured

    differences = {layer: mx.zeros((width,), dtype=mx.float32) for layer in layers}
    for index, pair in enumerate(pairs, 1):
        positive, negative = capture(pair, "positive"), capture(pair, "negative")
        for layer in layers:
            differences[layer] = differences[layer] + positive[layer] - negative[layer]
        mx.eval(*differences.values())
        print(f"Captured pair {index}/{len(pairs)}", flush=True)
    for layer in layers:
        difference = differences[layer] / len(pairs)
        norm = float(mx.linalg.norm(difference).item())
        if not math.isfinite(norm) or norm == 0:
            raise RuntimeError("The contrast produced no finite residual direction")
        result = {
            "format": "residual-direction-v1", "model": args.model,
            "layer": layer, "layer_count": count, "width": width,
            "injection": "post-block-add", "pairs": len(pairs),
            "mean_difference_norm": norm, "direction": (difference / norm).tolist(),
            "capture": "response-token-mean" if args.responses else "prompt-last-token",
        }
        for name in ("responses", "target", "contrast"):
            path = getattr(args, name)
            if path:
                result[name + "_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        output = args.output if len(layers) == 1 else str(Path(args.output).with_name(Path(args.output).stem + f"-layer{layer}.json"))
        write_json(output, result)
        print(f"Wrote {output}; mean difference norm {norm:.6g}", flush=True)


def request(args, prompt, edits):
    body = {
        "model": args.model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "seed": 123, "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if edits:
        body["activation_edits"] = edits
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(args.api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        args.url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, allow_nan=False).encode(), headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()}") from exc
    choice = result["choices"][0]
    return {
        "text": choice["message"].get("content"),
        "reasoning": choice["message"].get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"), "usage": result.get("usage"),
    }


def compare(args):
    vector = json.loads(Path(args.direction).read_text())
    if vector.get("format") != "residual-direction-v1":
        raise ValueError("Expected a residual-direction-v1 file")
    direction = vector["direction"]
    if len(direction) != vector["width"] or not all(math.isfinite(x) for x in direction):
        raise ValueError("Direction shape or values are invalid")
    scales = [float(item) for item in args.scales.split(",")]
    if not scales or not all(math.isfinite(scale) for scale in scales):
        raise ValueError("Scales must be finite numbers")
    prompts = read_prompts(args.prompts)
    if not prompts:
        raise ValueError("At least one evaluation prompt is required")
    report = {"url": args.url, "model": args.model, "direction": args.direction, "trials": []}
    failures = []
    changed = False
    for prompt in prompts:
        baseline = request(args, prompt, [])
        for scale in scales:
            edits = [
                {"layer": vector["layer"], "neuron": neuron, "add": scale * value}
                for neuron, value in enumerate(direction) if scale * value != 0
            ]
            edited = request(args, prompt, edits)
            restored = request(args, prompt, [])
            same = restored["text"] == baseline["text"]
            if not same:
                failures.append(f"Baseline changed after scale {scale}: {prompt}")
            difference = edited["text"] != baseline["text"]
            changed |= difference
            trial = {
                "prompt": prompt, "scale": scale, "baseline": baseline, "edited": edited,
                "restored": restored, "baseline_restored": same,
                "edited_differs": difference,
            }
            if args.concurrent and edits:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    edited_future = pool.submit(request, args, prompt, edits)
                    plain_future = pool.submit(request, args, prompt, [])
                    trial["concurrent_edited"] = edited_future.result()
                    trial["concurrent_plain"] = plain_future.result()
                trial["concurrent_plain_matches"] = trial["concurrent_plain"]["text"] == baseline["text"]
                if not trial["concurrent_plain_matches"]:
                    failures.append(f"Concurrent baseline changed at scale {scale}: {prompt}")
            report["trials"].append(trial)
            write_json(args.output, report)
            print(f"Scale {scale:g}: changed={difference}, baseline_restored={same}", flush=True)
    report["any_output_changed"] = changed
    report["failures"] = failures
    write_json(args.output, report)
    if failures or not changed:
        raise SystemExit("Comparison failed: " + ("; ".join(failures) or "no edited output changed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract", help="Capture paired prompts with a local MLX model")
    extraction.add_argument("--model", required=True)
    extraction.add_argument("--target")
    extraction.add_argument("--contrast")
    extraction.add_argument("--responses", help="JSONL prompt/positive/negative teacher-forced response pairs")
    extraction.add_argument("--layer", required=True, help="Comma-separated zero-based layers; captured in one forward")
    extraction.add_argument("--pairs", type=int)
    extraction.add_argument("--output", required=True)
    extraction.set_defaults(run=extract)
    comparison = commands.add_parser("compare", help="Exercise the same edit schema on either HTTP backend")
    comparison.add_argument("--url", required=True, help="API base URL including /v1")
    comparison.add_argument("--model", required=True, help="Served model ID")
    comparison.add_argument("--direction", required=True)
    comparison.add_argument("--prompts", required=True)
    comparison.add_argument("--scales", default="-0.25,0,0.25")
    comparison.add_argument("--max-tokens", type=int, default=192)
    comparison.add_argument("--timeout", type=float, default=180)
    comparison.add_argument("--api-key-env", default="ACTIVATION_API_KEY")
    comparison.add_argument("--concurrent", action="store_true")
    comparison.add_argument("--output", required=True)
    comparison.set_defaults(run=compare)
    args = parser.parse_args()
    if getattr(args, "pairs", 1) is not None and getattr(args, "pairs", 1) <= 0:
        parser.error("--pairs must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
