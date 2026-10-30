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
    from mlx_lm import load
    from omlx.activation_edits import dimensions, residual_edits

    target = read_prompts(args.target)
    contrast = read_prompts(args.contrast)
    if not target or len(target) != len(contrast):
        raise ValueError("Target and contrast files must contain equal, nonzero prompt counts")
    if args.pairs is not None:
        target, contrast = target[:args.pairs], contrast[:args.pairs]
    model, tokenizer = load(args.model)
    count, width = dimensions(model)
    if not 0 <= args.layer < count:
        raise ValueError(f"Layer must be in 0..{count - 1}")

    def capture(prompt):
        captured = []

        def observe(_layer, residual):
            captured.append(residual[0, -1].astype(mx.float32))

        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        with residual_edits(model, (), observer=observe, layer_ids=[args.layer]):
            logits = model(mx.array([tokens]))
            mx.eval(logits, *captured)
        if len(captured) != 1 or captured[0].shape != (width,):
            raise RuntimeError("The selected layer did not produce one residual capture")
        return captured[0]

    difference = mx.zeros((width,), dtype=mx.float32)
    for index, (positive, negative) in enumerate(zip(target, contrast), 1):
        difference = difference + capture(positive) - capture(negative)
        mx.eval(difference)
        print(f"Captured pair {index}/{len(target)}", flush=True)
    difference = difference / len(target)
    norm = float(mx.linalg.norm(difference).item())
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError("The contrast produced no finite residual direction")
    direction = (difference / norm).tolist()
    result = {
        "format": "residual-direction-v1", "model": args.model,
        "layer": args.layer, "layer_count": count, "width": width,
        "injection": "post-block-add", "pairs": len(target),
        "mean_difference_norm": norm, "direction": direction,
        "target_sha256": hashlib.sha256(Path(args.target).read_bytes()).hexdigest(),
        "contrast_sha256": hashlib.sha256(Path(args.contrast).read_bytes()).hexdigest(),
    }
    write_json(args.output, result)
    print(f"Wrote {width} channels at layer {args.layer}; mean difference norm {norm:.6g}")


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
    extraction.add_argument("--target", required=True)
    extraction.add_argument("--contrast", required=True)
    extraction.add_argument("--layer", required=True, type=int)
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
