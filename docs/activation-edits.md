# Residual activation edits

`POST /v1/chat/completions` accepts an optional `activation_edits` list:

```json
{
  "model": "Qwen3.5-0.8B-MLX-8bit",
  "messages": [{"role": "user", "content": "Explain rain."}],
  "temperature": 0,
  "max_tokens": 64,
  "activation_edits": [{"layer": 12, "neuron": 32, "add": 0.5}]
}
```

Each entry adds a finite float32 value to one residual channel after the
zero-based transformer block, at every prompt and generated token. Negative
values subtract. A dense direction is a list of entries for its channels.
Duplicate layer/channel pairs, non-integer indices, out-of-bounds indices and
non-finite or float32-overflowing values are rejected before inference. An
empty or omitted list uses ordinary generation.

`GET /v1/models` includes `activation_edit_dimensions` for models with authored
layer count and hidden width in `config.json`: `layers`, `neurons`,
`position: "post_block_residual"`, and `index_base: 0`. These describe the
architecture, not whether an unloaded model's engine can run edits. The loaded
resident model supplies final bounds validation.

Edited requests use the existing scheduler and generator. They wait for active
requests, run alone, bypass prefix reuse, and skip completion, abort and boundary
cache persistence. Later requests wait until the edited request finishes. Layer
instances are restored after every scheduler step, including failures and
cancellation; the next ordinary request sees the original model and cache.
Streaming uses the same request isolation.

Disable speculative decoding (MTP, DFlash and SpecPrefill) before requesting
edits. These paths are rejected when active, because their draft and replay
semantics have not been verified for residual edits. Ordinary text through a
VLM-backed language model is supported. Layer outputs must expose their residual
as an array or the first element of a tuple.

For local contrastive extraction, `omlx.activation_edits.residual_edits` also
accepts `observer=callback, layer_ids=[layer]` with an empty edits list. The
callback receives `(layer_index, post_block_array)`. Consume/evaluate the forward
inside the context. The context restores the exact owning layer instances even
when an exposed pipeline layer list is a slice.

## Contrastive probe

From this repository, use the same layer hook to extract a direction from paired
prompt files. The following example uses an existing local model and the DS4
verbosity corpus. Its positive direction is verbose minus succinct.

```sh
python -m tools.activation_probe extract \
  --model /Users/dr/Models/Text/mlx-community/Qwen3.5-0.8B-MLX-8bit \
  --target /Users/dr/Models/Clients/ds4/dir-steering/examples/verbose.txt \
  --contrast /Users/dr/Models/Clients/ds4/dir-steering/examples/succinct.txt \
  --layer 12 --pairs 16 --output /tmp/verbosity.json

python -m tools.activation_probe compare \
  --url http://127.0.0.1:18731/v1 --model Qwen3.5-0.8B-MLX-8bit \
  --direction /tmp/verbosity.json \
  --prompts /Users/dr/Models/Clients/ds4/dir-steering/examples/eval_prompts.txt \
  --scales=-0.25,0,0.25 --concurrent --output /tmp/comparison.json
```

The comparison tool also calls llama.cpp: change `--url` and `--model` to that
server's API base and served model ID. It compares ordinary output before and
after each edit, and optionally sends edited and ordinary requests concurrently.
An API key can be read from the environment variable named by `--api-key-env`.

On October 30, the 16-pair layer-12 direction changed greedy text on both local
Qwen3.5-0.8B backends at strengths -0.25 and +0.25. Across five evaluation prompts,
all subsequent and concurrent ordinary requests matched their original output.
This establishes a causal output change and request isolation. It does not
establish a reliable verbosity control. Strengths of magnitude 5 produced
repetitive or incoherent output on this small model.

## Isolated local server

Use a fresh base path, a directory containing only the test model, an explicit
cluster shim override, and disable discovery of the shared Hugging Face cache:

```sh
OMLX_CLUSTER_PYTHON_SHIM_PATH=/tmp/omlx-edit-test/cluster-python \
  .venv/bin/python -m omlx.cli serve \
  --base-path /tmp/omlx-edit-test \
  --model-dir /tmp/omlx-edit-models \
  --host 127.0.0.1 --port 18731 --headless --no-hf-cache --no-cache
```

`--base-path` alone does not isolate the cluster interpreter shim. The explicit
shim override keeps a temporary server from replacing the installed app's helper.
