#!/bin/bash
# End-to-end run of the port on the model in $MODEL_DIR. Prints "prefill: X tok/s" and "decode: Y tok/s" ONLY from a real
# run of the model on the GPU: build/generate_model (src/model/generate_model.cu) loads the safetensors checkpoint, runs an
# invariant check (chunked vs sequential prefill) and then times real prefill and decode. It exits non-zero without printing
# rates if logits are non-finite or the check fails.
set -uo pipefail
MODEL_DIR="${MODEL_DIR:-/tmp/model}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ROOT"
[ -d "$MODEL_DIR" ] || { echo "e2e: MODEL_DIR=$MODEL_DIR does not exist"; exit 2; }
{ compgen -G "$MODEL_DIR/*.gguf" || compgen -G "$MODEL_DIR/*.safetensors"; } >/dev/null || { echo "e2e: no .gguf or .safetensors weights in $MODEL_DIR"; exit 2; }   # the checkpoint on the test machine is HF safetensors (config.json + model-NNNNN-of-MMMMM.safetensors)
if [ -x build/generate_model ]; then            # a real model backend, when one exists, is built under this name
  exec build/generate_model --model "$MODEL_DIR" --bench
fi
echo "e2e: no real-model backend in this tree (only the synthetic generate backend); not reporting any tok/s"
exit 3
