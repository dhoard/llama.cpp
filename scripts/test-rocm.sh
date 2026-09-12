#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

DOCKER=${DOCKER:-docker}
IMAGE=${IMAGE:-local/llama.cpp:mtp-dev}
HF_CACHE=${HF_CACHE:-${HOME}/.cache/huggingface}
MODEL=${MODEL:-ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M}
ROCM_DOCKER_ARCH=${ROCM_DOCKER_ARCH:-gfx1101}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
BUILD_TEST_IMAGE=${BUILD_TEST_IMAGE:-1}
BACKEND_OPS=${BACKEND_OPS:-MUL_MAT,FLASH_ATTN_EXT}
BACKEND_TEST_MODE=${BACKEND_TEST_MODE:-support}
RUN_RECURRENT_ROLLBACK=${RUN_RECURRENT_ROLLBACK:-1}

cd "$ROOT_DIR"

if [[ "$BUILD_TEST_IMAGE" == 1 ]]; then
    IMAGE="$IMAGE" \
    TARGET=dev \
    LLAMA_BUILD_TESTS=ON \
    ROCM_DOCKER_ARCH="$ROCM_DOCKER_ARCH" \
    "$ROOT_DIR/scripts/build-image.sh"
fi

"$DOCKER" run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --env "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" \
    --env "BACKEND_OPS=$BACKEND_OPS" \
    --env "BACKEND_TEST_MODE=$BACKEND_TEST_MODE" \
    "$IMAGE" \
    -lc '
set -euo pipefail

test_bin=/app/build/bin
if [[ -x "$test_bin/test-backend-ops" ]]; then
    echo "running test-backend-ops ($BACKEND_TEST_MODE) for $BACKEND_OPS"
    "$test_bin/test-backend-ops" "$BACKEND_TEST_MODE" -o "$BACKEND_OPS"
else
    echo "skipping test-backend-ops: binary not present"
fi
for test_name in test-llama-archs; do
    test_path="$test_bin/$test_name"
    if [[ -x "$test_path" ]]; then
        echo "running $test_name"
        "$test_path"
    else
        echo "skipping $test_name: binary not present"
    fi
done
'

model_repo=${MODEL%%:*}
model_dir=${model_repo//\//--}
model_file=$(find -L "$HF_CACHE/hub/models--$model_dir" \
    -type f -name '*.gguf' -print -quit 2>/dev/null || true)
if [[ -z "$model_file" ]]; then
    echo "No cached GGUF found for $MODEL; skipping server smoke test"
    exit 0
fi

model_in_container="/root/.cache/huggingface${model_file#"$HF_CACHE"}"
if [[ "$RUN_RECURRENT_ROLLBACK" != 0 ]] && "$DOCKER" run --rm --pull=never "$IMAGE" -lc 'test -x /app/build/bin/test-recurrent-state-rollback'; then
    echo "running test-recurrent-state-rollback"
    "$DOCKER" run --rm --pull=never \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --group-add render \
        --ipc=host \
        --env "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        --entrypoint /app/build/bin/test-recurrent-state-rollback \
        "$IMAGE" \
        -m "$model_in_container" \
        -ngl all \
        -c 512 \
        -b 256 \
        -ub 128
fi

container_name=${CONTAINER_NAME:-rocm-test}
port=${PORT:-18000}
cleanup() {
    "$DOCKER" rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"$DOCKER" run -d --name "$container_name" --pull=never \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --env "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" \
    -p "$port:8000" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    --entrypoint /app/full/llama-server \
    "$IMAGE" \
    -hf "$MODEL" \
    --no-mmproj \
    --host 0.0.0.0 \
    --port 8000 \
    -c 4096 \
    -np 1 \
    -ngl all \
    -fa on \
    -b 512 \
    -ub 256 \
    -t 4 \
    --threads-batch 4 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --reasoning-budget 256 >/dev/null

for attempt in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
        break
    fi
    if (( attempt == 60 )); then
        "$DOCKER" logs "$container_name"
        exit 1
    fi
    sleep 2
done

curl -fsS "http://127.0.0.1:$port/completion" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Say exactly: ROCm smoke ok","n_predict":8,"temperature":0,"seed":123}' \
    | python3 -c '
import json
import sys

response = json.load(sys.stdin)
if not response.get("content"):
    raise SystemExit("completion returned no content")
if response.get("timings", {}).get("predicted_n", 0) <= 0:
    raise SystemExit("completion returned no generated tokens")
if response.get("timings", {}).get("draft_n", 0) <= 0:
    raise SystemExit("completion did not exercise MTP")
print("server smoke test passed")
'
