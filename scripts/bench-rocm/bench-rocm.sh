#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

DOCKER=${DOCKER:-docker}
IMAGE=${IMAGE:-local/llama.cpp:mtp}
HF_CACHE=${HF_CACHE:-${HOME}/.cache/huggingface}
MODEL=${MODEL:-ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M}
PORT=${PORT:-18080}
CTX=${CTX:-262144}
CONTEXTS=${CONTEXTS:-8192,32768,65536,131072,196608,262144}
PARALLEL=${PARALLEL:-1}
BATCH=${BATCH:-2048}
UBATCH=${UBATCH:-512}
THREADS=${THREADS:-8}
THREADS_BATCH=${THREADS_BATCH:-8}
CACHE_TYPE_K=${CACHE_TYPE_K:-q4_0}
CACHE_TYPE_V=${CACHE_TYPE_V:-q4_0}
CACHE_REUSE=${CACHE_REUSE:-1024}
REASONING_BUDGET=${REASONING_BUDGET:-8192}
MTP_N_MAX=${MTP_N_MAX:-0}
GDN_OPT=${GDN_OPT:-1}
SPARSE_ATTN_MODE=${SPARSE_ATTN_MODE:-off}
REPETITIONS=${REPETITIONS:-3}
WARMUPS=${WARMUPS:-1}
N_PREDICT=${N_PREDICT:-256}
TIMEOUT=${TIMEOUT:-1800}
OUTPUT=${OUTPUT:-benchmarks/rocm/results.json}
CONTAINER_NAME=${CONTAINER_NAME:-rocm-bench}

if [[ "$SPARSE_ATTN_MODE" != off ]]; then
    echo "error: sparse attention is not implemented for Qwen3.5 in this checkout" >&2
    exit 2
fi

if ! [[ "$MTP_N_MAX" =~ ^[0-9]+$ ]]; then
    echo "error: MTP_N_MAX must be a non-negative integer" >&2
    exit 2
fi

mkdir -p "$HF_CACHE" "$(dirname "$ROOT_DIR/$OUTPUT")"
cd "$ROOT_DIR"

commit=$(git rev-parse HEAD)
image_id=$("$DOCKER" image inspect --format '{{.Id}}' "$IMAGE")
metadata=$(python3 - "$IMAGE" "$image_id" "$commit" "$MODEL" "$CTX" "$PARALLEL" "$BATCH" "$UBATCH" "$THREADS" "$THREADS_BATCH" "$CACHE_TYPE_K" "$CACHE_TYPE_V" "$CACHE_REUSE" "$REASONING_BUDGET" "$MTP_N_MAX" "$GDN_OPT" "$SPARSE_ATTN_MODE" "$REPETITIONS" "$WARMUPS" "$N_PREDICT" "${GGML_HIP_FA_DEBUG:-}" "${GGML_HIP_FA_Q4_MTP_VEC:-}" <<'PY'
import json
import sys

keys = (
    "image",
    "image_id",
    "commit",
    "model",
    "ctx",
    "parallel",
    "batch",
    "ubatch",
    "threads",
    "threads_batch",
    "cache_type_k",
    "cache_type_v",
    "cache_reuse",
    "reasoning_budget",
    "mtp_n_max",
    "gdn_opt",
    "sparse_attn_mode",
    "repetitions",
    "warmups",
    "n_predict",
    "fa_debug",
    "fa_q4_mtp_vec",
)
values = sys.argv[1:]
data = dict(zip(keys, values))
for key in ("ctx", "parallel", "batch", "ubatch", "threads", "threads_batch", "reasoning_budget", "mtp_n_max", "repetitions", "warmups", "n_predict"):
    data[key] = int(data[key])
print(json.dumps(data))
PY
)

cleanup() {
    "$DOCKER" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker_args=(
    run -d --name "$CONTAINER_NAME" --pull=never
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    --ipc=host
    -p "$PORT:8000"
    -v "$HF_CACHE:/root/.cache/huggingface"
)
if [[ -n ${HF_TOKEN:-} ]]; then
    docker_args+=(--env HF_TOKEN)
fi
if [[ -n ${GGML_HIP_FA_DEBUG:-} ]]; then
    docker_args+=(--env GGML_HIP_FA_DEBUG)
fi
if [[ -n ${GGML_HIP_FA_Q4_MTP_VEC:-} ]]; then
    docker_args+=(--env GGML_HIP_FA_Q4_MTP_VEC)
fi
if [[ "$GDN_OPT" == 0 ]]; then
    docker_args+=(--env GGML_HIP_GDN_OPT=0)
fi

docker_args+=(
    "$IMAGE"
    -hf "$MODEL"
    --no-mmproj
    --host 0.0.0.0
    --port 8000
    -c "$CTX"
    -np "$PARALLEL"
    -ngl all
    -fa on
    -b "$BATCH"
    -ub "$UBATCH"
    -t "$THREADS"
    --threads-batch "$THREADS_BATCH"
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --cache-reuse "$CACHE_REUSE"
    --reasoning-budget "$REASONING_BUDGET"
)
if (( MTP_N_MAX > 0 )); then
    docker_args+=(--spec-type draft-mtp --spec-draft-n-max "$MTP_N_MAX")
fi

echo "Starting $IMAGE on port $PORT"
"$DOCKER" "${docker_args[@]}" >/dev/null

for attempt in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
        break
    fi
    if (( attempt == 120 )); then
        "$DOCKER" logs "$CONTAINER_NAME"
        exit 1
    fi
    sleep 2
done

python3 "$ROOT_DIR/scripts/bench-rocm/bench_server.py" \
    --url "http://127.0.0.1:$PORT" \
    --contexts "$CONTEXTS" \
    --repetitions "$REPETITIONS" \
    --warmups "$WARMUPS" \
    --n-predict "$N_PREDICT" \
    --timeout "$TIMEOUT" \
    --metadata "$metadata" \
    "--server-arg=-c=$CTX" \
    "--server-arg=-b=$BATCH" \
    "--server-arg=-ub=$UBATCH" \
    "--server-arg=--cache-type-k=$CACHE_TYPE_K" \
    "--server-arg=--cache-type-v=$CACHE_TYPE_V" \
    "--server-arg=--cache-reuse=$CACHE_REUSE" \
    "--server-arg=--reasoning-budget=$REASONING_BUDGET" \
    "--server-arg=--mtp-n-max=$MTP_N_MAX" \
    --output "$ROOT_DIR/$OUTPUT"
