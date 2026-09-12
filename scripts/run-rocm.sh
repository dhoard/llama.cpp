#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

DOCKER=${DOCKER:-docker}
IMAGE=${IMAGE:-local/llama.cpp:mtp}
PORT=${PORT:-8000}
HF_CACHE=${HF_CACHE:-${HOME}/.cache/huggingface}
MODEL=${MODEL:-ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M}
CTX=${CTX:-262144}
PARALLEL=${PARALLEL:-1}
BATCH=${BATCH:-512}
UBATCH=${UBATCH:-512}
THREADS=${THREADS:-8}
THREADS_BATCH=${THREADS_BATCH:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
N_GPU_LAYERS=${N_GPU_LAYERS:-all}
FIT=${FIT:-off}
FIT_TARGET=${FIT_TARGET:-1024}
FIT_CTX=${FIT_CTX:-4096}
LOAD_MODE=${LOAD_MODE:-}
CACHE_TYPE_K=${CACHE_TYPE_K:-q4_0}
CACHE_TYPE_V=${CACHE_TYPE_V:-q4_0}
CACHE_REUSE=${CACHE_REUSE:-1024}
REASONING_BUDGET=${REASONING_BUDGET:-8192}
MTP_N_MAX=${MTP_N_MAX:-3}
GDN_OPT=${GDN_OPT:-1}
SPARSE_ATTN_MODE=${SPARSE_ATTN_MODE:-off}
OPTIMIZED=${OPTIMIZED:-1}

if [[ "$OPTIMIZED" == 0 ]]; then
    GDN_OPT=0
    MTP_N_MAX=0
    SPARSE_ATTN_MODE=off
fi

if [[ "$SPARSE_ATTN_MODE" != off ]]; then
    echo "error: sparse attention is not implemented for Qwen3.5 in this checkout" >&2
    echo "       use SPARSE_ATTN_MODE=off" >&2
    exit 2
fi

if ! [[ "$MTP_N_MAX" =~ ^[0-9]+$ ]]; then
    echo "error: MTP_N_MAX must be a non-negative integer" >&2
    exit 2
fi

mkdir -p "$HF_CACHE"

docker_args=(
    run --rm --pull=never
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    --group-add render
    --ipc=host
    --env "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
    -p "$PORT:$PORT"
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

server_args=(
    -hf "$MODEL"
    --no-mmproj
    --host 0.0.0.0
    --port "$PORT"
    -c "$CTX"
    -np "$PARALLEL"
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

if [[ "$FIT" == on ]]; then
    server_args+=(--fit on --fit-target "$FIT_TARGET" --fit-ctx "$FIT_CTX")
else
    server_args+=(-ngl "$N_GPU_LAYERS")
fi
if [[ -n "$LOAD_MODE" ]]; then
    server_args+=(--load-mode "$LOAD_MODE")
fi

if (( MTP_N_MAX > 0 )); then
    server_args+=(--spec-type draft-mtp --spec-draft-n-max "$MTP_N_MAX")
fi

cd "$ROOT_DIR"
exec "$DOCKER" "${docker_args[@]}" "$IMAGE" "${server_args[@]}" "$@"
