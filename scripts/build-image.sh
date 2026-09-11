#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

DOCKER=${DOCKER:-docker}
IMAGE=${IMAGE:-local/llama.cpp:mtp}
TARGET=${TARGET:-server}
ROCM_DOCKER_ARCH=${ROCM_DOCKER_ARCH:-gfx1101}
LLAMA_BUILD_TESTS=${LLAMA_BUILD_TESTS:-OFF}

cd "$ROOT_DIR"

commit=$(git rev-parse HEAD)
describe=$(git describe --always --dirty)

build_args=(
    --build-arg "ROCM_DOCKER_ARCH=$ROCM_DOCKER_ARCH"
    --build-arg "LLAMA_BUILD_TESTS=$LLAMA_BUILD_TESTS"
    --build-arg "APP_REVISION=$commit"
    --build-arg "APP_VERSION=$describe"
)

if [[ -n ${ROCM_VERSION:-} ]]; then
    build_args+=(--build-arg "ROCM_VERSION=$ROCM_VERSION")
fi

echo "Building $IMAGE"
echo "  target: $TARGET"
echo "  architecture: $ROCM_DOCKER_ARCH"
echo "  commit: $commit"

"$DOCKER" build \
    --target "$TARGET" \
    -f .devops/rocm.Dockerfile \
    "${build_args[@]}" \
    -t "$IMAGE" \
    .

echo "Built $IMAGE from $commit"
