# ROCm MTP optimization

This checkout packages the existing llama.cpp Qwen3.5 execution path for
Docker-only ROCm use. The work reuses the current HIP Gated DeltaNet cache
fusion and quantized FlashAttention implementation. It adds reproducible
image, test, run, and benchmark entry points around that path.

The implementation was audited from source revision `41fc7584`. The initial
machine was an RX 7700 XT (`gfx1101`) with ROCm 7.2.1 supplied by the
repository Dockerfile. The tested model was a Qwen3.5-based 9B GGUF in
Q4_K_M.

## Build

The host only needs Docker, access to `/dev/kfd` and `/dev/dri`, and the
model cache. ROCm compilation and tests run inside the image.

```sh
ROCM_DOCKER_ARCH=gfx1101 ./scripts/build-image.sh
```

The default target is `server`. Set `TARGET=dev` and
`LLAMA_BUILD_TESTS=ON` when building the development image:

```sh
TARGET=dev LLAMA_BUILD_TESTS=ON ./scripts/build-image.sh
```

The script records the source revision in the image labels and uses the
checked-in `.devops/rocm.Dockerfile`. Set `ROCM_VERSION` to override the
Dockerfile ROCm version when needed.

## Run

The normal launch uses the same settings used for the baseline comparison:

```sh
./scripts/run-rocm.sh
```

The wrapper uses the local image
`local/llama.cpp:mtp`, mounts
`${HOME}/.cache/huggingface`, enables all model layers on the GPU, enables
FlashAttention, uses Q4 K/V cache, and starts with a 262144 token context.

Useful overrides are:

```sh
PORT=8001 CTX=32768 ./scripts/run-rocm.sh
MTP_N_MAX=0 ./scripts/run-rocm.sh
GDN_OPT=0 ./scripts/run-rocm.sh
```

`GDN_OPT=0` passes `GGML_HIP_GDN_OPT=0` into the container and disables the GDN cache fusion A/B path. MTP defaults to three draft tokens after the rollback fix described below; `MTP_N_MAX=0` disables it. The default batch and ubatch sizes are both 512. `SPARSE_ATTN_MODE` accepts only `off`; other values fail closed because this checkout has no validated Qwen3.5 sparse attention path.

Set `OPTIMIZED=0` to force the safe baseline wrapper settings. This
also disables GDN fusion, MTP, and sparse mode.

## Tests

Build and run the generic tests and a model smoke test in the ROCm development
container:

```sh
./scripts/test-rocm.sh
```

The script runs the existing `test-backend-ops` and `test-llama-archs` binaries when they are present, then starts the cached GGUF with a small context and checks health plus a generated completion. It also runs `test-recurrent-state-rollback` against the cached model by default; `RUN_RECURRENT_ROLLBACK=0` skips that check. The rollback test now passes on CPU and ROCm. No new test source files were added.

## MTP rollback fix and validation

Short decodes previously overwrote old convolution snapshots and left old GDN snapshots in the wrong slots. DeltaNet state loading now gathers the required history before cache writes, advances that history by the number of decoded tokens, and preserves snapshots for idle sequences when cache cells move. Checkpoint restore records the oldest available position so rollback cannot read history that was not restored.

The existing rollback test failed on the test model with a maximum logit difference of 2.09621. After the fix, split replay, idle-sequence independence, and rollback after cache-cell movement all match the reference exactly, with both zero-filled and deliberately nonzero cache buffers. These checks passed on CPU and ROCm, including Q4 K/V caches on ROCm.

With the RX 7700 XT, a 262144-token context allocation, Q4 K/V caches, batch/ubatch 512, and eight threads, three short prompts generating 128 tokens measured 82.6, 87.3, and 98.7 tokens/s with MTP. The baseline measured approximately 56.7 tokens/s. These are short-prompt measurements, not throughput at a filled 262K context. A repeated prompt after unrelated requests produced the same tokens and draft acceptance counts.

MTP output is still not guaranteed to match the non-MTP greedy token sequence. Identical continuations evaluated individually and in batches differ numerically on this ROCm configuration, even without drafting or rollback. Two of the three test completions differed from the baseline. The rollback fix addresses invalid state history; it does not make the GPU kernels invariant to batch size.

`~/ai/run-ai-4.sh` enables MTP by default. Run `SPEC_TYPE=none ~/ai/run-ai-4.sh` for its baseline mode, or use `MTP_N_MAX` to change the draft length.

## Benchmark method

The benchmark wrapper starts `llama-server` in a local ROCm container and the
Python helper sends HTTP requests from the host. It records the image ID,
source revision, server arguments, prompt token count, raw repetitions, and
medians:

```sh
./scripts/bench-rocm/bench-rocm.sh
```

A shorter validation run is:

```sh
CONTEXTS=8192,32768 REPETITIONS=1 WARMUPS=0 N_PREDICT=32 \
    ./scripts/bench-rocm/bench-rocm.sh
```

Results are written to
`benchmarks/rocm/results.json`, which is ignored by Git. The default
sweep uses 8K, 32K, 64K, 128K, 192K, and 256K requested context budgets, with
256 generated tokens reserved in each request. Each prompt is generated and
tokenized through the running server, then completed with
`temperature=0`, `cache_prompt=false`, and a fixed seed. This avoids
measuring prompt construction and avoids using a prior request's prompt cache
when comparing context sizes.

## Baseline observations

The baseline server was built before source changes with:

```text
-c 262144 -np 1 -ngl all -fa on -b 2048 -ub 512
-t 8 --threads-batch 8 --cache-type-k q4_0 --cache-type-v q4_0
--cache-reuse 1024 --reasoning-budget 8192
```

On the RX 7700 XT:

- VRAM use after model load was about 10.39 GiB of 12.87 GiB.
- A short deterministic completion measured 54.98 generated tokens per second.
- The server reported that `cache_reuse` is unsupported by this context and
  disabled it. The wrapper preserves the requested option so the server's
  behavior stays visible in logs.
- The short smoke completion measured 52.35 generated tokens per second.

The initial GGUF test exposed MTP tensors and the server loaded the draft head with
`--spec-type draft-mtp --spec-draft-n-max 3`. It measured 76.11 generated
tokens per second in the short run and accepted 19 of 33 draft tokens.
However, the same fixed-seed greedy prompt produced different committed text
from the non-MTP run. MTP was initially left opt-in pending the rollback investigation above.

The comparable three-repetition short sweep measured the following decode
throughput on the RX 7700 XT:

| Requested context | Baseline median TG/s | Local image median TG/s | Repetitions |
| --- | ---: | ---: | ---: |
| 8K | 48.78 | 48.67 | 3 |
| 32K | 36.64 | 36.69 | 3 |
| 64K | not run | 26.63 | 3 |
| 128K | not run | 17.41 | 3 |
| 192K | not run | 13.06 | 1 |
| 256K | not run | not completed | 0 |

The 64K and 128K local-image points came from the longer sweep. The 192K
point completed one measured repetition before the run was stopped because
each long-context repetition took several minutes. The short baseline and
local-image results are within measurement noise, so this checkout does not
claim a throughput gain from the existing GDN fusion path.

Direct `llama-bench` tuning for the RX 7700 XT used the development image
`local/llama.cpp:rocm-bench`, `ROCm0`, Q4 K/V cache, FlashAttention,
8 CPU threads, and two repetitions per setting. At an 8K prompt, batch `512`
with ubatch `512` measured 1647 prompt tokens/s and 56.4 generated tokens/s;
batch `2048` measured 1560 prompt tokens/s and 56.5 generated tokens/s. At a
32K prompt, batch `512` measured 1288 prompt tokens/s versus 1231 for batch
`2048`, with generation unchanged at about 56.5 tokens/s. The launcher in
`~/ai/run-ai-4.sh` therefore uses `-b 512 -ub 512`.

## Optimization decisions

The repository already contains the relevant execution paths:

- HIP uses the existing fused Gated DeltaNet operator and recurrent-cache
  snapshot fusion.
- Q4 K/V cache formats use the existing quantized FlashAttention support.
- Qwen3.5 model construction already supports the hybrid GDN and full
  attention layout.
- Current llama.cpp MTP support already handles the available draft tensors.

The new GDN switch is an A/B control around the existing cache fusion. It does
not change model weights or recurrent-state layout.

Sparse historical attention was not retained. The existing CUDA/HIP sparse
gather support is tied to other model-specific paths and does not provide a
validated Qwen3.5 block index or quality gate. The launch and benchmark
wrappers reject sparse modes instead of silently changing attention semantics.

TurboQuant was not retained. No new cache representation was added without a
Docker benchmark and a quality comparison against the existing Q4 K/V
FlashAttention path. GDN recurrent state remains in its existing format.

The benchmark JSON is the performance record for the exact image and revision.
Long-context results should be collected on the target GPU before changing
the default batch, context, cache, or MTP settings.

## FlashAttention dispatch experiment

The HIP FlashAttention selector keeps the existing default behavior for quantized
K/V caches. Set `GGML_HIP_FA_DEBUG=1` to log the first 32 HIP FlashAttention
dispatches, including query width, cache types, selected kernel, and F16
conversion requirements.

Set `GGML_HIP_FA_Q4_MTP_VEC=1` to enable the experimental Q4_0/Q4_0 vector path
for up to four query columns. This reuses the existing two-column vector kernel
and is disabled by default. Use both variables for an A/B run:

```sh
BATCH=512 GGML_HIP_FA_DEBUG=1 GGML_HIP_FA_Q4_MTP_VEC=1 ./scripts/bench-rocm/bench-rocm.sh
```

The switch is HIP-only and applies only when both cache tensors are Q4_0. It
does not add a four-column kernel or change CUDA dispatch. Compare its output,
acceptance statistics, and sustained decode throughput with the variables unset
before considering a default change.

## MTP draft length sweep

An exploratory sweep used the Ornith model from the Hugging Face cache with
batch 512, Q4_0 K/V, and one repetition at each context. The decode results
were:

| `MTP_N_MAX` | 8K tokens/s | 32K tokens/s |
| ---: | ---: | ---: |
| 1 | 62.6 | 46.6 |
| 2 | 75.0 | 73.9 |
| 3 | 77.6 | 85.4 |
| 4 | 68.0 | 82.4 |
| 5 | 59.3 | 82.6 |

This points to `MTP_N_MAX=3` for the next repeated run. It is not a default
change until acceptance and draft overhead statistics are collected.

The focused two-repetition MTP=3 run recorded aggregate acceptance through the
benchmark helper: 40 of 66 draft tokens accepted at 8K and 46 of 51 accepted
at 32K. The corresponding median decode rates were 76.9 and 85.0 tokens/s.
The helper now preserves `draft_n` and `draft_n_accepted` in each raw run.

## Chunked GDN prefill investigation

The current HIP Gated DeltaNet kernel keeps each recurrent state shard in
registers while processing tokens in order. This preserves the state dependency
without intermediate global-memory writes. Splitting the loop into chunks would
either add kernel launches and state traffic or require a separate parallel scan
kernel. Both choices increase complexity and risk changing rollback snapshots.

The 64K prompt measurement completed at 907.8 prompt tokens/s. The 128K
measurement was stopped after several minutes without completing. The current
evidence does not justify adding a second HIP GDN implementation, so no
chunked kernel was added.
