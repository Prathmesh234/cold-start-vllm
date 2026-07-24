# vLLM cold-start experiments

This project will measure and improve vLLM startup latency one layer at a time.
The central rule is to preserve the same model, serving configuration, and
steady-state performance while changing only one startup mechanism per
experiment.

Research snapshot: **2026-07-24**.

## Iteration 0: basic GPT-OSS-20B server

The first executable baseline serves the pinned
[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) MXFP4 model
through vLLM's OpenAI-compatible API. The model has about 21B total parameters
and fits its MXFP4 weights within 16 GB of GPU memory. The baseline fixes tensor
parallelism to 1, maximum context to 8,192 tokens, and GPU-memory utilization to
90% so later serving files can change one startup mechanism at a time.

The project will grow as separate, directly runnable serving iterations:

| Iteration | File | Purpose |
| --- | --- | --- |
| 0 | `serve.py` | Basic vLLM server; derived compile cache disabled |
| 1 | `serve_01_compile_cache.py` | Persist and reuse vLLM compile artifacts |
| 2 | `serve_02_safetensors_prefetch.py` | Prefetch weight shards into the OS page cache |
| 3 | `serve_03_instanttensor.py` | Pipeline safetensors loading directly to CUDA |
| 4 | not implemented yet | Checkpoint/restore experiment |

Later optimization files are directly runnable and are not hidden behind one
configurable framework.

Install the platform-specific server dependency on the NVIDIA GPU host:

```shell
uv sync --extra server
```

Inspect the exact pinned command without launching it:

```shell
uv run python serve.py --dry-run
```

Start the OpenAI-compatible server:

```shell
uv run --extra server python serve.py
```

Additional vLLM flags can be passed after a standalone `--`:

```shell
uv run --extra server python serve.py -- --disable-log-stats
```

This file intentionally contains no readiness probe, benchmark framework,
specialized loader, sleep mode, CUDA checkpoint, or custom cache hierarchy.
`VLLM_DISABLE_COMPILE_CACHE=1` prevents reuse of vLLM's derived compile cache.
`VLLM_NO_USAGE_STATS=1` disables telemetry, avoiding writes to the user's vLLM
configuration directory on restricted hosts.
`VLLM_USE_FLASHINFER_SAMPLER=0` selects vLLM's native sampler because this host
does not expose a system CUDA toolkit (`nvcc`) required by FlashInfer's JIT
sampler. This does not change model loading or the cold-start cache experiment.
The normal Hugging Face model cache remains available because downloaded model
weights are inputs rather than derived startup artifacts.

## Iteration 1: persistent compile cache

Iteration 1 changes exactly one startup mechanism: it removes
`VLLM_DISABLE_COMPILE_CACHE=1` and points `VLLM_CACHE_ROOT` at the persistent,
project-local `.cache/vllm-compile` directory. Model, revision, engine flags,
telemetry setting, and native-sampler compatibility setting remain identical to
iteration 0.

The first run creates the compile artifacts and is the cold-cache measurement:

```shell
uv run --extra server python serve_01_compile_cache.py
```

Stop the server and run the same command again for the warm-cache measurement.
The second startup should report cache reuse and spend less time compiling. To
repeat the cold-cache case, remove `.cache/vllm-compile` before launching
iteration 1. Do not compare the first iteration-1 run against a later warm run;
record both separately.

## Iteration 2: safetensors prefetch

Iteration 2 is cumulative: it preserves iteration 1's compile cache and adds
only `--safetensors-load-strategy=prefetch`. vLLM reads the three checkpoint
files into the OS page cache in background worker threads while the default
loader starts consuming them. This targets storage page faults and does not
change the model representation or inference path.

```shell
uv run --extra server python serve_02_safetensors_prefetch.py
```

Compare its `Loading weights took` value with iteration 1 under the same page
cache state. Prefetching is most useful when model files are not already in the
Linux page cache or are on high-latency/network storage. It may provide little
benefit for an already-warm local cache. The sub-second target is only 0.563
seconds above this machine's 0.437-second physical H2D floor, so prefetching is
a low-risk measurement—not a guarantee. Reaching that target may require a
prepared RAM-resident layout and a loader that pipelines pinned-memory DMA,
similar to the later Doubleword design.

## Iteration 3: InstantTensor loader

Iteration 3 returns to iteration 1's warm compile-cache setup and replaces the
default safetensors loader with `--load-format=instanttensor`. InstantTensor is
a specialized CUDA loader with pipelined prefetching and direct I/O; unlike
iteration 2, it changes the weight-loading implementation rather than merely
warming file pages. It requires the `instanttensor` package included in the
`server` dependency group.

```shell
uv run --extra server python serve_03_instanttensor.py
```

Compare both loader metrics. `Loading weights took` measures the checkpoint
loader, while `Model loading took` also includes model construction and
post-load processing. The 0.437-second speed of light bounds only movement of
13.761 GB across the maximum PCIe link; it is not a lower bound for all work in
the latter metric.

## Model-weight speed of light

The pinned Hugging Face revision contains three indexed MXFP4 weight shards:

| Shard | Bytes |
| --- | ---: |
| `model-00000-of-00002.safetensors` | 4,792,272,488 |
| `model-00001-of-00002.safetensors` | 4,798,702,184 |
| `model-00002-of-00002.safetensors` | 4,170,342,232 |
| **Total** | **13,761,316,904** |

The total is 13.761 GB, or 12.818 GiB. `gpu_speed_of_light.py` detects the GPU
and its **maximum** PCIe capability with `nvidia-smi`; it does not use a generic
bandwidth default. The GPU currently in this workspace is an **NVIDIA RTX 6000
Ada Generation** with a maximum **PCIe Gen4 x16** link. Gen4 transfers at 16
GT/s/lane and uses 128b/130b encoding, so its theoretical one-way payload limit
is:

$$
R_{\text{PCIe}} = \frac{16 \times 16 \times (128/130)}{8}
= 31.508\ \text{GB/s}
$$

$$
T_{\text{H2D,min}} = \frac{13.761\ \text{GB}}{31.508\ \text{GB/s}}
= 0.437\ \text{s}
$$

Run the hardware-derived calculation without installing vLLM:

```shell
uv run python gpu_speed_of_light.py
uv run python gpu_speed_of_light.py --json
```

The 0.437-second result is an irreducible **host-RAM-to-VRAM model-weight
movement** floor, not a prediction of server readiness. A GPU may report a lower
currently negotiated generation while idle; the calculation correctly uses the
maximum physical link capability. It excludes storage/network reads, Python
imports, tokenizer work, deserialization, allocation, compilation, KV-cache
profiling, graph capture, protocol overhead, and HTTP startup. The gap between
measured readiness and this floor is the optimization budget.

## What the Doubleword posts establish

The first post reduces an SGLang startup for a 117 GB model from 695 seconds to
9.6 seconds:

| Path | Startup time |
| --- | ---: |
| Fully cold SGLang startup | 695.0 s |
| Warm filesystem and compiler caches | 88.0 s |
| CRIU + CUDA restore baseline | 32.1 s |
| Patched containerd + CRIU | 24.5 s |
| RAM-staged weight restore | 9.6 s |

The final 9.6-second path consists mainly of about 3.5 seconds in
`cuda-checkpoint` and 3.1 seconds moving weights from host RAM to the GPU at an
effective 38 GB/s. It is "cold(ish)," not a network- or disk-cold start: the
checkpoint already exists and the model weights have been staged in node RAM.

The important architectural choice is to avoid putting weights and KV cache in
the CRIU image. The full GPU image would be roughly GPU-memory-sized and would
be expensive to serialize. Instead, the application releases those large,
reconstructible allocations before checkpointing. CRIU preserves the smaller
initialized process and CUDA stub, and the application reloads weights after
restore.

The second post explains why a raw CUDA checkpoint is slower than PCIe
bandwidth suggests:

- `cuda-checkpoint` creates a large anonymous `MAP_POPULATE` staging mapping in
	the target process.
- On checkpoint, faulting and zeroing every 4 KiB host page can dominate the
	device-to-host copy.
- On restore, rebuilding the CUDA context and unmapping the staging pages add
	overhead around the host-to-device copy.
- Transparent Huge Pages reduce page-management overhead substantially.
- A fragile `LD_PRELOAD` experiment that preallocates the staging mapping and
	defers its unmap reduces an 8,578 MiB checkpoint/restore cycle from 4.5 to
	1.2 seconds on the author's test system.

The 4x staging-buffer result is an advanced optimization, not the first step:
it intercepts an undocumented allocation made by closed-source `libcuda` and
therefore has a much higher maintenance and correctness risk than enabling huge
pages.

## How this maps to vLLM

A current vLLM startup can be decomposed into six engine phases:

1. Framework bootstrapping
2. Tokenizer initialization
3. Model loading
4. `torch.compile`
5. KV-cache profiling
6. CUDA graph capture

Recent research finds that the first four phases are predominantly CPU-bound
under a warm page cache; only KV-cache profiling and CUDA graph capture are
primarily GPU-bound. Model loading becomes storage-sensitive when weights are
not already in the Linux page cache. This means a faster GPU alone generally
does not solve startup latency.

vLLM already exposes a useful intermediate mechanism with
`--enable-sleep-mode`:

- Level 1 offloads weights to CPU RAM and discards KV cache. It is fast to wake,
	but the process and enough host RAM remain allocated.
- Level 2 discards weights and KV cache. Waking requires allocation followed by
	an explicit weight reload. This is closer to the small-checkpoint strategy in
	the first Doubleword post.
- Online sleep endpoints require `VLLM_SERVER_DEV_MODE=1`; they must stay behind
	a trusted control plane and must not be exposed to model clients.

CUDA checkpoint/restore is not yet a released vLLM feature. As of this research
date, upstream RFC #34303 and implementation PRs #35934, #37921, and #37925 are
open and unmerged. Experiments must use a pinned branch or an external
orchestrator rather than assuming `vllm serve` contains `/suspend` and `/resume`.

## NVIDIA support matrix

| Driver | Relevant checkpoint support |
| --- | --- |
| 550+ | `cuda-checkpoint` utility for a single process |
| 570+ | Driver C API, lock timeout, NVML, CRIU 4.0 process-tree integration |
| 580+ | GPU UUID remapping and improved container passthrough |
| 595+ | ARM CPU support |
| 610+ | `cuIpcGetMemHandle`-based CUDA IPC support |

For a new multi-process vLLM experiment, driver 610+ is the safest target. Start
with tensor parallelism 1 before attempting TP > 1. The restore GPU must have
enough memory and, for migration, must be the same chip type as the original.

CUDA checkpoint currently has important constraints:

- It copies device memory into host RAM before GPU resources are released.
- UVM and exported shareable IPC allocations are unsupported; IPC support is
	driver-version-dependent.
- Checkpoint waits for submitted CUDA work to complete.
- Some checkpoint errors can leave the target process unusable.
- Full CRIU restore requires Linux, elevated host/container privileges, and
	careful handling of process sockets, shared memory, and device identity.

## Experimental ladder

Each stage must produce a result file containing the exact command, model and
revision, vLLM/container version, driver, CUDA version, GPU, CPU, storage,
cache state, and at least five timings.

### Stage 0: define comparable metrics

Record these separately:

- Process spawn to HTTP readiness
- Process spawn to first successful generated token
- Model loading time
- Compilation time
- CUDA graph capture time
- Restore or wake time
- First-request TTFT after readiness
- Steady-state throughput and latency after startup

Use at least three named cache states:

1. **Cold:** new vLLM/Inductor caches and model bytes read from storage
2. **Warm-cache restart:** persistent compiler caches and warm filesystem pages
3. **Restore:** a prepared sleep or checkpoint artifact

Do not call a page-cache or RAM-staged restore "cold". Do not use
`drop_caches` on a shared or production node.

### Stage 1: reproducible vLLM baseline

- Pin the vLLM version or container digest and model revision.
- Keep all engine flags fixed.
- Run vLLM's built-in `vllm.benchmarks.startup` benchmark for isolated cold and
	warm engine measurements.
- Add an end-to-end measurement from process/container launch through `/health`
	and a one-token generation, because the built-in benchmark does not include
	every orchestration and HTTP cost.
- Save raw logs; do not rely only on a single total duration.

### Stage 2: persist ordinary caches

- Pre-download weights to node-local NVMe and pin the Hugging Face revision.
- Persist the Hugging Face cache and `VLLM_CACHE_ROOT` across process and
	container restarts.
- Keep vLLM's compile cache enabled. Its default compile artifacts live under
	`$VLLM_CACHE_ROOT/torch_compile_cache`.
- Warm every production execution shape needed for compilation and graph
	capture before recording the warm baseline.
- Use `HF_HUB_OFFLINE=1` for checkpoint experiments after all model assets are
	local, preventing stale outbound connections from entering a CRIU image.

This stage is low risk and should be completed before checkpoint work.

### Stage 3: optimize weight loading

Benchmark the default safetensors loader against loaders supported by the
pinned vLLM release, such as `runai_streamer`, `runai_streamer_sharded`,
`instanttensor`, or `tensorizer`. Results depend heavily on storage topology,
sharding, quantization, and tensor parallelism; no loader should be selected
without an A/B measurement and output-correctness check.

### Stage 4: quantify startup/runtime tradeoffs

- Measure `--enforce-eager` as a diagnostic lower-startup baseline, not as an
	automatic production recommendation. It gives up CUDA graphs and can reduce
	steady-state performance.
- Benchmark the pinned release's CUDA graph modes and capture sizes.
- Reduce captured batch shapes only if the production workload permits it.
- Re-run throughput, TTFT, and latency tests after every startup optimization.

### Stage 5: vLLM sleep-mode baseline

Measure level 1 first, then level 2:

- Drain in-flight requests before sleep.
- Track GPU and host memory before sleep, while asleep, and after wake.
- Verify generation correctness after every wake.
- Test the exact quantization and model architecture. Upstream discussion notes
	that some quantized models have historically had different initial-load and
	reload paths, so level 2 cannot be assumed correct.
- Keep the development endpoints private.

Sleep mode is an in-process GPU release mechanism, not full scale-to-zero: the
server and its CPU-side state remain alive.

### Stage 6: in-process CUDA checkpoint prototype

Start with one GPU, one worker, and no CRIU:

1. Fully initialize and warm vLLM.
2. Stop admission and drain requests.
3. Synchronize CUDA work.
4. Lock and checkpoint the GPU process.
5. Confirm GPU resources were released.
6. Restore, unlock, and run correctness plus performance probes.

Measure host-memory growth and each state transition independently. Ensure free
host RAM is at least the live GPU footprint, with additional OS headroom. Then
compare normal 4 KiB pages with a controlled Transparent Huge Pages experiment.

Do not introduce the staging-buffer `LD_PRELOAD` interception until the normal
API path is correct and profiled; it is intentionally the last optimization in
this stage.

### Stage 7: CRIU process restore with a small GPU stub

The target design mirrors the first Doubleword post:

1. Warm the process, compiler artifacts, and CUDA graphs.
2. Release reconstructible weights and KV cache using a validated level-2-like
	 path.
3. CUDA-checkpoint the small remaining GPU state.
4. CRIU-checkpoint the process tree.
5. Restore the process and CUDA state.
6. Reload weights, allocate KV cache, and become ready.

Known vLLM/CRIU investigation points include:

- Quiesce TorchInductor workers with
	`TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=1` or constrain compilation with
	`TORCHINDUCTOR_COMPILE_THREADS=1` if idle CUDA-owning workers block a dump.
- Bind single-node distributed control traffic to stable interfaces such as
	loopback where appropriate.
- Test `tcp-established` versus `tcp-close`. The upstream RFC proposes the
	latter, while field reports found PyTorch TCPStore/NCCL needed preserved
	established connections. This must be settled by experiment for the exact
	vLLM topology.
- Orchestrate CRIU and CUDA checkpointing from a privileged host agent rather
	than granting the serving container broad host capabilities.
- Treat multi-GPU NCCL teardown/reinitialization and CUDA graphs containing
	NCCL operations as a separate milestone after TP=1 succeeds.

### Stage 8: RAM-staged weight reload

Only after Stage 7 identifies weight transfer as the dominant remainder:

- Keep a node-level cache of the serialized GPU allocation layout or optimized
	weight files in RAM.
- Transfer shared-memory file descriptors to the restored worker over a Unix
	socket.
- Back staging buffers with huge pages.
- Pipeline host registration and asynchronous H2D copies.
- Fall back safely to local NVMe when the RAM cache misses.

This stage trades substantial host RAM for restore latency and needs eviction,
admission-control, NUMA, and failure-recovery policies.

### Stage 9: infrastructure and driver surgery

Containerd checkpoint-image caching, CRIU zero-copy page restore, parallel CUDA
plugin execution, direct CUDA pipe protocols, and `LD_PRELOAD` staging-buffer
replacement are invasive optimizations. Attempt them only after profiling shows
that their specific overhead dominates and after a portable baseline exists.

## Decision gates

Advance only when all checks pass:

- At least five comparable runs and reported median/p95
- Same model outputs for deterministic probes
- No steady-state throughput or latency regression outside the agreed budget
- GPU and host memory return to expected levels after repeated cycles
- Failed checkpoint/restore has a tested process-restart fallback
- The result is labeled accurately as cold, warm-cache, sleep wake, or restore

## Sources

- [Cloudburst: 70x faster cold(ish) starts for SGLang](https://blog.doubleword.ai/fast-sglang-starts)
- [Reverse-engineering NVIDIA's cuda-checkpoint for faster cold starts](https://blog.doubleword.ai/what-happens-when-you-checkpoint-a-cuda-process)
- [vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
- [vLLM startup benchmark](https://docs.vllm.ai/en/latest/api/vllm/benchmarks/startup/)
- [vLLM CUDA checkpoint RFC #34303](https://github.com/vllm-project/vllm/issues/34303)
- [NVIDIA CUDA checkpoint utility](https://github.com/NVIDIA/cuda-checkpoint)
- [NVIDIA CUDA Driver checkpoint API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html)
- [Breaking the Ice: Analyzing Cold Start Latency in vLLM](https://arxiv.org/abs/2606.07362)
