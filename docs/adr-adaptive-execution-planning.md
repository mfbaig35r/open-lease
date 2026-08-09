# ADR: adaptive model execution planning

**Status:** Rejected as proposed. One narrow capability accepted in its place.
**Date:** 2026-08-09
**Decision:** Do not build a Model Execution Planner. Ship `llama.cpp` as runtime #2 with a
documented `cpu_moe_offload` option, and change cost reporting from $/hour to $/token *and* $/hour.

---

## Context

A design brief proposed evolving open-lease from "provision a GPU and serve a model" into a system
that plans model execution across the whole memory hierarchy (GPU HBM, CPU RAM, local NVMe, remote
storage), generates candidate execution plans, benchmarks them on real hardware, and selects the
cheapest one meeting an SLO. It proposed seven new abstractions (`ModelProfile`, `HardwareProfile`,
`RuntimeCapabilities`, `ModelExecutionPlan`, `WorkloadProfile`, `ExecutionPlanner`,
`BenchmarkProfile`) and a benchmark knowledge base.

The central hypothesis:

> GPU VRAM should be treated as one tier in a model-serving memory hierarchy rather than as a binary
> model-fit constraint.

The immediate inspiration was a demonstration of AirLLM running Kimi K3 (2.8T parameters, 93 layers,
896 experts per layer, MXFP4) on a single RTX 6000 Ada with a 3.72 GB peak working set.

The hypothesis was tested in two gates before any code was written.

---

## Gate 0: desk analysis

### There are two tiering architectures, not one

**Stream weights to the GPU** (AirLLM). Every non-resident byte crosses PCIe per token:

| Model | GB/token | PCIe4 x16 (25 GB/s) | PCIe5 x16 (50 GB/s) |
| --------------------------------- | -------: | ------------------: | ------------------: |
| DeepSeek-R1 671B Q4, 37B active   |     20.4 |        1.2 tok/s    |        2.5 tok/s    |
| Kimi K2 1T Q4, 32B active         |     17.6 |        1.4 tok/s    |        2.8 tok/s    |
| Qwen3-235B-A22B Q4                |     12.1 |        2.1 tok/s    |        4.1 tok/s    |

These are ceilings before overhead, and they land at or below the brief's own 2 tok/s target. This
architecture is not viable.

**Compute experts where they live** (ktransformers, `llama.cpp --n-cpu-moe`). Attention and KV cache
stay on the GPU, routed experts stay in system RAM, and the expert matmul runs on the CPU. Only
activation vectors cross PCIe, so the ceiling becomes host RAM bandwidth. Published measurement:
~14 tok/s decode for DeepSeek-R1 671B on one RTX 4090 plus dual Xeon with AMX and 512 GB DDR5.

The brief's memory-tier abstraction models transfer paths between tiers. What actually determines
viability is *where the compute happens*. That is a smaller and different question.

### The inspiring demonstration does not support the claim

The AirLLM K3 result is 5 minutes per token, which is 0.0033 tok/s. The writeup reports it as
"roughly 0.2 tokens/second", off by a factor of 60. Even 0.2 tok/s is 10x below the brief's target.

More importantly it is nowhere near a hardware limit. At ~25-48 GB read per token, 5 min/token is a
sustained ~0.16 GB/s, roughly 40x slower than a single consumer NVMe drive. The independent Rust
reimplementation reaches ~0.7 GB/s, still ~8x under one drive. The demonstration is a correctness
proof that a checkpoint can be paged through a small buffer. It is not a performance result, and it
says nothing about memory hierarchies because it never reached the speed of its slowest tier.

### Batching destroys expert locality

This is the load-bearing finding against workload-specific expert placement.

- **Lynx** (Mixtral 8x7B, top-2 of 8): batch 1 touches ~2 experts per layer, batch 8 touches 7-8,
  and by batch 16 every expert is activated. Decode latency scales linearly with the number of
  *distinct* experts touched. The union across the batch, not the token count, governs cost.
- **MoE-Infinity** finds strong per-request skew (under 5% of experts repeatedly activated within
  one sequence) and then states that "after processing multiple requests, the skew disappears, and
  all experts tend to be activated uniformly." The system is explicitly designed for batch size one.

Locality is real and exists only at batch 1. Batching is where cheap throughput comes from.
Therefore expert caching and cheap serving are mutually exclusive, and the two proposed strategies
"expert cache" and "weight streaming" are one strategy at different hit rates.

### Gate 0 pass mark

Using real RunPod API prices (not the marketing page), for DeepSeek-R1 671B Q4 the cheapest resident
configuration is 6x A100 PCIe at $7.14/hr and the cheapest tiered configuration is 5x RTX 4090 at
$1.70/hr. Tiering is 4.2x cheaper per hour, so:

> **To win on cost per token, tiering must hold at least 24% of resident throughput.**

---

## Gate 1: measurement

Rented one RunPod L40S pod (secure, $0.99/hr) and measured directly. Total spend: **$0.36**.

Proxy model: Qwen3-30B-A3B-Instruct Q4_K_M (17.3 GiB) rather than DeepSeek-R1 Q4 (404 GB). The
quantities that needed measuring are host RAM bandwidth and llama.cpp's efficiency against it, both
of which extrapolate by expert-bytes-per-token. This cut the experiment from ~$25 and six hours to
$0.36 and one hour.

### What a rented pod actually provides

`free` and `nproc` inside the container report the whole host (1511 GB, 128 vCPU) and are
misleading. The enforced cgroup limits are what matter:

| | Value |
| ------------------------------- | ------------------------------------- |
| `memory.limit_in_bytes`         | **175 GiB** (API reports 188 GB)      |
| `cpu.cfs_quota_us`              | 1360000 => **13.6 cores**             |
| Host CPU                        | 2x AMD EPYC 9354 (32c each), 2 NUMA   |
| STREAM triad achievable         | **~290 GB/s** (270-311, 5 runs)       |
| Single-thread triad             | 28.7 GB/s                             |

RAM bandwidth was the unknown Gate 0 said would decide this. It came back high, roughly an 8-channel
DDR5 server and 11x PCIe 4.0 x16. **Bandwidth is not the constraint.**

### The constraint is CPU cores

Decode throughput with all 48 MoE layers' experts on CPU, sweeping thread count:

| threads |    4 |      8 |   13 |   16 |   26 |   32 |
| ------- | ---: | -----: | ---: | ---: | ---: | ---: |
| tok/s   | 25.8 | **37.6** | 33.3 | 28.7 | 17.7 | 11.0 |

Throughput peaks at 8 threads and collapses past the quota under CFS throttling. The CPU expert path
achieves only **17-37% of the 290 GB/s available**. The bandwidth is present and cannot be used,
because the cores needed to drive it are metered.

**Implication for open-lease:** the binding resource for tiered execution is vCPU per GB of
offloaded weight, not the RAM:VRAM ratio. `ProviderCapabilities` models neither.

### The decisive result

`llama-batched-bench`, 256-token prompt, 128-token generation, decode throughput in tok/s:

| concurrency | resident | tiered | tiered as % of resident |
| ----------: | -------: | -----: | ----------------------: |
|           1 |    211.4 |   32.4 |               **15.3%** |
|           2 |    315.8 |   38.8 |                   12.3% |
|           4 |    454.3 |   53.3 |                   11.7% |
|           8 |    686.2 |   63.8 |                    9.3% |
|          16 |    851.7 |   72.2 |                **8.5%** |

Resident scales 4.0x from batch 1 to 16. Tiered scales 2.2x. The Lynx and MoE-Infinity result
reproduces on rented hardware: the gap widens exactly as batching pulls in more distinct experts.

**Pass mark was 24%. Measured 15.3% at batch 1, falling to 8.5% at batch 16. Fails at every
concurrency level.**

### Extrapolated to DeepSeek-R1 671B Q4

Scaling by expert bytes per token (~1.5 GB for Qwen3-30B-A3B, ~20 GB for DeepSeek-R1 Q4):

| | resident (6x A100 PCIe, $7.14/hr) | tiered (5x RTX 4090, $1.70/hr) |
| ------------------------ | ------: | ---------: |
| decode @ batch 1         | 33 tok/s | **~2.4 tok/s** |
| $/M output tokens @ batch 1 | $60.10 | $194.33 (**3.2x worse**) |
| $/M output tokens @ conc 16 | $3.97 | $87.21 (**22x worse**) |
| $/hour                   | $7.14 | $1.70 (**4.2x cheaper**) |

---

## Findings

1. **Tiering is never the cheapest way to produce tokens.** It is 3.2x to 22x more expensive per
   token than resident serving, and the penalty grows with concurrency.
2. **Tiering is sometimes the cheapest way to have an endpoint exist**, by 4.2x per hour. That is a
   different product question, and it is the one the brief's own 2 tok/s example was really asking.
3. **DeepSeek-R1 671B at ~2.4 tok/s for $1.70/hr is real and buildable today** with one llama.cpp
   flag. It needs no planner, no execution-plan abstraction, and no benchmark knowledge base.
4. **The search space collapsed to a single interesting configuration.** A planner that searches by
   provisioning and benchmarking candidates would spend more on the search than the search saves,
   and would arrive at a config a human picks in thirty seconds.
5. **Kimi K3 (1.56 TB) has no resident option on RunPod within 8 GPUs.** The only configuration that
   holds it is 8x H100 SXM at $21.52/hr with 2008 GB host RAM, necessarily tiered, and it is also
   the slowest case. Tiering there is not an optimization, it is the only option.

### Corrections to earlier analysis in this investigation

- An earlier pass used RunPod's marketing page for host RAM. The API disagrees: RAM:VRAM ratios are
  commonly 3-4:1, not 1.5-2:1. Tiering saves 4.2x on DeepSeek-R1, not the 8.1x first calculated, and
  the framing "buying enough RAM means buying enough VRAM" was too strong.
- Gate 0 predicted RAM bandwidth would be the deciding unknown. It was not. CPU core quota was.

---

## Decision

**Rejected:** `ModelExecutionPlan`, `ExecutionPlanner`, `HardwareProfile`, `WorkloadProfile`,
`BenchmarkProfile`, the benchmark knowledge base, memory-tier abstraction, and expert-placement
policy. The hypothesis that motivated them does not survive measurement, and the reconcile loop's
converge-to-desired-state design is the wrong shape for a provision-benchmark-reject search anyway
(`next_step()` is pure by constraint; a search loop that provisions candidates is a cost-safety
hazard that belongs nowhere near it).

**Accepted in its place**, each justified on its own merits rather than as a step toward the vision:

| Item | Why | Size |
| ---- | --- | ---- |
| `llama.cpp` as runtime #2 with a `cpu_moe_offload` profile option | Delivers the one viable configuration; validates the `Runtime` ABC, which has only ever had one implementation | ~1 day |
| `ModelProfile` derived from HF config and safetensors metadata | `deploy_adhoc()` currently demands an explicit `--gpu` because nothing can infer requirements | ~2 days |
| Report **both** $/token and $/hour in `estimate_cost()` | Today it is `gpu.hourly_usd * hours`, catalog-only, and does not work for ad-hoc deploys at all. The correct configuration flips between the two metrics based on duty cycle, which an hourly model cannot express | ~2 days |
| Add host RAM and vCPU to `ProviderCapabilities` | RunPod publishes both; open-lease models neither. An L40S pod has 5x the RAM of an A5000 pod at similar price with no way to represent the difference | ~half day |

The `cpu_moe_offload` option must be documented as: **single stream, low duty cycle, expect
single-digit tokens per second.** It is not a general serving mode.

---

## What would reverse this decision

- A runtime whose CPU expert path saturates available RAM bandwidth rather than 17-37% of it. The
  bandwidth headroom measured here is real; only the cores to drive it are missing.
- A provider that sells vCPU and RAM decoupled from GPU count, changing the vCPU-per-offloaded-GB
  ratio that currently binds.
- Sustained demand for always-on frontier-MoE endpoints where hourly cost dominates and throughput
  is irrelevant, in enough volume to justify more than one documented flag.

---

## Reproducing

```
# host limits (inside the pod, not `free`/`nproc`)
cat /sys/fs/cgroup/memory/memory.limit_in_bytes    # 187999997952 = 175 GiB
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us            # 1360000 = 13.6 cores

# resident vs tiered decode
llama-bench -m Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf -ngl 99 -ncmoe 0  -t 13 -p 512 -n 128 -r 3
llama-bench -m Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf -ngl 99 -ncmoe 48 -t 13 -p 512 -n 128 -r 3

# concurrency sweep
llama-batched-bench -m <model> -ngl 99 -ncmoe {0,48} -t 8 -c 16384 -b 2048 -ub 512 \
    -npp 256 -ntg 128 -npl 1,2,4,8,16
```

Raw `llama-bench` sweep over how many layers are offloaded (`-ncmoe`, tg128 tok/s, 13 threads):

```
ncmoe= 0  220.68 +/- 2.56     ncmoe=24   78.53 +/- 9.46
ncmoe=12  127.11 +/- 5.48     ncmoe=36   60.56 +/- 1.44
                              ncmoe=48   48.46 +/- 4.24
```

---

## Sources

- KTransformers, SOSP'25: <https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf>
- Lynx, dynamic batch-aware expert selection: <https://arxiv.org/html/2411.08982v1>
- MoE-Infinity: <https://arxiv.org/html/2401.14361v3>
- llama.cpp MoE offload guide: <https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide>
- DeepSeek-V3/R1 on 8xH100 throughput: <https://github.com/dzhsurf/deepseek-v3-r1-deploy-and-benchmarks>
- RunPod GPU specs and pricing: `gpuTypes { memoryInGb lowestPrice { minMemory minVcpu ... } }`
