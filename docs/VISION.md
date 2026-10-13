# Elysium: the target architecture

> **This is the destination, not what runs today.** It describes the heterogeneous fabric of the roadmap's later
> phases: GPUs, TPUs, Trainium and other accelerators in one run, behind one PyTorch-facing API. Today Elysium trains one
> model across **homogeneous GPUs in different sites** over an ordinary WAN, byte-exact; the `els.fit` API below does not
> exist yet. See the [README](../README.md#current-capabilities) for what is built.

## Why Elysium?

AI infrastructure is becoming increasingly heterogeneous.

A single organization may have:

* NVIDIA GPUs
* Google TPUs
* AWS Trainium
* AMD accelerators
* custom AI chips
* on-premise servers
* cloud infrastructure
* geographically distributed machines

Today, these resources are generally treated as separate environments.

Elysium takes a different approach:

**The model should not care what chip is underneath.**

**The infrastructure should determine how to execute it.**

## The Core Idea

Elysium uses heterogeneous pipeline parallelism to execute a single model across different accelerator architectures.

For example:

```text
                    Transformer Model

Input
  │
  ▼
┌──────────────────┐
│ GPU Stage        │
│ Layers 0 – 15    │
└────────┬─────────┘
         │
         │ activation
         ▼
┌──────────────────┐
│ TPU Stage        │
│ Layers 16 – 31   │
└────────┬─────────┘
         │
         │ activation
         ▼
┌──────────────────┐
│ Trainium Stage   │
│ Layers 32 – 47   │
└────────┬─────────┘
         │
         ▼
       Loss
```

Each hardware family remains internally optimized for its own architecture.

Elysium connects the stages.

This allows:

```text
GPU ↔ GPU
    native collective communication

TPU ↔ TPU
    native TPU communication

Trainium ↔ Trainium
    native accelerator communication

GPU ↔ TPU ↔ Trainium
    Elysium transport
```

The result is a hierarchy:

**native parallelism inside a hardware domain + heterogeneous pipeline parallelism between domains.**

## Developer Experience

Developers should not need to learn three different distributed runtimes.

Elysium aims to preserve a familiar PyTorch programming model.

```python
import torch
import torch.nn as nn
import elysium as els


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([...])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


model = Model()

els.fit(
    model=model,
    hardware="auto",
    pipeline=True,
)
```

The developer writes the model.

Elysium determines:

* where the model should run
* how the model should be partitioned
* which accelerator should execute each stage
* how many microbatches to use
* how stages should communicate
* how resources should be replicated
* how failures should be handled

The goal is simple:

```text
           Developer
               │
               │ PyTorch
               ▼
            Elysium
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
      GPU     TPU    Trainium
```

## Architecture

### 1. Elysium Frontend

The frontend accepts a standard PyTorch model and training configuration.

```text
PyTorch
   │
   ▼
Model capture
   │
   ▼
Graph representation
```

The frontend should eventually support model capture through technologies such as `torch.export` and FX-style graph representations.

### 2. Graph Partitioner

The partitioner converts the model into executable stages.

```text
Model Graph

Embedding
    │
Block 0
    │
Block 1
    │
 ...
    │
Block 15
    │
──── boundary ────
    │
Block 16
    │
 ...
    │
Block 31
    │
──── boundary ────
    │
Block 32
    │
 ...
    │
Block 47
```

Elysium evaluates candidate partitions using:

```text
compute cost
memory capacity
communication bandwidth
communication latency
device utilization
availability
cost
fault risk
```

The goal is to minimize end-to-end training time and infrastructure cost.

Conceptually:

$$
\min_{\text{partition, placement, schedule}} T_{\text{training}}
$$

subject to:

$$
M_i \leq M_i^{\text{capacity}}
$$

and:

$$
T_i^{\text{compute}} + T_i^{\text{communication}}
$$

being approximately balanced across pipeline stages.

### 3. Native Backend Layer

Elysium should not replace accelerator-specific software stacks.

Instead, it orchestrates them.

```text
Elysium Stage
      │
      ├── GPU
      │    └── CUDA / PyTorch
      │
      ├── TPU
      │    └── PyTorch + TPU/XLA stack
      │
      └── Trainium
           └── PyTorch + Neuron
```

This means Elysium can evolve as accelerator ecosystems evolve.

The Elysium API stays stable.

The backend implementation can change.

### 4. Tensor Contract

Cross-device communication uses a common tensor contract.

A transmitted object contains information such as:

```text
Tensor
├── shape
├── dtype
├── layout
├── device
├── microbatch_id
├── sequence metadata
├── gradient metadata
└── integrity metadata
```

For example:

```text
TensorContract
-------------------------
shape:        [1, 4096, 8192]
dtype:        BF16
layout:       contiguous
microbatch:   42
requires_grad: true
```

The tensor contract is independent of the accelerator that produced it.

### 5. Elysium Transport

The transport layer moves activations and gradients between stages.

```text
GPU
 │
 │ activation
 ▼
Elysium Transport
 │
 ▼
TPU
 │
 │ activation
 ▼
Elysium Transport
 │
 ▼
Trainium
```

The transport layer can support different networking technologies depending on the environment:

```text
TCP
RDMA
RoCE
EFA
high-speed datacenter fabrics
WAN links
```

The scheduler incorporates communication cost into placement decisions.

### 6. Pipeline Engine

Elysium uses microbatching to keep different stages busy simultaneously.

```text
time →

GPU       MB1  MB2  MB3  MB4  MB5
TPU            MB1  MB2  MB3  MB4  MB5
Trainium            MB1  MB2  MB3  MB4  MB5
```

For training, the runtime uses one family of schedules, [zero-bubble pipeline parallelism](engine/ARCHITECTURE.md#one-execution-model-zero-bubble-pipeline-parallelism).

### 7. Hierarchical Parallelism

Elysium is not limited to one accelerator per stage.

A stage can itself contain a parallel group.

```text
                     Elysium
                        │
       ┌────────────────┼────────────────┐
       │                │                │
   GPU Stage         TPU Stage       Trainium Stage
       │                │                │
   ┌───┼───┐         ┌──┼──┐       ┌────┼────┐
   GPU GPU GPU        TPU TPU       TN   TN   TN
```

This creates multiple levels of parallelism:

```text
Model Parallelism
       +
Pipeline Parallelism
       +
Data Parallelism
       +
Hardware-native Parallelism
```

### 8. Dynamic Scheduling

Hardware is not homogeneous.

A GPU may be faster for one workload.

A TPU may be more efficient for another.

A Trainium instance may have better economics for a different stage.

Elysium continuously profiles the system and estimates:

```text
compute
memory
communication
cost
availability
```

Then the scheduler can choose:

```text
GPU:
layers 0–18

TPU:
layers 19–35

Trainium:
layers 36–47
```

instead of blindly splitting:

```text
16 / 16 / 16
```

The scheduler can also replicate bottleneck stages:

```text
                 ┌── Trainium A
TPU ─────────────┤
                 └── Trainium B
```

Microbatches can then be routed dynamically.

### 9. Fault Tolerance

Distributed infrastructure assumes failures.

Machines can disappear.

Networks can degrade.

Cloud instances can terminate.

Elysium therefore treats failure as a normal scheduling event.

```text
Worker failure
      │
      ▼
Detect
      │
      ▼
Reschedule
      │
      ▼
Restore checkpoint
      │
      ▼
Resume pipeline
```

Checkpoints are coordinated independently from the execution of the pipeline.

## Example

Imagine an organization has:

```text
4 × NVIDIA GPUs
8 × TPU devices
16 × Trainium devices
```

Elysium analyzes the model and cluster.

It might produce:

```text
MODEL
│
├── GPU
│   └── Layers 0–14
│
├── TPU
│   └── Layers 15–31
│
└── Trainium
    └── Layers 32–47
```

with:

```text
microbatches = 32
pipeline schedule = zero-bubble
precision = BF16
```

The developer still sees:

```python
els.fit(model)
```

The complexity remains inside Elysium.

## Design Principles

### Hardware should be heterogeneous

Elysium does not assume every device is the same.

### Software should be portable

Developers should not rewrite their models for every accelerator.

### Communication is a resource

Network bandwidth and latency are treated as part of the compute model.

### Native execution matters

Each accelerator should use the runtime it is best optimized for.

### Scheduling should be economic

The fastest configuration is not always the cheapest configuration.

### Failures are expected

The system should recover rather than assume perfect infrastructure.

## Vision

The future of AI infrastructure should not be defined by a single chip, cloud, or datacenter.

Compute is everywhere.

GPUs are everywhere.

TPUs are everywhere.

Custom accelerators are everywhere.

The missing layer is the infrastructure that makes them work together.

Elysium is building that layer.

**One model. Any hardware. One computing fabric.**
