# Proposed Memory Metric

## Goal

We want a memory metric that reflects **practical usability for long-form generation**.

A single raw peak VRAM number is often not enough, because it does not show whether memory stays roughly constant as generation length increases or grows with sequence length.

For this reason, the proposed primary metric is a **memory scaling metric**.

---

## Primary Metric: Decode Memory Growth Rate (DMGR)

### Definition

Let:

- `M(L)` = peak incremental memory usage when generating a sequence of length `L`
- `L` = generated sequence length in tokens

Then define:

\[
\mathrm{DMGR} = \frac{M(L_2) - M(L_1)}{L_2 - L_1}
\]

This measures the **additional memory required per extra generated token**.

### Recommended reporting units

- **MB / 1k generated tokens**, or
- **GiB / 10k generated tokens**

These units are easier to interpret than bytes per token.

---

## Definition of `M(L)`

To isolate generation-related memory from the static model footprint, define:

\[
M(L) = \text{Peak VRAM during generation of length } L - \text{Idle VRAM after model load}
\]

This makes the metric reflect **generation overhead**, rather than total memory usage including the loaded model parameters.

---

## Why this metric is useful

DMGR is useful because it answers the practical question:

> As generation gets longer, how much extra memory does the model need?

This is especially important in long-form symbolic music generation, where sequence length can become very large.

### Interpretation

- **DMGR ≈ 0**  
  Memory remains effectively constant as sequence length grows.

- **Low DMGR**  
  Memory grows slowly and scales well to long generations.

- **High DMGR**  
  Memory usage increases substantially with generation length.

This makes DMGR well aligned with claims about recurrent-state inference, constant recurrent memory, and practical scalability.

---

## Secondary Supporting Metric

Alongside DMGR, report:

### Peak Incremental VRAM at target length

For example:

- Peak incremental VRAM at 4k tokens
- Peak incremental VRAM at 8k tokens
- Peak incremental VRAM at 12k tokens

This gives an absolute memory reference point, while DMGR captures the scaling trend.

---

## Recommended evaluation setup

### Step 1
Load the model and record the **idle VRAM after model load**.

### Step 2
Generate sequences at several target lengths, for example:

- 1k
- 2k
- 4k
- 8k
- 12k tokens

### Step 3
For each target length, record:

- peak VRAM during generation
- incremental VRAM = peak VRAM - idle VRAM

### Step 4
Compute DMGR between two selected lengths, for example:

\[
\mathrm{DMGR}_{1k \rightarrow 8k} = \frac{M(8000)-M(1000)}{8000-1000}
\]

### Step 5
Report both:

- **DMGR**
- **Peak Incremental VRAM at selected target lengths**

---

## Why not use only peak memory

Raw peak memory alone has two limitations:

1. It does not show how memory scales with generation length.
2. It can mix together model footprint and generation overhead.

A model with a moderate peak at one length may still scale poorly as length grows.  
DMGR makes this visible.

---

## Suggested paper-style wording

### Short version

> We measure memory efficiency using decode memory growth rate (DMGR), defined as the increase in peak incremental VRAM per additional generated token. This captures whether memory usage remains approximately constant or grows with sequence length during generation.

### Longer version

> To evaluate memory efficiency during generation, we define decode memory growth rate (DMGR) as the increase in peak incremental VRAM per additional generated token. Here, incremental VRAM is measured relative to the idle memory footprint after model loading. Unlike a single peak-memory value, DMGR captures how memory usage scales with generation length, making it more suitable for evaluating practical long-form generation.

---

## Recommended use in this paper

For this paper, memory efficiency can be presented under **inference efficiency**, alongside:

- generation speed
- seconds per bar
- throughput
- out-of-memory behavior

DMGR is especially suitable if the paper wants to show that recurrent-state inference avoids strong length-dependent memory growth.

---

## One-line summary

> Use **Decode Memory Growth Rate (DMGR)** as the primary memory metric, supported by **Peak Incremental VRAM at target lengths**.
