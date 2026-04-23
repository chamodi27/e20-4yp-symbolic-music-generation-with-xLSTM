
# Critical Structure Set Selection Criteria (Refined)

This document defines the **exact rules used to select the Critical Structure Set** for human evaluation of music generation models.

The objective of the critical set is to include **samples that stress long‑term structural coherence**, which is the central research focus of the study.

Selection is **metric-based** to avoid researcher bias and ensure reproducibility.

---

# Metrics Available

The following metrics are computed for each generated MIDI file:

- `bars_N` — Number of bars
- `global_key` — Estimated key
- `key_stability` — Key stability score
- `key_change_rate` — Rate of key changes
- `chord_diversity` — Chord diversity
- `pitch_range` — Pitch range
- `stepwise_ratio` — Stepwise motion ratio
- `note_density_mean` — Mean note density
- `note_density_std` — Note density standard deviation
- `duration_entropy` — Duration entropy
- `avg_offdiag` — Average off‑diagonal similarity in SSM
- `rep_density` — Repetition density
- `struct_entropy` — Structural entropy
- `block_coherence` — Block coherence

Percentiles are computed over the **candidate pool of generated clips**.

### Path to CSV files with all metrics per each midi of a model
**LookBack RNN** - /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/evaluation/results/all_metrics_lookback_rnn.csv
---

# Step 0 — Eligibility Filter

Before selecting critical samples, remove clips that meet any of the following:

- `bars_N < 16`
- `note_density_mean == 0`
- `duration_entropy < 0.5`
- obvious degeneracy (single‑note repetition, silence, etc.)

This step removes **invalid outputs only**, not low-quality music.

---

# Critical Clip 1 — Strong Repetition Structure

## Purpose
Evaluate whether the model produces **clear repeating musical patterns**.

## Criteria

Select clips satisfying:

- `rep_density ≥ 75th percentile`
- `avg_offdiag ≥ 75th percentile`
- `block_coherence ≥ median`
- `bars_N ≥ 16`

From these candidates choose the clip with the **highest `rep_density`**.

## Rationale
High repetition density and strong off‑diagonal similarity indicate recurring musical patterns and structural repetition.

---

# Critical Clip 2 — Clear Sectional Organization

## Purpose
Evaluate whether the model creates **distinct musical sections** rather than random variation.

## Criteria

Select clips satisfying:

- `block_coherence ≥ 75th percentile`
- `struct_entropy between 40th and 80th percentile`
- `avg_offdiag ≥ median`
- `bars_N ≥ 16`

From these candidates choose the clip with the **highest `block_coherence`**.

## Rationale
High block coherence indicates structured similarity blocks in the self‑similarity matrix, suggesting sectional organization.

---

# Critical Clip 3 — Long‑Sequence Memory Stress

## Purpose
Evaluate whether the model maintains structure over **long musical spans**.

## Criteria

Select clips satisfying:

- `bars_N ≥ 75th percentile`
- `avg_offdiag ≥ median`
- `rep_density ≥ median`

Choose the clip with the **largest `bars_N`**.

## Rationale
Longer sequences increase the difficulty of maintaining coherent structure.

---

# Critical Clip 4 — Local vs Global Structure Conflict

## Purpose
Identify cases where music appears **locally coherent but lacks global structure**.

## Criteria

Select clips satisfying:

- `block_coherence ≥ 75th percentile`
- `avg_offdiag ≤ 25th percentile`
- `rep_density ≤ median`

From these candidates choose the clip maximizing:

```
block_coherence − avg_offdiag
```

## Rationale
High block coherence combined with low off‑diagonal similarity indicates strong local patterns but weak long‑range repetition.

---

# Final Critical Set

For each model:

| Critical Type | Count |
|---|---|
| Strong repetition structure | 1 |
| Clear sectional organization | 1 |
| Long‑sequence stress | 1 |
| Local vs global structure conflict | 1 |

Total critical samples per model: **4**

---

# Methodology Statement (for Paper)

Example description:

> A critical structure subset was selected using predefined metric-based criteria derived from bar-level self-similarity analysis. Four categories were used: strong repetition structure, clear sectional organization, long-sequence stress, and local–global structure conflict. Selection relied on repetition density, off-diagonal similarity, block coherence, structural entropy, and sequence length. A brief listening check was performed only to remove corrupted renders, not to judge model quality.
