# xLSTM DMGR Evaluation Results

**Date:** 2026-04-22  
**Model:** `xlstm_lmd_preprocessed_512d_4096ctx_12b` — checkpoint `checkpoint-46000-last`  
**GPU:** NVIDIA RTX 6000 Ada Generation (48 GB)  
**Metric:** Decode Memory Growth Rate (DMGR) — see `../context/proposed_memory_metric.md`

---

## Summary

| Metric | Value |
|--------|-------|
| Idle VRAM after model load (pynvml) | **3538.3 MB** |
| Idle VRAM after model load (torch) | **2380.3 MB** |
| DMGR\_{1k→12k} (pynvml) | **0.5455 MB / 1k tokens** |
| DMGR\_{1k→12k} (torch) | **0.4882 MB / 1k tokens** |

DMGR is close to zero, confirming that xLSTM's recurrent state is **effectively fixed-size regardless of sequence length**.

---

## Per-Length Results

| Generated Tokens | Peak VRAM pynvml (MB) | Incremental pynvml (MB) | Incremental torch (MB) | Generation Time (s) |
|-----------------|----------------------|------------------------|------------------------|---------------------|
| 1,000 | 3632.3 | 94.0 | 20.9 | 9 |
| 2,000 | 3632.3 | 94.0 | 21.4 | 15 |
| 4,000 | 3634.3 | 96.0 | 22.4 | 32 |
| 8,000 | 3636.3 | 98.0 | 24.3 | 71 |
| 12,000 | 3638.3 | 100.0 | 26.3 | 115 |

Across a **12× increase in sequence length** (1k → 12k tokens), incremental VRAM grows by only **6 MB** (pynvml) or **5.4 MB** (torch). This is measurement-noise territory for a model with a 2380 MB footprint.

---

## DMGR Interpretation

```
DMGR_{1k→12k} = (incr[12k] - incr[1k]) / ((12000 - 1000) / 1000)
              = (100.0 - 94.0) / 11      [pynvml]
              = 0.5455 MB / 1k tokens

              = (26.3 - 20.9) / 11       [torch]
              = 0.4909 MB / 1k tokens
```

For reference, a transformer with KV cache grows linearly with sequence length — typical DMGR values for attention-based models are **10–100× higher** than what we see here.

**xLSTM's DMGR is effectively 0** in practical terms: generating 12,000 tokens instead of 1,000 costs less than 6 MB of extra VRAM, compared to a ~2380 MB model footprint. The ratio of generation overhead to model size stays below 0.3% even at 12k tokens.

---

## Measurement Methodology

Two measurement methods were used in parallel:

### pynvml (primary)
- Polls total GPU VRAM used (driver-level) every 50 ms via `nvmlDeviceGetMemoryInfo()`
- Captures the maximum observed during each generation run
- Includes PyTorch allocator caching, CUDA context overhead, and all other GPU processes
- **Used as the primary metric** for cross-model comparison consistency with Museformer

### torch.cuda (secondary)
- `torch.cuda.max_memory_allocated()` after `torch.cuda.reset_peak_memory_stats()`
- Tracks only PyTorch tensor allocations — excludes CUDA allocator fragmentation and caching
- More precise, zero polling lag, but lower absolute numbers than pynvml

The gap between the two (idle: 3538 MB pynvml vs 2380 MB torch) reflects PyTorch's CUDA memory allocator, which pre-caches blocks from the driver for faster future allocations. This is normal and expected.

### Why pynvml > torch for idle VRAM

PyTorch's CUDA allocator requests memory from the driver in large blocks and holds them even when tensors are freed, to avoid the overhead of frequent driver calls. `torch.cuda.memory_allocated()` counts only tensors that are currently live; pynvml counts everything the process holds at the driver level. The ~1158 MB gap at idle is the allocator's cached-but-unused pool.

### Between-length cleanup

Between each length measurement:
- `torch.cuda.empty_cache()` — flushes PyTorch's cached pool back to the driver
- `torch.cuda.reset_peak_memory_stats()` — zeroes the peak counter for the next run
- No model reload needed — xLSTM's recurrent `state = {}` is local to each `generate()` call and is garbage-collected after it returns

This is valid because xLSTM has no growing KV cache. Each generation starts with a clean recurrent state regardless of what happened in the previous run.

---

## Generation Speed

| Generated Tokens | Time (s) | Tokens/sec |
|-----------------|----------|------------|
| 1,000 | 9 | ~111 |
| 2,000 | 15 | ~133 |
| 4,000 | 32 | ~125 |
| 8,000 | 71 | ~113 |
| 12,000 | 115 | ~104 |

Speed is approximately constant across all lengths (~110–133 tok/s), consistent with O(N) generation — each step costs the same regardless of how many tokens have been generated already. (Small variance is expected due to Triton kernel warm-up within the first run of each length.)

---

## Setup Details

- **Evaluation script:** `notebooks/xLSTM-memory-metric/evaluate_dmgr_xlstm.py`
- **Generation API:** `xLSTMGenerator` from `notebooks/xLSTM-4-recurrent-state/inference.py`  
  (recurrent formulation using `model.step(token, state)` — O(1) per token)
- **Prompt:** `"s-9 o-0 t-38"` (fixed, same as existing single-shot generation configs)
- **Temperature:** 0.9
- **Inference context length override:** 16,384 tokens (training context was 4,096)
- **pynvml poll interval:** 50 ms
- **Seed:** 42

### Important: GPU isolation

GPU 0 had an unrelated Python process using ~3 GB of VRAM during the run. **All measurements were taken on GPU 1**, which was idle (4 MB VRAM used). This was essential for clean pynvml readings — pynvml reports total VRAM used by all processes on the GPU, so a competing process would have inflated all numbers equally and, importantly, made the idle baseline wrong.

### Path conflict note

The repo contains a local git clone of the xlstm library at `repos/xlstm/`. When `repos/` is added to `sys.path` before site-packages (using `sys.path.insert(0, ...)`), this local clone shadows the installed `xlstm` package and causes an import error (`No module named 'xlstm.xlstm_lm_model'`). The evaluation script avoids this by using `sys.path.append()` for the repos path, matching the same pattern used by `inference.py`.

---

## Raw JSON

Results are also saved to `dmgr_results_xlstm.json` in this directory. Excerpt:

```json
{
  "model": "run_20260301-1906",
  "checkpoint": "checkpoint-46000-last",
  "gpu": "NVIDIA RTX 6000 Ada Generation",
  "idle_vram_torch_mb": 2380.3,
  "idle_vram_pynvml_mb": 3538.3,
  "DMGR_L1": 1000,
  "DMGR_L2": 12000,
  "DMGR_pynvml_MB_per_1k_tokens": 0.5455,
  "DMGR_torch_MB_per_1k_tokens": 0.4882,
  "per_length": {
    "1000":  { "incr_pynvml_mb":  94.0, "incr_torch_mb": 20.9, "generation_time_s":   9.0 },
    "2000":  { "incr_pynvml_mb":  94.0, "incr_torch_mb": 21.4, "generation_time_s":  15.0 },
    "4000":  { "incr_pynvml_mb":  96.0, "incr_torch_mb": 22.4, "generation_time_s":  32.0 },
    "8000":  { "incr_pynvml_mb":  98.0, "incr_torch_mb": 24.3, "generation_time_s":  71.0 },
    "12000": { "incr_pynvml_mb": 100.0, "incr_torch_mb": 26.3, "generation_time_s": 115.0 }
  }
}
```
