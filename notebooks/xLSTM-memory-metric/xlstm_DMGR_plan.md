# DMGR Evaluation Plan — xLSTM

## Overview

This document describes the approach for measuring Decode Memory Growth Rate (DMGR) for the xLSTM model trained on the LMD REMIGEN dataset. It mirrors the Museformer DMGR plan (`context/Museformer_DMGR_Plan.md`) with adaptations for xLSTM's direct Python API and recurrent-state generation.

See `context/proposed_memory_metric.md` for the full DMGR metric definition.

---

## Key Difference vs. Museformer

| Aspect | Museformer | xLSTM |
|--------|-----------|-------|
| Generation API | CLI subprocess (`fairseq-interactive`) | Direct Python API (`xLSTMGenerator`) |
| VRAM measurement | pynvml polling thread only | pynvml polling thread **+** `torch.cuda` in-process |
| Memory growth expected | High — KV cache grows O(N) | ≈ 0 — fixed recurrent state regardless of N |
| Model reload per length | Yes (clean CUDA context per run) | No — `torch.cuda.empty_cache()` + reset stats suffices |
| Warm-up | Triton kernel compilation (~5 min) | Minimal (model already compiled on first forward) |

Since xLSTM is called in-process, we can use two complementary measurement methods:
- **pynvml** — total driver-level VRAM (consistent with Museformer, captures fragmentation)
- **`torch.cuda.max_memory_allocated()`** — exact PyTorch tensor tracking, no polling lag

Both are reported. The pynvml value is used as the primary metric for cross-model comparison.

---

## Why No Model Reload Between Lengths

Museformer reloads the model for each length because residual CUDA allocations from KV buffers
accumulate across generations in a single process.

xLSTM does **not** have this problem:
- The recurrent state (`state = {}`) is local to each `generate()` call in `xLSTMGenerator`
- After `generate()` returns, the state dict goes out of scope and is freed
- `torch.cuda.empty_cache()` flushes any cached blocks before the next run
- The model weights stay loaded — only one load overhead for all 5 measurements

This makes the xLSTM evaluation significantly faster than Museformer (no repeated 30–60s model loads).

---

## Paths Reference

| Item | Path |
|------|------|
| Generator class | `notebooks/xLSTM-4-recurrent-state/inference.py` |
| Model output dir | `repos/helibrunna/output/xlstm_lmd_preprocessed_512d_4096ctx_12b/run_20260301-1906` |
| Checkpoint | `checkpoint-46000-last` |
| Evaluation script | `notebooks/xLSTM-memory-metric/evaluate_dmgr_xlstm.py` |
| Results output | `notebooks/xLSTM-memory-metric/output/dmgr_results_xlstm.json` |

All commands run from the repo root (`fyp-musicgen/`) inside the conda environment that has xLSTM and helibrunna installed.

---

## Pre-flight Checks

Run these before executing the evaluation script:

```bash
# Check pynvml
python -c "import pynvml; pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0); print(pynvml.nvmlDeviceGetMemoryInfo(h).used // 1024**2, 'MB used on GPU 0'); pynvml.nvmlShutdown()"

# Check torch CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# Check model checkpoint exists
ls repos/helibrunna/output/xlstm_lmd_preprocessed_512d_4096ctx_12b/run_20260301-1906/checkpoint-46000-last/

# Check no competing GPU processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

- [ ] pynvml available and returns a plausible MB value
- [ ] torch.cuda.is_available() == True
- [ ] checkpoint-46000-last/ exists and contains `model.safetensors`
- [ ] No other jobs on the target GPU

---

## Script Architecture

```
evaluate_dmgr_xlstm.py
│
├── Step 0: Parse args (--gpu-index, --warmup, --skip-lengths)
│           pynvml.nvmlInit(), get handle for GPU_INDEX
│           Set CUDA_VISIBLE_DEVICES so torch and pynvml see the same GPU
│
├── Step 1: Load xLSTMGenerator once
│           xLSTMGenerator(MODEL_PATH, checkpoint_name=CHECKPOINT_NAME,
│                          config_overrides={"context_length": 16384})
│           torch.cuda.empty_cache()
│           idle_torch_bytes  = torch.cuda.memory_allocated()
│           idle_pynvml_bytes = nvmlDeviceGetMemoryInfo(handle).used
│
├── Step 2: Optional warm-up
│           generate(PROMPT, max_length=200)
│           torch.cuda.empty_cache()
│
├── Step 3: For each L in [1000, 2000, 4000, 8000, 12000]:
│           torch.cuda.empty_cache()
│           torch.cuda.reset_peak_memory_stats()
│           Start PeakMemoryMonitor thread
│           generator.generate(PROMPT, temperature=0.9, max_length=L)
│           peak_torch  = torch.cuda.max_memory_allocated()
│           peak_pynvml = monitor.stop()
│           Record incremental = peak - idle, generation_time
│
└── Step 4: Compute DMGR_{1k→12k} for both methods
            Save JSON to output/dmgr_results_xlstm.json
            Print results table
```

---

## Generation Config

```python
MODEL_PATH              = "<repo_root>/repos/helibrunna/output/xlstm_lmd_preprocessed_512d_4096ctx_12b/run_20260301-1906"
CHECKPOINT_NAME         = "checkpoint-46000-last"
INFERENCE_CONTEXT_LENGTH = 16_384
PROMPT                  = "s-9 o-0 t-38"
TEMPERATURE             = 0.9
TARGET_LENGTHS          = [1000, 2000, 4000, 8000, 12000]
DMGR_L1, DMGR_L2       = 1000, 12000
```

---

## DMGR Computation

```
DMGR_{1k→12k} = (M(12000) - M(1000)) / ((12000 - 1000) / 1000)

Reported in MB / 1k tokens.
```

Two values are reported — one using pynvml peak, one using torch peak.

---

## Expected Results

xLSTM's recurrent state is a **fixed-size** dictionary of tensors that is independent of
sequence length. The state is fully determined by the model architecture (embedding dim,
number of layers, head dim), not by how many tokens have been generated. This is fundamentally
different from a transformer KV cache, which grows as O(N·layers·head_dim).

| Length (tokens) | Expected incr. VRAM | Rationale |
|----------------|---------------------|-----------|
| 1,000 | Low, ~constant | Recurrent state fixed |
| 2,000 | ≈ same | State does not grow |
| 4,000 | ≈ same | State does not grow |
| 8,000 | ≈ same | State does not grow |
| 12,000 | ≈ same | State does not grow |

**Expected DMGR ≈ 0 MB / 1k tokens.**

Any non-zero DMGR would indicate residual tensor accumulation (unlikely) or measurement noise.

---

## Run Instructions

```bash
cd /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen
conda activate <xlstm-env>

# Quick test (skip longest length)
python notebooks/xLSTM-memory-metric/evaluate_dmgr_xlstm.py --skip-lengths 12000

# Full run (inside tmux — may take 1–2 hours for 12k)
python notebooks/xLSTM-memory-metric/evaluate_dmgr_xlstm.py > notebooks/xLSTM-memory-metric/output/dmgr_eval.log 2>&1 &

# Watch live output
tail -f notebooks/xLSTM-memory-metric/output/dmgr_eval.log
```

---

## Output Format

Results are saved to `notebooks/xLSTM-memory-metric/output/dmgr_results_xlstm.json`:

```json
{
  "model": "xlstm_lmd_preprocessed_512d_4096ctx_12b",
  "checkpoint": "checkpoint-46000-last",
  "gpu": "NVIDIA ...",
  "idle_vram_torch_mb": ...,
  "idle_vram_pynvml_mb": ...,
  "DMGR_L1": 1000,
  "DMGR_L2": 12000,
  "DMGR_torch_MB_per_1k_tokens": ...,
  "DMGR_pynvml_MB_per_1k_tokens": ...,
  "per_length": {
    "1000": {
      "peak_torch_mb": ...,
      "peak_pynvml_mb": ...,
      "incr_torch_mb": ...,
      "incr_pynvml_mb": ...,
      "generation_time_s": ...
    },
    ...
  }
}
```

---

## Results Table (to fill after run)

| Generated Length (tokens) | Peak VRAM pynvml (MB) | Incremental VRAM pynvml (MB) | Incremental VRAM torch (MB) | Time (s) |
|--------------------------|----------------------|------------------------------|------------------------------|----------|
| 1,000                    |                      |                              |                              |          |
| 2,000                    |                      |                              |                              |          |
| 4,000                    |                      |                              |                              |          |
| 8,000                    |                      |                              |                              |          |
| 12,000                   |                      |                              |                              |          |

**Idle VRAM after model load (pynvml): ______ MB**
**Idle VRAM after model load (torch):  ______ MB**
**DMGR_{1k→12k} (pynvml): ______ MB / 1k tokens**
**DMGR_{1k→12k} (torch):  ______ MB / 1k tokens**

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `pynvml` not found | `pip install pynvml` in active conda env |
| `import xLSTMGenerator` fails | Check sys.path includes `notebooks/xLSTM-4-recurrent-state/` |
| `model.safetensors` not found | Check MODEL_PATH and CHECKPOINT_NAME constants in script |
| OOM at 12k tokens | Add `--skip-lengths 12000`, note OOM in paper |
| All incremental values are 0 | pynvml is watching wrong GPU — confirm `CUDA_VISIBLE_DEVICES` matches `--gpu-index` |
| Incremental values non-zero and growing | Unexpected — check for background GPU processes; run `nvidia-smi` during generation |

---

## Notes for the Paper

> We measure memory efficiency using Decode Memory Growth Rate (DMGR), defined as the
> increase in peak incremental VRAM per additional generated token (MB per 1k tokens).
> Incremental VRAM is measured relative to the idle GPU memory footprint after model loading.
> For xLSTM, VRAM was measured in-process using both the NVIDIA Management Library (NVML)
> polled at 50 ms intervals and PyTorch's `torch.cuda.max_memory_allocated()`.
> Unlike transformer models whose KV cache grows linearly with sequence length, xLSTM
> maintains a fixed-size recurrent state, yielding DMGR ≈ 0 across all tested lengths.

Report alongside:
- Idle model VRAM footprint (MB)
- Peak incremental VRAM at 4k, 8k, and 12k tokens
- DMGR_{1k→12k} in MB / 1k tokens
- Comparison table with Museformer DMGR values
