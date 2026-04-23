# DMGR Evaluation Plan — Museformer

## Revised Feasibility Assessment

### What we know works (proven)

- `fairseq-interactive` and `fairseq-generate` CLI generation works end-to-end (samples exist in `output/generation/`)
- The model checkpoint at `checkpoints/mf-lmd6remi-1/checkpoint_best.pt` is valid and has been used

### What we do NOT know

- Whether the fairseq Python API can be called directly to load and run Museformer generation
- `musformer_ppl_eval.py` was AI-generated and **never tested** — it cannot be used as a baseline

### Why the naive Python API approach is risky

Museformer generation through the CLI involves several layers:

1. `fairseq-interactive` loads model + task internally via its own CLI bootstrapping
2. The `MuseformerLanguageModelingTask` registers custom datasets and preprocessing
   (`ChunkSequenceDataset2`, `AddBeatDataset`, `ExtendedWrapperDataset`, etc.)
3. The decoder's `forward()` expects structured `chunk_points`, `beat_ids`, etc. —
   these are auto-constructed from tokens during inference, but only if the sample dict
   is formatted exactly as the task expects
4. There is no tested Python entry point that bypasses the CLI

Attempting to replicate the CLI setup in Python from scratch would require reverse-engineering
undocumented fairseq internal call conventions and risks silent failures (wrong shapes, wrong
sample format) that produce incorrect VRAM measurements.

### Verdict: Feasible via a different approach

**Use the proven CLI path and measure VRAM externally using `pynvml`.**

`pynvml` (NVIDIA Management Library Python bindings) exposes `nvmlDeviceGetMemoryInfo()` which
polls actual GPU VRAM usage at the driver level — independent of fairseq, PyTorch, or any Python
internals. Running it in a background thread at ~100 ms intervals while the CLI generates is
sufficient to capture peak VRAM for each target length.

This approach:
- Requires no changes to museformer source code
- Uses only the proven CLI generation path
- Gives the same VRAM numbers that `torch.cuda.max_memory_allocated()` would give inside the process
- Is straightforward to verify: numbers should match `nvidia-smi` spot checks

---

## Architecture of the Evaluation Script

```
evaluate_dmgr.py
│
├── Step 1: Load model idle VRAM
│     Run: fairseq-interactive (just load model, send nothing, exit)
│     Measure: GPU memory after model load settles, before any generation
│
├── Step 2: For each target length L in [1k, 2k, 4k, 8k, 12k]:
│     ├── Start pynvml polling thread (100ms interval, records max seen)
│     ├── Run: fairseq-interactive subprocess with --max-len-b=L and --min-len=L
│     │         Feed: one empty prompt via stdin (as the working generation scripts do)
│     ├── Wait for subprocess to exit
│     ├── Stop polling thread, record peak VRAM
│     └── incremental_vram[L] = peak_vram - idle_vram
│
└── Step 3: Compute and save DMGR
```

---

## Paths Reference

| Item | Path |
|------|------|
| Repo root | `repos/muzic/museformer/` |
| Checkpoint | `checkpoints/mf-lmd6remi-1/checkpoint_best.pt` |
| Data bin | `data-bin/lmd6remi/` |
| Evaluation script (to create) | `repos/muzic/museformer/evaluate_dmgr.py` |
| Results output | `repos/muzic/museformer/output_log/dmgr_results.json` |

All commands run from the **repo root** inside the `museformer` conda environment:

```bash
conda activate museformer
cd /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/muzic/museformer
```

---

## Pre-flight Checks

Before writing the script, verify these manually:

```bash
# Check pynvml is available in the museformer env
python -c "import pynvml; pynvml.nvmlInit(); print('pynvml OK')"

# If missing:
pip install pynvml

# Confirm which GPU index museformer uses (typically 0)
nvidia-smi --list-gpus

# Confirm the generation command works for a short run (quick sanity check)
printf '\n' | fairseq-interactive data-bin/lmd6remi \
  --path checkpoints/mf-lmd6remi-1/checkpoint_best.pt \
  --user-dir museformer \
  --task museformer_language_modeling \
  --sampling --sampling-topk 8 --beam 1 --nbest 1 \
  --min-len 100 --max-len-b 200 \
  --seed 42 2>&1 | tail -5
```

- [ ] `pynvml` available in `museformer` env
- [ ] GPU index confirmed
- [ ] Short generation test passes (produces output, no crash)

---

## Stage 1 — Measure Idle VRAM After Model Load

The idle VRAM baseline must be measured after the model is loaded but before any generation
tokens are processed. The CLI loads the model before reading stdin, so:

```bash
# Load model, immediately close stdin, capture GPU memory reading
printf '' | fairseq-interactive data-bin/lmd6remi \
  --path checkpoints/mf-lmd6remi-1/checkpoint_best.pt \
  --user-dir museformer \
  --task museformer_language_modeling \
  --sampling --sampling-topk 8 --beam 1 --nbest 1 \
  --min-len 1 --max-len-b 1 \
  --seed 42
```

Measure GPU memory during the quiet period between model-load prints and first generation output.
In the script this is handled by:

1. Starting the subprocess
2. Watching stderr/stdout for the "| model params" line (fairseq prints this after model load)
3. Recording `nvmlDeviceGetMemoryInfo(handle).used` at that moment as `idle_vram`

---

## Stage 2 — Per-Length Measurement Loop

Target lengths: **1000, 2000, 4000, 8000, 12000** tokens

For each length `L`:

```bash
printf '\n' | fairseq-interactive data-bin/lmd6remi \
  --path checkpoints/mf-lmd6remi-1/checkpoint_best.pt \
  --user-dir museformer \
  --task museformer_language_modeling \
  --sampling --sampling-topk 8 --beam 1 --nbest 1 \
  --min-len L --max-len-b L \
  --num-workers 4 \
  --seed 42
```

The pynvml polling thread records the maximum `used` bytes observed while this subprocess is
running. After the subprocess exits, `peak_vram[L]` is that maximum.

```
incremental_vram[L] = peak_vram[L] - idle_vram   (in bytes)
```

---

## Stage 3 — DMGR Computation

```
DMGR_{1k→8k} = (incremental_vram[8000] - incremental_vram[1000]) / (8000 - 1000)

Report in MB / 1k tokens:
  DMGR_MB_per_1k = DMGR_{1k→8k} * 1000 / (1024^2)
```

---

## Script Structure: `evaluate_dmgr.py`

```python
#!/usr/bin/env python3
"""
DMGR evaluation for Museformer.
Runs fairseq-interactive as a subprocess for each target length,
monitors GPU VRAM via pynvml, computes Decode Memory Growth Rate.
"""
import subprocess, threading, time, json, os, sys

import pynvml

# ── CONFIG ──────────────────────────────────────────────────────────────────
CHECKPOINT   = "checkpoints/mf-lmd6remi-1/checkpoint_best.pt"
DATA_BIN     = "data-bin/lmd6remi"
GPU_INDEX    = 0
POLL_INTERVAL_S = 0.1          # 100 ms pynvml polling
SEED         = 42
TARGET_LENGTHS = [1000, 2000, 4000, 8000, 12000]
DMGR_L1, DMGR_L2 = 1000, 8000

# ── pynvml helpers ───────────────────────────────────────────────────────────
def get_gpu_used_bytes(handle):
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used

class PeakMemoryMonitor:
    """Background thread that polls pynvml and records the peak."""
    def __init__(self, handle, interval=0.1):
        self.handle = handle
        self.interval = interval
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._peak = get_gpu_used_bytes(self.handle)
        self._stop.clear()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self._peak

    def _run(self):
        while not self._stop.is_set():
            used = get_gpu_used_bytes(self.handle)
            if used > self._peak:
                self._peak = used
            time.sleep(self.interval)

# ── Generation command builder ───────────────────────────────────────────────
def make_cmd(min_len, max_len):
    return [
        "fairseq-interactive", DATA_BIN,
        "--path", CHECKPOINT,
        "--user-dir", "museformer",
        "--task", "museformer_language_modeling",
        "--sampling", "--sampling-topk", "8",
        "--beam", "1", "--nbest", "1",
        "--min-len", str(min_len),
        "--max-len-b", str(max_len),
        "--num-workers", "4",
        "--seed", str(SEED),
    ]

# ── Idle VRAM measurement ─────────────────────────────────────────────────────
def measure_idle_vram(handle):
    """
    Run a minimal generation (1 token) and capture VRAM after the model-load
    print line, before generation output begins.
    """
    print("Measuring idle VRAM (model load baseline) ...")
    monitor = PeakMemoryMonitor(handle)
    proc = subprocess.Popen(
        make_cmd(1, 1),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    idle_bytes = None
    for line in proc.stdout:
        if "model params" in line.lower() or "| num. model params" in line.lower():
            # Model is loaded — sample VRAM right now before generation starts
            idle_bytes = get_gpu_used_bytes(handle)
            break
    proc.stdin.write("\n")
    proc.stdin.flush()
    proc.wait()
    if idle_bytes is None:
        # Fallback: use memory before generation as idle
        idle_bytes = get_gpu_used_bytes(handle)
    return idle_bytes

# ── Per-length peak VRAM measurement ─────────────────────────────────────────
def measure_peak_vram(handle, target_len):
    print(f"  Generating {target_len} tokens ...")
    monitor = PeakMemoryMonitor(handle)
    monitor.start()
    proc = subprocess.Popen(
        make_cmd(target_len, target_len),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(b"\n")
    proc.stdin.flush()
    proc.wait()
    peak = monitor.stop()
    return peak

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)

    # Warm-up: let Triton kernels compile (not measured)
    print("Warm-up pass (Triton kernel compilation) ...")
    proc = subprocess.run(
        make_cmd(200, 300),
        input=b"\n", capture_output=True,
    )

    idle_bytes = measure_idle_vram(handle)
    print(f"Idle VRAM after model load: {idle_bytes / 1024**2:.1f} MB")

    results = {}
    for L in TARGET_LENGTHS:
        peak = measure_peak_vram(handle, L)
        incr_mb = (peak - idle_bytes) / 1024**2
        results[L] = {
            "peak_vram_mb":        peak / 1024**2,
            "incremental_vram_mb": incr_mb,
        }
        print(f"    L={L:>6}  peak={results[L]['peak_vram_mb']:.1f} MB  "
              f"incremental={incr_mb:.1f} MB")

    dm = results[DMGR_L2]["incremental_vram_mb"] - results[DMGR_L1]["incremental_vram_mb"]
    dmgr = dm / ((DMGR_L2 - DMGR_L1) / 1000)

    summary = {
        "idle_vram_mb":            idle_bytes / 1024**2,
        "DMGR_MB_per_1k_tokens":   dmgr,
        "DMGR_L1":                 DMGR_L1,
        "DMGR_L2":                 DMGR_L2,
        "per_length":              results,
    }

    os.makedirs("output_log", exist_ok=True)
    with open("output_log/dmgr_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DMGR Result ===")
    print(f"Idle VRAM             : {summary['idle_vram_mb']:.1f} MB")
    print(f"DMGR ({DMGR_L1}→{DMGR_L2}) : {dmgr:.3f} MB / 1k tokens")
    for L, v in results.items():
        print(f"  Peak incr. @ {L:>6} tokens: {v['incremental_vram_mb']:.1f} MB")

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
```

---

## Stage 4 — Run

Launch in the background (from repo root, inside tmux):

```bash
python evaluate_dmgr.py --gpu-index 0 > output_log/dmgr_eval.log 2>&1 &
```

Watch live output:

```bash
tail -f output_log/dmgr_eval.log
```

Expected wall-clock time: several hours (12k-token autoregressive decoding is slow).
Run inside `tmux` or `screen` to avoid losing the session.

### Stopping the process

```bash
# Kill the Python process (replace PID with the number printed when you backgrounded it)
kill <PID>

# Also kill any fairseq-interactive child it may have spawned
pkill -f "fairseq-interactive"
```

### Checking whether the process has stopped

```bash
# If the process is gone, this prints only the header line (no data row)
ps -p <PID>

# Confirm no fairseq-interactive is still running
pgrep -a fairseq-interactive
```

---

## Stage 5 — Validate Results

- [ ] `output_log/dmgr_results.json` contains all 5 length entries
- [ ] `idle_vram_mb` is plausible (~1–3 GB for a 233 MB checkpoint after fairseq overhead)
- [ ] `incremental_vram_mb` values are non-negative and non-decreasing with L
- [ ] Spot-check: run `nvidia-smi` manually during one generation and confirm the peak is in the same ballpark as recorded values
- [ ] `DMGR_MB_per_1k_tokens` is computed and reported

---

## Stage 6 — Fill Results Table

| Generated Length (tokens) | Peak VRAM (MB) | Incremental VRAM (MB) |
|--------------------------|---------------|----------------------|
| 1,000                    |               |                       |
| 2,000                    |               |                       |
| 4,000                    |               |                       |
| 8,000                    |               |                       |
| 12,000                   |               |                       |

**Idle VRAM after model load: ______ MB**  
**DMGR_{1k→8k} = ______ MB / 1k tokens**

---

## Known Gotchas

### `--gpu-index` does not control which GPU fairseq uses

**Symptom:** All `incremental_vram` values are `0.0 MB` regardless of generation length.
Generation completes and the log looks normal, but DMGR is zero.

**Root cause:** `--gpu-index` only tells pynvml which GPU to *watch*. Without
`CUDA_VISIBLE_DEVICES`, fairseq-interactive always defaults to `device_id=0` regardless of
what `--gpu-index` is set to. So if you pass `--gpu-index 1`, pynvml monitors GPU 1 while
generation silently runs on GPU 0 — the two are out of sync and peak readings stay flat.

**Fix already applied in the script:** `evaluate_dmgr.py` sets `CUDA_VISIBLE_DEVICES=<gpu_index>`
in the subprocess environment inside `main()` so fairseq and pynvml always refer to the same
physical device. No manual action needed — just pass the correct `--gpu-index`.

**How to verify it is working:** after starting the script, run `nvidia-smi` and confirm
the `museformer/bin/python` process appears on the GPU index you specified, not GPU 0.

---

### The model reloads for every sequence length — this is intentional

**Observation:** Each measurement spawns a fresh `fairseq-interactive` subprocess, loading
the model 6 times in total (1 idle baseline + 5 lengths).

**Why this is correct:** Reusing one subprocess across lengths would leave residual CUDA
allocations from the previous generation (unflushed KV buffers, temporary tensors) live in
GPU memory when the next length starts. Each length's peak reading would be inflated by the
previous run, making incremental VRAM values meaningless.

Fresh subprocesses give every length a clean CUDA context and an uncontaminated baseline.
The model load overhead (~2 seconds per reload) is negligible against generation times of
30–60 minutes per length.

**Effect on DMGR:** Any constant overhead from deferred first-pass CUDA allocations (cuBLAS
workspace, Triton runtime buffers) is present in every subprocess equally. It cancels out of
the DMGR formula:

```
DMGR = (peak[12k] - idle) - (peak[1k] - idle)  /  11000
     = (peak[12k] - peak[1k])  /  11000
```

So DMGR reflects only the length-dependent memory growth, not fixed per-process overhead.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `pynvml` not found | `pip install pynvml` inside `museformer` env |
| `fairseq-interactive` not found in subprocess | Use full path: `$(which fairseq-interactive)`, or activate conda env before launching script |
| Idle VRAM detection fails (`model params` line not found) | Print all lines and search for the correct fairseq model-load marker; update the `if` condition |
| Peak VRAM same for all lengths (polling too slow) | Reduce `POLL_INTERVAL_S` to 0.05; confirm subprocess is actually finishing generation |
| OOM at 12k tokens | Remove 12k from `TARGET_LENGTHS_DEFAULT`, note OOM in paper |
| Generation exits immediately without output | Run the sanity-check command from Pre-flight Checks manually to debug |
| VRAM numbers seem too low (under-counting) | `nvmlDeviceGetMemoryInfo().used` reports all processes on the GPU; if other processes are running, isolate on a dedicated GPU |

---

## Notes for the Paper

Report results under **inference efficiency**, alongside generation speed:

> We measure memory efficiency using Decode Memory Growth Rate (DMGR), defined as the
> increase in peak incremental VRAM per additional generated token (MB per 1k tokens).
> Incremental VRAM is computed relative to the idle GPU memory footprint after model loading,
> isolating generation overhead from model parameter storage. VRAM was measured using
> the NVIDIA Management Library (NVML) polled at 100 ms intervals during generation.

Report alongside:
- Idle model VRAM footprint (MB)
- Peak incremental VRAM at 4k and 8k tokens
- DMGR_{1k→8k} in MB / 1k tokens

---

## What You Need to Do Before Implementation

Run each check below, paste the output back, and confirm it passes.
These are the four things that would cause the script to silently fail or give wrong numbers
if left unverified.

---

### Check 1 — pynvml availability

```bash
conda activate museformer
python -c "import pynvml; pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0); print(pynvml.nvmlDeviceGetMemoryInfo(h).used // 1024**2, 'MB used on GPU 0'); pynvml.nvmlShutdown()"
```

**Expected:** prints a number of MB (e.g. `1200 MB used on GPU 0`).  
**If it fails with ModuleNotFoundError:** run `pip install pynvml` then re-test.  
**Report back:** the printed MB value, or the error message.

---

### Check 2 — fairseq-interactive is on PATH inside the conda env

The script launches `fairseq-interactive` as a subprocess. It needs to resolve to the
binary in the `museformer` env, not some other env.

```bash
conda activate museformer
which fairseq-interactive
```

**Expected:** a path containing `museformer` or `envs/museformer`, e.g.
`/home/.../miniconda3/envs/museformer/bin/fairseq-interactive`  
**Report back:** the full path printed.

---

### Check 3 — Short generation sanity check

This confirms the CLI works with the checkpoint and that Triton kernels compile cleanly.
Run it and wait for it to finish (may take up to 5 minutes on first run due to Triton compile).

```bash
conda activate museformer
cd /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/muzic/museformer
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

printf '\n' | fairseq-interactive data-bin/lmd6remi \
  --path checkpoints/mf-lmd6remi-1/checkpoint_best.pt \
  --user-dir museformer \
  --task museformer_language_modeling \
  --sampling --sampling-topk 8 --beam 1 --nbest 1 \
  --min-len 200 --max-len-b 300 \
  --num-workers 4 \
  --seed 42 2>&1 | tail -10
```

**Expected:** last few lines contain a `H-0` hypothesis line (token sequence) with no errors.  
**Report back:** the last 10 lines of output, or any error/traceback.

> Note: `CC`/`CXX` env vars are set because your `run_generation.sh` sets them —
> they are needed for Triton CUDA kernel compilation on this machine.

---

### Check 4 — GPU is exclusively available

`pynvml` reports total VRAM used by **all processes** on the GPU.
If another job is running on GPU 0 during the evaluation, the idle baseline and peak
readings will be inflated and the DMGR will be wrong.

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**Expected:** empty table (no other processes), or only your own shell.  
**Report back:** the output of the command.

If other jobs are present, either wait for them to finish or identify a free GPU index
and report which one to use (so the script's `GPU_INDEX` can be updated accordingly).
