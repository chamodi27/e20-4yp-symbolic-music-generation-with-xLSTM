#!/usr/bin/env python3
"""
xLSTM DMGR (Decode Memory Growth Rate) evaluation.

Loads xLSTMGenerator once, generates sequences at multiple target lengths,
measures peak GPU VRAM via both pynvml and torch.cuda, and computes DMGR.

Usage (from repo root, xlstm conda env active):
    python notebooks/xLSTM-memory-metric/evaluate_dmgr_xlstm.py
    python notebooks/xLSTM-memory-metric/evaluate_dmgr_xlstm.py --gpu-index 1
    python notebooks/xLSTM-memory-metric/evaluate_dmgr_xlstm.py --warmup
    python notebooks/xLSTM-memory-metric/evaluate_dmgr_xlstm.py --skip-lengths 12000

Flags:
    --gpu-index N          GPU device index (default: 0)
    --warmup               Force a warm-up pass before measurement
    --skip-lengths L [L …] Exclude these lengths from the sweep (e.g. 12000)
"""
import argparse
import gc
import json
import os
import sys
import threading
import time

import pynvml
import torch

# ── Repo-relative path resolution ─────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))           # xLSTM-memory-metric/
_NB_ROOT  = os.path.dirname(_HERE)                               # notebooks/
_REPO_ROOT = os.path.dirname(_NB_ROOT)                           # fyp-musicgen/

# Add paths needed by xLSTMGenerator and helibrunna.
# repos/ must be appended (not inserted at 0) so the installed xlstm package in
# site-packages takes precedence over the local repos/xlstm/ git clone.
sys.path.insert(0, os.path.join(_NB_ROOT, "xLSTM-4-recurrent-state"))
sys.path.append(os.path.join(_REPO_ROOT, "repos"))

from inference import xLSTMGenerator  # noqa: E402 (path set above)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    _REPO_ROOT, "repos", "helibrunna", "output",
    "xlstm_lmd_preprocessed_512d_4096ctx_12b", "run_20260301-1906"
)
CHECKPOINT_NAME          = "checkpoint-46000-last"
INFERENCE_CONTEXT_LENGTH = 16_384
PROMPT                   = "s-9 o-0 t-38"
TEMPERATURE              = 0.9
POLL_S                   = 0.05          # pynvml polling interval (50 ms)
TARGET_LENGTHS_DEFAULT   = [1000, 2000, 4000, 8000, 12000]
DMGR_L1, DMGR_L2        = 1000, 12000

OUTPUT_DIR  = os.path.join(_HERE, "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "dmgr_results_xlstm.json")

# ── CLI ARGS ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="xLSTM DMGR evaluation")
    p.add_argument("--gpu-index", type=int, default=0,
                   help="GPU device index to monitor and run on (default: 0)")
    p.add_argument("--warmup", action="store_true",
                   help="Force a warm-up generation pass before measurement")
    p.add_argument("--skip-lengths", type=int, nargs="*", default=[],
                   help="Token lengths to exclude from the sweep (e.g. --skip-lengths 12000)")
    return p.parse_args()

# ── pynvml helpers ────────────────────────────────────────────────────────────
def get_used_bytes(handle):
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used

def check_gpu_exclusive(handle, gpu_index):
    own_pid = os.getpid()
    others  = [p for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
               if p.pid != own_pid]
    if not others:
        return
    print(f"\n[WARNING] Other processes on GPU {gpu_index}:")
    for p in others:
        print(f"  PID {p.pid}   VRAM {p.usedGpuMemory // 1024**2} MB")
    print("  pynvml reports total GPU VRAM — their footprint inflates all readings.")
    print("  Use --gpu-index to target a free GPU, or wait for those jobs to finish.\n")
    ans = input("  Continue anyway? [y/N]: ").strip().lower()
    if ans != "y":
        print("Aborted.")
        sys.exit(0)

# ── Peak memory monitor (background thread) ───────────────────────────────────
class PeakMemoryMonitor:
    """Polls pynvml at POLL_S intervals and records the maximum VRAM seen."""
    def __init__(self, handle):
        self.handle  = handle
        self._peak   = 0
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._peak = get_used_bytes(self.handle)
        self._stop.clear()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self._peak

    def _run(self):
        while not self._stop.is_set():
            used = get_used_bytes(self.handle)
            if used > self._peak:
                self._peak = used
            time.sleep(POLL_S)

# ── Per-length measurement ────────────────────────────────────────────────────
def measure_length(generator, handle, target_len, idle_torch, idle_pynvml):
    """Generate target_len tokens and return peak VRAM from both measurement methods."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    monitor = PeakMemoryMonitor(handle)
    monitor.start()

    t0 = time.time()
    generator.generate(
        prompt=PROMPT,
        temperature=TEMPERATURE,
        max_length=target_len,
        return_structured_output=False,
    )
    elapsed = time.time() - t0

    peak_torch  = torch.cuda.max_memory_allocated()
    peak_pynvml = monitor.stop()

    return {
        "peak_torch_mb":   round(peak_torch  / 1024**2, 2),
        "peak_pynvml_mb":  round(peak_pynvml / 1024**2, 2),
        "incr_torch_mb":   round((peak_torch  - idle_torch)  / 1024**2, 2),
        "incr_pynvml_mb":  round((peak_pynvml - idle_pynvml) / 1024**2, 2),
        "generation_time_s": round(elapsed, 1),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Make torch use the same GPU that pynvml will monitor
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    pynvml.nvmlInit()
    handle   = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    total_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).total // 1024**2

    target_lengths = [L for L in TARGET_LENGTHS_DEFAULT if L not in args.skip_lengths]

    print("=" * 60)
    print("xLSTM DMGR Evaluation")
    print("=" * 60)
    print(f"GPU {args.gpu_index}: {gpu_name}  ({total_mb} MB total)")
    print(f"Model      : {MODEL_PATH}")
    print(f"Checkpoint : {CHECKPOINT_NAME}")
    print(f"Lengths    : {target_lengths}")
    print(f"DMGR range : {DMGR_L1} → {DMGR_L2} tokens")
    print()

    check_gpu_exclusive(handle, args.gpu_index)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading xLSTMGenerator...")
    generator = xLSTMGenerator(
        model_path_or_repo=MODEL_PATH,
        checkpoint_name=CHECKPOINT_NAME,
        config_overrides={"context_length": INFERENCE_CONTEXT_LENGTH},
        device="cuda",
    )
    print("Model loaded.\n")

    # Idle VRAM baseline — after load, before any generation
    torch.cuda.empty_cache()
    idle_torch_bytes  = torch.cuda.memory_allocated()
    idle_pynvml_bytes = get_used_bytes(handle)
    idle_torch_mb     = idle_torch_bytes  / 1024**2
    idle_pynvml_mb    = idle_pynvml_bytes / 1024**2
    print(f"  Idle VRAM (torch):  {idle_torch_mb:.1f} MB")
    print(f"  Idle VRAM (pynvml): {idle_pynvml_mb:.1f} MB\n")

    # ── Optional warm-up ──────────────────────────────────────────────────────
    if args.warmup:
        print("Warm-up pass (200 tokens) ...")
        t0 = time.time()
        generator.generate(PROMPT, temperature=TEMPERATURE, max_length=200)
        print(f"  Warm-up done ({time.time()-t0:.0f}s)\n")
        torch.cuda.empty_cache()

    # ── Per-length sweep ──────────────────────────────────────────────────────
    print("Running generation sweep...")
    results = {}
    for L in target_lengths:
        print(f"  L={L:>6} tokens  ...", end="", flush=True)
        v = measure_length(generator, handle, L, idle_torch_bytes, idle_pynvml_bytes)
        results[L] = v
        print(f"  peak_pynvml={v['peak_pynvml_mb']:.1f} MB  "
              f"incr_pynvml={v['incr_pynvml_mb']:+.1f} MB  "
              f"incr_torch={v['incr_torch_mb']:+.1f} MB  "
              f"({v['generation_time_s']:.0f}s)")

    # ── DMGR computation ──────────────────────────────────────────────────────
    if DMGR_L1 in results and DMGR_L2 in results:
        delta_tokens = (DMGR_L2 - DMGR_L1) / 1000
        dmgr_pynvml  = (results[DMGR_L2]["incr_pynvml_mb"] - results[DMGR_L1]["incr_pynvml_mb"]) / delta_tokens
        dmgr_torch   = (results[DMGR_L2]["incr_torch_mb"]  - results[DMGR_L1]["incr_torch_mb"])  / delta_tokens
    else:
        dmgr_pynvml = dmgr_torch = None
        print(f"\n[WARNING] DMGR_L1={DMGR_L1} or DMGR_L2={DMGR_L2} not in measured lengths — DMGR not computed.")

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        "model":                          os.path.basename(os.path.dirname(MODEL_PATH)),
        "checkpoint":                     CHECKPOINT_NAME,
        "gpu":                            gpu_name,
        "idle_vram_torch_mb":             round(idle_torch_mb,  2),
        "idle_vram_pynvml_mb":            round(idle_pynvml_mb, 2),
        "DMGR_L1":                        DMGR_L1,
        "DMGR_L2":                        DMGR_L2,
        "DMGR_pynvml_MB_per_1k_tokens":   round(dmgr_pynvml, 4) if dmgr_pynvml is not None else None,
        "DMGR_torch_MB_per_1k_tokens":    round(dmgr_torch,  4) if dmgr_torch  is not None else None,
        "per_length":                     {str(k): v for k, v in results.items()},
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Final report ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("DMGR RESULTS — xLSTM")
    print("=" * 60)
    print(f"GPU                            : {gpu_name}")
    print(f"Idle VRAM (torch)              : {idle_torch_mb:.1f} MB")
    print(f"Idle VRAM (pynvml)             : {idle_pynvml_mb:.1f} MB")
    print()
    hdr = f"  {'Length':>8}   {'Peak pynvml':>14}   {'Incr pynvml':>14}   {'Incr torch':>12}   {'Time':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for L, v in results.items():
        print(f"  {L:>8}   {v['peak_pynvml_mb']:>12.1f} MB"
              f"   {v['incr_pynvml_mb']:>12.1f} MB"
              f"   {v['incr_torch_mb']:>10.1f} MB"
              f"   {v['generation_time_s']:>6.0f}s")
    print()
    if dmgr_pynvml is not None:
        print(f"DMGR ({DMGR_L1}→{DMGR_L2} tokens, pynvml) : {dmgr_pynvml:.4f} MB / 1k tokens")
        print(f"DMGR ({DMGR_L1}→{DMGR_L2} tokens, torch)  : {dmgr_torch:.4f} MB / 1k tokens")
    print()
    print(f"Results saved → {OUTPUT_JSON}")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
