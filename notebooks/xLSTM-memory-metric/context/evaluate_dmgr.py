#!/usr/bin/env python3
"""
Museformer DMGR (Decode Memory Growth Rate) evaluation.

Runs fairseq-interactive as a subprocess at multiple target token lengths,
monitors GPU VRAM via pynvml, and computes the DMGR metric.

Usage (from museformer repo root, museformer env active):
    python evaluate_dmgr.py                     # default: 1k/2k/4k/8k, warmup skipped
    python evaluate_dmgr.py --gpu-index 1       # use a different GPU
    python evaluate_dmgr.py --full              # also measure 12k tokens (~1.5h extra)
    python evaluate_dmgr.py --warmup            # force warm-up (e.g. on a fresh machine)

Flags:
    --gpu-index N    GPU device index to monitor (default: 0)
    --full           Include 12k token measurement in the sweep
    --warmup         Force a warm-up pass to trigger Triton kernel compilation
                     (skip-warmup is the default since kernels persist in cache)
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import pynvml

# ── CONFIG ────────────────────────────────────────────────────────────────────
FAIRSEQ_BIN  = "/home/e20363/miniconda3/envs/museformer/bin/fairseq-interactive"
CHECKPOINT   = "checkpoints/mf-lmd6remi-1/checkpoint_best.pt"
DATA_BIN     = "data-bin/lmd6remi"
POLL_S       = 0.05      # pynvml polling interval (50 ms)
SEED         = 42
TARGET_LENGTHS_DEFAULT = [1000, 2000, 4000, 8000, 12000]
TARGET_LENGTHS_FULL    = [1000, 2000, 4000, 8000, 12000]
DMGR_L1, DMGR_L2 = 1000, 12000
READY_MARKER = "Type the input sentence and press return"

# Triton CUDA kernel compilation requires these on this machine
_ENV = dict(os.environ, CC="/usr/bin/gcc", CXX="/usr/bin/g++")

# ── CLI ARGS ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Museformer DMGR evaluation")
    p.add_argument("--gpu-index", type=int, default=0,
                   help="GPU device index to monitor (default: 0)")
    p.add_argument("--full", action="store_true",
                   help="Include 12k token measurement (adds ~1.5h to run time)")
    p.add_argument("--skip-warmup", action="store_true", default=True,
                   help="Skip warm-up pass (default: True — Triton caches persist across runs)")
    p.add_argument("--warmup", dest="skip_warmup", action="store_false",
                   help="Force a warm-up pass even if Triton is already compiled")
    return p.parse_args()

# ── pynvml helpers ────────────────────────────────────────────────────────────
def get_used_bytes(handle):
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used

def check_gpu_exclusive(handle, gpu_index):
    """Warn if other processes are on this GPU; ask whether to continue."""
    own_pid = os.getpid()
    others = [p for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
              if p.pid != own_pid]
    if not others:
        return
    print(f"\n[WARNING] Other processes detected on GPU {gpu_index}:")
    for p in others:
        print(f"  PID {p.pid}   VRAM {p.usedGpuMemory // 1024**2} MB")
    print("  pynvml reports total GPU VRAM — their footprint will inflate all readings.")
    print("  For accurate DMGR, wait for those jobs to finish or pick a free GPU with")
    print("  --gpu-index.\n")
    ans = input("  Continue anyway? [y/N]: ").strip().lower()
    if ans != "y":
        print("Aborted.")
        sys.exit(0)

# ── Peak memory monitor ────────────────────────────────────────────────────────
class PeakMemoryMonitor:
    """Background thread: polls pynvml and tracks the maximum VRAM seen."""
    def __init__(self, handle):
        self.handle = handle
        self._peak  = 0
        self._stop  = threading.Event()
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

# ── Subprocess helpers ────────────────────────────────────────────────────────
def _make_cmd(min_len, max_len):
    return [
        FAIRSEQ_BIN, DATA_BIN,
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

def _drain(pipe):
    """Consume a pipe to prevent the subprocess from blocking on a full buffer."""
    try:
        for _ in pipe:
            pass
    except Exception:
        pass

# ── Idle VRAM measurement ──────────────────────────────────────────────────────
def measure_idle_vram(handle):
    """
    Sample VRAM at the exact moment fairseq signals it is ready for input —
    after the model is fully loaded but before any generation token is produced.
    """
    print("Measuring idle VRAM (model-load baseline)...")
    proc = subprocess.Popen(
        _make_cmd(1, 1),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_ENV,
    )

    idle_bytes = None
    for line in proc.stdout:
        if READY_MARKER in line:
            idle_bytes = get_used_bytes(handle)
            # Send the empty prompt so the process can finish and exit cleanly
            try:
                proc.stdin.write("\n")
                proc.stdin.flush()
                proc.stdin.close()
            except BrokenPipeError:
                pass
            break

    # Drain remaining stdout so the subprocess is not blocked on a full pipe
    drain_thread = threading.Thread(target=_drain, args=(proc.stdout,), daemon=True)
    drain_thread.start()
    proc.wait()
    drain_thread.join(timeout=15)

    if idle_bytes is None:
        print("[WARNING] Ready marker not found in fairseq output.")
        print("          Falling back to current VRAM as idle baseline.")
        idle_bytes = get_used_bytes(handle)

    return idle_bytes

# ── Per-length peak VRAM measurement ─────────────────────────────────────────
def measure_peak_vram(handle, target_len):
    """
    Generate target_len tokens and return the peak VRAM observed during the run.
    The monitor starts before the subprocess so it catches the full memory curve.
    Output is discarded; only VRAM is recorded.
    """
    monitor = PeakMemoryMonitor(handle)
    monitor.start()

    proc = subprocess.Popen(
        _make_cmd(target_len, target_len),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_ENV,
    )
    try:
        proc.stdin.write(b"\n")
        proc.stdin.flush()
        proc.stdin.close()
    except BrokenPipeError:
        pass

    # Poll instead of blocking wait — print elapsed time every 60s so the
    # terminal doesn't look frozen during long generations (30-60 min each).
    t0 = time.time()
    while proc.poll() is None:
        time.sleep(60)
        if proc.poll() is None:
            print(f" {int(time.time()-t0)}s...", end="", flush=True)

    return monitor.stop()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Make fairseq-interactive use the same GPU that pynvml will monitor.
    # Without this, fairseq always defaults to device_id=0 regardless of --gpu-index.
    _ENV['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)

    pynvml.nvmlInit()
    handle   = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    total_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).total // 1024**2

    target_lengths = TARGET_LENGTHS_FULL if args.full else TARGET_LENGTHS_DEFAULT

    print("=" * 55)
    print("Museformer DMGR Evaluation")
    print("=" * 55)
    print(f"GPU {args.gpu_index}: {gpu_name}  ({total_mb} MB total)")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Lengths    : {target_lengths}")
    print(f"DMGR range : {DMGR_L1} → {DMGR_L2} tokens")
    print()

    check_gpu_exclusive(handle, args.gpu_index)

    # ── Warm-up ──────────────────────────────────────────────────────────────
    if not args.skip_warmup:
        print("Warm-up pass (triggers Triton kernel compilation if not cached)...")
        print("This may take several minutes on first run — subsequent runs are faster.")
        t0 = time.time()
        subprocess.run(
            _make_cmd(200, 400),
            input=b"\n",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_ENV,
        )
        print(f"  Warm-up complete ({time.time() - t0:.0f}s)\n")
    else:
        print("Skipping warm-up (--skip-warmup).\n")

    # ── Idle VRAM baseline ───────────────────────────────────────────────────
    idle_bytes = measure_idle_vram(handle)
    idle_mb    = idle_bytes / 1024**2
    print(f"  → Idle VRAM after model load: {idle_mb:.1f} MB\n")

    # ── Per-length sweep ─────────────────────────────────────────────────────
    print("Running generation sweep...")
    results = {}
    for L in target_lengths:
        print(f"  L={L:>6} tokens  ...", end="", flush=True)
        t0 = time.time()
        peak_bytes = measure_peak_vram(handle, L)
        elapsed    = time.time() - t0
        incr_mb    = (peak_bytes - idle_bytes) / 1024**2

        results[L] = {
            "peak_vram_mb":        round(peak_bytes / 1024**2, 2),
            "incremental_vram_mb": round(incr_mb, 2),
            "generation_time_s":   round(elapsed, 1),
        }
        print(f"  peak={results[L]['peak_vram_mb']:.1f} MB  "
              f"incr={incr_mb:+.1f} MB  "
              f"({elapsed:.0f}s)")

    # ── DMGR computation ─────────────────────────────────────────────────────
    dm   = results[DMGR_L2]["incremental_vram_mb"] - results[DMGR_L1]["incremental_vram_mb"]
    dmgr = dm / ((DMGR_L2 - DMGR_L1) / 1000)

    summary = {
        "gpu":                   gpu_name,
        "checkpoint":            CHECKPOINT,
        "idle_vram_mb":          round(idle_mb, 2),
        "DMGR_L1":               DMGR_L1,
        "DMGR_L2":               DMGR_L2,
        "DMGR_MB_per_1k_tokens": round(dmgr, 4),
        "per_length":            results,
    }

    os.makedirs("output_log", exist_ok=True)
    out_path = "output_log/dmgr_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Final report ─────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("DMGR RESULTS — Museformer")
    print("=" * 55)
    print(f"GPU                        : {gpu_name}")
    print(f"Idle VRAM (model loaded)   : {idle_mb:.1f} MB")
    print()
    print(f"  {'Length (tokens)':<20} {'Peak VRAM (MB)':<18} {'Incremental (MB)'}")
    print(f"  {'-'*18:<20} {'-'*14:<18} {'-'*16}")
    for L, v in results.items():
        print(f"  {L:<20} {v['peak_vram_mb']:<18.1f} {v['incremental_vram_mb']:.1f}")
    print()
    print(f"DMGR ({DMGR_L1}→{DMGR_L2} tokens)     : {dmgr:.4f} MB / 1k tokens")
    print()
    print(f"Results saved → {out_path}")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
