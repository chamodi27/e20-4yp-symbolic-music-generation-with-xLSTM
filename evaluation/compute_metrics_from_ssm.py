#!/usr/bin/env python3
"""
Compute structure-related metrics from Self-Similarity Matrices (SSMs).

Inputs:
- Folder of .npy SSM files (NxN) produced by compute_bar_ssm_midi_folder.py

Outputs:
- metrics.csv containing:
  - N (bars)
  - avg_offdiag
  - rep_density (threshold tau, ignoring near-diagonal band k)
  - struct_entropy
  - block_coherence (KMeans-based)

Usage:
  python compute_metrics_from_ssm_folder.py \
    --ssm_dir "/path/to/bar_ssm_outputs" \
    --out_csv "/path/to/metrics.csv" \
    --tau 0.75 \
    --ignore_band 1 \
    --k_clusters 4
"""

import argparse
from pathlib import Path
import numpy as np
import csv

from sklearn.cluster import KMeans


def load_ssm(path: Path) -> np.ndarray:
    S = np.load(path)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"SSM must be square, got {S.shape} for {path.name}")
    return S.astype(np.float32)


def avg_offdiag(S: np.ndarray) -> float:
    n = S.shape[0]
    if n <= 1:
        return float("nan")
    total = S.sum() - np.trace(S)
    return float(total / (n * (n - 1)))


def rep_density(S: np.ndarray, tau: float = 0.75, ignore_band: int = 1) -> float:
    """
    Fraction of strong similarities off-diagonal, excluding a |i-j| <= ignore_band band.
    """
    n = S.shape[0]
    if n <= 2:
        return float("nan")

    mask = np.ones((n, n), dtype=bool)

    # remove diagonal
    np.fill_diagonal(mask, False)

    # remove near-diagonal band (local continuity)
    if ignore_band > 0:
        for d in range(1, ignore_band + 1):
            idx = np.arange(n - d)
            mask[idx, idx + d] = False
            mask[idx + d, idx] = False

    vals = S[mask]
    if vals.size == 0:
        return float("nan")

    return float((vals >= tau).mean())


def structural_entropy(S: np.ndarray, ignore_band: int = 1, eps: float = 1e-12) -> float:
    """
    Shannon entropy of off-diagonal similarities (upper triangle), treated as a distribution.
    """
    n = S.shape[0]
    if n <= 2:
        return float("nan")

    # build mask for upper triangle excluding diagonal and near-diagonal band
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= ignore_band:
                continue
            mask[i, j] = True

    vals = S[mask]
    vals = np.clip(vals, 0.0, None)  # keep non-negative for probability construction
    total = vals.sum()
    if total <= eps:
        return float("nan")

    p = vals / total
    H = -np.sum(p * np.log(p + eps))
    return float(H)


def block_coherence(S: np.ndarray, k_clusters: int = 4, ignore_band: int = 1, random_state: int = 0) -> float:
    """
    Cluster bars into K groups using SSM row-vectors, then compute:
      mean(within-cluster similarity) - mean(between-cluster similarity)
    excluding diagonal and near-diagonal band.
    """
    n = S.shape[0]
    if n < k_clusters or n <= 2:
        return float("nan")

    # Use rows as embeddings
    X = S.copy()

    km = KMeans(n_clusters=k_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)

    mask_valid = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask_valid, False)

    if ignore_band > 0:
        for d in range(1, ignore_band + 1):
            idx = np.arange(n - d)
            mask_valid[idx, idx + d] = False
            mask_valid[idx + d, idx] = False

    # within vs between
    within_mask = np.zeros((n, n), dtype=bool)
    between_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            if not mask_valid[i, j]:
                continue
            if labels[i] == labels[j]:
                within_mask[i, j] = True
            else:
                between_mask[i, j] = True

    within_vals = S[within_mask]
    between_vals = S[between_mask]

    if within_vals.size == 0 or between_vals.size == 0:
        return float("nan")

    return float(within_vals.mean() - between_vals.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssm_dir", required=True, type=str, help="Folder containing .npy SSM files")
    ap.add_argument("--out_csv", required=True, type=str, help="Output CSV path")
    ap.add_argument("--tau", type=float, default=0.75, help="Threshold for repetition density")
    ap.add_argument("--ignore_band", type=int, default=1, help="Ignore |i-j| <= k band around diagonal")
    ap.add_argument("--k_clusters", type=int, default=4, help="K for block coherence clustering")
    args = ap.parse_args()

    ssm_dir = Path(args.ssm_dir).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(ssm_dir.glob("*.npy"))
    if not npy_files:
        print(f"[INFO] No .npy files found in {ssm_dir}")
        return

    rows = []
    for f in npy_files:
        try:
            S = load_ssm(f)
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")
            continue

        n = S.shape[0]
        row = {
            "file": f.name,
            "bars_N": n,
            "avg_offdiag": avg_offdiag(S),
            "rep_density": rep_density(S, tau=args.tau, ignore_band=args.ignore_band),
            "struct_entropy": structural_entropy(S, ignore_band=args.ignore_band),
            "block_coherence": block_coherence(S, k_clusters=args.k_clusters, ignore_band=args.ignore_band),
        }
        rows.append(row)

    # write CSV
    fieldnames = ["file", "bars_N", "avg_offdiag", "rep_density", "struct_entropy", "block_coherence"]
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
