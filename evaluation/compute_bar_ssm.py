#!/usr/bin/env python3
"""
Compute BAR-LEVEL Self-Similarity Matrices (SSMs) for a folder of MIDI files.

Core idea (as used in music structure analysis literature):
Given feature vectors x_1..x_N for successive time units, the Self-Similarity Matrix is:
    S(i, j) = s(x_i, x_j)
Here:
- time units = bars
- features   = bar-level pitch-class (12-D "chroma-like") or piano-roll (128-D)
- s          = cosine similarity

Inputs:
- A directory containing MIDI files (searched recursively)

Outputs (per MIDI):
- .npy : bar-level SSM matrix
- .png : visualization heatmap

Usage example:
python compute_bar_ssm_midi_folder.py \
  --input_dir "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/notebooks/xLSTM-2/generated_batch_20260208_041715-xlstm_lmd_512d_2048ctx_12b-checkpoint-84000" \
  --output_dir "./bar_ssm_outputs" \
  --feature chroma \
  --binarize

Notes:
- Bar boundaries use pretty_midi.get_downbeats(). If a MIDI has no downbeats, we fall back to
  an approximate 4/4 bar grid using the estimated tempo (best-effort).
"""

import argparse
from pathlib import Path
import numpy as np
import pretty_midi
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def list_midi_files(input_dir: Path) -> list[Path]:
    exts = {".mid", ".midi"}
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def get_bar_boundaries(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    """
    Return an array of bar boundary times [t0, t1, ..., tK]
    such that bars are [t0,t1), [t1,t2), ... .
    Prefer get_downbeats(). If unavailable, fallback to an approximate 4/4 grid.
    """
    downbeats = np.array(midi.get_downbeats(), dtype=np.float64)

    # If downbeats exist and look usable
    if downbeats.size >= 2:
        # Ensure sorted + unique
        downbeats = np.unique(np.sort(downbeats))
        return downbeats

    # Fallback: approximate bars using tempo and 4/4
    end_time = midi.get_end_time()
    if end_time <= 0:
        return np.array([], dtype=np.float64)

    # Estimate tempo (pretty_midi returns array; take first / median)
    tempos, tempo_times = midi.get_tempo_changes()
    if tempos.size == 0:
        tempo = 120.0  # last-resort fallback
    else:
        tempo = float(np.median(tempos))

    seconds_per_beat = 60.0 / tempo
    beats_per_bar = 4.0
    seconds_per_bar = seconds_per_beat * beats_per_bar

    # Build bar grid from 0 to end_time
    num_bars = int(np.ceil(end_time / seconds_per_bar))
    boundaries = np.arange(0, num_bars + 1, dtype=np.float64) * seconds_per_bar
    # Clip last boundary to end_time (optional)
    boundaries[-1] = max(boundaries[-1], end_time)
    return boundaries


def piano_roll_from_midi(midi: pretty_midi.PrettyMIDI, fs: int = 50) -> np.ndarray:
    """
    Returns piano roll as (frames, 128) float32, with frame i centered at time i/fs.
    """
    pr = midi.get_piano_roll(fs=fs).T.astype(np.float32)  # (frames, 128)
    return pr


def bar_features_from_pianoroll(
    pr: np.ndarray,
    bar_times: np.ndarray,
    fs: int,
    feature: str,
    binarize: bool,
) -> np.ndarray:
    """
    Build one feature vector per bar by aggregating piano-roll frames that fall within each bar.

    pr: (frames, 128)
    bar_times: [t0..tK] boundaries, K bars
    feature:
      - "chroma": 12D pitch-class sum per bar
      - "pianoroll": 128D pitch activity sum per bar
    """
    if pr.size == 0 or bar_times.size < 2:
        return np.zeros((0, 12 if feature == "chroma" else 128), dtype=np.float32)

    if binarize:
        pr = (pr > 0).astype(np.float32)

    num_bars = bar_times.size - 1
    out_dim = 12 if feature == "chroma" else 128
    X = np.zeros((num_bars, out_dim), dtype=np.float32)

    # For each bar: collect frames whose time in [t_start, t_end)
    # frame index i corresponds to time i/fs
    for b in range(num_bars):
        t0, t1 = float(bar_times[b]), float(bar_times[b + 1])
        i0 = int(np.floor(t0 * fs))
        i1 = int(np.ceil(t1 * fs))
        i0 = max(i0, 0)
        i1 = min(i1, pr.shape[0])

        if i1 <= i0:
            continue

        chunk = pr[i0:i1, :]  # (frames_in_bar, 128)
        bar_sum = chunk.sum(axis=0)  # (128,)

        if feature == "pianoroll":
            X[b, :] = bar_sum
        else:
            # fold to pitch classes
            chroma = np.zeros((12,), dtype=np.float32)
            for pitch in range(128):
                chroma[pitch % 12] += bar_sum[pitch]
            X[b, :] = chroma

    return X


def compute_ssm(X: np.ndarray) -> np.ndarray:
    """
    Cosine similarity SSM.
    """
    if X.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    Xn = l2_normalize(X)
    return cosine_similarity(Xn).astype(np.float32)


def plot_ssm(ssm: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(ssm, origin="lower", aspect="auto")
    plt.title(title)
    plt.xlabel("Bar index")
    plt.ylabel("Bar index")
    plt.colorbar(label="Cosine similarity")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def process_one_midi(
    midi_path: Path,
    out_dir: Path,
    fs: int,
    feature: str,
    binarize: bool,
) -> None:
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"[WARN] Failed to read {midi_path.name}: {e}")
        return

    bar_times = get_bar_boundaries(midi)
    if bar_times.size < 2:
        print(f"[WARN] No usable bar boundaries for {midi_path.name} (skipping).")
        return

    pr = piano_roll_from_midi(midi, fs=fs)
    X = bar_features_from_pianoroll(pr, bar_times, fs=fs, feature=feature, binarize=binarize)
    ssm = compute_ssm(X)

    stem = midi_path.stem
    out_npy = out_dir / f"{stem}__barSSM_{feature}_fs{fs}.npy"
    out_png = out_dir / f"{stem}__barSSM_{feature}_fs{fs}.png"

    np.save(out_npy, ssm)
    plot_ssm(ssm, out_png, title=f"Bar-level SSM ({feature}) - {stem} | bars={ssm.shape[0]}")

    print(f"[OK] {midi_path.name} -> bars={ssm.shape[0]} | saved: {out_npy.name}, {out_png.name}")


def main():
    parser = argparse.ArgumentParser(description="Compute BAR-level SSMs for MIDI files in a folder.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input folder (searched recursively).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for .npy and .png.")
    parser.add_argument("--fs", type=int, default=50, help="Piano-roll frames per second (mapping time->frames).")
    parser.add_argument("--feature", choices=["chroma", "pianoroll"], default="chroma",
                        help="Bar feature type.")
    parser.add_argument("--binarize", action="store_true",
                        help="Binarize piano roll before aggregation (often cleaner structure).")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    safe_mkdir(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    midi_files = list_midi_files(input_dir)
    if not midi_files:
        print(f"[INFO] No MIDI files found in: {input_dir}")
        return

    print(f"[INFO] Found {len(midi_files)} MIDI files under: {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Settings: fs={args.fs}, feature={args.feature}, binarize={args.binarize}")

    for mp in midi_files:
        process_one_midi(
            midi_path=mp,
            out_dir=output_dir,
            fs=args.fs,
            feature=args.feature,
            binarize=args.binarize,
        )


if __name__ == "__main__":
    main()
