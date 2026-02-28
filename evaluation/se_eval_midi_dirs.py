"""
Similarity Error (SE) Evaluation for Symbolic Music Generation

This script computes a structure-based Similarity Error (SE) metric between
two sets of MIDI files:
    (1) real / validation music
    (2) generated music produced by a model

The evaluation is performed directly in the symbolic (MIDI) domain and is
independent of the tokenization scheme or training framework used by the
models. This makes the metric suitable for comparing outputs from different
architectures (e.g., Museformer, xLSTM-based models, and other baselines).

Methodology:
- Each MIDI file is segmented into bars using time-signature information
  (defaulting to 4/4 when unavailable).
- Each bar is represented by a 12-dimensional pitch-class histogram weighted
  by note duration.
- For each bar-distance t = 1..T, cosine similarity is computed between bar i
  and bar (i - t), and averaged over all valid bar pairs and pieces.
- Similarity Error (SE) is defined as the mean absolute difference between the
  similarity curves of real and generated music.

Lower SE values indicate that the generated music exhibits structural
similarity patterns closer to those observed in real music.

Directory Structure (default):
- evaluation/data/real_dir : validation/reference MIDI files
- evaluation/data/gen_dir  : generated MIDI files
- evaluation/results/      : output JSON with SE and similarity curves
"""

from pathlib import Path
import os
import json
import argparse
from typing import List
import numpy as np
import miditoolkit


# =========================
# PATH SETUP (PROJECT-AWARE)
# =========================

BASE_DIR = Path(__file__).resolve().parent          # evaluation/
DATA_DIR = BASE_DIR / "data"
REAL_DIR_DEFAULT = DATA_DIR / "real_dir"
GEN_DIR_DEFAULT = DATA_DIR / "gen_dir"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =========================
# UTILITIES
# =========================

def list_midis(folder: Path) -> List[Path]:
    return sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in [".mid", ".midi"]
    ])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_end_tick(midi: miditoolkit.MidiFile) -> int:
    return max(
        (int(n.end) for inst in midi.instruments for n in inst.notes),
        default=0
    )


def beats_per_bar(num, den):
    return num * (4.0 / den)


def bar_length_ticks(midi, num, den):
    return int(round(beats_per_bar(num, den) * midi.ticks_per_beat))


# =========================
# BAR FEATURE EXTRACTION
# =========================

def compute_bar_pitch_features(midi_path: Path, max_bars=256):
    midi = miditoolkit.MidiFile(str(midi_path))
    notes = [n for inst in midi.instruments for n in inst.notes]
    notes.sort(key=lambda n: int(n.start))

    if not notes:
        return []

    ts = midi.time_signature_changes or [
        miditoolkit.TimeSignature(4, 4, 0)
    ]

    end_tick = get_end_tick(midi)
    bars = []
    bar_count = 0

    for i, tsc in enumerate(ts):
        seg_start = int(tsc.time)
        seg_end = int(ts[i + 1].time) if i + 1 < len(ts) else end_tick
        blen = bar_length_ticks(midi, tsc.numerator, tsc.denominator)

        cur = seg_start
        while cur < seg_end and bar_count < max_bars:
            hist = np.zeros(12)
            bar_start = cur
            bar_end = min(cur + blen, seg_end)

            for n in notes:
                if n.end <= bar_start:
                    continue
                if n.start >= bar_end:
                    break
                overlap = max(0, min(n.end, bar_end) - max(n.start, bar_start))
                if overlap > 0:
                    hist[n.pitch % 12] += overlap

            if hist.sum() > 0:
                hist /= hist.sum()

            bars.append(hist)
            bar_count += 1
            cur += blen

    return bars


# =========================
# SIMILARITY COMPUTATION
# =========================

def similarity_curve(midi_files: List[Path], T=40, max_bars=256):
    sum_sim = np.zeros(T)
    count = np.zeros(T)

    for m in midi_files:
        bars = compute_bar_pitch_features(m, max_bars)
        B = len(bars)
        for t in range(1, min(T + 1, B)):
            for i in range(t, B):
                s = cosine_sim(bars[i], bars[i - t])
                sum_sim[t - 1] += s
                count[t - 1] += 1

    return np.divide(sum_sim, count, out=np.zeros_like(sum_sim), where=count != 0)


def similarity_error(L_real, L_gen):
    return float(np.mean(np.abs(L_real - L_gen)))


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=Path, default=REAL_DIR_DEFAULT)
    parser.add_argument("--gen_dir", type=Path, default=GEN_DIR_DEFAULT)
    parser.add_argument("--T", type=int, default=40)
    parser.add_argument("--max_bars", type=int, default=256)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "se_results.json")
    args = parser.parse_args()

    real_midis = list_midis(args.real_dir)
    gen_midis = list_midis(args.gen_dir)

    if not real_midis:
        raise RuntimeError(f"No MIDI files in {args.real_dir}")
    if not gen_midis:
        raise RuntimeError(f"No MIDI files in {args.gen_dir}")

    print(f"Real MIDIs: {len(real_midis)}")
    print(f"Gen  MIDIs: {len(gen_midis)}")

    L_real = similarity_curve(real_midis, args.T, args.max_bars)
    L_gen = similarity_curve(gen_midis, args.T, args.max_bars)

    se = similarity_error(L_real, L_gen)

    results = {
        "SE": se,
        "T": args.T,
        "max_bars": args.max_bars,
        "real_dir": str(args.real_dir),
        "gen_dir": str(args.gen_dir),
        "L_real": L_real.tolist(),
        "L_gen": L_gen.tolist()
    }

    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n==============================")
    print(f"Similarity Error (SE): {se:.6f}")
    print(f"Saved to: {args.out}")
    print("==============================\n")


if __name__ == "__main__":
    main()
