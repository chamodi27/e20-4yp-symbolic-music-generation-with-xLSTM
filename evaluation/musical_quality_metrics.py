#!/usr/bin/env python3
"""
Compute MUSICAL-QUALITY metrics from a folder of MIDI files.

Implements a compact set of MIDI-friendly musical quality metrics:
HARMONY / TONALITY
  1) Key stability:
       - Estimate a GLOBAL key from the whole piece using Krumhansl-Schmuckler profiles
       - Estimate key per bar
       - Report:
           * key_stability = fraction of bars whose key == global key
           * key_change_rate = (# key changes between consecutive bars) / (bars-1)

  2) Chord / pitch-class diversity:
       - For each bar, build a 12-D pitch-class histogram (from all notes)
       - Binarize to a pitch-class set fingerprint
       - chord_diversity = (# unique fingerprints) / (# bars)

MELODY (heuristic)
  3) Pitch range:
       - Extract a melody line by taking the highest pitch note at each onset time
       - pitch_range = max_pitch - min_pitch

  4) Stepwise motion ratio:
       - From consecutive melody notes, compute interval in semitones
       - stepwise_ratio = fraction with |Δpitch| <= 2

RHYTHM
  5) Note density:
       - notes_per_bar = number of note onsets per bar (all instruments)
       - report mean and std across bars

  6) Duration entropy:
       - Collect note durations (in beats) (all instruments)
       - Quantize to nearest common duration bin (e.g., 1/16, 1/8, 1/4, 1/2, 1, 2, 4, etc.)
       - duration_entropy = Shannon entropy of the discrete duration distribution

Outputs:
  - A single CSV with one row per MIDI.

Usage:
  python midi_quality_metrics.py \
    --input_dir "/path/to/generated_midis" \
    --out_csv "/path/to/midi_quality_metrics.csv"

Notes:
- Bar boundaries use pretty_midi.get_downbeats(). If missing, falls back to an approximate 4/4 grid.
- Key estimation is approximate (pitch-class only). It’s consistent for comparisons across models.
"""

import argparse
from pathlib import Path
import numpy as np
import pretty_midi
import csv
import math


# -------------------------
# Utilities
# -------------------------

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_midi_files(input_dir: Path) -> list[Path]:
    exts = {".mid", ".midi"}
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def shannon_entropy(counts: np.ndarray, eps: float = 1e-12) -> float:
    total = float(np.sum(counts))
    if total <= eps:
        return float("nan")
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + eps)))


def estimate_median_tempo(midi: pretty_midi.PrettyMIDI, default: float = 120.0) -> float:
    tempos, _ = midi.get_tempo_changes()
    if tempos is None or len(tempos) == 0:
        return default
    t = float(np.median(tempos))
    return t if t > 1e-3 else default


def get_bar_boundaries(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    """
    Return bar boundary times [t0, t1, ..., tK] -> bars are [t0,t1), [t1,t2), ...
    Prefer downbeats, else fallback to approximate 4/4 grid.
    """
    downbeats = np.array(midi.get_downbeats(), dtype=np.float64)

    if downbeats.size >= 2:
        downbeats = np.unique(np.sort(downbeats))
        return downbeats

    end_time = midi.get_end_time()
    if end_time <= 0:
        return np.array([], dtype=np.float64)

    tempo = estimate_median_tempo(midi)
    seconds_per_beat = 60.0 / tempo
    beats_per_bar = 4.0
    seconds_per_bar = seconds_per_beat * beats_per_bar

    num_bars = int(np.ceil(end_time / seconds_per_bar))
    boundaries = np.arange(0, num_bars + 1, dtype=np.float64) * seconds_per_bar
    boundaries[-1] = max(boundaries[-1], end_time)
    return boundaries


def collect_all_notes(midi: pretty_midi.PrettyMIDI):
    notes = []
    for inst in midi.instruments:
        for n in inst.notes:
            notes.append(n)
    return notes


# -------------------------
# Pitch-class features
# -------------------------

def pitch_class_hist_for_interval(notes, t0: float, t1: float) -> np.ndarray:
    """
    Build a 12-D pitch-class histogram for notes that overlap [t0, t1).
    Uses velocity-weighted overlap duration as weight (simple, stable).
    """
    pc = np.zeros(12, dtype=np.float32)
    if t1 <= t0:
        return pc

    for n in notes:
        # overlap with bar interval
        start = max(n.start, t0)
        end = min(n.end, t1)
        if end > start:
            dur = end - start
            pc[n.pitch % 12] += float(dur) * float(n.velocity if n.velocity is not None else 100.0)
    return pc


# -------------------------
# Key estimation (Krumhansl-Schmuckler profiles)
# -------------------------

# Standard K-S key profiles (major/minor) for pitch classes C..B
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def rotate_profile(profile: np.ndarray, k: int) -> np.ndarray:
    return np.roll(profile, k)


def estimate_key_from_pc(pc: np.ndarray) -> str:
    """
    Estimate key label like 'C:maj' or 'A:min' from a 12D pitch-class vector
    using correlation with rotated K-S profiles.
    """
    if pc is None or pc.size != 12:
        return "UNK"

    if np.allclose(pc, 0):
        return "UNK"

    # Normalize
    x = pc.astype(np.float32)
    x = x / (np.linalg.norm(x) + 1e-8)

    best_score = -1e9
    best_key = "UNK"

    for k in range(12):
        maj = rotate_profile(KS_MAJOR, k)
        minp = rotate_profile(KS_MINOR, k)

        maj = maj / (np.linalg.norm(maj) + 1e-8)
        minp = minp / (np.linalg.norm(minp) + 1e-8)

        score_maj = float(np.dot(x, maj))
        score_min = float(np.dot(x, minp))

        if score_maj > best_score:
            best_score = score_maj
            best_key = f"{KEY_NAMES[k]}:maj"
        if score_min > best_score:
            best_score = score_min
            best_key = f"{KEY_NAMES[k]}:min"

    return best_key


def key_stability_metrics(notes, bar_times: np.ndarray):
    """
    Compute:
      - global_key
      - key_stability: fraction of bars matching global
      - key_change_rate: changes between consecutive bars
    """
    if bar_times.size < 2:
        return "UNK", float("nan"), float("nan")

    # global PC over whole piece
    global_pc = pitch_class_hist_for_interval(notes, float(bar_times[0]), float(bar_times[-1]))
    global_key = estimate_key_from_pc(global_pc)
    if global_key == "UNK":
        return "UNK", float("nan"), float("nan")

    # per-bar keys
    bar_keys = []
    for i in range(bar_times.size - 1):
        pc = pitch_class_hist_for_interval(notes, float(bar_times[i]), float(bar_times[i + 1]))
        bar_keys.append(estimate_key_from_pc(pc))

    # remove UNK bars from stability denominator? (better: count them as mismatch)
    matches = sum(1 for k in bar_keys if k == global_key)
    key_stability = matches / max(len(bar_keys), 1)

    # key change rate (ignore UNK transitions by treating UNK as its own label)
    changes = 0
    for i in range(1, len(bar_keys)):
        if bar_keys[i] != bar_keys[i - 1]:
            changes += 1
    key_change_rate = changes / max((len(bar_keys) - 1), 1)

    return global_key, float(key_stability), float(key_change_rate)


def chord_diversity(notes, bar_times: np.ndarray, bin_thresh: float = 1e-6) -> float:
    """
    Compute unique pitch-class set fingerprints per bar / #bars.
    """
    if bar_times.size < 2:
        return float("nan")

    fingerprints = []
    for i in range(bar_times.size - 1):
        pc = pitch_class_hist_for_interval(notes, float(bar_times[i]), float(bar_times[i + 1]))
        # binarize (presence/absence)
        fp = tuple((pc > bin_thresh).astype(np.int8).tolist())
        fingerprints.append(fp)

    if len(fingerprints) == 0:
        return float("nan")

    unique = len(set(fingerprints))
    return float(unique / len(fingerprints))


# -------------------------
# Melody extraction + metrics (heuristic)
# -------------------------

def extract_melody_pitches(notes):
    """
    Simple melody heuristic:
    - group notes by onset time (rounded), take highest pitch at each onset.
    - returns a list of pitches in onset order.

    Works reasonably for many generated MIDIs without explicit melody track.
    """
    if not notes:
        return []

    # Group by onset time with rounding for stability
    onset_map = {}
    for n in notes:
        t = round(float(n.start), 3)  # 1ms-ish granularity
        onset_map.setdefault(t, []).append(n.pitch)

    melody = []
    for t in sorted(onset_map.keys()):
        melody.append(int(max(onset_map[t])))
    return melody


def pitch_range_and_stepwise_ratio(melody_pitches):
    if melody_pitches is None or len(melody_pitches) == 0:
        return float("nan"), float("nan")

    prange = int(max(melody_pitches) - min(melody_pitches))

    if len(melody_pitches) < 2:
        return float(prange), float("nan")

    intervals = [abs(melody_pitches[i] - melody_pitches[i - 1]) for i in range(1, len(melody_pitches))]
    stepwise = sum(1 for d in intervals if d <= 2)
    stepwise_ratio = stepwise / len(intervals)

    return float(prange), float(stepwise_ratio)


# -------------------------
# Rhythm metrics
# -------------------------

def note_density(notes, bar_times: np.ndarray):
    """
    Notes per bar using note onsets.
    Return mean and std across bars.
    """
    if bar_times.size < 2:
        return float("nan"), float("nan")

    # For faster lookup, get onset times
    onsets = np.array([float(n.start) for n in notes], dtype=np.float64)
    if onsets.size == 0:
        return 0.0, 0.0

    counts = []
    for i in range(bar_times.size - 1):
        t0, t1 = float(bar_times[i]), float(bar_times[i + 1])
        c = int(np.sum((onsets >= t0) & (onsets < t1)))
        counts.append(c)

    if len(counts) == 0:
        return float("nan"), float("nan")

    return float(np.mean(counts)), float(np.std(counts, ddof=0))


DURATION_BINS_BEATS = np.array([
    1/16, 1/8, 3/16, 1/4, 3/8, 1/2,
    3/4, 1.0, 1.5, 2.0, 3.0, 4.0
], dtype=np.float32)


def duration_entropy(notes, midi: pretty_midi.PrettyMIDI):
    """
    Compute entropy over quantized note durations in beats.
    """
    if not notes:
        return float("nan")

    tempo = estimate_median_tempo(midi)
    sec_per_beat = 60.0 / tempo

    # Collect durations (beats)
    durs = []
    for n in notes:
        dur_sec = float(max(0.0, n.end - n.start))
        if dur_sec <= 0:
            continue
        dur_beats = dur_sec / sec_per_beat
        durs.append(dur_beats)

    if len(durs) == 0:
        return float("nan")

    durs = np.array(durs, dtype=np.float32)

    # Quantize to nearest duration bin
    # For each duration, choose index with minimal absolute error
    idx = np.argmin(np.abs(durs[:, None] - DURATION_BINS_BEATS[None, :]), axis=1)
    counts = np.bincount(idx, minlength=len(DURATION_BINS_BEATS)).astype(np.float32)

    return shannon_entropy(counts)


# -------------------------
# Main processing per MIDI
# -------------------------

def compute_metrics_for_midi(midi_path: Path):
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        return {"file": midi_path.name, "error": f"read_fail: {e}"}

    notes = collect_all_notes(midi)
    bar_times = get_bar_boundaries(midi)
    bars_N = int(max(0, bar_times.size - 1))

    # Key / harmony
    global_key, key_stab, key_change = key_stability_metrics(notes, bar_times)
    cdiv = chord_diversity(notes, bar_times)

    # Melody
    melody_pitches = extract_melody_pitches(notes)
    prange, step_ratio = pitch_range_and_stepwise_ratio(melody_pitches)

    # Rhythm
    nd_mean, nd_std = note_density(notes, bar_times)
    dent = duration_entropy(notes, midi)

    return {
        "file": midi_path.name,
        "bars_N": bars_N,
        "global_key": global_key,
        "key_stability": key_stab,
        "key_change_rate": key_change,
        "chord_diversity": cdiv,
        "pitch_range": prange,
        "stepwise_ratio": step_ratio,
        "note_density_mean": nd_mean,
        "note_density_std": nd_std,
        "duration_entropy": dent,
        "error": ""
    }


def main():
    ap = argparse.ArgumentParser(description="Compute musical-quality metrics from MIDI folder.")
    ap.add_argument("--input_dir", required=True, type=str, help="Folder containing MIDI files (recursive).")
    ap.add_argument("--out_csv", required=True, type=str, help="Output CSV path.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    safe_mkdir(out_csv.parent)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    midi_files = list_midi_files(input_dir)
    if not midi_files:
        print(f"[INFO] No MIDI files found under: {input_dir}")
        return

    rows = []
    for mp in midi_files:
        rows.append(compute_metrics_for_midi(mp))

    fieldnames = [
        "file", "bars_N", "global_key",
        "key_stability", "key_change_rate",
        "chord_diversity",
        "pitch_range", "stepwise_ratio",
        "note_density_mean", "note_density_std",
        "duration_entropy",
        "error"
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
