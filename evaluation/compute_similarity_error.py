
"""
Similarity Error (SE) Evaluation for Symbolic Music Generation

Computes Similarity Error (SE) metric between a set of real (validation) MIDI files
and generated MIDI files.

Logic extracted from `similarity_error_evaluation_xlstm.ipynb` and `se_eval_midi_dirs.py`.
"""

import argparse
from pathlib import Path
import os
import json
import numpy as np
import miditoolkit
import sys

def list_midis(folder: Path) -> list:
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

def compute_bar_pitch_features(midi_path: Path, max_bars=256):
    try:
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception as e:
        print(f"[WARN] Failed to load {midi_path}: {e}", file=sys.stderr)
        return []

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

        if blen <= 0: continue

        cur = seg_start
        while cur < seg_end and bar_count < max_bars:
            hist = np.zeros(12)
            bar_start = cur
            bar_end = min(cur + blen, seg_end)

            # Collect notes overlapping this bar
            # Simple optimization: filtering in loop vs pre-filtering
            # Given typically small file size, simple loop is okay or bisect
            
            # Using loop similar to original script logic for fidelity
            count_notes = 0
            for n in notes:
                if n.end <= bar_start:
                    continue
                if n.start >= bar_end:
                    # notes are sorted by start, but end can vary. 
                    # Can break if strictly sorted? No, long note could start earlier.
                    # notes sorted by start. if n.start >= bar_end, all subsequent notes also >= bar_end.
                    break
                
                overlap = max(0, min(n.end, bar_end) - max(n.start, bar_start))
                if overlap > 0:
                    hist[n.pitch % 12] += overlap
                    count_notes += 1

            if hist.sum() > 0:
                hist /= hist.sum()

            bars.append(hist)
            bar_count += 1
            cur += blen
            
            if bar_count >= max_bars:
                break
        if bar_count >= max_bars:
            break

    return bars

def similarity_curve(midi_files: list, T=40, max_bars=256):
    sum_sim = np.zeros(T)
    count = np.zeros(T)

    for m in midi_files:
        bars = compute_bar_pitch_features(m, max_bars)
        B = len(bars)
        # Compute self-similarity for lags 1..T
        for t in range(1, min(T + 1, B)):
            # Average cosine sim between bar i and bar i-t
            # i starts at t, goes to B-1
            # bars[i] vs bars[i-t]
            val_sum = 0.0
            val_count = 0
            
            for i in range(t, B):
                s = cosine_sim(bars[i], bars[i - t])
                val_sum += s
                val_count += 1
            
            if val_count > 0:
                sum_sim[t - 1] += val_sum
                count[t - 1] += val_count

    # Average over all occurances in dataset
    # Note: Logic in original script seemed to accumulate sum_sim and count across files?
    # Yes: sum_sim[t-1] accumulates all similarity values for lag t across all files?
    # Or average per file then average?
    # Original:
    # for t in range...
    #   for i in range...
    #     sum_sim[t-1] += s, count[t-1] += 1
    # This pools all bar-pairs of lag t from all files.
    
    return np.divide(sum_sim, count, out=np.zeros_like(sum_sim), where=count != 0)

def similarity_error(L_real, L_gen):
    # Mean Absolute Difference
    # Ensure same length, though T should be fixed
    min_len = min(len(L_real), len(L_gen))
    return float(np.mean(np.abs(L_real[:min_len] - L_gen[:min_len])))

def get_real_midis(valid_split_file, midi_data_dir, limit=None):
    real_midis = []
    if not valid_split_file.exists():
        print(f"Error: Validation split file not found at {valid_split_file}", file=sys.stderr)
        return []
    
    try:
        with open(valid_split_file, "r") as f:
            for line in f:
                filename = line.strip()
                if filename:
                    full_path = midi_data_dir / filename
                    # Fallback check
                    if not full_path.exists() and not full_path.suffix:
                         full_path = full_path.with_suffix(".mid")
                    
                    if full_path.exists():
                        real_midis.append(full_path)
                        if limit is not None and len(real_midis) >= limit:
                            break
    except Exception as e:
        print(f"Error reading valid split: {e}", file=sys.stderr)

    return real_midis

def main():
    parser = argparse.ArgumentParser(description="Compute Similarity Error (SE)")
    parser.add_argument("--real_list", type=str, required=True, help="Path to text file listing real MIDI filenames")
    parser.add_argument("--real_dir", type=str, required=True, help="Base directory for real MIDIs")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory containing generated MIDIs")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=100, help="Max number of real MIDI files to use")
    parser.add_argument("--T", type=int, default=40, help="Max bar lag")
    parser.add_argument("--max_bars", type=int, default=256, help="Max bars per file")
    
    args = parser.parse_args()

    real_list_path = Path(args.real_list).resolve()
    real_dir_path = Path(args.real_dir).resolve()
    gen_dir_path = Path(args.gen_dir).resolve()
    out_path = Path(args.out).resolve()

    print(f"[INFO] Loading real MIDIs from list: {real_list_path} (limit={args.limit})")
    real_midis = get_real_midis(real_list_path, real_dir_path, limit=args.limit)
    
    if not real_midis:
        print("[ERROR] No real MIDI files found/loaded.")
        sys.exit(1)

    print(f"[INFO] Listing generated MIDIs from: {gen_dir_path}")
    gen_midis = list_midis(gen_dir_path)
    
    if not gen_midis:
        print("[ERROR] No generated MIDI files found.")
        sys.exit(1)

    print(f"[INFO] Found {len(real_midis)} real, {len(gen_midis)} generated files.")

    print("[INFO] Computing curve for Real Data...")
    L_real = similarity_curve(real_midis, T=args.T, max_bars=args.max_bars)
    
    print("[INFO] Computing curve for Generated Data...")
    L_gen = similarity_curve(gen_midis, T=args.T, max_bars=args.max_bars)

    se = similarity_error(L_real, L_gen)
    
    print("="*40)
    print(f"Similarity Error (SE): {se:.6f}")
    print("="*40)

    results = {
        "SE": se,
        "T": args.T,
        "max_bars": args.max_bars,
        "limit": args.limit,
        "real_count": len(real_midis),
        "gen_count": len(gen_midis),
        "real_dir": str(real_dir_path),
        "gen_dir": str(gen_dir_path),
        "L_real": L_real.tolist(),
        "L_gen": L_gen.tolist()
    }

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Saved results to {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save output: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
