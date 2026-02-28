"""
MuseFormer MIDI Filtering and Pitch Normalization Pipeline (Improved)

This script implements the filtering rules from Table 4 of the MuseFormer paper
with improvements for robustness and configurability.

Usage:
    python filter_and_normalize.py [OPTIONS]
    
Options:
    --preset strict        Use strict MuseFormer configuration
    --preset permissive    Use permissive configuration
    --config FILE          Use custom configuration file
    --help                 Show this help message

Configuration:
    Edit filter_config.py to customize filtering parameters
"""

from pathlib import Path
from dataclasses import dataclass
import json
import math
import shutil
from datetime import datetime
from typing import Optional, Tuple, List
import argparse
import sys

import pandas as pd
import miditoolkit
from collections import defaultdict

# Import configuration
try:
    import filter_config_v2 as config
except ImportError:
    print("ERROR: filter_config.py not found!")
    print("Please ensure filter_config.py is in the same directory as this script.")
    exit(1)


# =============================================================================
# KEY DETECTION (Krumhansl-Kessler)
# =============================================================================

def rotate(lst: list, k: int) -> list:
    """Rotate list by k positions"""
    k %= len(lst)
    return lst[k:] + lst[:k]


def correlation(a: list, b: list) -> float:
    """Compute cosine-like correlation between two lists"""
    sa = sum(x * x for x in a) ** 0.5
    sb = sum(x * x for x in b) ** 0.5
    if sa == 0 or sb == 0:
        return -1e9
    return sum(x * y for x, y in zip(a, b)) / (sa * sb)


def minimal_mod12_shift(target_pc: int, tonic_pc: int) -> int:
    """
    Calculate minimal semitone shift from tonic_pc to target_pc
    Returns value in range [-6, +5]
    """
    d = (target_pc - tonic_pc) % 12
    if d > 6:
        d -= 12
    return d


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MidiStats:
    """Statistics extracted from a MIDI file for filtering"""
    basename: str
    path: Path
    ticks_per_beat: int
    time_sigs: List[str]
    is_time_sig_valid: bool
    tempo_min: float
    tempo_max: float
    has_tempo_data: bool
    pitch_min: int
    pitch_max: int
    max_note_dur_beats: float
    num_notes: int
    distinct_onsets: int
    programs_sig: tuple
    nonempty_tracks: int
    has_required_track: bool
    empty_bars: int
    consecutive_empty_bars: int
    unique_pitch_count: int
    unique_dur_count: int
    num_bars: int
    duration_beats: float

    def dup_signature(self) -> tuple:
        """Generate signature for duplicate detection"""
        return (
            self.num_bars,
            round(self.duration_beats, 4),
            self.num_notes,
            self.distinct_onsets,
            self.programs_sig,
        )


# =============================================================================
# MIDI LOADING AND STATS EXTRACTION
# =============================================================================

def load_midi(path: Path) -> miditoolkit.MidiFile:
    """Load MIDI file using miditoolkit"""
    return miditoolkit.MidiFile(str(path))


def count_empty_bars_sounding(midi: miditoolkit.MidiFile, tpb: int, num_bars: int, 
                               count_drums: bool = True) -> Tuple[int, int]:
    """
    Count empty bars using "sounding notes" method.
    
    A bar is empty if NO notes are sounding during any part of that bar.
    
    Args:
        midi: MidiFile object
        tpb: Ticks per beat
        num_bars: Total number of bars
        count_drums: Whether to include drum notes
    
    Returns:
        (total_empty_bars, max_consecutive_empty_bars)
    """
    bar_ticks = 4 * tpb  # Assuming 4/4 time
    
    if bar_ticks == 0 or num_bars == 0:
        return 0, 0
    
    # Collect all notes
    all_notes = []
    for inst in midi.instruments:
        if inst.is_drum and not count_drums:
            continue
        all_notes.extend(inst.notes)
    
    if not all_notes:
        return num_bars, num_bars
    
    # Check each bar
    empty_count = 0
    consecutive = 0
    max_consecutive = 0
    
    for bar_idx in range(num_bars):
        bar_start = bar_idx * bar_ticks
        bar_end = (bar_idx + 1) * bar_ticks
        
        # Check if any note is sounding during this bar
        # Note is sounding if: note.start < bar_end AND note.end > bar_start
        has_sound = False
        for note in all_notes:
            if note.start < bar_end and note.end > bar_start:
                has_sound = True
                break
        
        if not has_sound:
            empty_count += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    
    return empty_count, max_consecutive


def count_empty_bars_onset(midi: miditoolkit.MidiFile, tpb: int, num_bars: int,
                           count_drums: bool = True) -> Tuple[int, int]:
    """
    Count empty bars using "onset" method (original implementation).
    
    A bar is empty if NO notes start in that bar.
    
    Args:
        midi: MidiFile object
        tpb: Ticks per beat
        num_bars: Total number of bars
        count_drums: Whether to include drum notes
    
    Returns:
        (total_empty_bars, max_consecutive_empty_bars)
    """
    bar_ticks = 4 * tpb
    
    if bar_ticks == 0 or num_bars == 0:
        return 0, 0
    
    # Collect onsets
    onsets = set()
    for inst in midi.instruments:
        if inst.is_drum and not count_drums:
            continue
        for note in inst.notes:
            onsets.add(int(note.start))
    
    if not onsets:
        return num_bars, num_bars
    
    # Sort onsets for efficient scanning
    starts = sorted(onsets)
    
    empty_count = 0
    consecutive = 0
    max_consecutive = 0
    si = 0
    
    for bar_idx in range(num_bars):
        bar_start = bar_idx * bar_ticks
        bar_end = (bar_idx + 1) * bar_ticks
        
        # Advance pointer to first onset >= bar_start
        while si < len(starts) and starts[si] < bar_start:
            si += 1
        
        # Check if there's an onset in this bar
        if si >= len(starts) or starts[si] >= bar_end:
            empty_count += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    
    return empty_count, max_consecutive


def extract_stats(path: Path) -> MidiStats:
    """Extract all statistics from a MIDI file"""
    midi = load_midi(path)
    tpb = midi.ticks_per_beat
    
    # -------------------------------------------------------------------------
    # Time Signatures
    # -------------------------------------------------------------------------
    ts = getattr(midi, "time_signature_changes", []) or []
    time_sigs = [f"{x.numerator}/{x.denominator}" for x in ts] if ts else []
    
    # Validate time signature
    if config.ALLOW_MISSING_TIME_SIGNATURE and len(time_sigs) == 0:
        # Treat as valid (assume 4/4)
        is_time_sig_valid = True
    elif len(time_sigs) == 0:
        # Strict mode: must have time signature
        is_time_sig_valid = False
    else:
        # Check all time sigs are in allowed list
        is_time_sig_valid = all(ts in config.ALLOWED_TIME_SIGNATURES for ts in time_sigs)
    
    # -------------------------------------------------------------------------
    # Tempo
    # -------------------------------------------------------------------------
    tempos = getattr(midi, "tempo_changes", []) or []
    if tempos:
        bpm_vals = [float(t.tempo) for t in tempos]
        tempo_min = min(bpm_vals)
        tempo_max = max(bpm_vals)
        has_tempo_data = True
    else:
        # No tempo data
        tempo_min = float("inf")
        tempo_max = float("-inf")
        has_tempo_data = False
    
    # -------------------------------------------------------------------------
    # Instruments Signature
    # -------------------------------------------------------------------------
    prog_list = []
    for inst in midi.instruments:
        if inst.is_drum:
            prog_list.append(("drum", 0))
        else:
            # Use improved track detection (handles MuseScore inconsistencies)
            if config.track_matches_requirement(inst):
                prog_list.append(("required", inst.program))
            else:
                prog_list.append(("inst", inst.program))
    programs_sig = tuple(sorted(prog_list))
    
    # -------------------------------------------------------------------------
    # Collect Notes
    # -------------------------------------------------------------------------
    all_notes = []
    non_drum_notes = []
    onsets = set()
    pitches = []
    durs = []
    max_end_tick = 0
    
    nonempty_tracks = 0
    has_required = False
    
    for inst in midi.instruments:
        if inst.notes:
            nonempty_tracks += 1
        
        # Use improved track detection (handles MuseScore inconsistencies)
        if config.track_matches_requirement(inst) and len(inst.notes) > 0:
            has_required = True
        
        for n in inst.notes:
            all_notes.append(n)
            onsets.add(int(n.start))
            max_end_tick = max(max_end_tick, int(n.end))
            
            if not inst.is_drum:
                non_drum_notes.append(n)
                pitches.append(int(n.pitch))
                durs.append(int(n.end) - int(n.start))
    
    num_notes = len(all_notes)
    distinct_onsets = len(onsets)
    
    # -------------------------------------------------------------------------
    # Pitch and Duration Stats
    # -------------------------------------------------------------------------
    if non_drum_notes:
        pitch_min = min(pitches)
        pitch_max = max(pitches)
        max_note_dur_beats = max(d / tpb for d in durs) if tpb > 0 else float("inf")
        unique_pitch_count = len(set(pitches))
        unique_dur_count = len(set(durs))
    else:
        pitch_min = 999
        pitch_max = -999
        max_note_dur_beats = float("inf")
        unique_pitch_count = 0
        unique_dur_count = 0
    
    # -------------------------------------------------------------------------
    # Duration and Bars
    # -------------------------------------------------------------------------
    duration_beats = max_end_tick / tpb if tpb > 0 else 0.0
    num_bars = int(math.ceil(duration_beats / 4.0)) if duration_beats > 0 else 0
    
    # -------------------------------------------------------------------------
    # Empty Bars
    # -------------------------------------------------------------------------
    if config.EMPTY_BAR_METHOD == "sounding":
        empty_bars, consecutive_empty = count_empty_bars_sounding(
            midi, tpb, num_bars, config.COUNT_DRUMS_FOR_EMPTY_BARS
        )
    else:  # "onset"
        empty_bars, consecutive_empty = count_empty_bars_onset(
            midi, tpb, num_bars, config.COUNT_DRUMS_FOR_EMPTY_BARS
        )
    
    return MidiStats(
        basename=path.name,
        path=path,
        ticks_per_beat=tpb,
        time_sigs=time_sigs,
        is_time_sig_valid=is_time_sig_valid,
        tempo_min=tempo_min,
        tempo_max=tempo_max,
        has_tempo_data=has_tempo_data,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        max_note_dur_beats=max_note_dur_beats,
        num_notes=num_notes,
        distinct_onsets=distinct_onsets,
        programs_sig=programs_sig,
        nonempty_tracks=nonempty_tracks,
        has_required_track=has_required,
        empty_bars=empty_bars,
        consecutive_empty_bars=consecutive_empty,
        unique_pitch_count=unique_pitch_count,
        unique_dur_count=unique_dur_count,
        num_bars=num_bars,
        duration_beats=duration_beats,
    )


# =============================================================================
# FILTERING LOGIC
# =============================================================================

def filter_reason(s: MidiStats) -> Optional[str]:
    """
    Determine if a MIDI file should be filtered out.
    
    Returns:
        None if file passes all filters
        String describing the reason for filtering if it fails
    """
    # Time signature check
    if not s.is_time_sig_valid:
        return "filter_invalid_time_signature"
    
    # Track count check
    if s.nonempty_tracks < config.MIN_NONEMPTY_TRACKS:
        return f"filter_lt_{config.MIN_NONEMPTY_TRACKS}_tracks"
    
    # Required track check
    if not s.has_required_track:
        return f"filter_no_melody_track"
    
    # Tempo check (improved)
    if not s.has_tempo_data:
        return "filter_no_tempo_data"
    
    if s.tempo_min < config.TEMPO_MIN:
        return "filter_tempo_too_slow"
    
    if s.tempo_max > config.TEMPO_MAX:
        return "filter_tempo_too_fast"
    
    # Pitch range check
    if s.pitch_min < config.PITCH_MIN:
        return "filter_pitch_too_low"
    
    if s.pitch_max > config.PITCH_MAX:
        return "filter_pitch_too_high"
    
    # Note duration check
    if s.max_note_dur_beats > config.MAX_NOTE_DURATION_BEATS:
        return "filter_note_too_long"
    
    # Empty bars check (using consecutive count for better detection)
    if s.consecutive_empty_bars > config.MAX_EMPTY_BARS_ALLOWED:
        return "filter_too_many_empty_bars"
    
    # Degenerate content check
    if config.DROP_DEGENERATE_CONTENT:
        if s.unique_pitch_count == 1:
            return "filter_degenerate_all_same_pitch"
        if s.unique_dur_count == 1:
            return "filter_degenerate_all_same_duration"
    
    return None


# =============================================================================
# PITCH NORMALIZATION
# =============================================================================

def pitch_class_histogram(midi: miditoolkit.MidiFile) -> List[float]:
    """
    Compute duration-weighted pitch class histogram over non-drum notes.
    
    Returns:
        List of 12 floats representing the weight of each pitch class (0-11)
    """
    hist = [0.0] * 12
    tpb = midi.ticks_per_beat
    
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pc = int(n.pitch) % 12
            dur_beats = (int(n.end) - int(n.start)) / tpb if tpb > 0 else 0.0
            hist[pc] += max(dur_beats, 0.0)
    
    return hist


def detect_key_and_shift(midi: miditoolkit.MidiFile) -> Tuple[Optional[str], Optional[int], int]:
    """
    Detect key using Krumhansl-Kessler profiles and calculate transposition.
    
    Returns:
        (mode, tonic_pc, semitone_shift)
        - mode: "major" or "minor" (None if no pitched content)
        - tonic_pc: Pitch class of detected tonic 0-11 (None if no content)
        - semitone_shift: Semitones to transpose (0 if no content)
    """
    hist = pitch_class_histogram(midi)
    
    if sum(hist) == 0:
        return None, None, 0  # No pitched content
    
    # Find best matching major key
    best_major = (-1e9, None)
    for tonic in range(12):
        score = correlation(hist, rotate(config.KK_MAJOR_PROFILE, tonic))
        if score > best_major[0]:
            best_major = (score, tonic)
    
    # Find best matching minor key
    best_minor = (-1e9, None)
    for tonic in range(12):
        score = correlation(hist, rotate(config.KK_MINOR_PROFILE, tonic))
        if score > best_minor[0]:
            best_minor = (score, tonic)
    
    # Choose major vs minor
    if best_major[0] >= best_minor[0]:
        mode = "major"
        tonic = best_major[1]
        target = config.MAJOR_TARGET_PC
    else:
        mode = "minor"
        tonic = best_minor[1]
        target = config.MINOR_TARGET_PC
    
    shift = minimal_mod12_shift(target, tonic)
    return mode, tonic, shift


def apply_pitch_norm(midi: miditoolkit.MidiFile, shift: int) -> bool:
    """
    Apply pitch normalization to MIDI file.
    
    1. Transpose all non-drum notes by 'shift' semitones
    2. Adjust octaves to fit within [PITCH_MIN, PITCH_MAX] range
    
    Args:
        midi: MidiFile object (modified in-place)
        shift: Semitones to transpose
    
    Returns:
        True if successful, False if unable to fit in range
    """
    # Apply key transposition
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            n.pitch = int(n.pitch) + int(shift)
    
    # Check if already in valid range
    all_pitched = [
        n.pitch 
        for inst in midi.instruments 
        if not inst.is_drum 
        for n in inst.notes
    ]
    
    if not all_pitched:
        return True  # No pitched notes, nothing to do
    
    mn = min(all_pitched)
    mx = max(all_pitched)
    
    # Early exit if already valid
    if config.PITCH_MIN <= mn and mx <= config.PITCH_MAX:
        return True
    
    # Try octave adjustments
    # Shift up if too low
    while mn < config.PITCH_MIN:
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                n.pitch += 12
        mn += 12
        mx += 12
        
        # Check if we've gone out of MIDI range
        if mx > 127:
            return False
    
    # Shift down if too high
    while mx > config.PITCH_MAX:
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                n.pitch -= 12
        mn -= 12
        mx -= 12
        
        # Check if we've gone out of MIDI range
        if mn < 0:
            return False
    
    # Final validation
    all_pitched = [
        n.pitch 
        for inst in midi.instruments 
        if not inst.is_drum 
        for n in inst.notes
    ]
    
    if any(p < 0 or p > 127 for p in all_pitched):
        return False
    
    return True


# =============================================================================
# MANIFEST MANAGEMENT
# =============================================================================

def update_manifest(df: pd.DataFrame, updates: dict):
    """
    Update manifest DataFrame with new data.
    
    Args:
        df: Manifest DataFrame
        updates: Dict mapping basename -> dict of column values
    """
    # Ensure raw_basename column exists
    if "raw_basename" not in df.columns:
        df["raw_basename"] = df["raw_path"].astype(str).map(lambda p: Path(p).name)
    
    # Add any new columns
    all_cols = set(k for u in updates.values() for k in u.keys())
    for col in all_cols:
        if col not in df.columns:
            df[col] = pd.NA
    
    # Apply updates
    for basename, cols in updates.items():
        mask = df["raw_basename"].astype(str) == basename
        for k, v in cols.items():
            df.loc[mask, k] = v


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='MuseFormer MIDI Filtering and Pitch Normalization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration (as defined in filter_config.py)
  python filter_and_normalize.py
  
  # Use strict MuseFormer preset
  python filter_and_normalize.py --preset strict
  
  # Use permissive preset
  python filter_and_normalize.py --preset permissive

  # Process in batches of 5000 files
  python filter_and_normalize.py --preset strict --batch-size 5000

  # Resume after a crash/reboot (skips already-processed files)
  python filter_and_normalize.py --preset strict --batch-size 5000 --resume
  
  # Use custom config file
  python filter_and_normalize.py --config my_config.py
  
Configuration Methods:
  The preset configurations can be used to quickly switch between different
  filtering strategies. See filter_config.py for details on each preset.
        """
    )
    
    parser.add_argument(
        '--preset',
        choices=['strict', 'permissive'],
        help='Load a preset configuration (overrides config file settings)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (alternative to filter_config.py)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually processing files'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=0,
        metavar='N',
        help='Process files in batches of N (0 = all at once, recommended: 5000)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip files already present in OUTPUT_PITCH_NORM_DIR and resume from last checkpoint'
    )
    
    return parser.parse_args()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def _make_json_safe(obj):
    """Recursively convert numpy scalars and tuples to JSON-safe Python types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    return obj  # str, int, float — already safe


def _restore_sig(lst):
    """Restore a dup_signature from its JSON list representation.
    Structure: [num_bars, duration_beats, num_notes, distinct_onsets, [[kind, prog], ...]]
    """
    num_bars, duration_beats, num_notes, distinct_onsets, prog_list = lst
    programs_sig = tuple((pair[0], pair[1]) for pair in prog_list)
    return (int(num_bars), float(duration_beats), int(num_notes), int(distinct_onsets), programs_sig)


def _load_progress(progress_path: Path) -> dict:
    """Load progress/checkpoint file from a previous run."""
    if not progress_path.exists():
        return {'global_sigs': [], 'processed_count': 0}
    try:
        import json as _json
        with open(progress_path) as f:
            data = _json.load(f)
        data['global_sigs'] = [_restore_sig(s) for s in data.get('global_sigs', [])]
        return data
    except Exception as e:
        print(f"WARNING: Could not load progress file ({e}), starting fresh")
        return {'global_sigs': [], 'processed_count': 0}


def _save_progress(progress_path: Path, global_sigs: set, processed_count: int):
    """Save progress checkpoint after each batch."""
    import json as _json
    data = {
        'global_sigs': [_make_json_safe(sig) for sig in global_sigs],
        'processed_count': processed_count,
        'timestamp': datetime.now().isoformat(),
    }
    with open(progress_path, 'w') as f:
        _json.dump(data, f)




def _process_batch(
    batch: list,
    global_sigs: set,
    df: 'pd.DataFrame',
    batch_idx: int,
    total_batches: int,
) -> dict:
    """
    Process a single batch of MIDI files through the full filter+pitch-norm pipeline.

    Modifies global_sigs in-place (adds new signatures).
    Returns a stats dict with keys: total, failed, dups, kept, dropped, pitch_ok, pitch_dropped.
    Also updates the manifest DataFrame df in-place.
    """
    print(f"\nBatch {batch_idx + 1}/{total_batches}: processing {len(batch)} files...")

    # ---- 1. Extract stats -----------------------------------------------
    stats_list = []
    failed = []
    for i, path in enumerate(batch, 1):
        try:
            s = extract_stats(path)
            stats_list.append(s)
        except Exception as e:
            failed.append((path.name, str(e)))
            print(f"  ERROR parsing {path.name}: {e}")
        if i % 500 == 0:
            print(f"  Extracted stats: {i}/{len(batch)}")

    # ---- 2. Duplicate detection (global) --------------------------------
    is_dup = {}
    for s in stats_list:
        sig = s.dup_signature()
        if sig not in global_sigs:
            global_sigs.add(sig)
            is_dup[s.basename] = False
        else:
            is_dup[s.basename] = True

    num_dups = sum(is_dup.values())

    # ---- 3. Filter -------------------------------------------------------
    keep = []
    drop_reason_map = {}
    drop_stats = defaultdict(int)

    for s in stats_list:
        if is_dup.get(s.basename, False):
            drop_reason_map[s.basename] = "filter_duplicate_signature"
            drop_stats["filter_duplicate_signature"] += 1
            continue
        reason = filter_reason(s)
        if reason is None:
            keep.append(s)
        else:
            drop_reason_map[s.basename] = reason
            drop_stats[reason] += 1

    print(f"  Batch {batch_idx + 1}: kept={len(keep)}, dropped={len(drop_reason_map)}, dups={num_dups}")

    # ---- 4. Copy filtered files ------------------------------------------
    config.OUTPUT_FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    for s in keep:
        shutil.copy2(s.path, config.OUTPUT_FILTERED_DIR / s.basename)

    # ---- 5. Pitch normalization ------------------------------------------
    config.OUTPUT_PITCH_NORM_DIR.mkdir(parents=True, exist_ok=True)
    pitch_updates = {}
    pitch_drops = {}

    for i, s in enumerate(keep, 1):
        src = config.OUTPUT_FILTERED_DIR / s.basename
        dst = config.OUTPUT_PITCH_NORM_DIR / s.basename
        try:
            midi = load_midi(src)
            mode, tonic, shift = detect_key_and_shift(midi)
            if mode is None:
                pitch_drops[s.basename] = "pitchnorm_no_pitched_content"
                continue
            ok = apply_pitch_norm(midi, shift)
            if not ok:
                pitch_drops[s.basename] = "pitchnorm_out_of_range"
                continue
            midi.dump(str(dst))
            pitch_updates[s.basename] = {
                "pitchnorm_mode": mode,
                "pitchnorm_tonic_pc": tonic,
                "pitchnorm_semitones": shift,
                "pitchnorm_path": str(dst),
            }
        except Exception as e:
            pitch_drops[s.basename] = f"pitchnorm_error:{e}"

        if i % 500 == 0:
            print(f"  Pitch normalized: {i}/{len(keep)}")

    print(f"  Pitch norm: {len(pitch_updates)} OK, {len(pitch_drops)} dropped")

    # ---- 6. Update manifest (in-place) -----------------------------------
    kept_basenames = set(s.basename for s in keep)
    updates = {}

    for s in stats_list:
        b = s.basename
        if b in kept_basenames:
            updates[b] = {
                "stage": config.STAGE_FILTER,
                "status": "ok",
                "drop_reason": pd.NA,
                "error_msg": pd.NA,
                "mscore_norm_path": str(config.INPUT_NORMALIZED_DIR / b),
                "filtered_path": str(config.OUTPUT_FILTERED_DIR / b),
                "filter_is_duplicate": bool(is_dup.get(b, False)),
                "filter_empty_bars": s.empty_bars,
                "filter_consecutive_empty_bars": s.consecutive_empty_bars,
                "filter_num_bars": s.num_bars,
                "filter_duration_beats": s.duration_beats,
                "filter_nonempty_tracks": s.nonempty_tracks,
                "filter_pitch_min": s.pitch_min,
                "filter_pitch_max": s.pitch_max,
                "filter_tempo_min": s.tempo_min,
                "filter_tempo_max": s.tempo_max,
                "filter_max_note_dur_beats": s.max_note_dur_beats,
                "filter_distinct_onsets": s.distinct_onsets,
                "filter_num_notes": s.num_notes,
            }
        else:
            reason = drop_reason_map.get(b, "filter_unknown")
            updates[b] = {
                "stage": config.STAGE_FILTER,
                "status": "dropped",
                "drop_reason": reason,
                "error_msg": pd.NA,
                "mscore_norm_path": str(config.INPUT_NORMALIZED_DIR / b),
                "filtered_path": pd.NA,
            }

    for b, cols in pitch_updates.items():
        updates.setdefault(b, {})
        updates[b].update({
            "stage": config.STAGE_PITCH_NORM,
            "status": "ok",
            "drop_reason": pd.NA,
            "error_msg": pd.NA,
        })
        updates[b].update(cols)

    for b, reason in pitch_drops.items():
        updates.setdefault(b, {})
        updates[b].update({
            "stage": config.STAGE_PITCH_NORM,
            "status": "dropped",
            "drop_reason": reason,
            "error_msg": pd.NA,
        })

    for b, row_data in updates.items():
        if "raw_basename" not in df.columns:
            df["raw_basename"] = df["raw_path"].astype(str).map(lambda p: Path(p).name)
        all_cols = set(row_data.keys())
        for col in all_cols:
            if col not in df.columns:
                df[col] = pd.NA
        mask = df["raw_basename"].astype(str) == b
        for k, v in row_data.items():
            df.loc[mask, k] = v

    return {
        'total': len(batch),
        'failed': len(failed),
        'dups': num_dups,
        'kept': len(keep),
        'dropped': len(drop_reason_map),
        'pitch_ok': len(pitch_updates),
        'pitch_dropped': len(pitch_drops),
    }


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load custom config if specified
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        if spec is None or spec.loader is None:
            print(f"ERROR: Could not load config file: {args.config}")
            sys.exit(1)
        global config
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        print(f"Loaded custom configuration from: {args.config}")
        print()
    
    # Load preset if specified
    if args.preset:
        if args.preset == 'strict':
            config.load_museformer_strict()
        elif args.preset == 'permissive':
            config.load_permissive()
        print()
    
    print("=" * 80)
    print("MuseFormer MIDI Filtering and Pitch Normalization Pipeline")
    print("=" * 80)
    print()

    batch_size = args.batch_size  # 0 means process all at once
    resume = args.resume

    # Display active configuration
    print("Active Configuration:")
    print(f"  Preset: {args.preset if args.preset else 'default'}")
    print(f"  Track detection: {config.TRACK_DETECTION_METHOD}")
    print(f"  Required program: {config.REQUIRED_PROGRAM_NUMBER}")
    print(f"  Tempo range: {config.TEMPO_MIN}-{config.TEMPO_MAX} BPM")
    print(f"  Pitch range: {config.PITCH_MIN}-{config.PITCH_MAX}")
    print(f"  Max note duration: {config.MAX_NOTE_DURATION_BEATS} beats")
    print(f"  Max empty bars: {config.MAX_EMPTY_BARS_ALLOWED}")
    print(f"  Empty bar method: {config.EMPTY_BAR_METHOD}")
    print(f"  Min tracks: {config.MIN_NONEMPTY_TRACKS}")
    print(f"  Batch size: {batch_size if batch_size > 0 else 'unlimited (single pass)'}")
    print(f"  Resume: {resume}")
    print()
    
    if args.dry_run:
        print("DRY RUN MODE: No files will be processed")
        print()
        return
    
    # Create output directories
    config.OUTPUT_FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_PITCH_NORM_DIR.mkdir(parents=True, exist_ok=True)

    # Progress/checkpoint file (stored alongside final output)
    progress_path = config.OUTPUT_PITCH_NORM_DIR / "progress.json"
    
    # Find all input files
    all_midi_files = sorted(
        list(config.INPUT_NORMALIZED_DIR.glob("*.mid")) + 
        list(config.INPUT_NORMALIZED_DIR.glob("*.midi"))
    )
    
    print(f"Input directory: {config.INPUT_NORMALIZED_DIR}")
    print(f"Output directory (filtered): {config.OUTPUT_FILTERED_DIR}")
    print(f"Output directory (pitch norm): {config.OUTPUT_PITCH_NORM_DIR}")
    print(f"Total MIDI files found: {len(all_midi_files)}")
    
    if len(all_midi_files) == 0:
        print("ERROR: No MIDI files found!")
        return

    # ---- Resume: skip already processed files ----------------------------
    progress = _load_progress(progress_path)
    global_sigs = set(progress['global_sigs'])

    if resume:
        already_done = {f.name for f in config.OUTPUT_PITCH_NORM_DIR.glob("*.mid")}
        midi_files = [f for f in all_midi_files if f.name not in already_done]
        print(f"Resume mode: {len(already_done)} already done, {len(midi_files)} remaining")
    else:
        midi_files = all_midi_files
        # Fresh start: clear progress state
        global_sigs = set()

    print()

    if not config.MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found: {config.MANIFEST_PATH}")
        return

    df = pd.read_csv(config.MANIFEST_PATH)
    backup_path = config.MANIFEST_PATH.with_suffix(
        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df.to_csv(backup_path, index=False)
    print(f"Manifest backup: {backup_path}")
    print()

    # ---- Split into batches ----------------------------------------------
    if batch_size > 0:
        batches = [midi_files[i:i + batch_size] for i in range(0, len(midi_files), batch_size)]
    else:
        batches = [midi_files]  # Single pass (original behaviour)

    total_batches = len(batches)
    print(f"Processing {len(midi_files)} files in {total_batches} batch(es)")
    print()

    # ---- Totals across all batches ---------------------------------------
    totals = dict(total=0, failed=0, dups=0, kept=0, dropped=0, pitch_ok=0, pitch_dropped=0)
    processed_so_far = progress.get('processed_count', 0)

    for batch_idx, batch in enumerate(batches):
        batch_stats = _process_batch(
            batch=batch,
            global_sigs=global_sigs,
            df=df,
            batch_idx=batch_idx,
            total_batches=total_batches,
        )
        for k in totals:
            totals[k] += batch_stats.get(k, 0)
        processed_so_far += len(batch)

        # Save manifest after each batch so progress is not lost on crash
        df.to_csv(config.MANIFEST_PATH, index=False)
        print(f"  Manifest saved ({processed_so_far}/{len(all_midi_files)} total processed)")

        # Save progress checkpoint
        _save_progress(progress_path, global_sigs, processed_so_far)

    # ---- Final summary ---------------------------------------------------
    print()
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total input files processed this run: {len(midi_files)}")
    print(f"  Parse failures:       {totals['failed']}")
    print(f"  Duplicates removed:   {totals['dups']}")
    print(f"  Filtered (kept):      {totals['kept']}")
    print(f"  Filtered (dropped):   {totals['dropped']}")
    print(f"  Pitch normalized:     {totals['pitch_ok']}")
    print(f"  Pitch norm failures:  {totals['pitch_dropped']}")
    
    total_in_output = len(list(config.OUTPUT_PITCH_NORM_DIR.glob("*.mid")))
    print(f"\nTotal files in {config.OUTPUT_PITCH_NORM_DIR.name}/: {total_in_output}")
    print("=" * 80)

    # Save final summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_files": len(all_midi_files),
        "this_run_files": len(midi_files),
        "parse_failures": totals['failed'],
        "duplicates_removed": totals['dups'],
        "filtered_kept": totals['kept'],
        "filtered_dropped": totals['dropped'],
        "pitch_normalized": totals['pitch_ok'],
        "pitch_norm_failures": totals['pitch_dropped'],
        "total_output_files": total_in_output,
        "resume": resume,
        "batch_size": batch_size,
    }
    summary_path = config.OUTPUT_PITCH_NORM_DIR / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()