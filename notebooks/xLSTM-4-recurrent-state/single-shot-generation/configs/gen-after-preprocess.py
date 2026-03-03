"""
xLSTM-4 Single-Shot Generation Pipeline — Configuration
========================================================
All hyperparameters are configurable. Modify this file to adjust generation settings.
"""

import os

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so scripts work from any cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))  # configs/
_SINGLE_SHOT_DIR = os.path.dirname(_HERE)  # single-shot-generation/
_XLSTM4_DIR = os.path.dirname(_SINGLE_SHOT_DIR)  # xLSTM-4-recurrent-state/
_NOTEBOOKS_DIR = os.path.dirname(_XLSTM4_DIR)  # notebooks/
_REPO_ROOT = os.path.dirname(_NOTEBOOKS_DIR)  # fyp-musicgen/

# Path to the MidiProcessor source (for MIDI conversion)
MIDIPROCESSOR_PATH = os.path.join(_REPO_ROOT, "repos", "MidiProcessor")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Model run directory — xLSTMGenerator auto-discovers checkpoint-*-last
MODEL_PATH = os.path.join(
    _REPO_ROOT,
    "repos", "helibrunna", "output",
    "xlstm_lmd_preprocessed_512d_4096ctx_12b", "run_20260301-1906"
)

# Checkpoint name (best PPL checkpoint)
CHECKPOINT_NAME = "checkpoint-46000-last"

# Override context_length at inference time
# Training context: 4096, inference context: 16384 (allows extrapolation testing)
INFERENCE_CONTEXT_LENGTH = 16_384

# ---------------------------------------------------------------------------
# Generation Hyperparameters (All Configurable)
# ---------------------------------------------------------------------------
# Fixed prompt for all runs (ensures fair comparison)
PROMPT = "s-9 o-0 t-38"

# Sampling temperature (higher = more creative, lower = more conservative)
TEMPERATURE = 0.9

# Target token lengths to generate
# Training context = 4096, so these test within-context and extrapolation
TARGET_TOKENS = [2048, 4096, 8192, 12288]

# Number of pieces to generate per target length
PIECES_PER_COND = 30

# Base random seed (actual seed per piece = SEED + piece_id * 1000 + target_idx)
SEED = 42

# ---------------------------------------------------------------------------
# Output Paths (automatically inferred from config filename)
# ---------------------------------------------------------------------------
# Auto-infer run name from this config file's name (e.g., "default" from "default.py")
_CONFIG_FILENAME = os.path.splitext(os.path.basename(__file__))[0]
RUN_NAME = _CONFIG_FILENAME if _CONFIG_FILENAME else "xlstm_recurrent_single_shot"

RESULTS_DIR = os.path.join(_SINGLE_SHOT_DIR, "results", RUN_NAME)
MIDI_DIR    = os.path.join(RESULTS_DIR, "midi")
TOKEN_DIR   = os.path.join(RESULTS_DIR, "tokens")
LOG_CSV     = os.path.join(RESULTS_DIR, "generation_metrics.csv")
