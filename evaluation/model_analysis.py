#!/usr/bin/env python3
"""
Model Analysis and Visualization
Compares: xLSTM (Model 6), Lookback RNN, Museformer  vs  Human Composed
Run: conda run -n muspy-env python evaluation/model_analysis.py
All figures saved to: evaluation/analysis_figures/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pretty_midi
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────
#  ▶  MANUALLY FILL THESE VALUES BEFORE RUNNING
# ─────────────────────────────────────────────────────

# Perplexity across sequence lengths (lower is better)
# Keys are sequence lengths; values are perplexity for each model
PERPLEXITY_SEQ_LENGTHS = [1024, 5120, 10240]
PERPLEXITY = {
    "xLSTM":      [1.67, 1.50, 1.89 ],   # fill in values for 5120, 10240
    "Museformer": [1.80, 1.62, 1.56],   # fill in values for 5120, 10240
}

# Generation time in seconds per bar – fill in your actual values
GEN_TIME = {
    "xLSTM": 2.652,      # e.g. 4.2
    "Lookback RNN":    0.559,      # e.g. 1.8
    "Museformer":      163.953,      # e.g. 6.5
}

# ─────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────
BASE   = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen")
RDIR   = BASE / "evaluation" / "results"
OUTDIR = BASE / "evaluation" / "analysis_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

MIDI_DIRS = {
    "xLSTM": BASE / "notebooks/xLSTM-4-recurrent-state/single-shot-generation/results/gen-after-preprocess-24k-dataset/midi",
    "Lookback RNN":    BASE / "melody_rnn/output/generated_samples_for_eval",
    "Museformer":      BASE / "evaluation/museformer_output",
    "Human Composed":  BASE / "evaluation/validation_sample_100",
}

SE_FILES = {
    "xLSTM": RDIR / "se_results_model6.json",
    "Lookback RNN":    RDIR / "se_results_lookback_rnn.json",
    "Museformer":      RDIR / "se_results_museformer_set1.json",
}

MUSICAL_CSVS = {
    "xLSTM": RDIR / "musical_metrics_model6.csv",
    "Lookback RNN":    RDIR / "musical_metrics_lookback_rnn.csv",
    "Museformer":      RDIR / "musical_metrics_museformer_set1.csv",
    "Human Composed":  RDIR / "musical_metrics_validation.csv",
}

SSM_CSVS = {
    "xLSTM": RDIR / "ssm_metrics_model6.csv",
    "Lookback RNN":    RDIR / "ssm_metrics_lookback_rnn.csv",
    "Museformer":      RDIR / "ssm_metrics_museformer_set1.csv",
    "Human Composed":  RDIR / "ssm_metrics_validation.csv",
}

# ─────────────────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────────────────
PALETTE = {
    "xLSTM": "#4C72B0",
    "Lookback RNN":    "#DD8452",
    "Museformer":      "#55A868",
    "Human Composed":  "#C44E52",
}
THREE_MODELS = ["xLSTM", "Lookback RNN", "Museformer"]

# ─────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────
import json

def load_se(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("SE")

def load_df(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def bar_chart(ax, labels, values, colors, ylabel, title, fmt=".4f"):
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f" {val:{fmt}}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

def style_fig(fig):
    fig.patch.set_facecolor("white")
    for ax in fig.get_axes():
        ax.set_facecolor("white")


# ═══════════════════════════════════════════════════════════════
#  1. Perplexity Line Graph (across sequence lengths)
# ═══════════════════════════════════════════════════════════════
print("▶ Plot 1: Perplexity (line graph)")
fig, ax = plt.subplots(figsize=(8, 5))
style_fig(fig)

markers = {"xLSTM": "o", "Museformer": "s"}

for model, vals in PERPLEXITY.items():
    x_pts = [PERPLEXITY_SEQ_LENGTHS[i] for i, v in enumerate(vals) if v is not None]
    y_pts = [v for v in vals if v is not None]
    if x_pts:
        ax.plot(x_pts, y_pts, color=PALETTE[model], linewidth=2.5,
                marker=markers.get(model, "o"), markersize=8, label=model)
        for xi, yi in zip(x_pts, y_pts):
            ax.annotate(f"{yi:.2f}", xy=(xi, yi),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold",
                        color=PALETTE[model])

ax.set_xlabel("Sequence Length (tokens)", fontsize=11)
ax.set_ylabel("Perplexity", fontsize=11)
ax.set_title("Model Perplexity vs. Sequence Length", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(PERPLEXITY_SEQ_LENGTHS)
ax.set_xticklabels([str(s) for s in PERPLEXITY_SEQ_LENGTHS], fontsize=10)
ax.legend(fontsize=10, framealpha=0.85)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(linestyle="--", alpha=0.4)

plt.tight_layout()
fig.savefig(OUTDIR / "1_perplexity.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved 1_perplexity.png")


# ═══════════════════════════════════════════════════════════════
#  2. Similarity Error Bar Chart
# ═══════════════════════════════════════════════════════════════
print("▶ Plot 2: Similarity Error")
fig, ax = plt.subplots(figsize=(7, 4))
style_fig(fig)

labels = THREE_MODELS
vals   = [load_se(SE_FILES[m]) for m in labels]
colors = [PALETTE[m] for m in labels]

bar_chart(ax, labels, vals, colors, "Similarity Error (SE)", "Similarity Error")
plt.tight_layout()
fig.savefig(OUTDIR / "2_similarity_error.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved 2_similarity_error.png")


# ═══════════════════════════════════════════════════════════════
#  3. Structural Coherence (grouped bar chart)
# ═══════════════════════════════════════════════════════════════
print("▶ Plot 3: Structural Coherence")
struct_metrics = ["block_coherence", "rep_density", "avg_offdiag"]
struct_labels  = ["Block\nCoherence", "Repetition\nDensity", "Avg Off-Diagonal\nSimilarity"]
all_models_struct = THREE_MODELS + ["Human Composed"]

struct_data = {}
for model in all_models_struct:
    df = load_df(SSM_CSVS[model])
    struct_data[model] = {m: df[m].mean() if not df.empty else 0 for m in struct_metrics}

x      = np.arange(len(struct_metrics))
width  = 0.18
fig, ax = plt.subplots(figsize=(9, 5))
style_fig(fig)

for i, model in enumerate(all_models_struct):
    offset = (i - len(all_models_struct)/2 + 0.5) * width
    vals   = [struct_data[model][m] for m in struct_metrics]
    bars   = ax.bar(x + offset, vals, width=width*0.9,
                    color=PALETTE[model], label=model, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(struct_labels, fontsize=10)
ax.set_ylabel("Mean Value", fontsize=11)
ax.set_title("Structural Coherence Analysis", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9, framealpha=0.8)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
plt.tight_layout()
fig.savefig(OUTDIR / "3_structural_coherence.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved 3_structural_coherence.png")


# ═══════════════════════════════════════════════════════════════
#  4. Musical Quality – Box Plots
# ═══════════════════════════════════════════════════════════════
print("▶ Plot 4: Musical Quality (box plots)")
mq_metrics = ["pitch_range", "duration_entropy", "key_stability"]
mq_labels  = ["Pitch Range", "Duration Entropy", "Key Stability"]
all_models_mq = THREE_MODELS + ["Human Composed"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
style_fig(fig)
fig.suptitle("Musical Quality Metrics", fontsize=14, fontweight="bold", y=1.01)

for ax, metric, label in zip(axes, mq_metrics, mq_labels):
    data   = []
    labels = []
    colors = []
    for model in all_models_mq:
        df = load_df(MUSICAL_CSVS[model])
        if not df.empty and metric in df.columns:
            vals = df[metric].dropna().values
            data.append(vals)
            labels.append(model.replace(" ", "\n"))
            colors.append(PALETTE[model])
        else:
            data.append(np.array([]))
            labels.append(model.replace(" ", "\n"))
            colors.append(PALETTE[model])

    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Value", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
fig.savefig(OUTDIR / "4_musical_quality_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved 4_musical_quality_boxplots.png")


# ═══════════════════════════════════════════════════════════════
#  5. Generation Time Bar Chart
# ═══════════════════════════════════════════════════════════════
print("▶ Plot 5: Generation Time")
fig, ax = plt.subplots(figsize=(7, 4))
style_fig(fig)

labels = THREE_MODELS
vals   = [GEN_TIME[m] for m in labels]
colors = [PALETTE[m] for m in labels]

if any(v is not None for v in vals):
    bar_chart(ax, labels, vals, colors, "Time (seconds per bar)",
              "Generation Time per Bar", fmt=".2f")
else:
    ax.text(0.5, 0.5, "Fill in GEN_TIME values at the top of the script",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
            color="gray", style="italic")
    ax.set_title("Generation Time per Bar", fontsize=12, fontweight="bold")

plt.tight_layout()
fig.savefig(OUTDIR / "5_generation_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved 5_generation_time.png")


# ═══════════════════════════════════════════════════════════════
#  6a. Pitch Distribution Histograms
# ═══════════════════════════════════════════════════════════════
print("▶ Plot 6a: Pitch Distribution")

def extract_pitches(midi_dir, max_files=None):
    pitches = []
    files   = sorted(Path(midi_dir).rglob("*.mid"))
    if max_files:
        files = files[:max_files]
    for fp in files:
        try:
            pm = pretty_midi.PrettyMIDI(str(fp))
            for inst in pm.instruments:
                if not inst.is_drum:
                    pitches.extend([n.pitch for n in inst.notes])
        except Exception:
            pass
    return np.array(pitches)

def extract_durations(midi_dir, max_files=None):
    durs = []
    files = sorted(Path(midi_dir).rglob("*.mid"))
    if max_files:
        files = files[:max_files]
    for fp in files:
        try:
            pm = pretty_midi.PrettyMIDI(str(fp))
            for inst in pm.instruments:
                if not inst.is_drum:
                    durs.extend([n.end - n.start for n in inst.notes])
        except Exception:
            pass
    return np.array(durs)

all_models_dist = THREE_MODELS + ["Human Composed"]
print("   Extracting pitches from MIDI files...")
pitch_data = {m: extract_pitches(MIDI_DIRS[m]) for m in all_models_dist}
print("   Done.")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
style_fig(fig)
fig.suptitle("Pitch Distribution Comparison", fontsize=14, fontweight="bold")

MIDI_NOTE_RANGE = (21, 108)

for ax, model in zip(axes.flat, all_models_dist):
    p = pitch_data[model]
    if len(p) > 0:
        ax.hist(p, bins=np.arange(MIDI_NOTE_RANGE[0], MIDI_NOTE_RANGE[1]+1),
                color=PALETTE[model], alpha=0.85, edgecolor="none")
        ax.axvline(np.mean(p), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean={np.mean(p):.1f}")
        ax.legend(fontsize=9)
    ax.set_title(model, fontsize=11, fontweight="bold")
    ax.set_xlabel("MIDI Pitch", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(MIDI_NOTE_RANGE)

plt.tight_layout()
fig.savefig(OUTDIR / "6a_pitch_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved 6a_pitch_distribution.png")


# ─── Overlay comparison (KDE smooth curves) ───
LINESTYLES = ["-", "--", "-.", ":"]
fig, ax = plt.subplots(figsize=(10, 5))
style_fig(fig)

kde_x_pitch = np.linspace(21, 108, 400)
for idx, model in enumerate(all_models_dist):
    p = pitch_data[model]
    if len(p) > 1:
        kde = gaussian_kde(p, bw_method=0.15)
        density = kde(kde_x_pitch)
        ls = LINESTYLES[idx % len(LINESTYLES)]
        ax.plot(kde_x_pitch, density, color=PALETTE[model], linewidth=2.5,
                linestyle=ls, label=model)

ax.set_xlabel("MIDI Pitch", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Pitch Distribution – Overlay Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.95, edgecolor="#cccccc")
ax.spines[["top","right"]].set_visible(False)
ax.grid(linestyle="--", alpha=0.35)
plt.tight_layout()
fig.savefig(OUTDIR / "6b_pitch_distribution_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved 6b_pitch_distribution_overlay.png")


# ═══════════════════════════════════════════════════════════════
#  6c. Note Duration Distribution
# ═══════════════════════════════════════════════════════════════
print("▶ Plot 6c: Note Duration Distribution")
print("   Extracting durations from MIDI files...")
dur_data = {m: extract_durations(MIDI_DIRS[m]) for m in all_models_dist}
print("   Done.")

# Per-model subplots
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
style_fig(fig)
fig.suptitle("Note Duration Distribution Comparison", fontsize=14, fontweight="bold")

for ax, model in zip(axes.flat, all_models_dist):
    d = dur_data[model]
    d = d[d < 4.0]          # clip to 4 s for clarity
    if len(d) > 0:
        ax.hist(d, bins=60, color=PALETTE[model], alpha=0.85, edgecolor="none")
        ax.axvline(np.median(d), color="black", linestyle="--", linewidth=1.5,
                   label=f"Median={np.median(d):.2f}s")
        ax.legend(fontsize=9)
    ax.set_title(model, fontsize=11, fontweight="bold")
    ax.set_xlabel("Duration (seconds)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUTDIR / "6c_duration_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved 6c_duration_distribution.png")

# Overlay comparison (KDE smooth curves)
fig, ax = plt.subplots(figsize=(10, 5))
style_fig(fig)

kde_x_dur = np.linspace(0, 4, 400)
for idx, model in enumerate(all_models_dist):
    d = dur_data[model]
    d = d[(d > 0) & (d < 4.0)]
    if len(d) > 1:
        kde = gaussian_kde(d, bw_method=0.2)
        density = kde(kde_x_dur)
        ls = LINESTYLES[idx % len(LINESTYLES)]
        ax.plot(kde_x_dur, density, color=PALETTE[model], linewidth=2.5,
                linestyle=ls, label=model)

ax.set_xlabel("Duration (seconds)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Note Duration Distribution – Overlay Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.95, edgecolor="#cccccc")
ax.spines[["top","right"]].set_visible(False)
ax.grid(linestyle="--", alpha=0.35)
plt.tight_layout()
fig.savefig(OUTDIR / "6d_duration_distribution_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved 6d_duration_distribution_overlay.png")


# ═══════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("All figures saved to:", OUTDIR)
print("="*55)
print("\nFiles generated:")
for f in sorted(OUTDIR.glob("*.png")):
    print(f"  {f.name}")

print("\n⚠  Remember to fill in PERPLEXITY and GEN_TIME values at")
print("   the top of this script and re-run to complete plots 1 & 5.")
