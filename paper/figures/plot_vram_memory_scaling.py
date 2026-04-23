"""
Generate vram_memory_scaling.pdf for the DMGR paper section.
Run with: /home/e20363/miniconda3/envs/pytorch_env/bin/python plot_vram_memory_scaling.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Data ─────────────────────────────────────────────────────────────────────
lengths = [1000, 2000, 4000, 8000, 12000]

# Incremental VRAM (peak - idle), in MB
# xLSTM: torch allocator values (more precise than pynvml)
# Lookback RNN: TF pre-allocated pool — constant, not sequence-dependent
# Museformer: pynvml values
xlstm_incr      = [20.90, 21.39, 22.36, 24.31, 26.27]
museformer_incr = [190.0, 956.0, 6250.0, 31874.0, 47986.0]
lookback_incr   = [47171.3, 47171.3, 47171.3, 47171.3, 47171.3]

# VRAM growth from L=1k baseline (used in left subplot)
xlstm_delta      = [v - xlstm_incr[0]      for v in xlstm_incr]
museformer_delta = [v - museformer_incr[0] for v in museformer_incr]
lookback_delta   = [v - lookback_incr[0]   for v in lookback_incr]  # all 0

# DMGR values (MB / 1k tokens or steps)
dmgr_values = {
    "Lookback\nRNN":  0.0,
    "xLSTM":          0.4882,
    "Museformer":     4345.0909,
}

# ── Style ─────────────────────────────────────────────────────────────────────
COLOR_LOOKBACK   = "#2196F3"   # blue
COLOR_XLSTM      = "#4CAF50"   # green  (proposed model)
COLOR_MUSEFORMER = "#F44336"   # red

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
fig.subplots_adjust(wspace=0.38)

# ── Subplot (a): VRAM growth from 1k baseline ─────────────────────────────────
x_ticks = [1, 2, 4, 8, 12]   # in thousands

# Use log1p so zero values sit at the bottom of a log-like axis
y_lookback   = np.log1p(lookback_delta)
y_xlstm      = np.log1p(xlstm_delta)
y_museformer = np.log1p(museformer_delta)

ax1.plot(x_ticks, y_lookback,   color=COLOR_LOOKBACK,   marker="s", linewidth=1.6,
         markersize=5, label="Lookback RNN (DMGR≈0)")
ax1.plot(x_ticks, y_xlstm,      color=COLOR_XLSTM,      marker="^", linewidth=1.6,
         markersize=5, label="xLSTM (DMGR=0.49)")
ax1.plot(x_ticks, y_museformer, color=COLOR_MUSEFORMER, marker="o", linewidth=1.6,
         markersize=5, label="Museformer (DMGR=4345)")

# Custom y-ticks: map log1p values back to original MB values
ref_vals = [0, 1, 10, 100, 1000, 10000, 50000]
ax1.set_yticks([np.log1p(v) for v in ref_vals])
ax1.set_yticklabels(["0", "1", "10", "100", "1k", "10k", "50k"])

ax1.set_xlabel("Generation Length (×1k tokens / steps)")
ax1.set_ylabel("VRAM Growth from 1k Baseline (MB)")
ax1.set_title("(a) Incremental VRAM Growth")
ax1.set_xticks(x_ticks)
ax1.set_xticklabels([f"{v}k" for v in x_ticks])
ax1.legend(loc="upper left", framealpha=0.9,
           bbox_to_anchor=(0.0, -0.22), ncol=1)
ax1.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Subplot (b): DMGR bar chart ───────────────────────────────────────────────
models      = list(dmgr_values.keys())
dmgr_raw    = list(dmgr_values.values())
colors_bar  = [COLOR_LOOKBACK, COLOR_XLSTM, COLOR_MUSEFORMER]

# Use log1p so DMGR=0 renders at the bottom without -inf
bar_heights = [np.log1p(v) for v in dmgr_raw]
bars = ax2.bar(models, bar_heights, color=colors_bar, width=0.5,
               edgecolor="white", linewidth=0.8)

# Y-ticks mapped back to DMGR values
dmgr_ticks = [0, 1, 10, 100, 1000, 5000]
ax2.set_yticks([np.log1p(v) for v in dmgr_ticks])
ax2.set_yticklabels(["0", "1", "10", "100", "1k", "5k"])

# Annotate bar tops
for bar, val in zip(bars, dmgr_raw):
    label = "≈0" if val == 0.0 else f"{val:.2f}" if val < 10 else f"{val:,.0f}"
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.05,
             label, ha="center", va="bottom", fontsize=8, fontweight="bold")

ax2.set_ylabel("DMGR (MB / 1k tokens or steps)")
ax2.set_title("(b) Decode Memory Growth Rate")
ax2.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "vram_memory_scaling.pdf"
fig.savefig(out, bbox_inches="tight", format="pdf")
print(f"Saved: {out}")
