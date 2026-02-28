"""
Update compare_models.ipynb to support three models by adding Model 3.
"""
import json
from pathlib import Path

NB_PATH = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/evaluation/compare_models.ipynb")
MODEL_3_PATH = "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/notebooks/xLSTM-2/generated_batch_20260225_155134"

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

def get_source(cell):
    return "".join(cell["source"])

def set_source(cell, lines):
    cell["source"] = lines
    cell["outputs"] = []
    cell["execution_count"] = None

# ── Cell 0: Markdown header ──────────────────────────────────────────────────
md_cell = cells[0]
src = get_source(md_cell)
src = src.replace(
    "# Model Comparison: xLSTM vs Baseline",
    "# Model Comparison: xLSTM Models"
).replace(
    "This notebook evaluates and compares two music generation models",
    "This notebook evaluates and compares three music generation models"
)
if "Model 3" not in src:
    src += (
        "- **Model 3**: "
        f"`{MODEL_3_PATH}`\n"
    )
md_cell["source"] = src.splitlines(keepends=True)

# ── Cell 1: Path definitions ─────────────────────────────────────────────────
paths_cell = cells[1]
old_lines = paths_cell["source"]
new_lines = []
inserted = False
for line in old_lines:
    new_lines.append(line)
    # Insert MODEL_3_PATH right after MODEL_2_PATH assignment
    if 'MODEL_2_PATH = Path(' in line and not inserted:
        new_lines.append(
            f'MODEL_3_PATH = Path("{MODEL_3_PATH}")\n'
        )
        inserted = True

# Update print statements
rebuilt = []
for line in new_lines:
    rebuilt.append(line)
    if 'print(f"Model 2: {MODEL_2_PATH}")' in line or "print(f\\\"Model 2: {MODEL_2_PATH}\\\")" in line:
        rebuilt.append('print(f"Model 3: {MODEL_3_PATH}")\n')
        break

# Merge: keep lines up to the MODEL_2 print, then append model3 print
# Actually, let's do it cleanly by rebuilding the source from scratch
src = get_source(paths_cell)

# Add MODEL_3_PATH after MODEL_2_PATH
if "MODEL_3_PATH" not in src:
    src = src.replace(
        'MODEL_2_PATH = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/notebooks/xLSTM-2/generated_batch_20260128_191749")',
        'MODEL_2_PATH = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/notebooks/xLSTM-2/generated_batch_20260128_191749")\n'
        f'MODEL_3_PATH = Path("{MODEL_3_PATH}")'
    )

# Add print for model3
if 'print(f"Model 3: {MODEL_3_PATH}")' not in src and 'Model 3' not in src:
    src = src.replace(
        'print(f"Model 2: {MODEL_2_PATH}")',
        'print(f"Model 2: {MODEL_2_PATH}")\nprint(f"Model 3: {MODEL_3_PATH}")'
    )

set_source(paths_cell, src.splitlines(keepends=True))

# ── Cell 2: Musical metrics ───────────────────────────────────────────────────
musical_cell = cells[2]
src = get_source(musical_cell)
if 'run_musical_metrics(MODEL_3_PATH, "model3")' not in src:
    src = src.rstrip("\n") + '\nrun_musical_metrics(MODEL_3_PATH, "model3")'
set_source(musical_cell, src.splitlines(keepends=True))

# ── Cell 4: SSM metrics ───────────────────────────────────────────────────────
ssm_cell = cells[4]
src = get_source(ssm_cell)
if 'run_ssm_metrics(MODEL_3_PATH, "model3")' not in src:
    src = src.rstrip("\n") + '\nrun_ssm_metrics(MODEL_3_PATH, "model3")'
set_source(ssm_cell, src.splitlines(keepends=True))

# ── Cell 6: SE metric ─────────────────────────────────────────────────────────
se_cell = cells[6]
src = get_source(se_cell)
if 'run_se_metric(MODEL_3_PATH, "model3")' not in src:
    src = src.rstrip("\n") + '\nrun_se_metric(MODEL_3_PATH, "model3")'
set_source(se_cell, src.splitlines(keepends=True))

# ── Cell 8: Aggregate comparison table ───────────────────────────────────────
agg_cell = cells[8]
src = get_source(agg_cell)

# Add data_m3 loading after data_m2 block
if "data_m3" not in src:
    data_m3_block = (
        '\ndata_m3 = load_summary(RESULTS_DIR / "musical_summary_model3.csv")\n'
        'data_m3.update(load_summary(RESULTS_DIR / "ssm_summary_model3.csv"))\n'
        "data_m3['Similarity Error (SE)'] = load_se(RESULTS_DIR / \"se_results_model3.json\")\n"
    )
    src = src.replace(
        "# Construct Comparison Table",
        data_m3_block + "\n# Construct Comparison Table"
    )

# Extend all_metrics union
src = src.replace(
    "all_metrics = sorted(list(set(data_m1.keys()) | set(data_m2.keys())))",
    "all_metrics = sorted(list(set(data_m1.keys()) | set(data_m2.keys()) | set(data_m3.keys())))"
)

# Add Model 3 column in comparison_data dict
if '"Model 3 (Feb 25 Batch)"' not in src:
    src = src.replace(
        '"Model 2 (Jan 28 Batch)": data_m2.get(m, "N/A")',
        '"Model 2 (Jan 28 Batch)": data_m2.get(m, "N/A"),\n        "Model 3 (Feb 25 Batch)": data_m3.get(m, "N/A")'
    )

set_source(agg_cell, src.splitlines(keepends=True))

# Save notebook
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")

# Verify
with open(NB_PATH) as f:
    nb2 = json.load(f)

for i, cell in enumerate(nb2["cells"]):
    src = "".join(cell["source"])
    if "MODEL_3_PATH" in src or "model3" in src or "Model 3" in src:
        print(f"  Cell {i} ({cell['cell_type']}): contains Model 3 references ✓")
