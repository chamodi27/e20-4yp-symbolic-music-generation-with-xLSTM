
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics from a CSV file.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input metrics CSV.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output summary CSV.")
    
    args = parser.parse_args()
    
    csv_path = Path(args.input_csv).resolve()
    out_path = Path(args.output_csv).resolve()
    
    if not csv_path.exists():
        print(f"[ERROR] Input CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        print(f"[ERROR] Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Filter out rows with errors
    if 'error' in df.columns:
        df_clean = df[df['error'].isna() | (df['error'] == "") | (df['error'].astype(str) == "nan")].copy()
        print(f"[INFO] Rows after filtering errors: {len(df_clean)}")
    else:
        df_clean = df.copy()

    # Define potential numeric columns
    # We aggregate whatever numeric columns are present from this list
    potential_cols = [
        "bars_N",
        "key_stability", 
        "key_change_rate", 
        "chord_diversity", 
        "pitch_range", 
        "stepwise_ratio", 
        "note_density_mean", 
        "note_density_std", 
        "duration_entropy",
        "avg_offdiag",
        "rep_density",
        "struct_entropy",
        "block_coherence"
    ]
    
    numeric_cols = [c for c in potential_cols if c in df_clean.columns]

    # Ensure they are numeric
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Compute Mean and Std
    summary_data = []
    print("\nMetrics Summary (Mean ± Std):")
    print("-" * 60)
    print(f"{'Metric':<25} | {'Formatted':<30}")
    print("-" * 60)

    for col in numeric_cols:
        mean_val = df_clean[col].mean()
        std_val = df_clean[col].std()
        
        # Handle cases with no valid data
        if pd.isna(mean_val):
            mean_val = 0.0
            std_val = 0.0
            
        formatted = f"{mean_val:.4f} ± {std_val:.4f}"
        
        summary_data.append({
            "Metric": col,
            "Mean": mean_val,
            "Std": std_val,
            "Formatted": formatted
        })
        print(f"{col:<25} | {formatted:<30}")

    try:
        summary_df = pd.DataFrame(summary_data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"[INFO] Summary saved to {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save summary: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
