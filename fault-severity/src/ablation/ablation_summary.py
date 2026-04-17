# ablation_summary.py
"""
Ablation Study Summary — All 4 Models × 6 Configurations
==========================================================
Consolidates results from:
  - Full pipeline (with SMOTE)      : {model}_smote_results.json
  - Full pipeline (without SMOTE)   : {model}_results.json          ← already done
  - Ablation I  (no scaler)         : ablation_i_no_scaler_{model}_results.json
  - Ablation II (no scaler+SMOTE)   : ablation_ii_no_scaler_no_smote_{model}_results.json
  - Ablation III (no metrics)       : ablation_iii_no_metrics_{model}_results.json
  - Ablation IV (source code only)  : ablation_iv_source_code_only_{model}_results.json

Run after all ablation scripts complete:
  python ablation_summary.py
"""

import os, json
import pandas as pd

ALL_MODELS  = ["codebert", "graphcodebert", "unixcoder", "codet5p"]
METRIC_KEYS = ["Accuracy", "Macro F1", "Weighted F1", "MCC", "G-Mean"]

CONFIG_ORDER = [
    "Full Pipeline (w/ SMOTE)",
    "Full Pipeline (no SMOTE)",
    "I: No Scaler",
    "II: No Scaler + No SMOTE",
    "III: No Metrics",
    "IV: Source Code Only",
]

CONFIG_NOTES = {
    "Full Pipeline (w/ SMOTE)":   "Scaler ✓ | SMOTE ✓ | Metrics ✓",
    "Full Pipeline (no SMOTE)":   "Scaler ✓ | SMOTE ✗ | Metrics ✓",
    "I: No Scaler":               "Scaler ✗ | SMOTE ✓ | Metrics ✓",
    "II: No Scaler + No SMOTE":   "Scaler ✗ | SMOTE ✗ | Metrics ✓",
    "III: No Metrics":            "Scaler ✓ | SMOTE ✓ | Metrics ✗",
    "IV: Source Code Only":       "Scaler ✗ | SMOTE ✗ | Metrics ✗",
}


# ── Load helper ───────────────────────────────────────────────────────────────
def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Collect all results into a flat DataFrame ─────────────────────────────────
def collect():
    rows = []
    for m in ALL_MODELS:
        sources = {
            "Full Pipeline (w/ SMOTE)": load(f"results/{m}_smote_results.json"),
            "Full Pipeline (no SMOTE)": load(f"results/{m}_results.json"),
            "I: No Scaler":             load(f"results/ablation/ablation_i_no_scaler_{m}_results.json"),
            "II: No Scaler + No SMOTE": load(f"results/ablation/ablation_ii_no_scaler_no_smote_{m}_results.json"),
            "III: No Metrics":          load(f"results/ablation/ablation_iii_no_metrics_{m}_results.json"),
            "IV: Source Code Only":     load(f"results/ablation/ablation_iv_source_code_only_{m}_results.json"),
        }
        for cfg, data in sources.items():
            if data is None:
                continue
            rows.append({
                "Config": cfg,
                "Model":  m.upper(),
                **{k: data.get(k, "-") for k in METRIC_KEYS},
                "Notes":  CONFIG_NOTES[cfg],
            })
    return pd.DataFrame(rows)


# ── Print per-model breakdown ─────────────────────────────────────────────────
def print_per_model(df):
    for model in [m.upper() for m in ALL_MODELS]:
        sub = df[df["Model"] == model]
        if sub.empty:
            continue

        print(f"\n  ┌─ {model} {'─'*(57 - len(model))}┐")
        print(f"  │ {'Configuration':<30} {'Acc':>7} {'MacF1':>7} {'WtdF1':>7} {'MCC':>7} {'GMean':>7}  │")
        print(f"  │ {'─'*68} │")

        for cfg in CONFIG_ORDER:
            row = sub[sub["Config"] == cfg]
            if row.empty:
                continue
            r = row.iloc[0]
            acc  = f"{r['Accuracy']:.4f}"   if isinstance(r['Accuracy'],   float) else "-"
            mf1  = f"{r['Macro F1']:.4f}"   if isinstance(r['Macro F1'],   float) else "-"
            wf1  = f"{r['Weighted F1']:.4f}" if isinstance(r['Weighted F1'], float) else "-"
            mcc  = f"{r['MCC']:.4f}"         if isinstance(r['MCC'],         float) else "-"
            gm   = f"{r['G-Mean']:.4f}"      if isinstance(r['G-Mean'],      float) else "-"
            print(f"  │ {cfg:<30} {acc:>7} {mf1:>7} {wf1:>7} {mcc:>7} {gm:>7}  │")

        print(f"  └{'─'*70}┘")


# ── Print delta table (Δ Macro F1 vs. Full Pipeline with SMOTE) ───────────────
def print_delta_table(df):
    print(f"\n\n{'='*80}")
    print("  Δ Macro F1 vs. Full Pipeline (with SMOTE)  [negative = worse]")
    print(f"{'='*80}")

    col_w = 14
    header = f"  {'Configuration':<30}"
    for m in ALL_MODELS:
        header += f"  {m.upper():>{col_w}}"
    print(header)
    print(f"  {'─'*(30 + (col_w + 2) * len(ALL_MODELS))}")

    for cfg in CONFIG_ORDER:
        if cfg == "Full Pipeline (w/ SMOTE)":
            continue
        row_str = f"  {cfg:<30}"
        for m in ALL_MODELS:
            full = df[(df["Model"] == m.upper()) & (df["Config"] == "Full Pipeline (w/ SMOTE)")]
            curr = df[(df["Model"] == m.upper()) & (df["Config"] == cfg)]
            if full.empty or curr.empty:
                row_str += f"  {'n/a':>{col_w}}"
                continue
            try:
                delta   = float(curr.iloc[0]["Macro F1"]) - float(full.iloc[0]["Macro F1"])
                sign    = "+" if delta >= 0 else ""
                row_str += f"  {sign}{delta:>+.4f}      "
            except Exception:
                row_str += f"  {'err':>{col_w}}"
        print(row_str)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = collect()

    if df.empty:
        print("\nNo results found. Run all ablation scripts first:\n")
        print("  python ablation_i_no_scaler.py             --models all")
        print("  python ablation_ii_no_scaler_no_smote.py   --models all")
        print("  python ablation_iii_no_metrics.py          --models all")
        print("  python ablation_iv_source_code_only.py     --models all")
        print("  python ablation_summary.py")
        return

    print(f"\n{'='*80}")
    print("  ABLATION STUDY — Per-Model Results")
    print(f"  {'─'*78}")
    print(f"  Legend: Acc = Accuracy | MacF1 = Macro F1 | WtdF1 = Weighted F1")
    print(f"          MCC = Matthews Corr. Coeff. | GMean = Geometric Mean")
    print(f"{'='*80}")
    print_per_model(df)
    print_delta_table(df)

    # Component contribution summary
    print(f"\n\n{'='*80}")
    print("  COMPONENT CONTRIBUTION ANALYSIS (averaged across all 4 models)")
    print(f"{'='*80}")
    components = {
        "RobustScaler alone":         ("Full Pipeline (w/ SMOTE)", "I: No Scaler"),
        "SMOTE alone":                ("Full Pipeline (w/ SMOTE)", "Full Pipeline (no SMOTE)"),
        "Code Metrics alone":         ("Full Pipeline (w/ SMOTE)", "III: No Metrics"),
        "All components combined":    ("Full Pipeline (w/ SMOTE)", "IV: Source Code Only"),
    }
    for label, (full_cfg, ablated_cfg) in components.items():
        deltas = []
        for m in ALL_MODELS:
            full = df[(df["Model"] == m.upper()) & (df["Config"] == full_cfg)]
            abl  = df[(df["Model"] == m.upper()) & (df["Config"] == ablated_cfg)]
            if full.empty or abl.empty:
                continue
            try:
                deltas.append(float(full.iloc[0]["Macro F1"]) - float(abl.iloc[0]["Macro F1"]))
            except Exception:
                pass
        if deltas:
            avg = sum(deltas) / len(deltas)
            print(f"  Contribution of {label:<30}: avg Δ Macro F1 = {avg:+.4f}")

    # Save CSV
    os.makedirs("results/ablation", exist_ok=True)
    out_path = "results/ablation/ablation_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Full table saved → {out_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
