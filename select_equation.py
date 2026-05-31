"""
select_equation.py — Step 3b: Pick the best equation per class from PySR hall-of-fame.

Selection policy (mirrors original NeuSym):
  1. Retain only rows with complexity ≤ Cmax
  2. Among those, pick minimum loss
  3. Break ties by total hardware operator cost (via op_counter)

Writes outputs/equations/{exp}/equations.json with one entry per output class.
"""
import argparse
import json

import numpy as np
import pandas as pd

import config
from op_counter import count_ops


def _select_for_class(hof_path, cmax: int) -> dict:
    df = (pd.read_csv(hof_path)
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["loss"]))

    eq_col   = next((c for c in ["equation", "sympy_format", "lambda_format"]
                     if c in df.columns), df.columns[0])
    loss_col = "loss" if "loss" in df.columns else "score"

    valid = df[df["complexity"] <= cmax]
    if valid.empty:
        valid = df

    min_loss = valid[loss_col].min()
    tied     = valid[valid[loss_col] == min_loss]

    if len(tied) > 1:
        hw_costs = tied[eq_col].apply(lambda e: count_ops(str(e), config.OP_COSTS))
        best_row = tied.iloc[hw_costs.values.argmin()]
    else:
        best_row = tied.iloc[0]

    return {
        "equation":   str(best_row[eq_col]),
        "loss":       float(best_row[loss_col]),
        "complexity": int(best_row["complexity"]),
        "op_cost":    count_ops(str(best_row[eq_col]), config.OP_COSTS),
    }


def select_experiment(exp_name: str, cmax: int) -> list[dict]:
    eq_dir = config.EQ_DIR / exp_name
    eq_dir.mkdir(parents=True, exist_ok=True)

    all_classes = []
    for cls in range(config.NUM_CLASSES):
        hof_path = config.PYSR_DIR / exp_name / f"class_{cls}" / "hall_of_fame.csv"
        if not hof_path.exists():
            print(f"[{exp_name}] class {cls}: hall_of_fame.csv not found — using fallback 0.0")
            all_classes.append({"class": cls, "equation": "0.0",
                                 "loss": 9999.0, "complexity": 1, "op_cost": 0})
            continue

        result          = _select_for_class(hof_path, cmax)
        result["class"] = cls
        all_classes.append(result)
        print(f"[{exp_name}] class {cls}: {result['equation'][:60]}  "
              f"loss={result['loss']:.6f}  cplx={result['complexity']}  "
              f"hw_cost={result['op_cost']}")

    out_path = eq_dir / "equations.json"
    with open(out_path, "w") as f:
        json.dump({"experiment": exp_name, "cmax": cmax, "classes": all_classes}, f, indent=2)
    print(f"Saved → {out_path}\n")
    return all_classes


def parse_args():
    p = argparse.ArgumentParser(description="Select best SR equations from hall-of-fame")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiment", type=str, choices=config.EXPERIMENTS)
    g.add_argument("--all",        action="store_true")
    p.add_argument("--cmax", type=int, default=config.PYSR_COMPLEXITY)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exps = config.EXPERIMENTS if args.all else [args.experiment]
    for exp in exps:
        select_experiment(exp, args.cmax)
