"""
eval_hybrid.py — Step 4b: Compare baseline vs. hybrid accuracy and resource savings.

Prints a table of accuracy, F1, accuracy drop, and hardware cost
for every experiment.  Saves JSON to outputs/results/eval_{exp}.json.
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import config
from data_utils import get_loaders
from models import load_baseline, build_hybrid
from op_counter import count_ops


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(yb.tolist())
    return np.array(preds), np.array(labels)


def eval_experiment(exp_name: str, device, smoke_test: bool):
    eq_path = config.EQ_DIR / exp_name / "equations.json"
    if not eq_path.exists():
        print(f"[{exp_name}] equations.json missing — run select_equation.py first.")
        return None

    with open(eq_path) as f:
        eq_data = json.load(f)
    equations = [c["equation"]
                 for c in sorted(eq_data["classes"], key=lambda c: c["class"])]
    layer     = config.get_layer(exp_name)

    _, test_loader = get_loaders(smoke_test=smoke_test)

    baseline = load_baseline(device=device)
    bl_preds, labels = _predict(baseline, test_loader, device)
    bl_acc = accuracy_score(labels, bl_preds)
    bl_f1  = f1_score(labels, bl_preds, average="macro", zero_division=0)

    hybrid    = build_hybrid(baseline, layer, equations).to(device)
    hy_preds, _ = _predict(hybrid, test_loader, device)
    hy_acc = accuracy_score(labels, hy_preds)
    hy_f1  = f1_score(labels, hy_preds, average="macro", zero_division=0)

    total_cost   = sum(count_ops(eq, config.OP_COSTS) for eq in equations)
    macs_remain  = config.SR_1L_MACS if layer == "1L" else config.SR_2L_MACS
    mac_reduction = (config.BASELINE_MACS - macs_remain) / config.BASELINE_MACS

    result = {
        "experiment":       exp_name,
        "baseline_acc":     round(bl_acc, 6),
        "baseline_f1":      round(bl_f1, 6),
        "hybrid_acc":       round(hy_acc, 6),
        "hybrid_f1":        round(hy_f1, 6),
        "acc_drop":         round(bl_acc - hy_acc, 6),
        "f1_drop":          round(bl_f1  - hy_f1,  6),
        "total_sr_op_cost": total_cost,
        "baseline_macs":    config.BASELINE_MACS,
        "macs_remaining":   macs_remain,
        "mac_reduction_pct": round(mac_reduction * 100, 2),
    }

    print(f"\n{'─'*60}")
    print(f"Experiment : {exp_name}")
    print(f"  Baseline  acc={bl_acc:.4f}  f1={bl_f1:.4f}  MACs={config.BASELINE_MACS:,}")
    print(f"  Hybrid    acc={hy_acc:.4f}  f1={hy_f1:.4f}  "
          f"hw_cost={total_cost}  MACs_remaining={macs_remain:,}  "
          f"MAC_reduction={mac_reduction*100:.1f}%")
    print(f"  Acc drop  {bl_acc - hy_acc:+.4f}")

    out_path = config.RESULTS_DIR / f"eval_{exp_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate baseline vs hybrid models")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiment", type=str, choices=config.EXPERIMENTS)
    g.add_argument("--all",        action="store_true")
    p.add_argument("--device",     type=str, default=config.DEVICE)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    device  = torch.device(args.device)
    exps    = config.EXPERIMENTS if args.all else [args.experiment]
    results = []
    for exp in exps:
        r = eval_experiment(exp, device, args.smoke_test)
        if r:
            results.append(r)

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print(f"\n{'='*60}")
        print(df[["experiment", "baseline_acc", "hybrid_acc",
                  "acc_drop", "total_sr_op_cost", "mac_reduction_pct"]].to_string(index=False))
        df.to_csv(config.RESULTS_DIR / "eval_summary.csv", index=False)
