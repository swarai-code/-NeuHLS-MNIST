"""
finetune_hybrid.py — Step 4a (optional): Fine-tune non-symbolic layers after drop-in.

After SR replaces one or two layers the frozen upstream weights may not be
perfectly adapted.  This script re-trains any trainable parameters while
keeping the symbolic expressions fixed.

For Hybrid1L: fc1, fc2 weights are trainable (fc3 → symbolic).
For Hybrid2L: fc1 weights are trainable  (fc2+fc3 → symbolic).
"""
from __future__ import annotations
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import config
from data_utils import get_loaders
from models import load_baseline, build_hybrid
from op_counter import count_ops


def _finetune(model, train_loader, test_loader, epochs: int, lr: float, device):
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr) if trainable else None

    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if optimizer:
                optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            if optimizer:
                loss.backward()
                optimizer.step()
            tr_loss    += loss.item() * xb.size(0)
            tr_total   += xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()

        model.eval()
        te_correct, te_total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds   = model(xb).argmax(1)
                te_correct  += (preds == yb).sum().item()
                te_total    += xb.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(yb.cpu().tolist())

        te_acc = te_correct / te_total
        f1     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        rows.append({"epoch": epoch, "train_acc": tr_correct / tr_total,
                     "test_acc": te_acc, "f1": f1})
        print(f"  epoch {epoch:3d}  test_acc={te_acc:.4f}  f1={f1:.4f}")

    return rows


def finetune_experiment(exp_name: str, epochs: int, lr: float, device, smoke_test: bool):
    eq_path = config.EQ_DIR / exp_name / "equations.json"
    if not eq_path.exists():
        print(f"[{exp_name}] equations.json missing — run select_equation.py first.")
        return None

    with open(eq_path) as f:
        eq_data = json.load(f)

    equations = [c["equation"]
                 for c in sorted(eq_data["classes"], key=lambda c: c["class"])]
    layer     = config.get_layer(exp_name)

    baseline  = load_baseline(device=device)
    model     = build_hybrid(copy.deepcopy(baseline), layer, equations).to(device)

    train_loader, test_loader = get_loaders(smoke_test=smoke_test)

    print(f"\n[{exp_name}] Fine-tuning hybrid-{layer} for {epochs} epochs …")
    t0   = time.time()
    rows = _finetune(model, train_loader, test_loader, epochs, lr, device)

    ckpt_path = config.CKPT_DIR / f"{exp_name}_finetuned.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "experiment": exp_name, "epochs": epochs}, ckpt_path)

    best       = max(rows, key=lambda r: r["test_acc"])
    total_cost = sum(count_ops(eq, config.OP_COSTS) for eq in equations)

    result = {
        "experiment":    exp_name,
        "layer":         layer,
        "best_test_acc": round(best["test_acc"], 6),
        "best_f1":       round(best["f1"], 6),
        "best_epoch":    best["epoch"],
        "total_op_cost": total_cost,
        "elapsed_sec":   round(time.time() - t0, 2),
    }
    print(f"  → best_test_acc={result['best_test_acc']:.4f}  "
          f"f1={result['best_f1']:.4f}  hw_cost={total_cost}")
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune non-symbolic layers of hybrid models")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiment", type=str, choices=config.EXPERIMENTS)
    g.add_argument("--all",        action="store_true")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--device",     type=str,   default=config.DEVICE)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    device  = torch.device(args.device)
    exps    = config.EXPERIMENTS if args.all else [args.experiment]
    results = []

    for exp in exps:
        r = finetune_experiment(exp, args.epochs, args.lr, device, args.smoke_test)
        if r:
            results.append(r)

    if results:
        import csv, pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(config.RESULTS_DIR / "finetuned_results.csv", index=False)
        with open(config.RESULTS_DIR / "finetuned_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {config.RESULTS_DIR}")
