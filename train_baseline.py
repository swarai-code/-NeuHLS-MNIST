"""
train_baseline.py — Step 1: Train BaselineMLP on MNIST to convergence.
Saves best checkpoint to outputs/checkpoints/baseline.pt.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

import config
from data_utils import get_loaders
from models import BaselineMLP


def set_seeds(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb    = xb.to(device), yb.to(device)
        logits    = model(xb)
        total_loss += criterion(logits, yb).item() * xb.size(0)
        correct   += (logits.argmax(1) == yb).sum().item()
        total     += xb.size(0)
    return total_loss / total, correct / total


def train(args):
    set_seeds()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable — falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    train_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        smoke_test=args.smoke_test,
    )

    model     = BaselineMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_rows            = []
    best_test_acc       = 0.0
    best_epoch          = 0
    t0                  = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * xb.size(0)
            tr_total   += xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()

        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)

        log_rows.append(dict(epoch=epoch,
                             train_loss=round(tr_loss, 6), train_acc=round(tr_acc, 6),
                             test_loss=round(te_loss, 6),  test_acc=round(te_acc, 6)))
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"test_loss={te_loss:.4f}  test_acc={te_acc:.4f}")

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_epoch    = epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "test_acc": te_acc, "test_loss": te_loss},
                       config.CKPT_DIR / "baseline.pt")

    elapsed = time.time() - t0
    print(f"\nBest test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
    print(f"Training complete in {elapsed:.1f}s")

    log_path = config.RESULTS_DIR / "baseline_training_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                          "test_loss", "test_acc"])
        w.writeheader()
        w.writerows(log_rows)

    # Re-evaluate best checkpoint
    ckpt = torch.load(config.CKPT_DIR / "baseline.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)

    metrics = {
        "model":               "BaselineMLP",
        "architecture":        f"{config.INPUT_DIM}→{config.H1_DIM}→{config.H2_DIM}→{config.NUM_CLASSES}",
        "best_epoch":          best_epoch,
        "final_test_accuracy": round(final_acc, 6),
        "final_test_loss":     round(final_loss, 6),
        "train_epochs":        args.epochs,
        "batch_size":          args.batch_size,
        "lr":                  args.lr,
        "smoke_test":          args.smoke_test,
        "elapsed_sec":         round(elapsed, 2),
        "baseline_macs":       config.BASELINE_MACS,
        "baseline_flops":      config.BASELINE_FLOPS,
    }
    with open(config.RESULTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFinal test accuracy : {final_acc:.4f}")
    print(f"Checkpoint          : {config.CKPT_DIR / 'baseline.pt'}")
    print(f"Metrics             : {config.RESULTS_DIR / 'baseline_metrics.json'}")


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline MLP on MNIST")
    p.add_argument("--epochs",     type=int,   default=config.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=config.BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=config.LEARNING_RATE)
    p.add_argument("--device",     type=str,   default=config.DEVICE)
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny data subset for quick pipeline check")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
