"""
extract_traces.py — Step 2: Run the trained model and capture I/O traces.

For a 10-class MLP we collect:
  1L traces: (h2 ∈ R^64,  logits ∈ R^10)  — one CSV per class under traces/1L/
  2L traces: (h1 ∈ R^128, logits ∈ R^10)  — one CSV per class under traces/2L/

Each CSV has columns x0…xD-1 (hidden activations) and y (raw logit for that class).
PySR fits one analytic function per CSV.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

import config
from data_utils import get_tensors
from models import load_baseline


@torch.no_grad()
def _collect_activations(model, X: torch.Tensor, device, batch_size: int = 512):
    model.eval()
    h1_buf, h2_buf, logit_buf = [], [], []
    for i in range(0, len(X), batch_size):
        xb            = X[i : i + batch_size].to(device)
        logits, h1, h2 = model(xb, return_hidden=True)
        h1_buf.append(h1.cpu())
        h2_buf.append(h2.cpu())
        logit_buf.append(logits.cpu())
    return (torch.cat(h1_buf).numpy(),
            torch.cat(h2_buf).numpy(),
            torch.cat(logit_buf).numpy())


def _sanitize(X: np.ndarray, Y: np.ndarray):
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return X[mask], Y[mask]


def _save_per_class(X: np.ndarray, Y: np.ndarray, tag: str, max_samples: int) -> list[str]:
    """
    Write one CSV per output class.  Returns list of saved paths.
    Columns: x0, x1, …, xD-1, y
    """
    if max_samples and len(X) > max_samples:
        rng = np.random.default_rng(config.SEED)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, Y = X[idx], Y[idx]

    out_dir = config.TRACE_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for cls in range(config.NUM_CLASSES):
        data = {f"x{i}": X[:, i] for i in range(X.shape[1])}
        data["y"] = Y[:, cls]
        path = out_dir / f"class_{cls}.csv"
        pd.DataFrame(data).to_csv(path, index=False)
        paths.append(str(path))
    return paths


def extract(args):
    device = torch.device(args.device)
    model  = load_baseline(device=device)

    X_train, _, X_test, _ = get_tensors(smoke_test=args.smoke_test)

    h1_tr, h2_tr, logits_tr = _collect_activations(model, X_train, device)
    h1_te, h2_te, logits_te = _collect_activations(model, X_test,  device)

    # Pool train + test for richer SR fitting data
    H1 = np.concatenate([h1_tr, h1_te])
    H2 = np.concatenate([h2_tr, h2_te])
    L  = np.concatenate([logits_tr, logits_te])

    H1, L1 = _sanitize(H1, L)
    H2, L2 = _sanitize(H2, L)

    paths_1L = _save_per_class(H2, L2, "1L", args.max_samples)
    paths_2L = _save_per_class(H1, L1, "2L", args.max_samples)

    summary = {
        "1L": {"input_dim": int(H2.shape[1]), "n_samples": int(H2.shape[0]),
               "n_classes": config.NUM_CLASSES, "paths": paths_1L},
        "2L": {"input_dim": int(H1.shape[1]), "n_samples": int(H1.shape[0]),
               "n_classes": config.NUM_CLASSES, "paths": paths_2L},
    }
    summary_path = config.TRACE_DIR / "trace_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"1L traces  h2 dim={H2.shape[1]}  N={H2.shape[0]}  → {config.TRACE_DIR / '1L'}")
    print(f"2L traces  h1 dim={H1.shape[1]}  N={H1.shape[0]}  → {config.TRACE_DIR / '2L'}")
    print(f"Summary    → {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Extract I/O traces for symbolic regression")
    p.add_argument("--device",      type=str, default=config.DEVICE)
    p.add_argument("--max-samples", type=int, default=5000,
                   help="Cap samples per trace file (PySR is slow on large N)")
    p.add_argument("--smoke-test",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    extract(parse_args())
