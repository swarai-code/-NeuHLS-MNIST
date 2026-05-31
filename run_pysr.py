"""
run_pysr.py — Step 3: Hardware-aware symbolic regression.

For each experiment and each of the 10 output classes, fits an analytic
function f_j(h) using PySR with an optional hardware latency cost bias.

Outputs per class:
  outputs/pysr/{exp}/class_{j}/hall_of_fame.csv
  outputs/pysr/{exp}/class_{j}/best_equation.txt
  outputs/pysr/{exp}/class_{j}/run_config.json
"""
from __future__ import annotations
import argparse
import io
import json
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

import config


def _build_pysr(op_set_name: str, cls_tempdir: Path, smoke_test: bool):
    from pysr import PySRRegressor

    ops       = config.OP_SETS[op_set_name]
    iters     = 5  if smoke_test else config.PYSR_ITERATIONS
    pops      = 4  if smoke_test else config.PYSR_POPULATIONS
    maxsize   = 10 if smoke_test else config.PYSR_COMPLEXITY

    binary_ops = [o for o in ops if o in ("+", "-", "*", "/")]
    unary_ops  = [o for o in ops if o not in ("+", "-", "*", "/")]

    # Bias SR search toward hardware-cheap expressions
    complexity_of_ops = {op: config.OP_COSTS.get(op, 1) for op in ops}

    return PySRRegressor(
        niterations=iters,
        populations=pops,
        maxsize=maxsize,
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        extra_sympy_mappings={
            "square": lambda x: x ** 2,
            "relu":   lambda x: (x + abs(x)) / 2,
        },
        complexity_of_operators=complexity_of_ops,
        verbosity=0,
        random_state=config.SEED,
        deterministic=True,
        parallelism="serial",
        tempdir=str(cls_tempdir),
    )


def run_experiment(exp_name: str, smoke_test: bool = False, max_samples: int = 0):
    layer     = config.get_layer(exp_name)
    op_set    = config.get_op_set(exp_name)
    trace_dir = config.TRACE_DIR / layer

    if not trace_dir.exists():
        print(f"[{exp_name}] Trace dir missing — run extract_traces.py first.")
        return

    exp_dir = config.PYSR_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for cls in range(config.NUM_CLASSES):
        csv_path = trace_dir / f"class_{cls}.csv"
        if not csv_path.exists():
            print(f"[{exp_name}] class {cls}: trace CSV missing, skipping.")
            results.append({"class": cls, "equation": "0.0", "loss": 9999.0, "complexity": 1})
            continue

        df = (pd.read_csv(csv_path)
                .replace([np.inf, -np.inf], np.nan)
                .dropna())

        x_cols = [c for c in df.columns if c.startswith("x")]
        X = df[x_cols].values.astype(np.float32)
        y = df["y"].values.astype(np.float32)

        # Down-sample if needed
        n_cap = 200 if smoke_test else (max_samples or len(X))
        if len(X) > n_cap:
            rng  = np.random.default_rng(config.SEED + cls)
            idx  = rng.choice(len(X), n_cap, replace=False)
            X, y = X[idx], y[idx]

        cls_dir = exp_dir / f"class_{cls}"
        cls_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{exp_name}] class {cls}: X={X.shape}  fitting …")
        try:
            model = _build_pysr(op_set, cls_dir, smoke_test)

            buf = io.StringIO()
            with redirect_stdout(buf):
                model.fit(X, y)

            hof = model.equations_
            hof.to_csv(cls_dir / "hall_of_fame.csv", index=False)

            # Pick best within complexity budget
            eq_col   = next((c for c in ["equation", "sympy_format"] if c in hof.columns),
                            hof.columns[0])
            valid    = hof[hof["complexity"] <= config.PYSR_COMPLEXITY]
            if valid.empty:
                valid = hof
            best_row = valid.loc[valid["loss"].idxmin()]

            best_eq   = str(best_row[eq_col])
            best_loss = float(best_row["loss"])
            best_cplx = int(best_row["complexity"])

            (cls_dir / "best_equation.txt").write_text(best_eq + "\n")
            with open(cls_dir / "run_config.json", "w") as f:
                json.dump({"experiment": exp_name, "class": cls, "op_set": op_set,
                           "layer": layer, "n_samples": len(y),
                           "equation": best_eq, "loss": best_loss,
                           "complexity": best_cplx}, f, indent=2)

            results.append({"class": cls, "equation": best_eq,
                            "loss": best_loss, "complexity": best_cplx})
            print(f"  → {best_eq[:70]}  loss={best_loss:.6f}  cplx={best_cplx}")

            with open(cls_dir / "run_log.txt", "w") as f:
                f.write(buf.getvalue())

        except Exception as e:
            print(f"[{exp_name}] class {cls} FAILED: {e}")
            traceback.print_exc()
            results.append({"class": cls, "equation": "0.0",
                            "loss": 9999.0, "complexity": 1})

    with open(exp_dir / "experiment_summary.json", "w") as f:
        json.dump({"experiment": exp_name, "classes": results}, f, indent=2)
    print(f"[{exp_name}] Done → {exp_dir / 'experiment_summary.json'}")


def parse_args():
    p = argparse.ArgumentParser(description="Hardware-aware symbolic regression (PySR)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiment", type=str, choices=config.EXPERIMENTS)
    g.add_argument("--all",        action="store_true")
    p.add_argument("--smoke-test",  action="store_true")
    p.add_argument("--max-samples", type=int, default=3000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exps = config.EXPERIMENTS if args.all else [args.experiment]
    for exp in exps:
        run_experiment(exp, smoke_test=args.smoke_test, max_samples=args.max_samples)
