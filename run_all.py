"""
run_all.py — Master pipeline runner.

Stages (in order):
  train      → train_baseline.py
  traces     → extract_traces.py
  pysr       → run_pysr.py --all
  select     → select_equation.py --all
  finetune   → finetune_hybrid.py --all
  eval       → eval_hybrid.py --all
  hls        → generate_hls.py --all

Usage:
  python run_all.py                    # full pipeline
  python run_all.py --smoke-test       # quick end-to-end check
  python run_all.py --start-from eval  # resume from eval stage
  python run_all.py --skip pysr        # skip one stage
"""
import argparse
import subprocess
import sys

import config

STAGES = [
    ("train",    ["python", "train_baseline.py"]),
    ("traces",   ["python", "extract_traces.py"]),
    ("pysr",     ["python", "run_pysr.py",          "--all"]),
    ("select",   ["python", "select_equation.py",   "--all"]),
    ("finetune", ["python", "finetune_hybrid.py",   "--all"]),
    ("eval",     ["python", "eval_hybrid.py",       "--all"]),
    ("hls",      ["python", "generate_hls.py",      "--all"]),
]
STAGE_NAMES = [s for s, _ in STAGES]


def run_stage(name: str, cmd: list, smoke_test: bool, skip: set) -> bool:
    if name in skip:
        print(f"[run_all] Skipping stage: {name}")
        return True
    if smoke_test:
        cmd = cmd + ["--smoke-test"]
    print(f"\n{'='*60}\n[run_all] Stage: {name}\n{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[run_all] Stage '{name}' failed (exit {result.returncode}).")
        return False
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Run full NeuHLS MNIST pipeline")
    p.add_argument("--smoke-test",  action="store_true",
                   help="Tiny data / 5 SR iterations for fast end-to-end check")
    p.add_argument("--skip",        nargs="*", default=[],
                   choices=STAGE_NAMES, metavar="STAGE")
    p.add_argument("--start-from",  type=str,  default=None,
                   choices=STAGE_NAMES, metavar="STAGE",
                   help="Skip all stages before this one")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    skip = set(args.skip)

    if args.start_from:
        skip.update(STAGE_NAMES[:STAGE_NAMES.index(args.start_from)])

    for name, cmd in STAGES:
        if not run_stage(name, cmd, args.smoke_test, skip):
            print(f"\n[run_all] Pipeline aborted at stage '{name}'.")
            sys.exit(1)

    print("\n[run_all] All stages completed successfully.")
