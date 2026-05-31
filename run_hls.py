"""
run_hls.py — Run Vitis HLS synthesis on all designs and collect resource reports.

Synthesises:
  hls_generated/baseline/
  hls_generated/1L_POL/  hls_generated/1L_SRL/  hls_generated/1L_SCE/
  hls_generated/2L_POL/  hls_generated/2L_SRL/  hls_generated/2L_SCE/

For symbolic experiments, each class is synthesised separately and results
are averaged across the 10 classes.

Usage:
  python run_hls.py
  python run_hls.py --vitis /path/to/vitis_hls   # override binary path
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
from pathlib import Path

import config

VITIS_DEFAULT = "/home/data/tools/Xilinx/Vitis_HLS/2024.1/bin/vitis_hls"

DESIGNS = ["baseline"] + config.EXPERIMENTS


def _parse_report(rpt_path: Path) -> dict:
    """Extract LUT, FF, DSP, latency from Vitis HLS csynth_design report."""
    if not rpt_path.exists():
        return {"LUT": "?", "FF": "?", "DSP": "?", "BRAM": "?", "Latency_cycles": "?"}

    txt = rpt_path.read_text()

    def grab(pattern):
        m = re.search(pattern, txt)
        return m.group(1).strip() if m else "?"

    return {
        "LUT":            grab(r"\|\s*LUT\s*\|\s*(\d+)"),
        "FF":             grab(r"\|\s*FF\s*\|\s*(\d+)"),
        "DSP":            grab(r"\|\s*DSP\s*\|\s*(\d+)"),
        "BRAM":           grab(r"\|\s*BRAM_18K\s*\|\s*(\d+)"),
        "Latency_cycles": grab(r"Latency \(cycles\)\s*\|\s*Min\s*\|\s*(\d+)"),
    }


def _synth(vitis: str, tcl: Path, work_dir: Path, log_path: Path) -> dict:
    """Run vitis_hls -f tcl in work_dir, return parsed resource dict."""
    result = subprocess.run(
        [vitis, "-f", str(tcl)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
    )
    log_path.write_text(result.stdout + result.stderr)

    if result.returncode != 0:
        print(f"    WARNING: synthesis returned exit {result.returncode}")

    # Vitis writes the report under work_dir/<project>/sol1/syn/report/
    rpt_files = list(work_dir.rglob("*_csynth.rpt"))
    if not rpt_files:
        # Try XML report fallback
        rpt_files = list(work_dir.rglob("csynth.xml"))

    if rpt_files:
        return _parse_report(rpt_files[0])

    # Last resort: parse stdout directly
    txt = result.stdout + result.stderr
    def grab(pattern):
        m = re.search(pattern, txt)
        return m.group(1).strip() if m else "?"

    return {
        "LUT":            grab(r"\|\s*LUT\s*\|\s*(\d+)"),
        "FF":             grab(r"\|\s*FF\s*\|\s*(\d+)"),
        "DSP":            grab(r"\|\s*DSP\s*\|\s*(\d+)"),
        "BRAM":           grab(r"\|\s*BRAM_18K\s*\|\s*(\d+)"),
        "Latency_cycles": grab(r"Latency \(cycles\)\s*\|\s*Min\s*\|\s*(\d+)"),
    }


def run_baseline(vitis: str) -> dict:
    work_dir = config.HLS_GEN_DIR / "baseline"
    tcl      = work_dir / "synth.tcl"
    if not tcl.exists():
        print("[baseline] synth.tcl missing — run generate_hls.py first.")
        return {}

    print("[baseline] Synthesising ...")
    log_path = work_dir / "synth.log"
    res = _synth(vitis, tcl, work_dir, log_path)
    print(f"  LUT={res['LUT']}  FF={res['FF']}  DSP={res['DSP']}  Lat={res['Latency_cycles']}")
    return {"design": "baseline", **res}


def run_experiment(exp_name: str, vitis: str) -> dict:
    """Synthesise all 10 classes and average numeric results."""
    exp_dir = config.HLS_GEN_DIR / exp_name
    all_res = []

    for cls in range(config.NUM_CLASSES):
        cls_dir = exp_dir / f"class_{cls}"
        tcl     = cls_dir / "synth.tcl"
        if not tcl.exists():
            print(f"  [{exp_name}] class {cls}: synth.tcl missing, skipping.")
            continue

        print(f"  [{exp_name}] class {cls} synthesising ...")
        log_path = cls_dir / "synth.log"
        res = _synth(vitis, tcl, cls_dir, log_path)
        all_res.append(res)
        print(f"    LUT={res['LUT']}  FF={res['FF']}  DSP={res['DSP']}  Lat={res['Latency_cycles']}")

    if not all_res:
        return {"design": exp_name, "LUT": "?", "FF": "?",
                "DSP": "?", "BRAM": "?", "Latency_cycles": "?"}

    # Average numeric fields across classes
    def avg(key):
        vals = [int(r[key]) for r in all_res if r.get(key, "?") != "?"]
        return str(round(sum(vals) / len(vals))) if vals else "?"

    return {
        "design":         exp_name,
        "LUT":            avg("LUT"),
        "FF":             avg("FF"),
        "DSP":            avg("DSP"),
        "BRAM":           avg("BRAM"),
        "Latency_cycles": avg("Latency_cycles"),
        "per_class":      all_res,
    }


def main(args):
    vitis = args.vitis
    try:
        v = subprocess.run([vitis, "-version"], capture_output=True, text=True)
        print(v.stdout.splitlines()[0])
    except FileNotFoundError:
        print(f"ERROR: vitis_hls not found at {vitis}")
        print("Pass correct path with --vitis /path/to/vitis_hls")
        return

    results = []

    bl = run_baseline(vitis)
    if bl:
        results.append(bl)

    for exp in config.EXPERIMENTS:
        print(f"\n[{exp}]")
        r = run_experiment(exp, vitis)
        results.append(r)

    # Print summary table
    print(f"\n{'='*65}")
    print(f"{'Design':<12} {'LUT':<8} {'FF':<8} {'DSP':<6} {'BRAM':<6} {'Latency(cy)'}")
    print(f"{'-'*65}")
    for r in results:
        print(f"{r['design']:<12} {r['LUT']:<8} {r['FF']:<8} {r['DSP']:<6} "
              f"{r.get('BRAM','?'):<6} {r['Latency_cycles']}")

    out_path = config.RESULTS_DIR / "hls_resources.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Run Vitis HLS synthesis on all designs")
    p.add_argument("--vitis", type=str, default=VITIS_DEFAULT,
                   help="Path to vitis_hls binary")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
