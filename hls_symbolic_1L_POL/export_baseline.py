#!/usr/bin/env python3
# export_baseline.py
# Export the ORIGINAL dense MLP (fc1, fc2, fc3) from baseline.pt into a single
# HLS ROM header, plus a NumPy reference for accuracy validation.
#
#   python export_baseline.py --ckpt .../checkpoints/baseline.pt --list   # show keys
#   python export_baseline.py --ckpt .../checkpoints/baseline.pt          # do export
#
# Produces:  baseline_params.h  (W1,B1,W2,B2,W3,B3)   and   baseline_ref.npz
#
# Uses the SAME quantization (ap_fixed<8,2>) as the hybrid, so the only
# difference vs the SR hybrid is the fc3 representation (dense MAC vs shift-add).

import argparse, sys
import numpy as np
import torch

KEY_W1, KEY_B1 = "fc1.weight", "fc1.bias"
KEY_W2, KEY_B2 = "fc2.weight", "fc2.bias"
KEY_W3, KEY_B3 = "fc3.weight", "fc3.bias"

W_TOTAL, W_INT = 8, 2   # must match w_t in mlp.h

def quantize(arr, total=W_TOTAL, intb=W_INT, name=""):
    frac  = total - intb
    scale = float(1 << frac)
    lo    = -(2 ** (intb - 1))
    hi    =  (2 ** (intb - 1)) - 1.0 / scale
    q     = np.round(arr.astype(np.float64) * scale) / scale
    n_clip = int(np.sum((q < lo) | (q > hi)))
    if n_clip:
        print(f"  [warn] {name}: {n_clip}/{q.size} clipped to [{lo}, {hi:.4f}] "
              f"(increase W_INT in this script AND mlp.h if many)")
    return np.clip(q, lo, hi)

def emit_matrix(f, name, M):
    rows, cols = M.shape
    f.write(f"static const w_t {name}[{rows}][{cols}] = {{\n")
    for r in range(rows):
        f.write("  {" + ", ".join(f"{v:.6f}" for v in M[r]) +
                ("}," if r < rows-1 else "}") + "\n")
    f.write("};\n\n")

def emit_vector(f, name, v):
    f.write(f"static const w_t {name}[{v.size}] = {{" +
            ", ".join(f"{x:.6f}" for x in v) + "};\n\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    obj = torch.load(args.ckpt, map_location="cpu")
    sd  = obj.get("model_state_dict", obj.get("state_dict", obj)) \
          if isinstance(obj, dict) else obj

    if args.list:
        print(list(sd.keys())); return

    for k in (KEY_W1,KEY_B1,KEY_W2,KEY_B2,KEY_W3,KEY_B3):
        if k not in sd:
            print(f"[error] '{k}' missing. Keys: {list(sd.keys())}"); sys.exit(1)

    W1=sd[KEY_W1].numpy(); b1=sd[KEY_B1].numpy()
    W2=sd[KEY_W2].numpy(); b2=sd[KEY_B2].numpy()
    W3=sd[KEY_W3].numpy(); b3=sd[KEY_B3].numpy()
    print(f"shapes: W1{W1.shape} W2{W2.shape} W3{W3.shape}")
    assert W1.shape==(128,784) and W2.shape==(64,128) and W3.shape==(10,64)

    print("quantizing...")
    q = lambda a,n: quantize(a, name=n)
    W1,b1,W2,b2,W3,b3 = q(W1,"W1"),q(b1,"B1"),q(W2,"W2"),q(b2,"B2"),q(W3,"W3"),q(b3,"B3")

    with open("baseline_params.h","w") as f:
        f.write('#ifndef BASELINE_PARAMS_H\n#define BASELINE_PARAMS_H\n#include "mlp.h"\n\n')
        emit_matrix(f,"W1",W1); emit_vector(f,"B1",b1)
        emit_matrix(f,"W2",W2); emit_vector(f,"B2",b2)
        emit_matrix(f,"W3",W3); emit_vector(f,"B3",b3)
        f.write("#endif\n")
    np.savez("baseline_ref.npz", W1=W1,b1=b1,W2=W2,b2=b2,W3=W3,b3=b3)
    print("wrote baseline_params.h, baseline_ref.npz")

if __name__ == "__main__":
    main()
