#!/usr/bin/env python3
# export_weights.py
# NeuSym-HLS: export trained fc1/fc2 weights to HLS ROM headers.
#
# Produces:
#   fc1_params.h   ->  W1[128][784], B1[128]
#   fc2_params.h   ->  W2[64][128],  B2[64]
#   mlp_ref.npz    ->  quantized weights for a NumPy reference forward pass
#
# Usage:
#   python export_weights.py --ckpt path/to/baseline.pt
#
# IMPORTANT: use the SAME checkpoint your fc3 equations were distilled from.
# If fc1/fc2 were fine-tuned after the symbolic swap (finetune_hybrid.py),
# point --ckpt at the fine-tuned weights.

import argparse, sys
import numpy as np
import torch


# 1) EDIT THESE if `print(list(sd.keys()))` shows different names.
KEY_W1, KEY_B1 = "fc1.weight", "fc1.bias"
KEY_W2, KEY_B2 = "fc2.weight", "fc2.bias"


# 2) Quantization must match the HLS types in mlp.h.
#    w_t  = ap_fixed<8,2>  -> 6 fractional bits, range [-2, 1.984375]
#    bias is stored in the same w_t ROM here; widen W_INT below if biases clip.
W_TOTAL, W_INT = 8, 2

def quantize(arr, total=W_TOTAL, intb=W_INT, name=""):
    frac  = total - intb
    scale = float(1 << frac)
    lo    = -(2 ** (intb - 1))
    hi    =  (2 ** (intb - 1)) - 1.0 / scale
    q     = np.round(arr.astype(np.float64) * scale) / scale
    n_clip = int(np.sum((q < lo) | (q > hi)))
    if n_clip:
        print(f"  [warn] {name}: {n_clip}/{q.size} values clipped to "
              f"[{lo}, {hi:.4f}] (consider increasing W_INT)")
    return np.clip(q, lo, hi)

def emit_matrix(f, ctype, name, M):
    rows, cols = M.shape
    f.write(f"static const {ctype} {name}[{rows}][{cols}] = {{\n")
    for r in range(rows):
        vals = ", ".join(f"{v:.6f}" for v in M[r])
        f.write(f"  {{{vals}}}{',' if r < rows-1 else ''}\n")
    f.write("};\n\n")

def emit_vector(f, ctype, name, v):
    vals = ", ".join(f"{x:.6f}" for x in v)
    f.write(f"static const {ctype} {name}[{v.size}] = {{{vals}}};\n\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to baseline checkpoint")
    ap.add_argument("--list", action="store_true",
                    help="just print the checkpoint keys and exit")
    args = ap.parse_args()

    obj = torch.load(args.ckpt, map_location="cpu")
    sd  = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

    if args.list:
        print(list(sd.keys())); return

    for k in (KEY_W1, KEY_B1, KEY_W2, KEY_B2):
        if k not in sd:
            print(f"[error] key '{k}' not in checkpoint. Keys are:\n"
                  f"  {list(sd.keys())}\nEdit KEY_* at the top of this script.")
            sys.exit(1)

    W1 = sd[KEY_W1].cpu().numpy(); b1 = sd[KEY_B1].cpu().numpy()
    W2 = sd[KEY_W2].cpu().numpy(); b2 = sd[KEY_B2].cpu().numpy()
    print(f"shapes: W1{W1.shape} b1{b1.shape}  W2{W2.shape} b2{b2.shape}")
    assert W1.shape == (128, 784) and W2.shape == (64, 128), \
        "Unexpected shapes — check the architecture / key order."

    print("quantizing...")
    W1q = quantize(W1, name="W1"); b1q = quantize(b1, name="B1")
    W2q = quantize(W2, name="W2"); b2q = quantize(b2, name="B2")

    ctype = f"ap_fixed<{W_TOTAL},{W_INT}>"
    with open("fc1_params.h", "w") as f:
        f.write('#ifndef FC1_PARAMS_H\n#define FC1_PARAMS_H\n#include "mlp.h"\n\n')
        emit_matrix(f, "w_t", "W1", W1q)
        emit_vector(f, "w_t", "B1", b1q)
        f.write("#endif\n")
    with open("fc2_params.h", "w") as f:
        f.write('#ifndef FC2_PARAMS_H\n#define FC2_PARAMS_H\n#include "mlp.h"\n\n')
        emit_matrix(f, "w_t", "W2", W2q)
        emit_vector(f, "w_t", "B2", b2q)
        f.write("#endif\n")

    np.savez("mlp_ref.npz", W1=W1q, b1=b1q, W2=W2q, b2=b2q)
    print("wrote fc1_params.h, fc2_params.h, mlp_ref.npz")
    print(f"(weights quantized as {ctype}; biases stored in same ROM type)")

if __name__ == "__main__":
    main()
