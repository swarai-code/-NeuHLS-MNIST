#!/usr/bin/env python3
# mlp_ref.py — NumPy reference for the full hybrid network.
# Uses the SAME quantized fc1/fc2 (from mlp_ref.npz) and the SAME fc3 symbolic
# equations as the HLS. Use it to validate top-1 accuracy and to generate the
# golden vectors for HLS csim.
#
#   python mlp_ref.py            # checks against MNIST test set (needs torchvision)
#   python mlp_ref.py --dump 5   # prints 5 sample (input, logits, argmax)

import argparse
import numpy as np

d = np.load("mlp_ref.npz")
W1, b1, W2, b2 = d["W1"], d["b1"], d["W2"], d["b2"]

def relu(x): return np.maximum(0.0, x)

# fc3: the 10 symbolic equations (affine form: coeff*h2[i] + bias), 1L_POL.
FC3 = {
 0: ({19:1,13:1,31:-1,40:-1,58:-1}, -2.6073067),
 1: ({57:0.81722426,12:-1,42:-1,31:-1,9:1,53:-1}, 1.055732684),
 2: ({38:1,48:-0.28745016,51:-1,52:-1,55:1,14:-1,7:-1}, -0.2888112),
 3: ({47:-0.9800235,44:-1,17:1,8:1,29:-1,54:-1}, -1.9390514),
 4: ({40:1,50:-1,19:-1.67230314,61:-1,37:1,12:-0.67230314}, 0.0),
 5: ({50:1,60:-1,3:-1,0:-1,4:1,54:-1,7:1}, -1.5606319),
 6: ({44:1.1927166,54:1.1927166,50:-1.1927166,52:-1.1927166,31:-1,7:1}, -0.3341241),
 7: ({9:1,48:-1,10:1,17:-1,27:-1,46:-0.56618065}, -0.104913995),
 8: ({48:1,34:-0.51528823,63:1,1:-1,14:-1}, 1.13450435),
 9: ({31:1,21:-1,45:1,24:0.680239,6:-0.680239,7:-1}, -7.6854916),
}

def forward(x):                      # x: (784,)
    h1 = relu(W1 @ x + b1)           # (128,)
    h2 = relu(W2 @ h1 + b2)          # (64,)
    out = np.zeros(10)
    for c,(co,bias) in FC3.items():
        out[c] = bias + sum(k*h2[i] for i,k in co.items())
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, default=0)
    args = ap.parse_args()

    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(root="./data", train=False, download=True,
                            transform=transforms.ToTensor())
        X = ds.data.numpy().reshape(-1,784).astype(np.float64)/255.0
        Y = ds.targets.numpy()
    except Exception as e:
        print(f"[info] torchvision unavailable ({e}); skipping accuracy check."); X=Y=None

    if X is not None:
        pred = np.array([forward(X[i]).argmax() for i in range(len(X))])
        print(f"hybrid (quantized fc1/fc2 + symbolic fc3) top-1: "
              f"{100*(pred==Y).mean():.2f}%  on {len(Y)} test images")

    for i in range(args.dump):
        x = X[i] if X is not None else np.random.rand(784)
        o = forward(x)
        print(f"sample {i}: argmax={o.argmax()}  logits={np.round(o,3)}")
