#!/usr/bin/env python3
# mlp_baseline_ref.py — quantized dense BL-MLP reference (accuracy gate).
#   CUDA_VISIBLE_DEVICES="" python mlp_baseline_ref.py

import numpy as np
d = np.load("baseline_ref.npz")
W1,b1,W2,b2,W3,b3 = d["W1"],d["b1"],d["W2"],d["b2"],d["W3"],d["b3"]

def relu(x): return np.maximum(0.0, x)
def forward(x):
    h1 = relu(W1 @ x + b1)
    h2 = relu(W2 @ h1 + b2)
    return W3 @ h2 + b3

try:
    from torchvision import datasets, transforms
    ds = datasets.MNIST(root="./data", train=False, download=True,
                        transform=transforms.ToTensor())
    X = ds.data.numpy().reshape(-1,784).astype(np.float64)/255.0
    X = (X - 0.1307) / 0.3081                      # SAME normalization as training
    Y = ds.targets.numpy()
    pred = np.array([forward(X[i]).argmax() for i in range(len(X))])
    print(f"dense baseline (quantized) top-1: {100*(pred==Y).mean():.2f}%  "
          f"on {len(Y)} test images")
except Exception as e:
    print(f"[info] torchvision unavailable ({e})")
