"""
data_utils.py — MNIST loading, normalisation, and DataLoader creation.
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import config


def _set_seeds(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_mnist(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Download MNIST, normalise, flatten → (N, 784) float32."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.MNIST_MEAN, config.MNIST_STD),
    ])
    ds = datasets.MNIST(
        root=str(config.DATA_DIR),
        train=train,
        download=True,
        transform=transform,
    )
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    images, labels = next(iter(loader))
    return images.view(len(ds), -1).float(), labels.long()


def get_loaders(
    batch_size: int = config.BATCH_SIZE,
    smoke_test: bool = False,
) -> tuple[DataLoader, DataLoader]:
    _set_seeds()
    X_tr, y_tr = _load_mnist(train=True)
    X_te, y_te = _load_mnist(train=False)

    if smoke_test:
        X_tr, y_tr = X_tr[:500], y_tr[:500]
        X_te, y_te = X_te[:200], y_te[:200]

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(TensorDataset(X_te, y_te),
                              batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def get_tensors(smoke_test: bool = False):
    """Return raw (X_train, y_train, X_test, y_test) tensors."""
    _set_seeds()
    X_tr, y_tr = _load_mnist(train=True)
    X_te, y_te = _load_mnist(train=False)
    if smoke_test:
        X_tr, y_tr = X_tr[:500], y_tr[:500]
        X_te, y_te = X_te[:200], y_te[:200]
    return X_tr, y_tr, X_te, y_te


if __name__ == "__main__":
    train_loader, _ = get_loaders()
    xb, yb = next(iter(train_loader))
    print(f"Batch shape : {xb.shape}")
    print(f"Input range : [{xb.min():.3f}, {xb.max():.3f}]")
    print(f"Label sample: {yb[:10].tolist()}")
