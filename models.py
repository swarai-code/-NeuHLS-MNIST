"""
models.py — BaselineMLP and Hybrid drop-in replacements for MNIST 784→128→64→10.
"""
import copy

import torch
import torch.nn as nn

import config


# ─── Baseline ─────────────────────────────────────────────────────────────────

class BaselineMLP(nn.Module):
    """784 → 128 (ReLU) → 64 (ReLU) → 10 logits."""

    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(config.INPUT_DIM,  config.H1_DIM)
        self.fc2  = nn.Linear(config.H1_DIM,     config.H2_DIM)
        self.fc3  = nn.Linear(config.H2_DIM,     config.NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        h1     = self.relu(self.fc1(x))    # (B, 128)
        h2     = self.relu(self.fc2(h1))   # (B,  64)
        logits = self.fc3(h2)              # (B,  10)
        if return_hidden:
            return logits, h1, h2
        return logits


# ─── Symbolic expression evaluator ───────────────────────────────────────────

def _make_torch_evaluator(expr: str, n_inputs: int):
    """
    Compile a PySR expression string into a callable (h: Tensor) → (B,) Tensor.

    PySR emits variable names x0, x1, … so we populate a namespace with
    xi = h[:, i] for each column before calling eval.
    """
    safe_ops = {
        "sin":    torch.sin,
        "cos":    torch.cos,
        "exp":    lambda t: torch.clamp(torch.exp(t), max=1e6),
        "square": lambda t: t ** 2,
        "relu":   torch.relu,
        "sqrt":   lambda t: torch.sqrt(torch.clamp(t, min=0.0)),
        "log":    lambda t: torch.log(torch.clamp(t.abs(), min=1e-8)),
    }

    def evaluator(h: torch.Tensor) -> torch.Tensor:
        ns = {f"x{i}": h[:, i] for i in range(n_inputs)}
        ns.update(safe_ops)
        ns["__builtins__"] = {}
        try:
            out = eval(expr, ns)  # noqa: S307
            if not torch.is_tensor(out):
                out = torch.full((h.size(0),), float(out),
                                 device=h.device, dtype=h.dtype)
            return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        except Exception:
            return torch.zeros(h.size(0), device=h.device, dtype=h.dtype)

    return evaluator


# ─── Hybrid models ────────────────────────────────────────────────────────────

class Hybrid1L(nn.Module):
    """
    Replace fc3 with NUM_CLASSES symbolic expressions f_j(h2), j=0..9.
    fc1, fc2 frozen from baseline.
    """

    def __init__(self, baseline: BaselineMLP, equations: list[str]):
        super().__init__()
        self.fc1  = copy.deepcopy(baseline.fc1)
        self.fc2  = copy.deepcopy(baseline.fc2)
        self.relu = nn.ReLU()

        for p in (*self.fc1.parameters(), *self.fc2.parameters()):
            p.requires_grad_(False)

        assert len(equations) == config.NUM_CLASSES, \
            f"Need {config.NUM_CLASSES} equations, got {len(equations)}"
        self._evals = [_make_torch_evaluator(eq, config.H2_DIM) for eq in equations]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        cols = [fn(h2).unsqueeze(1) for fn in self._evals]
        return torch.cat(cols, dim=1)   # (B, 10)


class Hybrid2L(nn.Module):
    """
    Replace fc2+fc3 with NUM_CLASSES symbolic expressions f_j(h1), j=0..9.
    fc1 frozen from baseline.
    """

    def __init__(self, baseline: BaselineMLP, equations: list[str]):
        super().__init__()
        self.fc1  = copy.deepcopy(baseline.fc1)
        self.relu = nn.ReLU()

        for p in self.fc1.parameters():
            p.requires_grad_(False)

        assert len(equations) == config.NUM_CLASSES
        self._evals = [_make_torch_evaluator(eq, config.H1_DIM) for eq in equations]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1   = self.relu(self.fc1(x))
        cols = [fn(h1).unsqueeze(1) for fn in self._evals]
        return torch.cat(cols, dim=1)   # (B, 10)


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_baseline(path=None, device=None) -> BaselineMLP:
    path   = path   or config.CKPT_DIR / "baseline.pt"
    device = device or torch.device(config.DEVICE)
    ckpt   = torch.load(path, map_location=device, weights_only=True)
    model  = BaselineMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_hybrid(baseline: BaselineMLP, layer: str, equations: list[str]):
    if layer == "1L":
        return Hybrid1L(baseline, equations)
    if layer == "2L":
        return Hybrid2L(baseline, equations)
    raise ValueError(f"Unknown layer spec: {layer!r}  (expected '1L' or '2L')")
