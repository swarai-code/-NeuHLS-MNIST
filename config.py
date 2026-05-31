"""
config.py — MNIST MLP NeuSym pipeline configuration.
Architecture: 784 → 128 → 64 → 10
"""
import torch
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 20
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── Architecture ──────────────────────────────────────────────────────────────
INPUT_DIM   = 784   # 28×28 flattened
H1_DIM      = 128
H2_DIM      = 64
NUM_CLASSES = 10

# ── MNIST normalisation statistics ────────────────────────────────────────────
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
CKPT_DIR    = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR = OUTPUTS_DIR / "results"
TRACE_DIR   = OUTPUTS_DIR / "traces"
PYSR_DIR    = OUTPUTS_DIR / "pysr"
EQ_DIR      = OUTPUTS_DIR / "equations"
HLS_GEN_DIR = ROOT_DIR / "hls_generated"
HLS_SRC_DIR = ROOT_DIR / "hls"

for _d in [DATA_DIR, CKPT_DIR, RESULTS_DIR, TRACE_DIR,
           PYSR_DIR, EQ_DIR, HLS_GEN_DIR, HLS_SRC_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Symbolic Regression ───────────────────────────────────────────────────────
# Three operator sets ordered by hardware friendliness
OP_SETS = {
    "POL": ["+", "-", "*"],
    "SRL": ["+", "-", "*", "square", "relu"],
    "SCE": ["+", "-", "*", "sin", "cos", "exp"],
}

PYSR_ITERATIONS  = 100
PYSR_POPULATIONS = 20
PYSR_COMPLEXITY  = 20   # Cmax — maximum expression tree size

# 6 experiments: 2 replacement depths × 3 operator sets
# 1L → replace fc3  (h2 ∈ R^64  → logits ∈ R^10)
# 2L → replace fc2+fc3 (h1 ∈ R^128 → logits ∈ R^10)
EXPERIMENTS = [
    f"{depth}_{ops}"
    for depth in ("1L", "2L")
    for ops   in ("POL", "SRL", "SCE")
]


def get_layer(exp_name: str) -> str:
    return exp_name.split("_")[0]


def get_op_set(exp_name: str) -> str:
    return exp_name.split("_")[1]


# ── MAC counts ────────────────────────────────────────────────────────────────
# fc1: 784×128  fc2: 128×64  fc3: 64×10
FC1_MACS = INPUT_DIM  * H1_DIM      # 100,352
FC2_MACS = H1_DIM     * H2_DIM      #   8,192
FC3_MACS = H2_DIM     * NUM_CLASSES #     640

BASELINE_MACS  = FC1_MACS + FC2_MACS + FC3_MACS  # 109,184
BASELINE_FLOPS = 2 * BASELINE_MACS

SR_1L_MACS = FC1_MACS + FC2_MACS   # fc3 replaced by SR
SR_2L_MACS = FC1_MACS               # fc2+fc3 replaced by SR

# ── Hardware operator latency costs (cycles) ──────────────────────────────────
OP_COSTS = {
    "+":      1,
    "-":      1,
    "*":      3,
    "square": 3,
    "relu":   2,
    "sin":    8,
    "cos":    8,
    "exp":   10,
    "/":      6,
}
