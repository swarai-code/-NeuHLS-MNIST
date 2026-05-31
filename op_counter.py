"""
op_counter.py — Weighted hardware cost of operators in a symbolic expression.
"""
import re
from typing import Optional


def count_ops(expr: str, cost_table: Optional[dict] = None) -> int:
    """
    Sum operator costs over all occurrences in a PySR expression string.
    Unrecognised operators default to cost 1.
    """
    if cost_table is None:
        cost_table = {}

    total = 0
    # Functional operators: sin(, cos(, exp(, square(, relu(, …
    for name, cost in cost_table.items():
        if name in ("+", "-", "*", "/"):
            continue
        total += len(re.findall(rf"\b{re.escape(name)}\s*\(", expr)) * cost

    # Infix arithmetic operators
    for sym in ("+", "-", "*", "/"):
        total += expr.count(sym) * cost_table.get(sym, 1)

    return total


if __name__ == "__main__":
    costs = {"+": 1, "-": 1, "*": 3, "sin": 8, "square": 3, "relu": 2, "exp": 10}
    for e in ["x0 + x1 * sin(x2)", "square(x0) - relu(x1) + 2.5", "x0 * x1 + x2 - x3"]:
        print(f"{e!r:45s}  cost={count_ops(e, costs)}")
