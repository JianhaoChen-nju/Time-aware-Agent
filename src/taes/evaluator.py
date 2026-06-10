"""Evaluation function V(S) = V_hard(S) * [ε + (1-ε) * V_soft(S)].

V_hard: product of binary hard constraint checks (0 or 1 each).
V_soft: soft quality score in [0, 1].
ε = 0.1: floor to avoid zero scores when V_soft is 0.
"""

from typing import Callable

from .base import State


def evaluate(
    state: State,
    v_hard_fn: Callable[[State], float],
    v_soft_fn: Callable[[State], float],
    epsilon: float = 0.1,
) -> float:
    """Compute V(S) = V_hard(S) * [ε + (1-ε) * V_soft(S)].

    Args:
        state: Current state to evaluate.
        v_hard_fn: Returns product of binary hard constraint checks. Should be in {0, 1}
                    or [0, 1] if partial satisfaction is modeled.
        v_soft_fn: Returns soft quality score in [0, 1].
        epsilon: Floor parameter to prevent zero scores. Default 0.1.

    Returns:
        Combined score in [0, 1].
    """
    v_hard = v_hard_fn(state)
    v_soft = v_soft_fn(state)
    return v_hard * (epsilon + (1 - epsilon) * v_soft)
