"""Neuro-Symbolic Beam Search (Algorithm 1 from thesis).

beam_search(initial_state, generate_fn, check_fn, evaluate_fn, B, K, max_depth)
  - generate_fn(state) -> List[Event]: LLM generates K candidate events
  - check_fn(state, event) -> bool: symbolic constraint checker
  - evaluate_fn(state) -> float: heuristic evaluation V(S)
  - B: beam width (default 5)
  - K: branch factor (default 3)
  - max_depth: maximum planning depth
"""

import logging
from typing import Callable, List, Optional, Tuple

from .base import Event, State

logger = logging.getLogger(__name__)


def beam_search(
    initial_state: State,
    generate_fn: Callable[[State], List[Event]],
    check_fn: Callable[[State, Event], bool],
    evaluate_fn: Callable[[State], float],
    B: int = 5,
    K: int = 3,  # noqa: N803 — kept for API clarity, caller controls K via generate_fn
    max_depth: int = 10,
    diversity_key: Optional[Callable[[State], object]] = None,
) -> Tuple[Optional[State], List[State]]:
    """Run neuro-symbolic beam search.

    Args:
        initial_state: Starting state S_0.
        generate_fn: Function that takes a state and returns up to K candidate events
                      (calls the LLM).
        check_fn: Symbolic checker that returns True if the event is valid for the state.
        evaluate_fn: Heuristic evaluation function V(S) -> float in [0, 1].
        B: Beam width — number of states to keep at each depth.
        K: Branch factor — number of candidates to generate per state.
        max_depth: Maximum search depth.
        diversity_key: Optional state bucketing function. When provided, the
                       beam keeps high-scoring states from distinct buckets
                       before filling remaining slots by score.

    Returns:
        (best_state, all_terminal_states): The best terminal state found, plus all
        terminal states for analysis. Returns (None, []) if no terminal state is reached.
    """
    beam: List[Tuple[float, State]] = [(evaluate_fn(initial_state), initial_state)]
    finished: List[Tuple[float, State]] = []

    for depth in range(max_depth):
        if not beam:
            break

        candidates: List[Tuple[float, State]] = []

        for _score, state in beam:
            if state.is_terminal():
                finished.append((evaluate_fn(state), state))
                continue

            # Generate K candidate events via LLM
            try:
                events = generate_fn(state)
            except Exception as e:
                logger.warning(f"generate_fn failed at depth {depth}: {e}")
                events = []

            if not events:
                # No candidates generated — treat current state as terminal
                finished.append((evaluate_fn(state), state))
                continue

            # Symbolic pruning + evaluation
            for event in events:
                if check_fn(state, event):
                    new_state = event.apply(state.copy())
                    score = evaluate_fn(new_state)
                    candidates.append((score, new_state))
                else:
                    logger.debug(f"Pruned event: {event.name}")

        if not candidates and not beam:
            break

        # Keep top-B states. For planning domains with route commitments,
        # score-only truncation can collapse all beams onto the same city path.
        # The optional diversity key preserves distinct symbolic states while
        # still ranking within each bucket by the heuristic score.
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = _select_beam(candidates, B, diversity_key)

        logger.info(
            f"Depth {depth}: {len(candidates)} candidates → beam size {len(beam)}, "
            f"finished {len(finished)}"
        )

    # Add remaining beam states to finished pool
    for score, state in beam:
        finished.append((evaluate_fn(state), state))

    if not finished:
        return None, []

    # Prefer genuinely terminal states over high-scoring partial states. Partial
    # states can retain more budget and otherwise outrank a complete plan, but
    # callers such as TravelPlanner must return full-depth plans.
    terminal_finished = [(score, state) for score, state in finished if state.is_terminal()]
    ranked = terminal_finished if terminal_finished else finished
    ranked.sort(key=lambda x: x[0], reverse=True)
    best_score, best_state = ranked[0]
    logger.info(f"Best state score: {best_score:.4f}")

    return best_state, [s for _, s in finished]


def _select_beam(
    candidates: List[Tuple[float, State]],
    B: int,
    diversity_key: Optional[Callable[[State], object]] = None,
) -> List[Tuple[float, State]]:
    if diversity_key is None or B <= 0:
        return candidates[:B]

    selected: List[Tuple[float, State]] = []
    selected_ids = set()
    seen_keys = set()

    for idx, (score, state) in enumerate(candidates):
        try:
            key = diversity_key(state)
        except Exception:
            key = None
        if key in seen_keys:
            continue
        selected.append((score, state))
        selected_ids.add(idx)
        seen_keys.add(key)
        if len(selected) >= B:
            return selected

    for idx, item in enumerate(candidates):
        if idx in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= B:
            break

    return selected
