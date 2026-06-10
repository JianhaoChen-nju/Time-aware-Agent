"""Evaluator functions for TravelPlanner TAES.

V(S) = V_hard(S) * [ε + (1-ε) * V_soft(S)]
"""

from .checker import estimate_future_budget_lower_bound


def v_hard(state) -> float:
    """Hard constraint satisfaction score."""
    if state.budget_remaining < 0:
        return 0.0
    return 1.0


def v_soft(state) -> float:
    """Soft quality score in [0, 1].

    Components: budget headroom, restaurant diversity, attraction diversity,
    completeness, and required city coverage.
    """
    scores = []

    # Budget headroom
    remaining_days = max(1, state.total_days - state.current_day + 1)
    if state.budget_total > 0:
        avg_budget_per_day = state.budget_total / state.total_days
        remaining_per_day = state.budget_remaining / remaining_days
        r_budget = min(1.0, remaining_per_day / max(1, avg_budget_per_day))
    else:
        r_budget = 1.0
    scores.append(r_budget)

    try:
        future_reserve = estimate_future_budget_lower_bound(state)
        budget_slack = state.budget_remaining - future_reserve
        r_budget_slack = max(0.0, min(1.0, budget_slack / max(1.0, state.budget_total * 0.2)))
    except Exception:
        r_budget_slack = 0.5
    scores.append(r_budget_slack)

    # Restaurant diversity
    planned_days = len(state.daily_plans)
    if planned_days > 0:
        expected_restaurants = planned_days * 3
        r_restaurant = min(1.0, len(state.used_restaurants) / max(1, expected_restaurants))
    else:
        r_restaurant = 1.0
    scores.append(r_restaurant)

    # Attraction diversity
    if planned_days > 0:
        expected_attractions = planned_days * 2
        r_attraction = min(1.0, len(state.used_attractions) / max(1, expected_attractions))
    else:
        r_attraction = 1.0
    scores.append(r_attraction)

    # Completeness
    r_completeness = planned_days / max(1, state.total_days)
    scores.append(r_completeness)

    # Required city coverage. This is critical for multi-city tasks because
    # otherwise states that stay too long in the first city can tie with states
    # that have already covered the next required destination city.
    required_cities = getattr(state, 'visiting_city_number', 0) or 0
    visited_cities = set()
    if required_cities:
        try:
            visited_cities = state._visited_destination_cities()
            r_city = min(1.0, len(visited_cities) / max(1, required_cities))
        except Exception:
            r_city = 0.0
    else:
        r_city = 1.0
    scores.append(r_city)

    # City deadline pressure. Covering two out of three required cities is not
    # equally good on day 2 and day 6; when non-final days are tight, preserve
    # states that still have enough slack to complete the city chain.
    if required_cities:
        missing = max(0, required_cities - len(visited_cities))
        non_final_days_left = max(0, state.total_days - state.current_day)
        if missing <= 0:
            r_city_slack = 1.0
        elif non_final_days_left < missing:
            r_city_slack = 0.0
        else:
            slack = non_final_days_left - missing
            r_city_slack = min(1.0, 0.55 + 0.25 * slack)
    else:
        r_city_slack = 1.0
    scores.append(r_city_slack)

    weights = [0.13, 0.15, 0.11, 0.11, 0.20, 0.20, 0.10]
    return sum(s * w for s, w in zip(scores, weights))


def evaluate_state(state) -> float:
    """Combined evaluation: V(S) = V_hard * [ε + (1-ε) * V_soft]."""
    epsilon = 0.1
    vh = v_hard(state)
    vs = v_soft(state)
    return vh * (epsilon + (1 - epsilon) * vs)
