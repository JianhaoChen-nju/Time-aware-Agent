"""Evaluator functions for TimeArena TAES."""


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def v_hard(state) -> float:
    return 1.0


def v_soft(state) -> float:
    alpha = 0.1

    completed_time, total_time = _progress_time(state)

    r_progress = completed_time / max(1, total_time) if total_time > 0 else 0

    if state.t_now > 0:
        r_efficiency = min(1.0, r_progress / (state.t_now / max(1, state.total_time)))
    else:
        r_efficiency = 1.0

    r_parallel = alpha if len(state.running_tasks) > 0 else 0
    r_temporal, r_critical, r_finish_speed, r_reachable = _temporal_rewards(state)

    return min(
        1.0,
        0.20 * r_progress
        + 0.10 * r_efficiency
        + 0.05 * min(1.0, r_parallel / alpha)
        + 0.15 * r_temporal
        + 0.10 * r_critical
        + 0.10 * r_finish_speed
        + 0.30 * r_reachable,
    )


def _progress_time(state):
    completed_time = 0
    total_time = state.total_action_time
    for obj, todos in state.object_todo.items():
        for action_key, remaining in todos.items():
            original = getattr(state, '_original_times', {}).get(obj, {}).get(action_key, remaining)
            completed_time += max(0, original - remaining)
    return completed_time, total_time


def _temporal_rewards(state):
    lower_bound, r_reachable = _remaining_analysis(state)
    time_left = max(0, state.total_time - state.t_now)

    if lower_bound <= 0:
        return 1.0, 1.0, 1.0, 1.0

    if time_left >= lower_bound:
        slack = time_left - lower_bound
        r_temporal = 0.7 + 0.3 * (slack / max(1, time_left))
    else:
        r_temporal = 0.7 * (time_left / max(1, lower_bound))

    r_critical = 1.0 - (lower_bound / max(1, state.total_time))
    estimated_finish = state.t_now + lower_bound
    if estimated_finish <= state.total_time:
        r_finish_speed = 1.0 - (estimated_finish / max(1, state.total_time))
    else:
        r_finish_speed = 0.0
    return _clamp(r_temporal), _clamp(r_critical), _clamp(r_finish_speed), _clamp(r_reachable)


def _remaining_makespan_lower_bound(state) -> int:
    lower_bound, _r_reachable = _remaining_analysis(state)
    return lower_bound


def _remaining_analysis(state):
    running = {}
    for obj, action_key, remaining in state.running_tasks:
        running[(obj, action_key)] = min(running.get((obj, action_key), remaining), remaining)

    memo = {}
    completed_time, total_time = _progress_time(state)
    reachable_remaining_time = 0

    def remaining_duration(obj, action_key):
        if (obj, action_key) in running:
            return max(0, running[(obj, action_key)])
        return max(0, state.object_todo.get(obj, {}).get(action_key, 0))

    def finish_time(obj, action_key, visiting):
        if (obj, action_key) in memo:
            return memo[(obj, action_key)]
        duration = remaining_duration(obj, action_key)
        if duration <= 0:
            memo[(obj, action_key)] = 0
            return 0
        if (obj, action_key) in running:
            memo[(obj, action_key)] = duration
            return duration
        if (obj, action_key) in visiting:
            return duration

        visiting = visiting | {(obj, action_key)}
        prereq_time = 0
        dep = state.object_dependencies.get(obj, {}).get(action_key)
        if isinstance(dep, str):
            prereq_time = max(prereq_time, finish_time(obj, dep, visiting))
        elif isinstance(dep, dict):
            for dep_obj, dep_action in dep.items():
                prereq_time = max(prereq_time, finish_time(dep_obj, dep_action, visiting))
        memo[(obj, action_key)] = prereq_time + duration
        return memo[(obj, action_key)]

    critical_path = 0
    agent_work = 0
    time_left = max(0, state.total_time - state.t_now)
    for obj, todos in state.object_todo.items():
        for action_key, remaining in todos.items():
            if remaining <= 0:
                continue
            action_finish_time = finish_time(obj, action_key, frozenset())
            critical_path = max(critical_path, action_finish_time)
            if action_finish_time <= time_left:
                reachable_remaining_time += remaining_duration(obj, action_key)
            if (obj, action_key) in running:
                continue
            if _is_blocking_action(state, action_key):
                agent_work += max(0, remaining)
            else:
                # Non-blocking actions still consume one agent timestep to start.
                agent_work += 1

    budget_cap = 1.0
    if agent_work > 0:
        budget_cap = min(1.0, time_left / max(1, agent_work))
    reachable = (completed_time + reachable_remaining_time * budget_cap) / max(1, total_time)
    return max(critical_path, agent_work), reachable


def _is_blocking_action(state, action_key: str) -> bool:
    base = action_key
    if action_key.endswith('_1') or action_key.endswith('_2'):
        base = action_key.rsplit('_', 1)[0]
    action_name = base.replace('_', ' ')
    return state.action_occupy.get(action_name, state.action_occupy.get(base, True))


def evaluate_state(state) -> float:
    epsilon = 0.1
    r_temporal, _r_critical, _r_finish_speed, r_reachable = _temporal_rewards(state)
    deadline_gate = 0.50 + 0.30 * r_temporal + 0.20 * r_reachable
    return v_hard(state) * deadline_gate * (epsilon + (1 - epsilon) * v_soft(state))
