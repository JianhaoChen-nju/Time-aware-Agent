"""Symbolic constraint checker for TimeArena TAES."""

import logging

logger = logging.getLogger(__name__)


def check_action_exists(state, event) -> bool:
    """Mirror environment.is_valid/action_object_valid for symbolic events."""
    if event.action_string == 'wait':
        return True
    if event.obj_name not in state.object_todo:
        return False
    if event.action_key not in state.object_todo.get(event.obj_name, {}):
        return False
    if event.secondary_obj:
        if event.secondary_obj not in state.object_todo:
            return False
        comp_key = event.action_key.replace('_1', '_2')
        if comp_key not in state.object_todo.get(event.secondary_obj, {}):
            return False
    return True


def _check_obj_dependency(state, obj_name, action_key) -> bool:
    """Check if all dependencies for a specific object's action are satisfied."""
    deps = state.object_dependencies.get(obj_name, {})
    if action_key not in deps:
        return True
    dep = deps[action_key]
    if isinstance(dep, str):
        return state.object_todo.get(obj_name, {}).get(dep, 1) == 0
    elif isinstance(dep, dict):
        for dep_obj, dep_action in dep.items():
            if state.object_todo.get(dep_obj, {}).get(dep_action, 1) != 0:
                return False
        return True
    return True


def check_dependency(state, event) -> bool:
    """Check dependencies for both subject and secondary object (container)."""
    if event.action_string == 'wait':
        return True
    # Check subject object dependency
    if not _check_obj_dependency(state, event.obj_name, event.action_key):
        return False
    # For two-object actions, also check the secondary object's dependency
    if event.secondary_obj:
        comp_key = event.action_key.replace('_1', '_2')
        if not _check_obj_dependency(state, event.secondary_obj, comp_key):
            return False
    return True


def check_resource_available(state, event) -> bool:
    if event.action_string == 'wait':
        return True
    if state.object_occupied.get(event.obj_name, False):
        return False
    if event.secondary_obj and state.object_occupied.get(event.secondary_obj, False):
        return False
    return True


def check_task_constraint(state, event) -> bool:
    """For merged multi-task objects, containers may accept only task-local subjects."""
    if event.action_string == 'wait' or not event.secondary_obj:
        return True
    allowed_subjects = state.task_constraint.get(event.secondary_obj)
    if allowed_subjects is None:
        return True
    return event.obj_name in allowed_subjects


def check_agent_available(state, event) -> bool:
    if event.action_string == 'wait':
        return True
    if event.is_blocking and state.agent_occupied:
        return False
    return True


def check_action_not_done(state, event) -> bool:
    if event.action_string == 'wait':
        return True
    remaining = state.object_todo.get(event.obj_name, {}).get(event.action_key, -1)
    return remaining > 0


def check_time_sufficient(state, event) -> bool:
    if event.action_string == 'wait':
        return True
    remaining_time = state.total_time - state.t_now
    return remaining_time >= event.duration


def check_wait_only_when_needed(state, event) -> bool:
    if event.action_string != 'wait':
        return True
    non_wait_actions = [action for action in state.get_available_actions() if action != 'wait']
    return not non_wait_actions


def check_all(state, event) -> bool:
    return all([
        check_action_exists(state, event),
        check_dependency(state, event),
        check_resource_available(state, event),
        check_task_constraint(state, event),
        check_agent_available(state, event),
        check_action_not_done(state, event),
        check_time_sufficient(state, event),
        check_wait_only_when_needed(state, event),
    ])
