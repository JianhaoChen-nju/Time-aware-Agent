"""TimeArena state and event definitions for TAES."""

from typing import Any, Dict, List, Optional, Tuple
from ..base import State, Event


class TimeArenaState(State):
    """Internal state model for TimeArena TAES planning."""

    def __init__(
        self,
        t_now: int,
        total_time: int,
        object_todo: Dict[str, Dict[str, int]],
        object_dependencies: Dict[str, Dict[str, Any]],
        action_occupy: Dict[str, bool],
        running_tasks: List[Tuple[str, str, int]],
        agent_occupied: bool,
        object_occupied: Dict[str, bool],
        completed_actions: List[Tuple[str, int]],
        total_action_time: int,
        action_descriptions: Dict[str, str],
        object_names: List[str],
        task_descriptions: List[str],
        task_constraint: Optional[Dict[str, List[str]]] = None,
    ):
        resources = {'time': total_time - t_now}
        constraints_met = {}
        super().__init__(t_now=t_now, resources=resources, constraints_met=constraints_met)

        self.total_time = total_time
        self.object_todo = object_todo
        self.object_dependencies = object_dependencies
        self.action_occupy = action_occupy
        self.running_tasks = running_tasks
        self.agent_occupied = agent_occupied
        self.object_occupied = object_occupied
        self.completed_actions = completed_actions
        self.total_action_time = total_action_time
        self.action_descriptions = action_descriptions
        self.object_names = object_names
        self.task_descriptions = task_descriptions
        self.task_constraint = task_constraint or {}  # {container_obj: [allowed_subject_objs]}

    def is_terminal(self) -> bool:
        if self.t_now >= self.total_time:
            return True
        for todos in self.object_todo.values():
            for remaining in todos.values():
                if remaining > 0:
                    return False
        return True

    def get_score(self) -> float:
        completed_time = 0
        for obj, todos in self.object_todo.items():
            for action_key, remaining in todos.items():
                if remaining == 0:
                    completed_time += self._get_original_time(obj, action_key)
        if self.total_action_time == 0:
            return 0
        return int(completed_time * 100 / self.total_action_time)

    def _get_original_time(self, obj: str, action_key: str) -> int:
        return getattr(self, '_original_times', {}).get(obj, {}).get(action_key, 0)

    def get_available_actions(self) -> List[str]:
        actions = ['wait']
        for obj_name, todos in self.object_todo.items():
            if self.object_occupied.get(obj_name, False):
                continue
            for action_key, remaining in todos.items():
                if remaining <= 0:
                    continue
                if not self._check_dependency(obj_name, action_key):
                    continue
                action_str = self._format_action(obj_name, action_key)
                if action_str:
                    actions.append(action_str)
        return actions

    def _check_dependency(self, obj_name: str, action_key: str) -> bool:
        deps = self.object_dependencies.get(obj_name, {})
        if action_key not in deps:
            return True
        dep = deps[action_key]
        if isinstance(dep, str):
            return self.object_todo.get(obj_name, {}).get(dep, 1) == 0
        elif isinstance(dep, dict):
            for dep_obj, dep_action in dep.items():
                if self.object_todo.get(dep_obj, {}).get(dep_action, 1) != 0:
                    return False
            return True
        return True

    def _format_action(self, obj_name: str, action_key: str) -> Optional[str]:
        if '_1' in action_key or '_2' in action_key:
            base_action = action_key.rsplit('_', 1)[0]
            action_name = base_action.replace('_', ' ')
            if action_key.endswith('_1'):
                for other_obj, other_todos in self.object_todo.items():
                    if other_obj == obj_name:
                        continue
                    complementary = action_key.replace('_1', '_2')
                    if complementary in other_todos:
                        # Skip if container object is occupied
                        if self.object_occupied.get(other_obj, False):
                            continue
                        # The environment checks dependencies for both sides
                        # of a two-object action, including dirty containers.
                        if not self._check_dependency(other_obj, complementary):
                            continue
                        # Check task constraint: container must allow this subject
                        if other_obj in self.task_constraint:
                            if obj_name not in self.task_constraint[other_obj]:
                                continue
                        parts = action_name.split()
                        preposition = parts[1] if len(parts) > 1 else 'in'
                        return f"{parts[0]} {obj_name} {preposition} {other_obj}"
            else:
                return None
        else:
            # Keep action_key as-is (e.g., "weed_with") to match environment's expected format
            return f"{action_key} {obj_name}"

    def summary(self) -> str:
        lines = [
            f"Time: {self.t_now}/{self.total_time}",
            f"Score: {self.get_score()}%",
            f"Tasks: {'; '.join(self.task_descriptions)}",
        ]
        if self.running_tasks:
            running = [f"{obj}.{action}({rem}min left)" for obj, action, rem in self.running_tasks]
            lines.append(f"Running: {', '.join(running)}")
        available = self.get_available_actions()[:10]
        lines.append(f"Available actions: {', '.join(available)}")
        remaining = []
        for obj, todos in self.object_todo.items():
            for action, time_left in todos.items():
                if time_left > 0:
                    remaining.append(f"{obj}.{action}({time_left}min)")
        if remaining:
            lines.append(f"Remaining: {', '.join(remaining[:15])}")
        return '\n'.join(lines)

    @staticmethod
    def from_environment(env, t_now: int, total_time: int) -> 'TimeArenaState':
        """Build a TAES state from the actual TimeArena environment."""
        object_todo = {}
        object_dependencies = {}
        object_occupied = {}
        running_tasks = []

        for obj in env.objects:
            name = obj.properties['name']
            object_todo[name] = dict(obj.properties['todo'])
            object_dependencies[name] = dict(obj.properties.get('dependency', {}))
            object_occupied[name] = obj.properties.get('occupy', False)

        for obj_name, action_keys in getattr(env, 'non_occupy', {}).items():
            for action_key in action_keys:
                remaining = object_todo.get(obj_name, {}).get(action_key, 0)
                if remaining <= 0 and '_2' in action_key:
                    action_text = getattr(env, 'non_occupy_conversation_info', {}).get(obj_name, {}).get(action_key, '')
                    parts = action_text.split()
                    if len(parts) >= 2:
                        subject = parts[1]
                        subject_key = action_key.replace('_2', '_1')
                        remaining = object_todo.get(subject, {}).get(subject_key, 0)
                if remaining > 0:
                    running_tasks.append((obj_name, action_key, remaining))

        action_occupy = {}
        action_descriptions = {}
        for action in env.actions:
            action_occupy[action.properties['name']] = action.properties['occupy']
            action_descriptions[action.properties['name']] = action.properties.get('usage', '')

        total_action_time = env.total_time
        object_names = [obj.properties['name'] for obj in env.objects]
        task_descriptions = [task.name for task in env.Task]

        # Capture task constraint (which subjects can act on which containers)
        task_constraint = dict(getattr(env, 'constraint_only_for_task_dict', {}))

        state = TimeArenaState(
            t_now=t_now, total_time=total_time,
            object_todo=object_todo, object_dependencies=object_dependencies,
            action_occupy=action_occupy, running_tasks=running_tasks,
            agent_occupied=False, object_occupied=object_occupied,
            completed_actions=[], total_action_time=total_action_time,
            action_descriptions=action_descriptions,
            object_names=object_names, task_descriptions=task_descriptions,
            task_constraint=task_constraint,
        )
        state._original_times = {
            obj.properties['name']: dict(obj.properties['todo'])
            for obj in env.objects
        }
        return state


class TimeArenaEvent(Event):
    """One action in TimeArena."""

    def __init__(self, action_string: str, obj_name: str, action_key: str,
                 duration: int, is_blocking: bool, secondary_obj: Optional[str] = None):
        super().__init__(name=action_string, duration=duration, cost={'time': 1})
        self.action_string = action_string
        self.obj_name = obj_name
        self.action_key = action_key
        self.is_blocking = is_blocking
        self.secondary_obj = secondary_obj

    def apply(self, state: TimeArenaState) -> TimeArenaState:
        new_state = state.copy()

        if self.action_string == 'wait':
            new_state.t_now = state.t_now + 1
            # Advance running tasks by 1 timestep
            new_state.running_tasks, new_state.completed_actions = self._advance_running(
                new_state.running_tasks, list(state.completed_actions),
                new_state.object_todo, new_state.object_occupied, 1, state.t_now,
            )
            # Record wait so _plan() can include it in the action sequence
            new_state.completed_actions = list(new_state.completed_actions) + [('wait', state.t_now)]
            return new_state

        obj_todo = new_state.object_todo.get(self.obj_name, {})
        action_time = obj_todo.get(self.action_key, 0)
        if action_time <= 0:
            new_state.t_now = state.t_now + 1
            new_state.running_tasks, new_state.completed_actions = self._advance_running(
                new_state.running_tasks, list(state.completed_actions),
                new_state.object_todo, new_state.object_occupied, 1, state.t_now,
            )
            return new_state

        if self.is_blocking:
            # Blocking actions occupy the agent until done, so in the internal
            # simulation we consume ALL remaining time at once (the agent will
            # keep repeating this action in the real env until it finishes).
            elapsed = action_time
            new_state.t_now = state.t_now + elapsed
            new_state.object_todo[self.obj_name][self.action_key] = 0
            new_state.agent_occupied = False
            new_state.completed_actions = list(state.completed_actions) + [(self.action_string, state.t_now)]
            if self.secondary_obj:
                comp_key = self.action_key.replace('_1', '_2')
                if comp_key in new_state.object_todo.get(self.secondary_obj, {}):
                    new_state.object_todo[self.secondary_obj][comp_key] = 0
            # Advance running tasks by the same elapsed time
            new_state.running_tasks, new_state.completed_actions = self._advance_running(
                new_state.running_tasks, new_state.completed_actions,
                new_state.object_todo, new_state.object_occupied, elapsed, state.t_now,
            )
        else:
            new_state.t_now = state.t_now + 1
            # Advance existing running tasks by 1 timestep first
            new_state.running_tasks, new_state.completed_actions = self._advance_running(
                new_state.running_tasks, list(state.completed_actions),
                new_state.object_todo, new_state.object_occupied, 1, state.t_now,
            )
            # Record action string at start time (for plan extraction)
            new_state.completed_actions = list(new_state.completed_actions) + [(self.action_string, state.t_now)]
            # Then start the new non-blocking task
            remaining_after_start = action_time - 1
            if remaining_after_start <= 0:
                new_state.object_todo[self.obj_name][self.action_key] = 0
                new_state.object_occupied[self.obj_name] = False
            else:
                new_state.running_tasks = list(new_state.running_tasks) + [
                    (self.obj_name, self.action_key, remaining_after_start)
                ]
                new_state.object_occupied[self.obj_name] = True
            new_state.agent_occupied = False
            if self.secondary_obj:
                comp_key = self.action_key.replace('_1', '_2')
                if comp_key in new_state.object_todo.get(self.secondary_obj, {}):
                    if remaining_after_start <= 0:
                        new_state.object_todo[self.secondary_obj][comp_key] = 0
                        new_state.object_occupied[self.secondary_obj] = False
                    else:
                        new_state.running_tasks.append((self.secondary_obj, comp_key, remaining_after_start))
                        new_state.object_occupied[self.secondary_obj] = True

        return new_state

    @staticmethod
    def _advance_running(running_tasks, completed_actions, object_todo, object_occupied, steps, base_t):
        """Advance all running (non-blocking) tasks by `steps` timesteps."""
        updated = []
        completed = list(completed_actions)
        for obj, action, remaining in running_tasks:
            new_remaining = remaining - steps
            if new_remaining <= 0:
                object_todo[obj][action] = 0
                object_occupied[obj] = False
            else:
                updated.append((obj, action, new_remaining))
        return updated, completed

    def describe(self) -> str:
        blocking_str = "blocking" if self.is_blocking else "non-blocking"
        return f"{self.action_string} ({blocking_str}, {self.duration}min)"
