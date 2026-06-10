"""Abstract base classes for TAES (Time-Aware Event Search)."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional


class State(ABC):
    """Abstract state S_t = (T_now, R, C_met).

    T_now: current time point (day number for TravelPlanner, timestep for TimeArena)
    resources: remaining resources (e.g., budget)
    constraints_met: which constraints have been satisfied so far
    """

    def __init__(self, t_now: float, resources: Dict[str, Any], constraints_met: Dict[str, bool]):
        self.t_now = t_now
        self.resources = resources
        self.constraints_met = constraints_met

    @abstractmethod
    def is_terminal(self) -> bool:
        """Whether this state represents a completed plan."""
        ...

    def copy(self) -> 'State':
        """Deep copy of this state."""
        return deepcopy(self)

    @abstractmethod
    def summary(self) -> str:
        """Human-readable summary for LLM context."""
        ...


class Event(ABC):
    """Abstract event (action) that transitions one state to the next.

    For TravelPlanner: one day's plan (transport, meals, attractions, accommodation).
    For TimeArena: one atomic action (e.g., 'put laundry in washer').
    """

    def __init__(self, name: str, duration: float = 1.0, cost: Optional[Dict[str, float]] = None):
        self.name = name
        self.duration = duration
        self.cost = cost or {}

    @abstractmethod
    def apply(self, state: State) -> State:
        """Apply this event to produce a new state. Does NOT mutate the input state."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description for LLM context."""
        ...
