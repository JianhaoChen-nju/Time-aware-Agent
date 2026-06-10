"""TravelPlanner state and event definitions for TAES."""

import math
import re
from typing import Any, Dict, List, Optional, Set
from ..base import State, Event
from .costs import lookup_accommodation, lookup_restaurant


def _clean_city(city: Optional[str]) -> Optional[str]:
    if city is None:
        return None
    match = re.search(r'^(.*?)\([^)]*\)', str(city))
    return (match.group(1) if match else str(city)).strip()


def _cities_for_plan(plan: dict) -> List[str]:
    current_city = plan.get('current_city', '')
    match = re.search(r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)", current_city or "")
    if match:
        return [_clean_city(match.group(1)), _clean_city(match.group(2))]
    city = _clean_city(current_city)
    return [city] if city else []


class TravelPlannerState(State):
    """State for TravelPlanner: tracks current day, city, budget, and plan so far.

    S_t = (current_day, current_city, budget_remaining, daily_plans, used_items, constraints)
    """

    def __init__(
        self,
        current_day: int,
        total_days: int,
        current_city: str,
        origin_city: str,
        dest_state: str,
        cities_to_visit: List[str],
        visiting_city_number: int,
        budget_total: float,
        budget_remaining: float,
        people_number: int,
        daily_plans: List[dict],
        used_restaurants: Set[str],
        used_attractions: Set[str],
        local_constraint: dict,
        transport_mode: Optional[str] = None,
        route_skeleton: Optional[dict] = None,
    ):
        resources = {'budget': budget_remaining}
        constraints_met = {}
        super().__init__(t_now=current_day, resources=resources, constraints_met=constraints_met)

        self.current_day = current_day
        self.total_days = total_days
        self.current_city = current_city
        self.origin_city = origin_city
        self.dest_state = dest_state
        self.cities_to_visit = cities_to_visit
        self.visiting_city_number = visiting_city_number
        self.budget_total = budget_total
        self.budget_remaining = budget_remaining
        self.people_number = people_number
        self.daily_plans = daily_plans
        self.used_restaurants = used_restaurants
        self.used_attractions = used_attractions
        self.local_constraint = local_constraint
        self.transport_mode = transport_mode
        self.route_skeleton = route_skeleton

    def is_terminal(self) -> bool:
        return self.current_day > self.total_days

    def summary(self) -> str:
        lines = [
            f"Day {self.current_day}/{self.total_days}",
            f"Current city: {self.current_city}",
            f"Origin: {self.origin_city}, Destination state: {self.dest_state}",
            f"Budget: ${self.budget_remaining:.0f} remaining of ${self.budget_total:.0f}",
            f"People: {self.people_number}",
            f"Cities to visit: {', '.join(self.cities_to_visit)}",
        ]
        if self.daily_plans:
            lines.append(f"Days planned so far: {len(self.daily_plans)}")
            lines.extend(self._recent_plan_summary())
            lines.extend(self._city_visit_progress_summary())
            lines.extend(self._cuisine_progress_summary())
            accommodation_memory = self._accommodation_memory_summary()
            if accommodation_memory:
                lines.extend(accommodation_memory)
        if self.used_restaurants:
            lines.append(f"Used restaurants ({len(self.used_restaurants)}): {', '.join(list(self.used_restaurants)[:5])}...")
        if self.local_constraint:
            constraints = []
            for k, v in self.local_constraint.items():
                if v is not None:
                    constraints.append(f"{k}: {v}")
            if constraints:
                lines.append(f"Constraints: {'; '.join(constraints)}")
        if self.route_skeleton:
            lines.extend(self._route_skeleton_summary())
        return '\n'.join(lines)

    def _route_skeleton_summary(self) -> List[str]:
        lines = ["Route skeleton:"]
        for segment in self.route_skeleton.get('segments', []):
            lines.append(
                "  "
                + f"days {segment.get('start_day')}-{segment.get('end_day')}: "
                + f"{segment.get('city')} for {segment.get('nights')} night(s), "
                + f"accommodation={segment.get('accommodation', '-')}"
            )
        return lines

    def _recent_plan_summary(self) -> List[str]:
        lines = ["Recent plan memory:"]
        for plan in self.daily_plans[-3:]:
            parts = [
                f"Day {plan.get('days', '?')}",
                f"city={plan.get('current_city', '-')}",
                f"transportation={plan.get('transportation', '-')}",
                f"accommodation={plan.get('accommodation', '-')}",
            ]
            lines.append("  " + "; ".join(parts))
        return lines

    def _city_visit_progress_summary(self) -> List[str]:
        visited = self._visited_destination_cities()
        required = self.visiting_city_number or 0
        if not required:
            return []

        missing = max(0, required - len(visited))
        remaining_non_final_days = max(0, self.total_days - self.current_day)
        lines = [
            "City visit progress:",
            f"  Required non-origin destination cities: {required}",
            f"  Visited non-origin destination cities: {len(visited)}"
            + (f" ({', '.join(sorted(visited))})" if visited else ""),
            f"  Missing destination cities: {missing}",
            f"  Non-final planning days left including today: {remaining_non_final_days}",
        ]
        if missing > 0 and remaining_non_final_days <= missing:
            lines.append("  Must travel to a new unvisited destination city today.")
        return lines

    def _visited_destination_cities(self) -> Set[str]:
        visited = set()
        for plan in self.daily_plans:
            for city in _cities_for_plan(plan):
                if city and city != self.origin_city:
                    visited.add(city)
        return visited

    def beam_diversity_key(self):
        """Bucket states by route-relevant commitments for beam retention."""
        visited = tuple(sorted(self._visited_destination_cities()))
        return (
            visited,
            self.current_city,
            self.transport_mode or 'none',
        )

    def _cuisine_progress_summary(self) -> List[str]:
        required = self.local_constraint.get('cuisine') if self.local_constraint else None
        if not required:
            return []

        covered = set()
        for plan in self.daily_plans:
            for meal in ['breakfast', 'lunch', 'dinner']:
                value = plan.get(meal, '-')
                if not value or value == '-' or self.origin_city in value:
                    continue
                row = lookup_restaurant(value)
                if row is None:
                    continue
                cuisines = str(row['Cuisines'])
                for cuisine in required:
                    if cuisine in cuisines:
                        covered.add(cuisine)

        missing = [cuisine for cuisine in required if cuisine not in covered]
        remaining_meals = max(0, self.total_days - self.current_day + 1) * 3
        lines = [
            "Cuisine progress:",
            f"  Required cuisines: {', '.join(required)}",
            f"  Covered cuisines: {', '.join(sorted(covered)) if covered else 'none'}",
            f"  Missing cuisines: {', '.join(missing) if missing else 'none'}",
            f"  Meal slots left including today: {remaining_meals}",
        ]
        if missing and self.current_day >= self.total_days - 1:
            lines.append("  Prioritize restaurants that cover missing cuisines now; do not leave cuisine coverage to the final return day.")
        return lines

    def _accommodation_memory_summary(self) -> List[str]:
        if self.current_day == self.total_days or not self.daily_plans:
            return []

        current_acc = self.daily_plans[-1].get('accommodation', '-')
        if not current_acc or current_acc == '-':
            return []

        consecutive = 0
        for plan in reversed(self.daily_plans):
            if plan.get('accommodation', '-') != current_acc:
                break
            consecutive += 1

        row = lookup_accommodation(current_acc)
        if row is None:
            return [
                "Accommodation memory:",
                f"  Current accommodation segment: {current_acc}",
                f"  Consecutive nights so far: {consecutive}",
            ]

        try:
            minimum = int(math.ceil(float(row['minimum nights'])))
        except Exception:
            minimum = None

        lines = [
            "Accommodation memory:",
            f"  Current accommodation segment: {current_acc}",
            f"  Consecutive nights so far: {consecutive}",
        ]
        if minimum is not None:
            remaining = max(0, minimum - consecutive)
            lines.append(f"  Minimum nights required: {minimum}")
            if remaining > 0:
                lines.append(
                    f"  Must continue this exact accommodation for {remaining} more night(s) "
                    "unless this is the final return day."
                )
        return lines

    @staticmethod
    def from_query_data(query_data: dict) -> 'TravelPlannerState':
        """Build initial state from a TravelPlanner query_data dict."""
        local_constraint = query_data.get('local_constraint', {})
        if isinstance(local_constraint, str):
            import ast
            local_constraint = ast.literal_eval(local_constraint)

        dest = query_data.get('dest', '')

        return TravelPlannerState(
            current_day=1,
            total_days=query_data['days'],
            current_city=query_data['org'],
            origin_city=query_data['org'],
            dest_state=dest,
            cities_to_visit=[],
            visiting_city_number=query_data.get('visiting_city_number', 0),
            budget_total=query_data['budget'],
            budget_remaining=query_data['budget'],
            people_number=query_data['people_number'],
            daily_plans=[],
            used_restaurants=set(),
            used_attractions=set(),
            local_constraint=local_constraint,
            transport_mode=None,
            route_skeleton=None,
        )


class TravelPlannerEvent(Event):
    """One day's plan: transportation, meals, attractions, accommodation."""

    def __init__(
        self,
        day_plan: dict,
        estimated_cost: float = 0,
        destination_city: Optional[str] = None,
    ):
        super().__init__(
            name="Day plan",
            duration=1.0,
            cost={'budget': estimated_cost},
        )
        self.day_plan = day_plan
        self.estimated_cost = estimated_cost
        self.destination_city = destination_city

    def apply(self, state: TravelPlannerState) -> TravelPlannerState:
        """Apply this day plan to produce a new state."""
        new_state = state.copy()
        new_state.daily_plans = list(state.daily_plans) + [self.day_plan]
        new_state.current_day = state.current_day + 1
        new_state.budget_remaining = state.budget_remaining - self.estimated_cost

        if self.destination_city:
            new_state.current_city = self.destination_city

        new_state.used_restaurants = set(state.used_restaurants)
        for meal in ['breakfast', 'lunch', 'dinner']:
            val = self.day_plan.get(meal, '-')
            if val and val != '-':
                new_state.used_restaurants.add(val)

        new_state.used_attractions = set(state.used_attractions)
        attr_val = self.day_plan.get('attraction', '-')
        if attr_val and attr_val != '-':
            for attr in attr_val.split(';'):
                attr = attr.strip()
                if attr:
                    new_state.used_attractions.add(attr)

        transport = self.day_plan.get('transportation', '-')
        if transport and transport != '-':
            if 'self-driving' in transport.lower():
                new_state.transport_mode = 'self-driving'
            elif 'flight' in transport.lower():
                if new_state.transport_mode is None:
                    new_state.transport_mode = 'flight'
            elif 'taxi' in transport.lower():
                if new_state.transport_mode is None:
                    new_state.transport_mode = 'taxi'

        return new_state

    def describe(self) -> str:
        parts = []
        for key in ['current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']:
            val = self.day_plan.get(key, '-')
            parts.append(f"  {key}: {val}")
        return '\n'.join(parts)
