"""TAESPlanner: Beam search planner for TravelPlanner using TAES algorithm."""

import logging
import ast
import math
from typing import List, Optional

from ..beam_search import beam_search
from ..llm_interface import get_llm, call_llm, parse_json_from_response
from .state import TravelPlannerState, TravelPlannerEvent
from .checker import (
    check_all,
    explain_failures,
    extract_from_to,
    _house_rule_allowed,
    _room_type_allowed,
)
from .evaluator import evaluate_state
from .costs import city_state, compute_day_cost, lookup_accommodation
from .prompts import (
    build_day_plan_prompt,
    _affordable_lodging_options_rule,
    _open_accommodation_rule,
)

logger = logging.getLogger(__name__)


class TAESPlanner:
    """TAES-based planner for TravelPlanner benchmark.

    Uses neuro-symbolic beam search:
    1. LLM generates K candidate day plans
    2. Symbolic checker prunes invalid ones
    3. Heuristic evaluator scores remaining
    4. Keep top-B beam states
    """

    def __init__(self, model_name: str = 'custom', B: int = 5, K: int = 3):
        self.model_name = model_name
        self.B = B
        self.K = K
        self.llm = get_llm()
        print(f"TAESPlanner loaded (B={B}, K={K})")

    def run(self, reference_information: str, query: str, query_data: dict) -> Optional[str]:
        """Run TAES beam search to generate a travel plan.

        Args:
            reference_information: Database lookup results (flights, hotels, etc.)
            query: The user query text
            query_data: Full query data dict with budget, days, people_number, etc.

        Returns:
            Travel plan as formatted text, or None if planning fails.
        """
        if isinstance(query_data.get('local_constraint'), str):
            query_data['local_constraint'] = ast.literal_eval(query_data['local_constraint'])

        initial_state = TravelPlannerState.from_query_data(query_data)
        initial_state.route_skeleton = self._generate_route_skeleton(
            query=query,
            reference_info=reference_information,
            query_data=query_data,
        )

        def generate_fn(state: TravelPlannerState) -> List[TravelPlannerEvent]:
            return self._generate_candidates(state, query, reference_information)

        def check_fn(state: TravelPlannerState, event: TravelPlannerEvent) -> bool:
            return check_all(state, event)

        best_state, _ = beam_search(
            initial_state=initial_state,
            generate_fn=generate_fn,
            check_fn=check_fn,
            evaluate_fn=evaluate_state,
            B=self.B,
            K=self.K,
            max_depth=query_data['days'],
            diversity_key=lambda state: state.beam_diversity_key(),
        )

        if best_state is None:
            logger.warning("TAES beam search returned no results")
            return None

        if len(best_state.daily_plans) < query_data['days']:
            logger.warning(
                "TAES beam search returned partial plan (%s/%s days)",
                len(best_state.daily_plans),
                query_data['days'],
            )
            return None

        return self._format_plan(best_state)

    def _generate_route_skeleton(
        self,
        query: str,
        reference_info: str,
        query_data: dict,
    ) -> Optional[dict]:
        """Generate a coarse city/night/accommodation skeleton for tight multi-city trips."""
        local_constraint = query_data.get('local_constraint') or {}
        required_cities = query_data.get('visiting_city_number') or 0
        has_lodging_constraint = bool(
            local_constraint.get('room type') or local_constraint.get('house rule')
        )
        if required_cities <= 1 or query_data.get('days', 0) < 5 or not has_lodging_constraint:
            return None

        lodging_options = self._skeleton_lodging_options(query_data)
        prompt = f"""
You are planning the route-level skeleton for a TravelPlanner task before daily details are generated.

Return 3 alternative JSON skeletons as a JSON array. Each skeleton must have this shape:
{{
  "segments": [
    {{"city": "CityA", "nights": 2, "accommodation": "AccommodationName, CityA"}},
    {{"city": "CityB", "nights": 2, "accommodation": "AccommodationName, CityB"}}
  ]
}}

Task:
{query}

Hard skeleton rules:
- Visit exactly {required_cities} different non-origin cities in {query_data.get('dest')}.
- The accommodation nights across all segments must sum to {query_data.get('days') - 1}.
- Each segment's accommodation must be selected exactly from the valid lodging options below, formatted "Name, City".
- Each accommodation must satisfy room type={local_constraint.get('room type')} and house rule={local_constraint.get('house rule')}.
- Each segment's nights must be >= that accommodation's minimum nights.
- Prefer balanced night allocations when minimum nights are tight, e.g. for 6 nights / 3 cities use 2+2+2.
- Do not include restaurants, attractions, or day-by-day plans here.

Valid lodging options:
{lodging_options}

Return ONLY the JSON array.
"""
        try:
            response = call_llm(self.llm, prompt)
            skeletons = parse_json_from_response(response)
        except Exception as e:
            logger.warning("Route skeleton generation failed: %s", e)
            return None

        if isinstance(skeletons, dict):
            skeletons = [skeletons]
        if not isinstance(skeletons, list):
            return None

        for skeleton in skeletons:
            normalized = self._normalize_route_skeleton(skeleton, query_data)
            if normalized:
                logger.info("Using route skeleton: %s", normalized)
                return normalized
        return None

    def _normalize_route_skeleton(self, skeleton: dict, query_data: dict) -> Optional[dict]:
        if not isinstance(skeleton, dict):
            return None
        segments = skeleton.get('segments')
        if not isinstance(segments, list):
            return None

        local_constraint = query_data.get('local_constraint') or {}
        required_cities = query_data.get('visiting_city_number') or 0
        total_nights = query_data.get('days', 0) - 1
        dest_state = query_data.get('dest')
        normalized_segments = []
        seen_cities = set()
        nights_sum = 0

        for segment in segments:
            if not isinstance(segment, dict):
                return None
            city = str(segment.get('city', '')).strip()
            accommodation = str(segment.get('accommodation', '')).strip()
            try:
                nights = int(math.ceil(float(segment.get('nights'))))
            except Exception:
                return None
            if not city or not accommodation or nights <= 0:
                return None
            if city in seen_cities or city_state(city) != dest_state:
                return None
            if not accommodation.endswith(f", {city}"):
                accommodation = f"{accommodation}, {city}"

            row = lookup_accommodation(accommodation)
            if row is None:
                return None
            if not _room_type_allowed(local_constraint.get('room type'), row['room type']):
                return None
            if not _house_rule_allowed(local_constraint.get('house rule'), row['house_rules']):
                return None
            try:
                minimum_nights = int(math.ceil(float(row['minimum nights'])))
            except Exception:
                minimum_nights = 1
            if nights < minimum_nights:
                return None

            seen_cities.add(city)
            nights_sum += nights
            normalized_segments.append({
                'city': city,
                'nights': nights,
                'accommodation': accommodation,
            })

        if len(normalized_segments) != required_cities or nights_sum != total_nights:
            return None

        day_targets = {}
        day = 1
        for segment in normalized_segments:
            start_day = day
            end_day = day + segment['nights'] - 1
            segment['start_day'] = start_day
            segment['end_day'] = end_day
            for target_day in range(start_day, end_day + 1):
                day_targets[target_day] = {
                    'city': segment['city'],
                    'accommodation': segment['accommodation'],
                    'segment_start': start_day,
                    'segment_end': end_day,
                }
            day = end_day + 1

        return {'segments': normalized_segments, 'day_targets': day_targets}

    def _skeleton_lodging_options(self, query_data: dict) -> str:
        from . import costs as costs_mod

        costs_mod._init_tools()
        local_constraint = query_data.get('local_constraint') or {}
        dest_state = query_data.get('dest')
        max_nights = max(1, query_data.get('days', 1) - 1)
        options = []
        try:
            rows = costs_mod._accommodations.data
        except Exception:
            return "(lodging table unavailable)"

        for _, row in rows.iterrows():
            city = str(row['city'])
            if city_state(city) != dest_state:
                continue
            if not _room_type_allowed(local_constraint.get('room type'), row['room type']):
                continue
            if not _house_rule_allowed(local_constraint.get('house rule'), row['house_rules']):
                continue
            try:
                minimum = int(math.ceil(float(row['minimum nights'])))
            except Exception:
                minimum = 1
            if minimum > max_nights:
                continue
            try:
                occupancy = max(1, float(row['maximum occupancy']))
                group_cost = float(row['price']) * math.ceil(query_data.get('people_number', 1) / occupancy)
            except Exception:
                group_cost = 0.0
            options.append((minimum, group_cost, f"{row['NAME']}, {city}"))

        options.sort(key=lambda item: (item[0], item[1], item[2]))
        lines = [
            f"- {name} (minimum nights {minimum}, group nightly cost ${cost:.0f})"
            for minimum, cost, name in options[:40]
        ]
        return "\n".join(lines) if lines else "(no valid lodging options found)"

    def _generate_candidates(
        self, state: TravelPlannerState, query: str, reference_info: str
    ) -> List[TravelPlannerEvent]:
        """Generate K candidate day plans via LLM."""
        prompt = build_day_plan_prompt(
            state=state, query=query, reference_info=reference_info, K=self.K,
        )

        try:
            response = call_llm(self.llm, prompt)
            candidates_json = parse_json_from_response(response)
            events = self._events_from_candidates(candidates_json, state)
            for _ in range(2):
                retry_reason = self._batch_retry_reason(state, events)
                if not retry_reason:
                    break
                retry_prompt = self._build_retry_prompt(prompt, state, events, retry_reason)
                retry_response = call_llm(self.llm, retry_prompt)
                retry_json = parse_json_from_response(retry_response)
                retry_events = self._events_from_candidates(retry_json, state)
                if not retry_events:
                    break
                events = retry_events
            return events

        except Exception as e:
            logger.error(f"LLM candidate generation failed: {e}")
            return []

    def _events_from_candidates(self, candidates_json, state: TravelPlannerState) -> List[TravelPlannerEvent]:
        if not candidates_json:
            logger.warning("Failed to parse candidates from LLM response")
            return []

        events = []
        for candidate in candidates_json[:self.K]:
            if not isinstance(candidate, dict):
                continue

            day_plan = {
                'days': state.current_day,
                'current_city': candidate.get('current_city', state.current_city),
                'transportation': candidate.get('transportation', '-'),
                'breakfast': candidate.get('breakfast', '-'),
                'attraction': candidate.get('attraction', '-'),
                'lunch': candidate.get('lunch', '-'),
                'dinner': candidate.get('dinner', '-'),
                'accommodation': candidate.get('accommodation', '-'),
            }

            try:
                cost = compute_day_cost(day_plan, state.people_number)
            except Exception as e:
                logger.warning(f"Cost computation failed: {e}")
                cost = 0

            dest_city = state.current_city
            current_city_field = day_plan['current_city']
            if 'from' in current_city_field.lower():
                _, dest = extract_from_to(current_city_field)
                if dest:
                    dest_city = dest.strip()
            else:
                dest_city = current_city_field.strip()

            events.append(TravelPlannerEvent(
                day_plan=day_plan,
                estimated_cost=cost,
                destination_city=dest_city,
            ))

        return events

    def _batch_retry_reason(
        self,
        state: TravelPlannerState,
        events: List[TravelPlannerEvent],
    ) -> Optional[str]:
        if not events:
            return None
        failure_sets = [set(explain_failures(state, event)) for event in events]
        retryable = [
            'minimum_nights',
            'budget',
            'future_budget',
            'transport_constraint',
            'house_rule',
            'room_type',
            'cuisine_coverage',
            'current_city_information',
            'complete_info',
            'sandbox_validity',
            'visiting_city_number',
            'city_after_accommodation',
            'self_driving_consistency',
        ]
        for reason in retryable:
            if all(reason in failures for failures in failure_sets):
                return reason
        return None

    def _build_retry_prompt(
        self,
        prompt: str,
        state: TravelPlannerState,
        events: List[TravelPlannerEvent],
        retry_reason: str,
    ) -> str:
        remaining_slots = max(0, state.total_days - state.current_day)
        rejected_accommodations = []
        for event in events:
            acc = event.day_plan.get('accommodation', '-')
            if acc and acc != '-' and acc not in rejected_accommodations:
                rejected_accommodations.append(acc)

        feedback_lines = [
            "",
            "=== SYMBOLIC VERIFIER FEEDBACK ===",
            f"All previous candidates were rejected by the symbolic verifier for: {retry_reason}.",
            f"Regenerate exactly {self.K} JSON candidates that fix this verifier failure.",
        ]
        if retry_reason == 'minimum_nights':
            feedback_lines.extend([
                f"This day can use at most {remaining_slots} consecutive accommodation night(s) before the final return day.",
                f"Choose accommodations whose minimum nights value is <= {remaining_slots}, or continue the exact previous accommodation if the current state says it is required.",
            ])
            lodging_hint = _affordable_lodging_options_rule(state) or _open_accommodation_rule(state)
            if lodging_hint:
                feedback_lines.append(lodging_hint)
        elif retry_reason == 'budget':
            remaining_days = state.total_days - state.current_day + 1
            transport_rule = state.local_constraint.get('transportation')
            if transport_rule == 'no self-driving':
                transport_hint = (
                    "Do not use self-driving. Use the cheapest valid flight when taxi exceeds the budget."
                )
            elif transport_rule == 'no flight':
                transport_hint = "Do not use flights. Use self-driving or taxi only."
            else:
                transport_hint = "Use the cheapest valid transportation mode from the provided data."
            feedback_lines.extend([
                f"The current remaining budget is ${state.budget_remaining:.0f} for {remaining_days} day(s).",
                "Use cheaper valid options from the provided data: inexpensive restaurants and the cheapest accommodation that still satisfies minimum nights and local constraints.",
                transport_hint,
            ])
        elif retry_reason == 'future_budget':
            remaining_days = state.total_days - state.current_day + 1
            feedback_lines.extend([
                f"The candidate day leaves too little budget for the remaining {remaining_days} day(s), including required accommodation nights and the final return trip.",
                "Choose substantially cheaper valid transportation, meals, and accommodation now.",
                "For accommodation, minimize total group cost: price * ceil(people / maximum occupancy), while still satisfying room type, house rule, and minimum nights.",
            ])
        elif retry_reason == 'transport_constraint':
            transport_rule = state.local_constraint.get('transportation')
            feedback_lines.extend([
                f"The transportation constraint is: {transport_rule}.",
                "Regenerate candidates that obey this transportation constraint exactly.",
            ])
            if transport_rule == 'no self-driving':
                feedback_lines.append("Do not use any Self-driving transportation. Use flights or taxi only.")
            elif transport_rule == 'no flight':
                feedback_lines.append("Do not use any Flight Number transportation. Use self-driving or taxi only.")
        elif retry_reason == 'house_rule':
            feedback_lines.extend([
                f"The accommodation house rule must allow: {state.local_constraint.get('house rule')}.",
                "Choose an accommodation whose house_rules field does not prohibit that requirement.",
            ])
            lodging_hint = _affordable_lodging_options_rule(state) or _open_accommodation_rule(state)
            if lodging_hint:
                feedback_lines.append(lodging_hint)
        elif retry_reason == 'room_type':
            feedback_lines.extend([
                f"The accommodation room type constraint is: {state.local_constraint.get('room type')}.",
                "Choose an accommodation whose room type satisfies this constraint and whose minimum nights fit the remaining accommodation nights.",
            ])
            lodging_hint = _affordable_lodging_options_rule(state) or _open_accommodation_rule(state)
            if lodging_hint:
                feedback_lines.append(lodging_hint)
            stay_hint = self._stay_on_current_accommodation_hint(state)
            if stay_hint:
                feedback_lines.append(stay_hint)
        elif retry_reason == 'cuisine_coverage':
            feedback_lines.extend([
                f"The trip must cover these cuisines: {state.local_constraint.get('cuisine')}.",
                "Use restaurants whose Cuisines field includes the missing cuisine preferences, while still obeying budget and duplicate-restaurant constraints.",
            ])
        elif retry_reason == 'current_city_information':
            feedback_lines.extend([
                "Every restaurant, attraction, and accommodation must be in the day destination/current city.",
                "If current_city is 'from A to B', then breakfast/lunch/dinner/attractions/accommodation must be in city B, not city A or a previous city.",
                "If you continue a previous accommodation because of minimum nights, do not travel to a different city on that day.",
            ])
        elif retry_reason == 'complete_info':
            feedback_lines.extend([
                "For a non-travel day, provide breakfast, lunch, dinner, attractions, and accommodation; do not use '-' for meals on a stay day.",
                "For a travel day, provide valid transportation and keep accommodation '-' only on the final return day.",
            ])
        elif retry_reason == 'sandbox_validity':
            feedback_lines.extend([
                "Use only exact flight numbers, restaurants, attractions, accommodations, and transport routes present in the provided data.",
                "For self-driving or taxi, the origin-destination route must exist in the distance matrix data.",
            ])
        elif retry_reason == 'visiting_city_number':
            feedback_lines.extend([
                f"The trip must visit exactly {state.visiting_city_number} non-origin destination city/cities in {state.dest_state}.",
                "This state is running out of non-final days to cover the required number of destination cities.",
                "Regenerate candidates that travel to a new unvisited destination city today when one is still missing.",
            ])
        elif retry_reason == 'city_after_accommodation':
            feedback_lines.extend([
                "The accommodation choice would consume too many remaining non-final days before all required cities are visited.",
                "Choose an accommodation with a shorter minimum-night requirement, or travel to a new required city now if the current accommodation segment is already satisfied.",
            ])
        elif retry_reason == 'self_driving_consistency':
            feedback_lines.extend([
                "The plan cannot mix self-driving with flights or taxis.",
                "If previous transportation used self-driving, regenerate candidates using only valid self-driving routes.",
                "If previous transportation used flights or taxis, do not use self-driving.",
            ])
        if rejected_accommodations:
            feedback_lines.append(
                "Do not reuse these rejected accommodations for this state: "
                + "; ".join(rejected_accommodations)
            )
        feedback_lines.append("Return ONLY a JSON array, no other text.")
        return prompt + "\n".join(feedback_lines)

    def _stay_on_current_accommodation_hint(self, state: TravelPlannerState) -> str:
        if state.current_day >= state.total_days or not state.daily_plans:
            return ""
        remaining_slots = max(0, state.total_days - state.current_day)
        if remaining_slots > 1:
            return ""
        previous_acc = state.daily_plans[-1].get('accommodation', '-')
        if not previous_acc or previous_acc == '-':
            return ""
        return (
            "Because opening a new accommodation segment is not feasible under the remaining-night constraint, "
            "the first JSON object in the returned array MUST be a stay-day that does not travel to a new overnight city. "
            f"For that first object, set current_city exactly to '{state.current_city}', transportation exactly to '-', "
            f"and accommodation exactly to '{previous_acc}'. "
            "Fill breakfast, lunch, dinner, and attraction with valid unused options in that same current city. "
            "Other candidates may differ only if they also avoid opening an infeasible accommodation segment."
        )

    def _format_plan(self, state: TravelPlannerState) -> str:
        """Format the plan from state's daily_plans into the expected text format."""
        lines = []
        for i, day_plan in enumerate(state.daily_plans):
            day_num = i + 1
            lines.append(f"Day {day_num}:")
            lines.append(f"Current City: {day_plan.get('current_city', '-')}")
            lines.append(f"Transportation: {day_plan.get('transportation', '-')}")
            lines.append(f"Breakfast: {day_plan.get('breakfast', '-')}")
            lines.append(f"Attraction: {day_plan.get('attraction', '-')}")
            lines.append(f"Lunch: {day_plan.get('lunch', '-')}")
            lines.append(f"Dinner: {day_plan.get('dinner', '-')}")
            lines.append(f"Accommodation: {day_plan.get('accommodation', '-')}")
            lines.append("")

        return '\n'.join(lines).strip()
