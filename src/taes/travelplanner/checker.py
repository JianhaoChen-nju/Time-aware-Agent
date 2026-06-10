"""Symbolic constraint checker for TravelPlanner TAES.

Checks constraints BEFORE committing an event (predictive checking).
Reuses logic from evaluation/hard_constraint.py and evaluation/commonsense_constraint.py.
"""

import re
import logging
import math

from . import costs as costs_mod
from .costs import (
    city_state,
    distance_available,
    extract_before_parenthesis,
    extract_flight_number,
    is_valid_city,
    lookup_accommodation,
    lookup_any_accommodation,
    lookup_attraction,
    lookup_flight,
    lookup_restaurant,
)

logger = logging.getLogger(__name__)
_MIN_ACCOMMODATION_COST_CACHE = {}
_MIN_ACCOMMODATION_NIGHTS_CACHE = {}
_MIN_RESTAURANT_COST_CACHE = {}
_HAS_ACCOMMODATION_CACHE = {}


def extract_from_to(text: str):
    pattern = r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)"
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)


def _clean_city(city: str):
    if city is None:
        return None
    return extract_before_parenthesis(city).strip()


def _cities_for_day(day_plan: dict):
    current_city = day_plan.get('current_city', '')
    if 'from' in current_city.lower():
        org, dest = extract_from_to(current_city)
        return [_clean_city(org), _clean_city(dest)] if org and dest else []
    city = _clean_city(current_city)
    return [city] if city else []


def _city_sequence(plans):
    cities = []
    for plan in plans:
        cities.extend(_cities_for_day(plan))
    return cities


def _transport_route(day_plan: dict):
    transport = day_plan.get('transportation', '-')
    org, dest = extract_from_to(transport) if transport and transport != '-' else (None, None)
    if org is None or dest is None:
        org, dest = extract_from_to(day_plan.get('current_city', ''))
    return _clean_city(org), _clean_city(dest)


def _nonempty(value):
    return value is not None and value not in ('', '-')


def _attractions(day_plan: dict):
    attr_val = day_plan.get('attraction', '-')
    if not _nonempty(attr_val):
        return []
    return [attr.strip() for attr in attr_val.split(';') if attr.strip()]


def _room_type_allowed(rule, room_type):
    if rule is None:
        return True
    if rule == 'not shared room':
        return room_type != 'Shared room'
    if rule == 'shared room':
        return room_type == 'Shared room'
    if rule == 'private room':
        return room_type == 'Private room'
    if rule == 'entire room':
        return room_type == 'Entire home/apt'
    return True


def _house_rule_allowed(rule, house_rules):
    if rule is None:
        return True
    rules = str(house_rules)
    disallowed = {
        'smoking': 'No smoking',
        'parties': 'No parties',
        'children under 10': 'No children under 10',
        'visitors': 'No visitors',
        'pets': 'No pets',
    }
    return disallowed.get(rule) not in rules


def check_budget(state, event) -> bool:
    """Check that this day's cost won't exceed remaining budget."""
    return event.estimated_cost <= state.budget_remaining


def _group_accommodation_cost(row, people_number: int) -> float:
    try:
        occupancy = max(1, float(row['maximum occupancy']))
        return float(row['price']) * math.ceil(people_number / occupancy)
    except Exception:
        return 0.0


def _valid_accommodation_row(row, state, slots_available=None) -> bool:
    if not _room_type_allowed(state.local_constraint.get('room type'), row['room type']):
        return False
    if not _house_rule_allowed(state.local_constraint.get('house rule'), row['house_rules']):
        return False
    if slots_available is not None:
        try:
            minimum = int(math.ceil(float(row['minimum nights'])))
        except Exception:
            minimum = 1
        if minimum > slots_available:
            return False
    return True


def _min_accommodation_cost(state, city=None, slots_available=None) -> float:
    cache_key = (
        city,
        slots_available,
        state.dest_state,
        state.people_number,
        state.local_constraint.get('room type'),
        state.local_constraint.get('house rule'),
    )
    if cache_key in _MIN_ACCOMMODATION_COST_CACHE:
        return _MIN_ACCOMMODATION_COST_CACHE[cache_key]

    costs_mod._init_tools()
    data = costs_mod._accommodations.data
    rows = []
    for _, row in data.iterrows():
        row_city = str(row['city'])
        if city and row_city != city:
            continue
        if not city and city_state(row_city) != state.dest_state:
            continue
        if _valid_accommodation_row(row, state, slots_available):
            rows.append(_group_accommodation_cost(row, state.people_number))
    result = min(rows) if rows else 0.0
    _MIN_ACCOMMODATION_COST_CACHE[cache_key] = result
    return result


def _has_valid_accommodation(state, city=None, slots_available=None) -> bool:
    """Whether at least one lodging option can satisfy local lodging rules."""
    cache_key = (
        city,
        slots_available,
        state.dest_state,
        state.people_number,
        state.local_constraint.get('room type'),
        state.local_constraint.get('house rule'),
    )
    if cache_key in _HAS_ACCOMMODATION_CACHE:
        return _HAS_ACCOMMODATION_CACHE[cache_key]

    costs_mod._init_tools()
    try:
        data = costs_mod._accommodations.data
    except Exception:
        _HAS_ACCOMMODATION_CACHE[cache_key] = False
        return False

    for _, row in data.iterrows():
        row_city = str(row['city'])
        if city and row_city != city:
            continue
        if not city and city_state(row_city) != state.dest_state:
            continue
        if _valid_accommodation_row(row, state, slots_available):
            _HAS_ACCOMMODATION_CACHE[cache_key] = True
            return True

    _HAS_ACCOMMODATION_CACHE[cache_key] = False
    return False


def _min_valid_accommodation_nights(state, city=None) -> float:
    """Minimum nights among lodging options satisfying room/house rules."""
    cache_key = (
        city,
        state.dest_state,
        state.local_constraint.get('room type'),
        state.local_constraint.get('house rule'),
    )
    if cache_key in _MIN_ACCOMMODATION_NIGHTS_CACHE:
        return _MIN_ACCOMMODATION_NIGHTS_CACHE[cache_key]

    costs_mod._init_tools()
    try:
        data = costs_mod._accommodations.data
    except Exception:
        _MIN_ACCOMMODATION_NIGHTS_CACHE[cache_key] = math.inf
        return math.inf

    values = []
    for _, row in data.iterrows():
        row_city = str(row['city'])
        if city and row_city != city:
            continue
        if not city and city_state(row_city) != state.dest_state:
            continue
        if not _valid_accommodation_row(row, state):
            continue
        try:
            values.append(int(math.ceil(float(row['minimum nights']))))
        except Exception:
            values.append(1)

    result = min(values) if values else math.inf
    _MIN_ACCOMMODATION_NIGHTS_CACHE[cache_key] = result
    return result


def _min_restaurant_cost(state, city=None) -> float:
    cache_key = (
        city,
        state.dest_state,
        state.people_number,
    )
    if cache_key in _MIN_RESTAURANT_COST_CACHE:
        return _MIN_RESTAURANT_COST_CACHE[cache_key]

    costs_mod._init_tools()
    data = costs_mod._restaurants.data
    values = []
    for _, row in data.iterrows():
        row_city = str(row['City'])
        if city and row_city != city:
            continue
        if not city and row_city != state.dest_state and city_state(row_city) != state.dest_state:
            continue
        try:
            values.append(float(row['Average Cost']) * state.people_number)
        except Exception:
            continue
    result = min(values) if values else 0.0
    _MIN_RESTAURANT_COST_CACHE[cache_key] = result
    return result


def estimate_future_budget_lower_bound(state) -> float:
    """Lower-bound remaining required spend from a partial TravelPlanner state."""
    if state.current_day > state.total_days:
        return 0.0

    lower_bound = 0.0
    future_accommodation_slots = max(0, state.total_days - state.current_day)
    open_acc, consecutive = _last_accommodation_segment(state)
    required_open_slots = 0

    if _nonempty(open_acc):
        minimum = _minimum_nights(open_acc)
        if minimum is not None and consecutive < minimum:
            required_open_slots = min(future_accommodation_slots, minimum - consecutive)
            row = lookup_any_accommodation(open_acc)
            if row is not None:
                lower_bound += required_open_slots * _group_accommodation_cost(row, state.people_number)

    remaining_slots = max(0, future_accommodation_slots - required_open_slots)
    if remaining_slots:
        lower_bound += remaining_slots * _min_accommodation_cost(
            state,
            slots_available=remaining_slots,
        )

    required = getattr(state, 'visiting_city_number', 0) or 0
    try:
        covered = len(state._visited_destination_cities())
    except Exception:
        covered = 0
    missing = max(0, required - covered) if required else 0

    if missing > 0:
        try:
            visited = state._visited_destination_cities()
        except Exception:
            visited = set()
        missing_transport = _min_missing_city_transport_chain_cost(state, visited)
        if not math.isfinite(missing_transport):
            return math.inf
        lower_bound += missing_transport

    future_non_final_days = max(0, state.total_days - state.current_day)
    minimum_stay_days = max(0, future_non_final_days - missing)
    if minimum_stay_days:
        lower_bound += minimum_stay_days * 3 * _min_restaurant_cost(state)

    must_be_able_to_return = not required or covered >= required or state.current_day == state.total_days
    if must_be_able_to_return:
        return_cost = _min_transport_cost(state.current_city, state.origin_city, state)
        if return_cost is None:
            return math.inf
        lower_bound += return_cost

    return lower_bound


def _min_transport_cost(origin, destination, state) -> float:
    origin = _clean_city(origin)
    destination = _clean_city(destination)
    if not origin or not destination or origin == destination:
        return 0.0

    costs_mod._init_tools()
    values = []
    transport_rule = state.local_constraint.get('transportation')

    allow_flight = transport_rule != 'no flight' and state.transport_mode != 'self-driving'
    allow_self_driving = transport_rule != 'no self-driving' and state.transport_mode not in ('flight', 'taxi')
    allow_taxi = state.transport_mode != 'self-driving'

    if allow_flight:
        try:
            flights = costs_mod._flights.data[
                (costs_mod._flights.data['OriginCityName'] == origin)
                & (costs_mod._flights.data['DestCityName'] == destination)
            ]
            if len(flights) > 0:
                values.append(float(flights['Price'].min()) * state.people_number)
        except Exception:
            pass

    if allow_self_driving:
        try:
            info = costs_mod._distance_matrix.run_for_evaluation(origin, destination, mode='self-driving')
            if info.get('cost') is not None:
                values.append(float(info['cost']) * math.ceil(state.people_number / 5))
        except Exception:
            pass

    if allow_taxi:
        try:
            info = costs_mod._distance_matrix.run_for_evaluation(origin, destination, mode='taxi')
            if info.get('cost') is not None:
                values.append(float(info['cost']) * math.ceil(state.people_number / 4))
        except Exception:
            pass

    return min(values) if values else None


def _last_accommodation_segment(state):
    if not state.daily_plans:
        return None, 0
    current = state.daily_plans[-1].get('accommodation', '-')
    if not _nonempty(current):
        return current, 0
    count = 0
    for plan in reversed(state.daily_plans):
        if plan.get('accommodation', '-') != current:
            break
        count += 1
    return current, count


def _candidate_missing_destination_cities(state, visited):
    if state.total_days <= 3 and state.dest_state:
        return [state.dest_state] if state.dest_state not in visited else []

    costs_mod._init_tools()
    return [
        city
        for city, city_state_value in costs_mod._city_state_map.items()
        if city_state_value == state.dest_state
        and city != state.origin_city
        and city not in visited
    ]


def _min_missing_city_transport_chain_cost(state, visited) -> float:
    values = []
    for city in _candidate_missing_destination_cities(state, visited):
        outbound = _min_transport_cost(state.current_city, city, state)
        if outbound is None:
            continue
        return_cost = _min_transport_cost(city, state.origin_city, state)
        if return_cost is None:
            continue
        values.append(outbound + return_cost)
    return min(values) if values else math.inf


def check_future_budget_feasibility(state, event) -> bool:
    """Admissible lower-bound budget check for the remaining unfinished trip."""
    if event.estimated_cost > state.budget_remaining:
        return False

    next_state = event.apply(state.copy())
    if next_state.current_day > next_state.total_days:
        return next_state.budget_remaining >= 0

    lower_bound = estimate_future_budget_lower_bound(next_state)

    return lower_bound <= next_state.budget_remaining


def check_no_duplicate_restaurants(state, event) -> bool:
    """No restaurant should repeat across the entire trip."""
    day_plan = event.day_plan
    new_restaurants = []
    for meal in ['breakfast', 'lunch', 'dinner']:
        val = day_plan.get(meal, '-')
        if val and val != '-':
            if val in state.used_restaurants or val in new_restaurants:
                return False
            new_restaurants.append(val)
    return True


def check_no_duplicate_attractions(state, event) -> bool:
    """No attraction should repeat across the entire trip."""
    attr_val = event.day_plan.get('attraction', '-')
    if attr_val and attr_val != '-':
        new_attrs = []
        for attr in attr_val.split(';'):
            attr = attr.strip()
            if attr:
                if attr in state.used_attractions or attr in new_attrs:
                    return False
                new_attrs.append(attr)
    return True


def check_city_route(state, event) -> bool:
    """Validate city transition using the official route constraints where safe."""
    current_city_field = event.day_plan.get('current_city', '')
    plans = list(state.daily_plans) + [event.day_plan]
    city_list = _city_sequence(plans)

    if state.current_day == 1:
        if 'from' in current_city_field:
            org, _ = extract_from_to(current_city_field)
            if org and _clean_city(org) != state.origin_city:
                return False
        elif _clean_city(current_city_field) != state.origin_city:
            return False

    if 'from' in current_city_field:
        transport = event.day_plan.get('transportation', '-')
        if not transport or transport == '-':
            return False

    for idx, city in enumerate(city_list):
        if not city or not is_valid_city(city):
            return False
        if state.total_days > 3 and city != state.origin_city and city_state(city) != state.dest_state:
            return False

    if len(city_list) >= 3:
        visited = set()
        i = 0
        while i < len(city_list):
            city = city_list[i]
            if city in visited and 0 < i < len(city_list) - 1:
                return False
            count = 0
            while i < len(city_list) and city_list[i] == city:
                count += 1
                i += 1
            is_last_open_segment = i == len(city_list) and state.current_day < state.total_days
            if count == 1 and 0 < i - 1 < len(city_list) - 1 and not is_last_open_segment:
                return False
            visited.add(city)

    return True


def check_return_trip(state, event) -> bool:
    """Last day must return to origin city."""
    if state.current_day == state.total_days:
        current_city_field = event.day_plan.get('current_city', '')
        if 'from' in current_city_field:
            _, dest = extract_from_to(current_city_field)
            if dest and _clean_city(dest) != state.origin_city:
                return False
        elif _clean_city(current_city_field) != state.origin_city:
            return False
    return True


def check_visiting_city_number(state, event) -> bool:
    plans = list(state.daily_plans) + [event.day_plan]
    city_set = {city for city in _city_sequence(plans) if city}
    city_set.discard(state.origin_city)
    required = getattr(state, 'visiting_city_number', 0)
    if not required:
        return True
    if len(city_set) > required:
        return False
    future_non_final_days = max(0, state.total_days - 1 - state.current_day)
    if len(city_set) + future_non_final_days < required:
        return False
    if state.current_day == state.total_days and len(city_set) != required:
        return False
    return True


def check_city_after_accommodation_feasibility(state, event) -> bool:
    """Ensure open accommodation commitments still leave days for missing cities."""
    required = getattr(state, 'visiting_city_number', 0) or 0
    if not required or state.current_day >= state.total_days:
        return True

    next_state = event.apply(state.copy())
    visited = next_state._visited_destination_cities()
    missing = max(0, required - len(visited))
    if missing <= 0:
        return True

    future_non_final_days = max(0, next_state.total_days - next_state.current_day)
    open_acc, consecutive = _last_accommodation_segment(next_state)
    required_open_slots = 0
    if _nonempty(open_acc):
        minimum = _minimum_nights(open_acc)
        if minimum is not None and consecutive < minimum:
            required_open_slots = max(0, minimum - consecutive)

    free_future_days = future_non_final_days - required_open_slots
    if free_future_days < missing:
        return False

    min_new_city_nights = _min_valid_accommodation_nights(next_state)
    if math.isfinite(min_new_city_nights) and free_future_days < missing * min_new_city_nights:
        return False

    # If every remaining non-final day must introduce a missing city, each
    # future missing city would only get one lodging night before final return.
    # With lodging constraints this is a brittle dead-end for TravelPlanner:
    # the next city must satisfy room type, house rule, and minimum nights all
    # at once. Force city coverage one day earlier instead of leaving it to the
    # final accommodation night.
    has_lodging_constraint = bool(
        next_state.local_constraint.get('room type')
        or next_state.local_constraint.get('house rule')
    )
    if free_future_days == missing and has_lodging_constraint:
        return False

    if free_future_days == missing and not _has_valid_accommodation(
        next_state,
        slots_available=1,
    ):
        return False

    return True


def check_transport_constraint(state, event, local_constraint: dict) -> bool:
    """Check transportation hard constraint (no flight / no self-driving)."""
    transport_rule = local_constraint.get('transportation')
    if transport_rule is None:
        return True

    transport = event.day_plan.get('transportation', '-')
    if not transport or transport == '-':
        return True

    if transport_rule == 'no flight' and 'flight' in transport.lower():
        return False
    if transport_rule == 'no self-driving' and 'self-driving' in transport.lower():
        return False
    return True


def check_self_driving_consistency(state, event) -> bool:
    """If self-driving is used, no flights allowed and vice versa."""
    transport = event.day_plan.get('transportation', '-')
    if not transport or transport == '-':
        return True

    if state.transport_mode == 'self-driving':
        if 'flight' in transport.lower():
            return False
    elif state.transport_mode in ('flight', 'taxi'):
        if 'self-driving' in transport.lower():
            return False
    return True


def check_accommodation_last_day(state, event) -> bool:
    """Non-last day should have accommodation."""
    if state.current_day != state.total_days:
        acc = event.day_plan.get('accommodation', '-')
        if not acc or acc == '-':
            return False
    return True


def check_complete_info(state, event) -> bool:
    """Mirror is_not_absent checks for the candidate day."""
    day_plan = event.day_plan
    for key in ['transportation', 'breakfast', 'lunch', 'dinner', 'attraction', 'accommodation']:
        if key not in day_plan:
            return False

    current_city = day_plan.get('current_city', '')
    is_travel_day = 'from ' in current_city or ' to ' in current_city

    if is_travel_day and not _nonempty(day_plan.get('transportation')):
        return False
    if not is_travel_day and not _nonempty(day_plan.get('attraction')):
        return False
    if state.current_day != state.total_days and not _nonempty(day_plan.get('accommodation')):
        return False
    if not is_travel_day:
        for meal in ['breakfast', 'lunch', 'dinner']:
            if not _nonempty(day_plan.get(meal)):
                return False
    return True


def check_current_city_information(state, event) -> bool:
    day_plan = event.day_plan
    cities = _cities_for_day(day_plan)
    if not cities:
        return False

    transport = day_plan.get('transportation', '-')
    if _nonempty(transport):
        for city in cities:
            if city not in transport:
                return False

    for meal in ['breakfast', 'lunch', 'dinner']:
        value = day_plan.get(meal, '-')
        if _nonempty(value) and not any(city in value for city in cities):
            return False

    for attraction in _attractions(day_plan):
        if not any(city in attraction for city in cities):
            return False

    acc = day_plan.get('accommodation', '-')
    if _nonempty(acc) and cities[-1] not in acc:
        return False

    return True


def check_sandbox_validity(state, event) -> bool:
    day_plan = event.day_plan
    transport = day_plan.get('transportation', '-')
    if _nonempty(transport):
        org, dest = _transport_route(day_plan)
        lower = transport.lower()
        if 'flight number' in lower:
            if lookup_flight(extract_flight_number(transport), org, dest) is None:
                return False
        elif 'self-driving' in lower:
            if not distance_available(org, dest, 'self-driving'):
                return False
        elif 'taxi' in lower:
            if not distance_available(org, dest, 'taxi'):
                return False

    for meal in ['breakfast', 'lunch', 'dinner']:
        value = day_plan.get(meal, '-')
        if _nonempty(value) and lookup_restaurant(value) is None:
            return False

    for attraction in _attractions(day_plan):
        if lookup_attraction(attraction) is None:
            return False

    acc = day_plan.get('accommodation', '-')
    if _nonempty(acc) and lookup_any_accommodation(acc) is None:
        return False

    return True


def check_room_type(state, event, local_constraint: dict) -> bool:
    room_rule = local_constraint.get('room type')
    if room_rule is None:
        return True
    acc = event.day_plan.get('accommodation', '-')
    if not _nonempty(acc):
        return True
    row = lookup_any_accommodation(acc)
    if row is None:
        return True
    return _room_type_allowed(room_rule, row['room type'])


def check_house_rule(state, event, local_constraint: dict) -> bool:
    house_rule = local_constraint.get('house rule')
    if house_rule is None:
        return True
    acc = event.day_plan.get('accommodation', '-')
    if not _nonempty(acc):
        return True
    row = lookup_any_accommodation(acc)
    if row is None:
        return True
    return _house_rule_allowed(house_rule, row['house_rules'])


def check_cuisine_coverage(state, event, local_constraint: dict) -> bool:
    required = local_constraint.get('cuisine')
    if not required:
        return True

    satisfied = set()
    for plan in list(state.daily_plans) + [event.day_plan]:
        for meal in ['breakfast', 'lunch', 'dinner']:
            value = plan.get(meal, '-')
            if not _nonempty(value) or state.origin_city in value:
                continue
            row = lookup_restaurant(value)
            if row is None:
                continue
            cuisines = str(row['Cuisines'])
            for cuisine in required:
                if cuisine in cuisines:
                    satisfied.add(cuisine)

    missing = len(required) - len(satisfied)
    remaining_meals = max(0, state.total_days - state.current_day) * 3
    if missing > remaining_meals:
        return False
    if state.current_day == state.total_days and missing:
        return False
    return True


def _minimum_nights(accommodation: str):
    if not accommodation or accommodation in ('-', ''):
        return None
    row = lookup_accommodation(accommodation)
    if row is None:
        return None
    try:
        return int(math.ceil(float(row['minimum nights'])))
    except Exception:
        return None


def _accommodation_segments(plans):
    segments = []
    current = None
    count = 0
    for plan in plans:
        acc = plan.get('accommodation', '-')
        if acc == current:
            count += 1
            continue
        if current is not None:
            segments.append((current, count))
        current = acc
        count = 1
    if current is not None:
        segments.append((current, count))
    return segments


def check_minimum_nights(state, event) -> bool:
    """Predictively enforce accommodation minimum-night constraints.

    The official evaluator groups consecutive identical accommodation strings
    and checks each group's length against the accommodation DB's
    ``minimum nights``. During partial search, closed segments must already
    satisfy the rule; the open final segment must still be satisfiable with
    the remaining non-final accommodation slots.
    """
    plans = list(state.daily_plans) + [event.day_plan]
    if not plans:
        return True

    segments = _accommodation_segments(plans)
    planned_days = len(plans)
    future_accommodation_slots = max(0, state.total_days - 1 - planned_days)

    for i, (acc, count) in enumerate(segments):
        if acc in ('-', ''):
            continue
        minimum = _minimum_nights(acc)
        if minimum is None:
            continue

        is_last_segment = i == len(segments) - 1
        if not is_last_segment or planned_days >= state.total_days:
            if count < minimum:
                logger.debug("Rejected accommodation segment %s: %s < %s nights", acc, count, minimum)
                return False
        elif count + future_accommodation_slots < minimum:
            logger.debug(
                "Rejected accommodation segment %s: only %s possible nights, needs %s",
                acc, count + future_accommodation_slots, minimum,
            )
            return False

    return True


def check_all(state, event) -> bool:
    """Run all symbolic checks. Returns True if event is valid."""
    return not explain_failures(state, event)


def explain_failures(state, event):
    """Return symbolic check names that reject an event."""
    local_constraint = state.local_constraint

    checks = [
        ("budget", check_budget(state, event)),
        ("future_budget", check_future_budget_feasibility(state, event)),
        ("no_duplicate_restaurants", check_no_duplicate_restaurants(state, event)),
        ("no_duplicate_attractions", check_no_duplicate_attractions(state, event)),
        ("city_route", check_city_route(state, event)),
        ("return_trip", check_return_trip(state, event)),
        ("visiting_city_number", check_visiting_city_number(state, event)),
        ("city_after_accommodation", check_city_after_accommodation_feasibility(state, event)),
        ("transport_constraint", check_transport_constraint(state, event, local_constraint)),
        ("self_driving_consistency", check_self_driving_consistency(state, event)),
        ("complete_info", check_complete_info(state, event)),
        ("current_city_information", check_current_city_information(state, event)),
        ("sandbox_validity", check_sandbox_validity(state, event)),
        ("room_type", check_room_type(state, event, local_constraint)),
        ("house_rule", check_house_rule(state, event, local_constraint)),
        ("cuisine_coverage", check_cuisine_coverage(state, event, local_constraint)),
        ("accommodation_last_day", check_accommodation_last_day(state, event)),
        ("minimum_nights", check_minimum_nights(state, event)),
    ]

    return [name for name, passed in checks if not passed]
