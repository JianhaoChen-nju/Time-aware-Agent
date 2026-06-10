"""LLM prompts for TravelPlanner TAES candidate generation."""

import math

from . import costs as costs_mod
from .costs import lookup_accommodation
from .checker import estimate_future_budget_lower_bound


TAES_DAY_PLAN_PROMPT = """You are a travel planning assistant. Given the current state of a trip and available information, generate {K} different candidate plans for Day {day}.

=== TRIP INFO ===
{query}

=== AVAILABLE INFORMATION ===
{reference_info}

=== CURRENT STATE ===
{state_summary}

=== CONSTRAINTS ===
{constraints}

=== INSTRUCTIONS ===
Generate exactly {K} different day plans as a JSON array. Each plan should be a JSON object with these keys:
- "current_city": The city for this day. If traveling, use format "from CityA to CityB"
- "transportation": Flight info (e.g., "Flight Number: F1234567, from A to B, Departure Time: HH:MM, Arrival Time: HH:MM"), or "Self-driving, from A to B", or "Taxi, from A to B", or "-" if staying
- "breakfast": "RestaurantName, City" or "-"
- "attraction": "Attraction1, City;Attraction2, City;" (semicolon-separated, each ending with semicolon) or "-"
- "lunch": "RestaurantName, City" or "-"
- "dinner": "RestaurantName, City" or "-"
- "accommodation": "AccommodationName, City" or "-" (use "-" on the last day)

IMPORTANT RULES:
1. ALL information (flight numbers, restaurant names, accommodation names, attraction names) MUST come from the provided data. Do NOT make up names.
2. Do NOT reuse any restaurant or attraction already used: {used_items}
3. Each candidate should be meaningfully different (different restaurants, attractions, or routes).
4. Restaurants, attractions, and accommodations must be in the current city.
   If "current_city" is "from CityA to CityB", the current city for meals, attractions, and accommodation is CityB.
   Never reuse an accommodation from CityA after traveling to CityB.
5. {special_day_instruction}

Return ONLY a JSON array with {K} objects, no other text.
"""


def build_day_plan_prompt(state, query: str, reference_info: str, K: int = 3) -> str:
    """Build the LLM prompt for generating K candidate day plans."""
    constraints_parts = []
    lc = state.local_constraint
    if lc.get('cuisine'):
        constraints_parts.append(f"Must include cuisine(s): {lc['cuisine']}")
    if lc.get('transportation'):
        constraints_parts.append(f"Transportation rule: {lc['transportation']}")
    if lc.get('room type'):
        constraints_parts.append(f"Room type: {lc['room type']}")
    if lc.get('house rule'):
        constraints_parts.append(f"House rule (must allow): {lc['house rule']}")
    constraints_parts.append(f"Total budget: ${state.budget_total:.0f}, remaining: ${state.budget_remaining:.0f}")
    constraints_str = '\n'.join(constraints_parts) if constraints_parts else 'No special constraints.'

    used_parts = []
    if state.used_restaurants:
        used_parts.append(f"Used restaurants: {', '.join(sorted(state.used_restaurants))}")
    if state.used_attractions:
        used_parts.append(f"Used attractions: {', '.join(sorted(state.used_attractions))}")
    used_str = '; '.join(used_parts) if used_parts else 'None used yet.'

    if state.current_day == 1:
        special = f"This is Day 1. You must depart from {state.origin_city}. Use 'from {state.origin_city} to <destination>' for current_city."
    elif state.current_day == state.total_days:
        special = f"This is the LAST day (Day {state.total_days}). You must return to {state.origin_city}. Use 'from <current> to {state.origin_city}' for current_city. Set accommodation to '-'."
    else:
        special = f"This is Day {state.current_day}. You should stay in or travel between destination cities in {state.dest_state}."

    accommodation_feasibility_rule = _accommodation_feasibility_rule(state)
    if accommodation_feasibility_rule:
        special += " " + accommodation_feasibility_rule

    budget_feasibility_rule = _budget_feasibility_rule(state)
    if budget_feasibility_rule:
        special += " " + budget_feasibility_rule

    budget_slack_rule = _budget_slack_rule(state)
    if budget_slack_rule:
        special += " " + budget_slack_rule

    transport_options_rule = _affordable_transport_options_rule(state)
    if transport_options_rule:
        special += " " + transport_options_rule

    restaurant_options_rule = _affordable_restaurant_options_rule(state)
    if restaurant_options_rule:
        special += " " + restaurant_options_rule

    city_visit_rule = _city_visit_feasibility_rule(state)
    if city_visit_rule:
        special += " " + city_visit_rule

    cuisine_rule = _cuisine_feasibility_rule(state)
    if cuisine_rule:
        special += " " + cuisine_rule

    lodging_rule = _lodging_preference_rule(state)
    if lodging_rule:
        special += " " + lodging_rule

    affordable_lodging_rule = _affordable_lodging_options_rule(state)
    if affordable_lodging_rule:
        special += " " + affordable_lodging_rule

    open_acc_rule = _open_accommodation_rule(state)
    if open_acc_rule:
        special += " " + open_acc_rule

    skeleton_rule = _route_skeleton_rule(state)
    if skeleton_rule:
        special += " " + skeleton_rule

    if state.transport_mode == 'self-driving':
        special += " You are using self-driving for this trip, so use 'Self-driving' for all transportation (no flights)."
    elif state.transport_mode in ('flight', 'taxi'):
        special += " Do not use self-driving (you have been using flights/taxi)."
    elif state.current_day == 1 and (state.visiting_city_number or 0) > 1:
        special += (
            " Choose a transportation mode that can remain consistent for every intercity move and the final return. "
            "Do not start with self-driving unless the provided distance data supports all later city-to-city moves and the final return; "
            "when flights are available, they are often safer for multi-city long-distance trips."
        )

    return TAES_DAY_PLAN_PROMPT.format(
        K=K,
        day=state.current_day,
        query=query,
        reference_info=reference_info,
        state_summary=state.summary(),
        constraints=constraints_str,
        used_items=used_str,
        special_day_instruction=special,
    )


def _route_skeleton_rule(state) -> str:
    skeleton = getattr(state, 'route_skeleton', None)
    if not skeleton:
        return ""
    day_targets = skeleton.get('day_targets') or {}
    target = day_targets.get(state.current_day) or day_targets.get(str(state.current_day))
    if not target:
        return ""

    city = target.get('city')
    accommodation = target.get('accommodation', '-')
    segment_start = target.get('segment_start')
    segment_end = target.get('segment_end')
    if state.current_day == state.total_days:
        return (
            "Follow the route skeleton: this is the final return day. "
            f"Return from the current city to {state.origin_city}, and set accommodation to '-'."
        )
    if not city:
        return ""
    if state.current_day == segment_start:
        travel_rule = (
            f"Follow the route skeleton: Day {state.current_day} starts the segment for {city}. "
            f"Travel to {city} today, use current_city 'from {state.current_city} to {city}' unless already there, "
            f"and set accommodation exactly to '{accommodation}'. "
        )
    else:
        travel_rule = (
            f"Follow the route skeleton: Day {state.current_day} is inside the {city} segment "
            f"(days {segment_start}-{segment_end}). Stay in {city}, set transportation to '-', "
            f"and set accommodation exactly to '{accommodation}'. "
        )
    return (
        travel_rule
        + "Meals and attractions must be valid unused entries in that same city. "
        + "Do not change the skeleton city or accommodation for this day."
    )


def _accommodation_feasibility_rule(state) -> str:
    if state.current_day == state.total_days:
        return ""

    remaining_slots = max(0, state.total_days - state.current_day)
    if remaining_slots <= 0:
        return ""

    return (
        f"There are {remaining_slots} accommodation night(s) left including this day before the final return day. "
        f"If you choose a new accommodation, its minimum nights value in the accommodation table must be <= {remaining_slots}; "
        "otherwise the symbolic verifier will reject the plan."
    )


def _budget_feasibility_rule(state) -> str:
    remaining_days = state.total_days - state.current_day + 1
    if remaining_days <= 0 or state.budget_remaining <= 0:
        return ""

    per_day = state.budget_remaining / remaining_days
    transport_rule = state.local_constraint.get('transportation')
    if transport_rule == 'no self-driving':
        transport_hint = (
            "Do not use self-driving. When taxi is too expensive for the remaining budget, "
            "prefer the cheapest valid flight from the provided data."
        )
    elif transport_rule == 'no flight':
        transport_hint = "Do not use flights. Prefer the cheaper valid option between self-driving and taxi."
    else:
        transport_hint = (
            "Prefer the cheapest valid transportation option from the provided data; "
            "self-driving is often cheaper than taxi or flights when allowed."
        )

    return (
        f"Budget feasibility is critical: only ${state.budget_remaining:.0f} remains for {remaining_days} day(s) "
        f"(about ${per_day:.0f}/day). Prefer the cheapest valid transportation, restaurants, and accommodations from the provided data; "
        "for accommodation minimize total group cost = price * ceil(people / maximum occupancy), not just the listed nightly price. "
        f"Do not choose expensive flights, taxis, restaurants, or hotels if a cheaper valid option is available. {transport_hint}"
    )


def _budget_slack_rule(state) -> str:
    try:
        reserve = estimate_future_budget_lower_bound(state)
    except Exception:
        return ""

    if not math.isfinite(reserve):
        return (
            "The remaining trip currently has no known cheap feasible return route under the constraints; "
            "choose transportation that restores a valid return path."
        )

    slack = state.budget_remaining - reserve
    remaining_days = max(1, state.total_days - state.current_day + 1)
    spend_cap = max(0.0, slack / remaining_days)
    tight = slack < state.budget_total * 0.15
    label = "TIGHT" if tight else "moderate"
    return (
        f"Coarse budget assessment: {label}. Reserve at least ${reserve:.0f} for required future return/accommodation/meals; "
        f"the rough safe spend for this candidate day is about ${spend_cap:.0f}. "
        "If the budget is tight, use the cheapest valid restaurants and avoid optional expensive meals/transport."
    )


def _affordable_restaurant_options_rule(state) -> str:
    if state.current_day == state.total_days:
        return ""

    candidate_cities = []
    if state.current_city and state.current_city != state.origin_city:
        candidate_cities.append(state.current_city)
    elif state.current_day == 1 and state.total_days <= 3 and state.dest_state:
        candidate_cities.append(state.dest_state)
    else:
        candidate_cities.extend(_likely_transport_destinations(state)[:3])

    options = []
    costs_mod._init_tools()
    seen = set()
    for city in candidate_cities:
        try:
            rows = costs_mod._restaurants.data[costs_mod._restaurants.data['City'] == city]
        except Exception:
            continue
        for _, row in rows.iterrows():
            name = f"{row['Name']}, {city}"
            if name in state.used_restaurants or name in seen:
                continue
            try:
                group_cost = float(row['Average Cost']) * state.people_number
            except Exception:
                continue
            seen.add(name)
            options.append((group_cost, name, str(row.get('Cuisines', ''))))

    if not options:
        return ""
    options.sort(key=lambda item: item[0])
    formatted = [
        f"{name} (group meal cost ${cost:.0f}, cuisines {cuisines})"
        for cost, name, cuisines in options[:6]
    ]
    return (
        "Low-cost valid restaurant options include: "
        + "; ".join(formatted)
        + ". Prefer these exact names when budget slack is tight."
    )


def _affordable_transport_options_rule(state) -> str:
    origin = state.current_city
    destinations = _likely_transport_destinations(state)
    if not origin or not destinations:
        return ""

    options = []
    for dest in destinations:
        if dest == origin:
            continue
        options.extend(_transport_options_between(origin, dest, state))

    if not options:
        return ""
    options.sort(key=lambda item: item[0])
    formatted = [text for _, text in options[:5]]
    return (
        "Low-cost valid transportation options for this state include: "
        + "; ".join(formatted)
        + ". Prefer one of these exact transportation strings when traveling."
    )


def _likely_transport_destinations(state):
    if state.current_day == state.total_days:
        return [state.origin_city]

    if state.current_day == 1 and state.total_days <= 3 and state.dest_state:
        return [state.dest_state]

    required = getattr(state, 'visiting_city_number', 0) or 0
    if not required:
        return []

    visited = state._visited_destination_cities()
    costs_mod._init_tools()
    cities = [
        city
        for city, city_state in costs_mod._city_state_map.items()
        if city_state == state.dest_state and city not in visited and city != state.origin_city
    ]
    return sorted(cities)


def _transport_options_between(origin: str, dest: str, state):
    costs_mod._init_tools()
    transport_rule = state.local_constraint.get('transportation') if state.local_constraint else None
    options = []

    allow_flight = transport_rule != 'no flight' and state.transport_mode != 'self-driving'
    allow_self_driving = transport_rule != 'no self-driving' and state.transport_mode not in ('flight', 'taxi')
    allow_taxi = state.transport_mode != 'self-driving'

    if allow_flight:
        try:
            flights = costs_mod._flights.data[
                (costs_mod._flights.data['OriginCityName'] == origin)
                & (costs_mod._flights.data['DestCityName'] == dest)
            ]
            if len(flights) > 0:
                row = flights.sort_values('Price').iloc[0]
                price = float(row['Price']) * state.people_number
                flight_number = str(row['Flight Number'])
                text = (
                    f"Flight Number: {flight_number}, from {origin} to {dest}, "
                    f"Departure Time: {row['DepTime']}, Arrival Time: {row['ArrTime']} "
                    f"(group cost ${price:.0f})"
                )
                options.append((price, text))
        except Exception:
            pass

    if allow_self_driving:
        try:
            info = costs_mod._distance_matrix.run_for_evaluation(origin, dest, mode='self-driving')
            if info.get('cost') is not None:
                price = float(info['cost']) * math.ceil(state.people_number / 5)
                options.append((price, f"Self-driving, from {origin} to {dest} (group cost ${price:.0f})"))
        except Exception:
            pass

    if allow_taxi:
        try:
            info = costs_mod._distance_matrix.run_for_evaluation(origin, dest, mode='taxi')
            if info.get('cost') is not None:
                price = float(info['cost']) * math.ceil(state.people_number / 4)
                options.append((price, f"Taxi, from {origin} to {dest} (group cost ${price:.0f})"))
        except Exception:
            pass

    return options


def _city_visit_feasibility_rule(state) -> str:
    required = getattr(state, 'visiting_city_number', 0) or 0
    if not required or state.current_day >= state.total_days:
        return ""

    visited = state._visited_destination_cities()
    missing = max(0, required - len(visited))
    remaining_non_final_days = max(0, state.total_days - state.current_day)
    if missing <= 0:
        return ""

    min_lodging_nights = _min_valid_lodging_nights_for_prompt(state)
    if min_lodging_nights and remaining_non_final_days <= missing * min_lodging_nights:
        return (
            f"You still need to visit {missing} new destination city/cities. "
            f"Under the lodging constraints, each new city needs about {min_lodging_nights} consecutive accommodation night(s), "
            f"and only {remaining_non_final_days} non-final planning day(s) remain including today. "
            "You MUST travel to a new unvisited destination city today and choose a valid accommodation there; "
            "do not spend an extra stay-day in an already visited city unless required by an unmet minimum-night segment."
        )

    try:
        can_defer_to_one_night = costs_mod_has_valid_one_night_lodging(state)
    except Exception:
        can_defer_to_one_night = True

    if remaining_non_final_days <= missing + 1 and not can_defer_to_one_night:
        return (
            f"You still need to visit {missing} new destination city/cities, and only "
            f"{remaining_non_final_days} non-final planning day(s) remain including today. "
            "Because the lodging constraints have no valid one-night future accommodation segment, "
            "you MUST travel to a new unvisited destination city today unless the current accommodation minimum-night rule forces a stay. "
            "Choose a valid accommodation in the arrival city whose room type, house rule, and minimum nights fit the remaining accommodation nights."
        )

    if remaining_non_final_days > missing + 1:
        return ""

    urgency = "must"
    return (
        f"You still need to visit {missing} new destination city/cities, and only "
        f"{remaining_non_final_days} non-final planning day(s) remain including today. "
        f"For this day, {urgency} travel to a new unvisited destination city from the provided data; "
        "a stay-day candidate is risky unless it is required to satisfy the current accommodation minimum-night rule."
    )


def costs_mod_has_valid_one_night_lodging(state) -> bool:
    costs_mod._init_tools()
    try:
        rows = costs_mod._accommodations.data
    except Exception:
        return True
    for _, row in rows.iterrows():
        try:
            if costs_mod.city_state(str(row['city'])) != state.dest_state:
                continue
        except Exception:
            continue
        if not _row_satisfies_lodging_constraints(row, state, 1):
            continue
        return True
    return False


def _min_valid_lodging_nights_for_prompt(state):
    costs_mod._init_tools()
    try:
        rows = costs_mod._accommodations.data
    except Exception:
        return None
    values = []
    for _, row in rows.iterrows():
        try:
            if costs_mod.city_state(str(row['city'])) != state.dest_state:
                continue
        except Exception:
            continue
        if not _row_satisfies_lodging_constraints(row, state, 99):
            continue
        try:
            values.append(int(math.ceil(float(row['minimum nights']))))
        except Exception:
            values.append(1)
    return min(values) if values else None


def _cuisine_feasibility_rule(state) -> str:
    required = state.local_constraint.get('cuisine') if state.local_constraint else None
    if not required:
        return ""

    covered = set()
    for plan in state.daily_plans:
        for meal in ['breakfast', 'lunch', 'dinner']:
            value = plan.get(meal, '-')
            if not value or value == '-' or state.origin_city in value:
                continue
            from .costs import lookup_restaurant
            row = lookup_restaurant(value)
            if row is None:
                continue
            cuisines = str(row['Cuisines'])
            for cuisine in required:
                if cuisine in cuisines:
                    covered.add(cuisine)

    missing = [cuisine for cuisine in required if cuisine not in covered]
    if not missing:
        return ""

    if state.current_day >= state.total_days - 1:
        return (
            f"Missing required cuisines: {', '.join(missing)}. "
            "Use this day's meals to cover all missing cuisines before the final return day if possible; "
            "choose low-cost restaurants that explicitly include those cuisine names in the provided table."
        )
    return (
        f"Required cuisines still missing: {', '.join(missing)}. "
        "Prefer restaurants that cover missing cuisines while staying within budget."
    )


def _lodging_preference_rule(state) -> str:
    if state.current_day == state.total_days:
        return ""
    house_rule = state.local_constraint.get('house rule') if state.local_constraint else None
    room_type = state.local_constraint.get('room type') if state.local_constraint else None
    parts = []
    if house_rule:
        parts.append(f"allows {house_rule}")
    if room_type:
        parts.append(f"room type satisfies {room_type}")
    if not parts:
        return ""
    return (
        "For accommodation, choose the cheapest option in the current city that "
        + " and ".join(parts)
        + " and also satisfies minimum nights. Avoid switching to accommodations that violate these lodging constraints."
    )


def _affordable_lodging_options_rule(state) -> str:
    if state.current_day == state.total_days:
        return ""

    candidate_cities = []
    if state.current_day == 1 and state.total_days == 3 and state.dest_state:
        candidate_cities.append(state.dest_state)
    elif state.current_day == 1 and (state.visiting_city_number or 0) > 1 and state.dest_state:
        costs_mod._init_tools()
        candidate_cities.extend(
            city
            for city, city_state in costs_mod._city_state_map.items()
            if city_state == state.dest_state
        )
    elif state.current_city and state.current_city != state.origin_city:
        candidate_cities.append(state.current_city)
        if state.current_day >= state.total_days - 1:
            candidate_cities.extend(_likely_transport_destinations(state)[:5])
    if not candidate_cities:
        return ""

    remaining_slots = max(1, state.total_days - state.current_day)
    costs_mod._init_tools()
    options = []
    for city in candidate_cities:
        try:
            rows = costs_mod._accommodations.data[costs_mod._accommodations.data['city'] == city]
        except Exception:
            continue
        for _, row in rows.iterrows():
            if not _row_satisfies_lodging_constraints(row, state, remaining_slots):
                continue
            try:
                occupancy = max(1, float(row['maximum occupancy']))
                group_cost = float(row['price']) * math.ceil(state.people_number / occupancy)
                minimum = int(math.ceil(float(row['minimum nights'])))
            except Exception:
                continue
            options.append((minimum, group_cost, str(row['NAME']), city))

    if not options:
        if state.current_day >= state.total_days - 1 and state.daily_plans:
            previous_acc = state.daily_plans[-1].get('accommodation', '-')
            if previous_acc and previous_acc != '-':
                return (
                    "Accommodation segment feasibility: under the room/house constraints, no valid new accommodation "
                    f"with minimum nights <= {remaining_slots} is available in the likely overnight cities. "
                    "A candidate that opens a new accommodation segment will be rejected by the symbolic verifier. "
                    f"Generate candidates that do not open a new accommodation segment, for example by continuing the current segment '{previous_acc}' when the city route allows it."
                )
        return ""
    options.sort(key=lambda item: (item[0], item[1]))
    formatted = [
        f"{name}, {city} (group nightly cost ${cost:.0f}, minimum nights {minimum})"
        for minimum, cost, name, city in options[:5]
    ]
    return (
        "Affordable valid accommodation options under the lodging constraints include: "
        + "; ".join(formatted)
        + ". Prefer these exact accommodation names when they fit the route."
    )


def _row_satisfies_lodging_constraints(row, state, remaining_slots: int) -> bool:
    house_rule = state.local_constraint.get('house rule') if state.local_constraint else None
    room_type = state.local_constraint.get('room type') if state.local_constraint else None
    if room_type == 'not shared room' and row['room type'] == 'Shared room':
        return False
    if room_type == 'shared room' and row['room type'] != 'Shared room':
        return False
    if room_type == 'private room' and row['room type'] != 'Private room':
        return False
    if room_type == 'entire room' and row['room type'] != 'Entire home/apt':
        return False
    disallowed = {
        'smoking': 'No smoking',
        'parties': 'No parties',
        'children under 10': 'No children under 10',
        'visitors': 'No visitors',
        'pets': 'No pets',
    }
    if house_rule and disallowed.get(house_rule) in str(row['house_rules']):
        return False
    try:
        minimum = int(math.ceil(float(row['minimum nights'])))
    except Exception:
        minimum = 1
    return minimum <= remaining_slots


def _open_accommodation_rule(state) -> str:
    if state.current_day == state.total_days or not state.daily_plans:
        return ""

    current_acc = state.daily_plans[-1].get('accommodation', '-')
    if not current_acc or current_acc == '-':
        return ""

    consecutive = 0
    for plan in reversed(state.daily_plans):
        if plan.get('accommodation', '-') != current_acc:
            break
        consecutive += 1

    row = lookup_accommodation(current_acc)
    if row is None:
        return ""

    try:
        minimum = int(math.ceil(float(row['minimum nights'])))
    except Exception:
        return ""

    if consecutive >= minimum:
        remaining_slots = max(0, state.total_days - state.current_day)
        if remaining_slots <= 1:
            return (
                "Accommodation segment feasibility: "
                f"{remaining_slots} accommodation night(s) remain before the final return. "
                "Opening a new accommodation segment is usually infeasible because it must satisfy room type, house rule, "
                "and minimum nights <= remaining accommodation nights. "
                "The first JSON object in your returned array MUST be a stay-day candidate: "
                f"set current_city exactly to '{state.current_city}', transportation exactly to '-', "
                f"and accommodation exactly to '{current_acc}'. "
                "Fill meals and attractions with valid unused options in that same city. "
                "Only other candidates may travel, and only if they use a new accommodation that explicitly satisfies all lodging constraints."
            )
        return ""

    return (
        f"The current accommodation segment has not met its minimum nights rule: "
        f"'{current_acc}' has been used for {consecutive} night(s), but requires {minimum}. "
        f"For this day, stay in {state.current_city}, set transportation to '-', and set accommodation exactly to '{current_acc}'. "
        "Do not travel to a new city until this minimum-night segment is satisfied."
    )
