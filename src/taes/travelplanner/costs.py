"""Cost computation for TravelPlanner TAES.

Computes the cost of a day plan by looking up actual prices from database tools.
"""

import math
import re
import os
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_flights = None
_accommodations = None
_restaurants = None
_distance_matrix = None
_attractions = None
_city_state_map = None


def _init_tools():
    """Initialize database tool singletons. Must be called after sys.path is set up."""
    global _flights, _accommodations, _restaurants, _distance_matrix, _attractions, _city_state_map
    if _flights is not None:
        return
    from tools.flights.apis import Flights
    from tools.accommodations.apis import Accommodations
    from tools.restaurants.apis import Restaurants
    from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
    from tools.attractions.apis import Attractions

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    db_root = os.path.join(repo_root, 'TravelPlanner', 'database')
    _flights = Flights(os.path.join(db_root, 'flights', 'clean_Flights_2022.csv'))
    _accommodations = Accommodations(os.path.join(db_root, 'accommodations', 'clean_accommodations_2022.csv'))
    _restaurants = Restaurants(os.path.join(db_root, 'restaurants', 'clean_restaurant_2022.csv'))
    _distance_matrix = GoogleDistanceMatrix(data_path=os.path.join(db_root, 'googleDistanceMatrix', 'distance.csv'))
    _attractions = Attractions(os.path.join(db_root, 'attractions', 'attractions.csv'))

    _city_state_map = {}
    city_state_path = os.path.join(db_root, 'background', 'citySet_with_states.txt')
    with open(city_state_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            city, state = line.split('\t', 1)
            _city_state_map[city] = state


def _extract_from_to(text: str):
    pattern = r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)"
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)


def _get_name_city(text: str):
    if not text or text == '-':
        return None, None
    pattern = r'(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip(), extract_before_parenthesis(match.group(2).strip()).strip()
    return text.strip(), None


def extract_before_parenthesis(text: str):
    if text is None:
        return None
    match = re.search(r'^(.*?)\([^)]*\)', str(text))
    return match.group(1) if match else str(text)


def city_state(city: str):
    _init_tools()
    if city is None:
        return None
    return _city_state_map.get(extract_before_parenthesis(city).strip())


def is_valid_city(city: str) -> bool:
    return city_state(city) is not None


def lookup_restaurant(text: str):
    _init_tools()
    name, city = _get_name_city(text)
    if not name or not city:
        return None
    try:
        res = _restaurants.data[
            (_restaurants.data['Name'].astype(str).str.contains(re.escape(name))) &
            (_restaurants.data['City'] == city)
        ]
    except Exception:
        return None
    if len(res) > 0:
        return res.iloc[0]
    return None


def lookup_attraction(text: str):
    _init_tools()
    name, city = _get_name_city(text)
    if not name or not city:
        return None
    try:
        res = _attractions.data[
            (_attractions.data['Name'].astype(str).str.contains(re.escape(name))) &
            (_attractions.data['City'] == city)
        ]
    except Exception:
        return None
    if len(res) > 0:
        return res.iloc[0]
    return None


def lookup_flight(flight_number: str, origin: str = None, destination: str = None):
    _init_tools()
    if not flight_number:
        return None
    try:
        res = _flights.data[_flights.data['Flight Number'] == flight_number]
        if origin is not None:
            res = res[res['OriginCityName'] == extract_before_parenthesis(origin).strip()]
        if destination is not None:
            res = res[res['DestCityName'] == extract_before_parenthesis(destination).strip()]
    except Exception:
        return None
    if len(res) > 0:
        return res.iloc[0]
    return None


def distance_available(origin: str, destination: str, mode: str) -> bool:
    _init_tools()
    if not origin or not destination:
        return False
    try:
        info = _distance_matrix.run_for_evaluation(
            extract_before_parenthesis(origin).strip(),
            extract_before_parenthesis(destination).strip(),
            mode=mode,
        )
    except Exception:
        return False
    return info.get('cost') is not None


def extract_flight_number(text: str):
    if not text:
        return None
    match = re.search(r'Flight Number:\s*([^,\s]+)', text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def lookup_accommodation(text: str):
    """Return the unique accommodation DB row for a formatted accommodation string.

    Mirrors the official evaluator's lookup behavior: only unique matches are
    treated as authoritative for constraints such as minimum nights.
    """
    _init_tools()
    name, city = _get_name_city(text)
    if not name or not city:
        return None
    try:
        res = _accommodations.data[
            (_accommodations.data['NAME'].astype(str).str.contains(re.escape(name))) &
            (_accommodations.data['city'] == city)
        ]
    except Exception:
        return None
    if len(res) == 1:
        return res.iloc[0]
    return None


def lookup_any_accommodation(text: str):
    _init_tools()
    name, city = _get_name_city(text)
    if not name or not city:
        return None
    try:
        res = _accommodations.data[
            (_accommodations.data['NAME'].astype(str).str.contains(re.escape(name))) &
            (_accommodations.data['city'] == city)
        ]
    except Exception:
        return None
    if len(res) > 0:
        return res.iloc[0]
    return None


def compute_day_cost(day_plan: dict, people_number: int) -> float:
    """Compute the cost for a single day plan."""
    _init_tools()
    total_cost = 0.0

    # Transportation
    transport = day_plan.get('transportation', '-')
    if transport and transport != '-':
        org_city, dest_city = _extract_from_to(transport)
        if org_city is None or dest_city is None:
            org_city, dest_city = _extract_from_to(day_plan.get('current_city', ''))

        if org_city and dest_city:
            if 'flight number' in transport.lower():
                try:
                    flight_num = transport.split('Flight Number: ')[1].split(',')[0]
                    res = _flights.data[_flights.data['Flight Number'] == flight_num]
                    if len(res) > 0:
                        total_cost += res['Price'].values[0] * people_number
                except (IndexError, KeyError):
                    pass
            elif 'self-driving' in transport.lower():
                try:
                    info = _distance_matrix.run_for_evaluation(org_city.strip(), dest_city.strip(), 'self-driving')
                    if info.get('cost') is not None:
                        total_cost += info['cost'] * math.ceil(people_number / 5)
                except Exception:
                    pass
            elif 'taxi' in transport.lower():
                try:
                    info = _distance_matrix.run_for_evaluation(org_city.strip(), dest_city.strip(), 'taxi')
                    if info.get('cost') is not None:
                        total_cost += info['cost'] * math.ceil(people_number / 4)
                except Exception:
                    pass

    # Meals
    for meal_key in ['breakfast', 'lunch', 'dinner']:
        meal = day_plan.get(meal_key, '-')
        if meal and meal != '-':
            name, city = _get_name_city(meal)
            if name and city:
                try:
                    res = _restaurants.data[
                        (_restaurants.data['Name'].astype(str).str.contains(re.escape(name))) &
                        (_restaurants.data['City'] == city)
                    ]
                    if len(res) > 0:
                        total_cost += res['Average Cost'].values[0] * people_number
                except Exception:
                    pass

    # Accommodation
    acc = day_plan.get('accommodation', '-')
    if acc and acc != '-':
        name, city = _get_name_city(acc)
        if name and city:
            try:
                res = _accommodations.data[
                    (_accommodations.data['NAME'].astype(str).str.contains(re.escape(name))) &
                    (_accommodations.data['city'] == city)
                ]
                if len(res) > 0:
                    total_cost += res['price'].values[0] * math.ceil(
                        people_number / res['maximum occupancy'].values[0]
                    )
            except Exception:
                pass

    return total_cost
