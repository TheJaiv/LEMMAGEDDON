"""The verifier: scores the final bookings against the trip request.

Programmatic only, no LLM. It reads the list of bookings from the sandbox, but
recomputes every price, fee, time and seat count from the harness's own copy of the
timetable, so a number reported by the sandbox is never trusted.

    reward = covered x budget_score x tidy
      covered       share of travellers on one shared valid trip (route, connections, deadline, class)
      budget_score  total spent (tickets kept + cancellation fees) within budget: 1;
                    up to 10% over: 0.5, so a trip over budget can never count as a success;
                    more than 10% over: 0
      tidy          1 - 0.1 per leftover or duplicate booking, never below 0.5
      best_choice   only when the request asks for the cheapest or best-value trip: 1 if the group's
                    trip is as good as the best valid trip the verifier finds itself, else 0.6

Booking nothing covers no one, so it scores 0. Booking everything goes far over budget, so it scores 0.
Travellers must travel together: if they are split across trains, only the biggest group counts.
"""
from collections import Counter

MAX_LEGS = 4          # the longest trip (number of trains) the verifier looks for
MAX_SEARCH = 20000    # search steps, so a flood of bookings can't make the verifier hang
OVER_BUDGET_GRACE = 0.10   # up to this share over budget halves the score; more gives 0
NOT_BEST = 0.6             # a valid trip that isn't the cheapest / best value: partial credit, never a success


def score_bookings(task, state):
    world, request = task.world, task.request
    trains, travellers = world["trains"], request["travellers"]
    notes, legs, spent, leftover = [], [], 0, 0
    seats_used = Counter()

    for booking in state.get("bookings", []):
        train = trains.get(booking.get("train"))
        fare = train["classes"].get(booking.get("class")) if train else None
        names = booking.get("travellers")
        if fare is None or not isinstance(names, list) or not names:
            notes.append(f"{booking.get('id')}: not a real train, class and traveller list")
            leftover += 1
            continue
        price = fare["price"] * len(names)
        if booking.get("status") == "cancelled":
            spent += round(price * world["cancel_fee"])
            continue
        spent += price
        seats_used[(train["number"], booking["class"])] += len(names)
        if len(set(names)) != len(names) or not set(names) <= set(travellers):
            notes.append(f"{booking.get('id')}: lists someone who is not on this trip")
            leftover += 1
            continue
        legs.append({**train, "id": booking.get("id"), "key": len(legs), "class": booking["class"],
                     "travellers": names})   # key: position, because ids from the sandbox may repeat

    # Seat limits. The sandbox enforces them, so this only fails if its state was tampered with.
    # Capacity = seats left at the start + seats held by bookings that existed at the start.
    capacity = Counter({(number, name): f["seats"] for number, t in trains.items() for name, f in t["classes"].items()})
    for booking in world.get("bookings", []):
        capacity[(booking["train"], booking["class"])] += len(booking["travellers"])
    for (number, name), used in seats_used.items():
        if used > capacity[(number, name)]:
            notes.append(f"train {number} class {name}: {used} seats booked but only {capacity[(number, name)]} "
                         f"exist; the booking state is not trusted")
            return _result(0, 0, 0, leftover, spent, request, notes)

    # Every trip (chain of train numbers) each traveller holds a seat on, in the required class.
    holders = {}
    for name in travellers:
        mine = {leg["number"]: leg for leg in legs
                if name in leg["travellers"] and leg["class"] == request["travel_class"]}
        for trip in set(_trips(list(mine.values()), request)):
            holders.setdefault(trip, []).append(name)
    # They travel together, so only the biggest group on one shared trip counts.
    best = max(sorted(holders), key=lambda trip: len(holders[trip]), default=())
    group = holders.get(best, [])
    for name in travellers:
        if name not in group:
            split = any(name in names for names in holders.values())
            notes.append(f"{name}: on different trains from the rest of the group" if split else
                         f"{name}: no valid trip from {request['origin']} to {request['destination']}")

    # The bookings that seat the group on that trip. Anything else, including a second
    # booking for someone who already has a seat, is a leftover.
    used = set()
    for number in best:
        need = set(group)
        options = [leg for leg in legs if leg["number"] == number and leg["class"] == request["travel_class"]]
        while need:
            pick = max(options, key=lambda leg: len(need & set(leg["travellers"])))
            if not need & set(pick["travellers"]):
                break
            used.add(pick["key"])
            need -= set(pick["travellers"])
    unused = [str(leg["id"]) for leg in legs if leg["key"] not in used]
    if unused:
        notes.append(f"leftover bookings: {', '.join(unused)}")
    leftover += len(unused)

    over = spent - request["budget"]
    if over <= 0:
        budget_score = 1
    elif over <= OVER_BUDGET_GRACE * request["budget"]:
        budget_score = 0.5
        notes.append(f"spent Rs {spent}, Rs {over} over the budget of Rs {request['budget']}: score halved")
    else:
        budget_score = 0
        notes.append(f"spent Rs {spent}, more than 10% over the budget of Rs {request['budget']}: score 0")
    tidy = max(0.5, 1 - 0.1 * leftover)
    best_choice = 1
    if request.get("objective") and group:
        top = best_trip(world, request)
        mine = trip_cost(trains, best, request)
        if top and mine > top[1] + 0.5:
            best_choice = NOT_BEST
            notes.append(f"a better valid trip existed ({' + '.join(top[0])}: Rs {top[1]:.0f} per traveller "
                         f"by the request's measure, this trip Rs {mine:.0f})")
    return _result(len(group) / len(travellers), budget_score, tidy, leftover, spent, request, notes, best_choice)


def trip_cost(trains, trip, request):
    """Per traveller, by the request's measure: ticket prices, plus time if the request values time."""
    fares = sum(trains[n]["classes"][request["travel_class"]]["price"] for n in trip)
    if request.get("objective") == "value":
        hours = (trains[trip[-1]]["arrive"] - trains[trip[0]]["depart"]) / 60
        return fares + request["time_value"] * hours
    return fares


def best_trip(world, request):
    """The best valid trip in the whole timetable: (train numbers, cost per traveller), or None.
    Valid = every rule, enough seats for the whole group, and within budget."""
    group, travel_class = len(request["travellers"]), request["travel_class"]
    legs = [{**t, "class": travel_class} for t in world["trains"].values()
            if t["classes"].get(travel_class, {"seats": 0})["seats"] >= group]
    trips = [(trip, trip_cost(world["trains"], trip, request)) for trip in set(_trips(legs, request))]
    trips = [(trip, cost) for trip, cost in trips
             if sum(world["trains"][n]["classes"][travel_class]["price"] for n in trip) * group <= request["budget"]]
    return min(trips, key=lambda tc: (tc[1], tc[0]), default=None)


def _trips(legs, request):
    """Every chain of these trains from the origin to the destination that keeps the rules."""
    found, steps = [], [0]

    def walk(here, earliest, path):
        steps[0] += 1
        if steps[0] > MAX_SEARCH:
            return
        if here == request["destination"] and path:
            found.append(tuple(path))
            return
        if len(path) >= MAX_LEGS:
            return
        for leg in legs:
            if (leg["number"] in path or leg["origin"] != here or leg["depart"] < earliest
                    or leg["class"] != request["travel_class"] or leg["arrive"] > request["deadline"]
                    or (request.get("needs_pantry") and not leg.get("pantry"))
                    or (request.get("max_stops") is not None and leg.get("stops", 0) > request["max_stops"])):
                continue
            walk(leg["destination"], leg["arrive"] + request["min_connection"], path + [leg["number"]])

    walk(request["origin"], 0, [])
    return found


def _result(covered, budget_score, tidy, leftover, spent, request, notes, best_choice=1):
    return {"covered": round(covered, 4), "budget_score": budget_score, "tidy": round(tidy, 4),
            "best_choice": best_choice, "total": round(covered * budget_score * tidy * best_choice, 4),
            "spent": spent, "budget": request["budget"], "leftover_bookings": leftover, "notes": notes}
