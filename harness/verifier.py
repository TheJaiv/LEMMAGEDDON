"""The verifier: scores the final bookings against the trip request.

Programmatic only, no LLM. It reads the list of bookings from the sandbox, but
recomputes every price, fee, time and seat count from the harness's own copy of the
timetable, so a number reported by the sandbox is never trusted.

    reward = covered x budget_ok x tidy
      covered    share of travellers with a valid trip (route, connections, deadline, class)
      budget_ok  1 if the total spent (tickets + cancellation fees) is within budget, else 0
      tidy       1 - 0.1 per leftover booking, never below 0.5

Booking nothing covers no one, so it scores 0. Booking everything goes over budget, so it scores 0.
"""
from collections import Counter


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
        legs.append({**train, "id": booking.get("id"), "class": booking["class"], "travellers": names})

    # The sandbox enforces seat limits, so this only fails if its state was tampered with.
    for (number, name), used in seats_used.items():
        if used > trains[number]["classes"][name]["seats"]:
            notes.append(f"train {number} class {name}: {used} seats booked but only "
                         f"{trains[number]['classes'][name]['seats']} exist; booking state is not trusted")
            return _result(0, 0, 0, leftover, spent, request, notes)

    used_ids, covered = set(), 0
    for name in travellers:
        trip = _find_trip([leg for leg in legs if name in leg["travellers"]],
                          request["origin"], 0, request, [])
        if trip:
            covered += 1
            used_ids.update(leg["id"] for leg in trip)
        else:
            notes.append(f"{name}: no valid trip from {request['origin']} to {request['destination']}")
    unused = [leg["id"] for leg in legs if leg["id"] not in used_ids]
    if unused:
        notes.append(f"leftover bookings: {', '.join(unused)}")
    leftover += len(unused)

    budget_ok = 1 if spent <= request["budget"] else 0
    if not budget_ok:
        notes.append(f"spent Rs {spent}, over the budget of Rs {request['budget']}")
    tidy = max(0.5, 1 - 0.1 * leftover)
    return _result(covered / len(travellers), budget_ok, tidy, leftover, spent, request, notes)


def _find_trip(legs, here, earliest, request, path):
    """Search the traveller's bookings for a chain of trains from `here` to the destination."""
    if here == request["destination"] and path:
        return path
    for leg in legs:
        if (leg in path or leg["origin"] != here or leg["depart"] < earliest
                or leg["class"] != request["travel_class"] or leg["arrive"] > request["deadline"]):
            continue
        found = _find_trip(legs, leg["destination"], leg["arrive"] + request["min_connection"],
                           request, path + [leg])
        if found:
            return found
    return None


def _result(covered, budget_ok, tidy, leftover, spent, request, notes):
    return {"covered": round(covered, 4), "budget_ok": budget_ok, "tidy": round(tidy, 4),
            "total": round(covered * budget_ok * tidy, 4),
            "spent": spent, "budget": request["budget"], "leftover_bookings": leftover, "notes": notes}
