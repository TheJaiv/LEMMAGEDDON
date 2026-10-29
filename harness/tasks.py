"""The tasks: seeded train-booking trips.

Every task is built from a seed, so the same (task_id, seed) always gives the same
timetable, trip request and reference solution. The generator first plants one trip
that meets every rule (this becomes the reference solution), then adds trap trains
that each break one rule. The verifier checks the rules, not the planted trip, so any
valid set of bookings scores.

Each task also lists its traps: wrong action sequences that must never pass.
tests/check_tasks.py plays them on many seeds to prove the verifier rejects them.
"""
import math
import random
from dataclasses import dataclass, field

from .verifier import best_trip, valid_trips

CITIES = ["Mumbai", "Pune", "Nashik", "Surat", "Vadodara", "Ahmedabad", "Nagpur", "Indore",
          "Bhopal", "Jaipur", "Delhi", "Agra", "Lucknow", "Hyderabad", "Bengaluru", "Chennai"]
NAMES = ["Jaiv", "Shrey", "Sahil", "Kabir", "Priya", "Arjun", "Tanisha", "Vikram", "Isha", "Rohan"]
CLASSES = {"SL": 1.6, "3A": 4.0, "2A": 5.8}   # class -> rupees per minute of travel, roughly
# train type -> how slow it is (x travel time), price factor, stops on the way, pantry car (None: either)
KINDS = {"Rajdhani": (1.0, 1.5, (2, 4), True), "Superfast": (1.15, 1.15, (4, 8), True),
         "Express": (1.3, 1.0, (8, 14), None), "Passenger": (1.7, 0.6, (18, 30), False)}
CANCEL_FEE = 0.2                               # share of the price kept when a booking is cancelled
MIN_CONNECTION = 30                            # minutes needed between trains at a change


@dataclass
class TaskInstance:
    task_id: str
    title: str
    difficulty: str
    instructions: str
    world: dict      # timetable, travellers, fees and existing bookings; a copy is sent to the sandbox
    request: dict    # the trip rules; the verifier checks the bookings against these
    reference: list  # (tool, args) actions that solve the task; used by the oracle
    traps: dict = field(default_factory=dict)  # name -> (tool, args) actions that must NOT pass


# ---------------------------------------------------------------- helpers
def hhmm(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def train_numbers(rng, count=20):
    return iter(str(n) for n in rng.sample(range(11001, 22999), count))


def make_train(rng, number, origin, destination, depart, duration, classes, seats=None, kind="Express",
               price_factor=1):
    """One train: a single leg from origin to destination. `seats` fixes chosen seat counts."""
    fares = {}
    for name in CLASSES:                          # keep a fixed class order: SL, 3A, 2A
        if name in classes:
            price = 5 * round(duration * CLASSES[name] * price_factor * rng.uniform(0.9, 1.1) / 5)
            fares[name] = {"price": price, "seats": rng.randint(4, 30)}
    for name, count in (seats or {}).items():
        fares[name]["seats"] = count
    return {"number": number, "origin": origin, "destination": destination,
            "depart": depart, "arrive": depart + duration, "classes": fares, **features(number, kind)}


def features(number, kind):
    """Type, stops and pantry. Drawn from the train number, not the task's random generator, so
    adding them did not change any earlier task's times, prices or seats."""
    rng = random.Random(f"features:{number}")
    _, _, stops, pantry = KINDS[kind]
    return {"kind": kind, "stops": rng.randint(*stops), "pantry": rng.random() < 0.6 if pantry is None else pantry}


def kind_train(rng, number, origin, destination, depart, base, kind, seats=None):
    """A train of a given type: `base` is how long the fastest type would take on this route."""
    slow, price_factor, _, _ = KINDS[kind]
    return make_train(rng, number, origin, destination, depart, 5 * round(base * slow / 5), list(CLASSES),
                      seats=seats, kind=kind, price_factor=price_factor)


def late_train(rng, number, origin, destination, deadline):
    """A direct train that has every class but arrives 30-120 minutes after the deadline."""
    duration = rng.randrange(150, 300, 5)
    return make_train(rng, number, origin, destination,
                      deadline - duration + rng.randrange(30, 120, 5), duration, list(CLASSES))


def plant_connection(rng, numbers, origin, hub, destination, travel_class, group):
    """Two trains origin -> hub -> destination with a 40-100 minute change, seats for the group."""
    leg1 = make_train(rng, next(numbers), origin, hub, rng.randrange(330, 600, 5),
                      rng.randrange(120, 240, 5), list(CLASSES),
                      seats={travel_class: len(group) + rng.randint(0, 4)})
    leg2 = make_train(rng, next(numbers), hub, destination, leg1["arrive"] + rng.randrange(40, 100, 5),
                      rng.randrange(120, 240, 5), list(CLASSES),
                      seats={travel_class: len(group) + rng.randint(0, 4)})
    return leg1, leg2


def tight_train(rng, number, leg1, leg2):
    """Same route as leg2 and arrives earlier, but leaves less than 30 minutes after leg1 arrives."""
    return make_train(rng, number, leg2["origin"], leg2["destination"],
                      leg1["arrive"] + rng.randrange(5, 25, 5), leg2["arrive"] - leg2["depart"], list(CLASSES))


def add_noise(rng, trains, numbers, stations, avoid, count):
    """Trains on other routes, so the timetable holds more than the answer."""
    while count:
        origin, destination = rng.sample(stations, 2)
        if (origin, destination) in avoid:
            continue
        classes = rng.sample(list(CLASSES), rng.randint(2, 3))
        train = make_train(rng, next(numbers), origin, destination, rng.randrange(300, 1020, 5),
                           rng.randrange(90, 360, 5), classes)
        trains[train["number"]] = train
        count -= 1


def deadline_after(arrive):
    """A deadline just after the planted arrival, rounded up to :00 or :30."""
    return 30 * math.ceil((arrive + 20) / 30)


def budget_for(rng, cost, low=1.15, high=1.4):
    """A budget a little above `cost`, rounded up to Rs 50."""
    return 50 * math.ceil(cost * rng.uniform(low, high) / 50)


def fare(train, travel_class):
    return train["classes"][travel_class]["price"]


def search(origin, destination=None):
    return ("search_trains", {"origin": origin, **({"destination": destination} if destination else {})})


def book(train, travel_class, travellers):
    return ("book", {"train": train["number"], "class": travel_class, "travellers": list(travellers)})


def other_class(train, travel_class):
    return next(c for c in train["classes"] if c != travel_class)


def describe(request, trains, bookings):
    """The trip request in plain words: the first observation the agent gets."""
    if bookings:
        current = "Current bookings (already paid):\n" + "\n".join(
            f"  {b['id']}: train {t['number']} {t['origin']} {hhmm(t['depart'])} -> {t['destination']} "
            f"{hhmm(t['arrive'])}, class {b['class']}, for {', '.join(b['travellers'])}, Rs {b['price']}"
            for b in bookings for t in [trains[b["train"]]])
    else:
        current = "Current bookings: none."
    return (f"Book train tickets for this trip.\n"
            f"Travellers: {', '.join(request['travellers'])} (all travel together on the same trains).\n"
            f"From: {request['origin']}   To: {request['destination']}\n"
            f"Arrive by: {hhmm(request['deadline'])} at the latest. All trains run on the same day.\n"
            f"Class: {request['travel_class']} on every train.\n"
            f"Budget: Rs {request['budget']} in total for all travellers. Everything counts: the tickets you keep "
            f"(including existing bookings) and cancellation fees ({round(CANCEL_FEE * 100)}% of a booking's price).\n"
            f"If you change trains, the next train must leave from the same station at least "
            f"{request['min_connection']} minutes after the previous one arrives.\n"
            f"{extra_rules(request)}"
            f"Every traveller needs a seat on every train of the trip. Leave no extra bookings.\n"
            f"{current}\n"
            f"Use the tools to search trains, check seats, book and cancel. When the bookings are done, call `finish`.")


def extra_rules(request):
    """Sentences for the optional rules: pantry car, stops, and which valid trip is best."""
    lines = []
    if request.get("needs_pantry"):
        lines.append("Every train must have a pantry car (meals on board).")
    if request.get("max_stops") is not None:
        lines.append(f"Every train may make at most {request['max_stops']} stops on the way.")
    if request.get("disruptions"):
        lines.append("Trains can be cancelled by the railway at short notice. A booking on a cancelled train is "
                     "refunded in full but gives no one a seat. Check your bookings before you finish.")
    if request.get("objective") == "cheapest":
        lines.append("Several trips may meet every rule: book the one with the lowest total ticket price.")
    if request.get("objective") == "value":
        lines.append(f"Several trips may meet every rule. The travellers value their time at Rs "
                     f"{request['time_value']} per hour each: book the trip with the lowest ticket price "
                     f"+ Rs {request['time_value']} x hours of travel (first departure to final arrival), "
                     f"per traveller.")
    return "".join(line + "\n" for line in lines)


def make_request(group, origin, destination, deadline, travel_class, budget, **rules):
    return {"travellers": list(group), "origin": origin, "destination": destination,
            "deadline": deadline, "min_connection": MIN_CONNECTION,
            "travel_class": travel_class, "budget": budget, **rules}


def build(task_id, title, difficulty, group, origin, destination, deadline, travel_class, budget,
          stations, trains, reference, traps, bookings=(), **rules):
    request = make_request(group, origin, destination, deadline, travel_class, budget, **rules)
    world = {"stations": sorted(stations), "trains": trains, "travellers": list(group),
             "cancel_fee": CANCEL_FEE, "bookings": [dict(b) for b in bookings]}
    return TaskInstance(task_id=task_id, title=title, difficulty=difficulty,
                        instructions=describe(request, trains, bookings), world=world, request=request,
                        reference=reference, traps=traps)


# ---------------------------------------------------------- direct_easy
def direct_task(seed):
    """1 traveller, one direct train fits. Traps: a cheaper train that arrives late; wrong class."""
    rng = random.Random(f"direct_easy:{seed}")
    stations = rng.sample(CITIES, 6)
    origin, destination = stations[0], stations[1]
    travel_class = rng.choice(list(CLASSES))
    traveller = rng.choice(NAMES)
    numbers = train_numbers(rng)

    # the planted trip: one direct train that meets every rule
    good = make_train(rng, next(numbers), origin, destination, rng.randrange(360, 720, 5),
                      rng.randrange(150, 360, 5), list(CLASSES), seats={travel_class: rng.randint(2, 9)})
    deadline = deadline_after(good["arrive"])
    budget = budget_for(rng, fare(good, travel_class))

    # traps on the same route, each breaking exactly one rule
    duration = max(120, good["arrive"] - good["depart"] - rng.randrange(0, 60, 5))
    late = make_train(rng, next(numbers), origin, destination,                # arrives too late
                      deadline - duration + rng.randrange(30, 120, 5), duration, list(CLASSES))
    depart = max(300, good["depart"] - rng.randrange(30, 180, 5))
    wrong = make_train(rng, next(numbers), origin, destination, depart,       # lacks the required class
                       min(rng.randrange(150, 360, 5), deadline - 10 - depart),
                       [c for c in CLASSES if c != travel_class])
    trains = {t["number"]: t for t in (good, late, wrong)}
    add_noise(rng, trains, numbers, stations, avoid={(origin, destination)}, count=8)

    return build("direct_easy", "Book a direct train", "easy", [traveller], origin, destination,
                 deadline, travel_class, budget, stations, trains,
                 reference=[search(origin, destination), book(good, travel_class, [traveller])],
                 traps={"late train": [book(late, travel_class, [traveller])],
                        "on-time train, wrong class": [book(wrong, other_class(wrong, travel_class), [traveller])]})


# --------------------------------------------------------- soldout_easy
def soldout_task(seed):
    """1-2 travellers. The earliest direct train is sold out in the class; a later one fits."""
    rng = random.Random(f"soldout_easy:{seed}")
    stations = rng.sample(CITIES, 6)
    origin, destination = stations[0], stations[1]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, rng.randint(1, 2))
    numbers = train_numbers(rng)

    # the obvious pick: the earliest train, but no seats left in the class
    first = make_train(rng, next(numbers), origin, destination, rng.randrange(330, 540, 5),
                       rng.randrange(150, 300, 5), list(CLASSES), seats={travel_class: 0})
    # the planted trip: a later direct train with enough seats
    good = make_train(rng, next(numbers), origin, destination, first["depart"] + rng.randrange(45, 150, 5),
                      first["arrive"] - first["depart"] + rng.randrange(0, 45, 5), list(CLASSES),
                      seats={travel_class: len(group) + rng.randint(0, 4)})
    deadline = deadline_after(good["arrive"])
    budget = budget_for(rng, fare(good, travel_class) * len(group))
    late = late_train(rng, next(numbers), origin, destination, deadline)
    trains = {t["number"]: t for t in (first, good, late)}
    traps = {"sold-out train": [book(first, travel_class, group)],
             "late train": [book(late, travel_class, group)]}
    if len(group) == 2:
        # one seat short for the group: splitting the travellers across two trains doesn't count
        short = make_train(rng, next(numbers), origin, destination, good["depart"] - rng.randrange(10, 40, 5),
                           good["arrive"] - good["depart"], list(CLASSES), seats={travel_class: 1})
        trains[short["number"]] = short
        traps["group split across trains"] = [book(short, travel_class, group[:1]),
                                              book(good, travel_class, group[1:])]
    add_noise(rng, trains, numbers, stations, avoid={(origin, destination)}, count=8)

    return build("soldout_easy", "Find a train with seats", "easy", group, origin, destination,
                 deadline, travel_class, budget, stations, trains,
                 reference=[search(origin, destination),
                            ("check_seats", {"train": good["number"], "class": travel_class}),
                            book(good, travel_class, group)],
                 traps=traps)


# ---------------------------------------------------- connection_medium
def connection_task(seed):
    """2 travellers, no direct train fits: change trains at a hub with at least 30 minutes."""
    rng = random.Random(f"connection_medium:{seed}")
    stations = rng.sample(CITIES, 7)
    origin, hub, destination = stations[0], stations[1], stations[2]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, 2)
    numbers = train_numbers(rng)

    leg1, leg2 = plant_connection(rng, numbers, origin, hub, destination, travel_class, group)
    deadline = deadline_after(leg2["arrive"])
    budget = budget_for(rng, (fare(leg1, travel_class) + fare(leg2, travel_class)) * len(group))
    tight = tight_train(rng, next(numbers), leg1, leg2)
    direct = late_train(rng, next(numbers), origin, destination, deadline)
    trains = {t["number"]: t for t in (leg1, leg2, tight, direct)}
    add_noise(rng, trains, numbers, stations, count=8,
              avoid={(origin, destination), (origin, hub), (hub, destination)})

    return build("connection_medium", "Change trains on the way", "medium", group, origin, destination,
                 deadline, travel_class, budget, stations, trains,
                 reference=[search(origin), search(hub, destination),
                            book(leg1, travel_class, group), book(leg2, travel_class, group)],
                 traps={"connection too tight": [book(leg1, travel_class, group), book(tight, travel_class, group)],
                        "direct but late": [book(direct, travel_class, group)],
                        "first train only": [book(leg1, travel_class, group)],
                        "only one traveller": [book(leg1, travel_class, group[:1]),
                                               book(leg2, travel_class, group[:1])]})


# -------------------------------------------------------- budget_medium
def budget_task(seed):
    """1-2 travellers. A fast direct express fits every rule but costs more than the budget."""
    rng = random.Random(f"budget_medium:{seed}")
    stations = rng.sample(CITIES, 7)
    origin, hub, destination = stations[0], stations[1], stations[2]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, rng.randint(1, 2))
    numbers = train_numbers(rng)

    leg1, leg2 = plant_connection(rng, numbers, origin, hub, destination, travel_class, group)
    deadline = deadline_after(leg2["arrive"])
    budget = budget_for(rng, (fare(leg1, travel_class) + fare(leg2, travel_class)) * len(group), 1.05, 1.2)
    # the obvious pick: fast, direct, on time, but its fare alone is 15-40% over the budget
    express = make_train(rng, next(numbers), origin, destination, leg1["depart"] + rng.randrange(0, 90, 5),
                         rng.randrange(120, 180, 5), list(CLASSES),
                         seats={travel_class: len(group) + rng.randint(2, 8)})
    express["classes"][travel_class]["price"] = 5 * math.ceil(budget * rng.uniform(1.15, 1.4) / len(group) / 5)
    slow = late_train(rng, next(numbers), origin, destination, deadline)
    trains = {t["number"]: t for t in (leg1, leg2, express, slow)}
    add_noise(rng, trains, numbers, stations, count=8,
              avoid={(origin, destination), (origin, hub), (hub, destination)})

    return build("budget_medium", "Stay within the budget", "medium", group, origin, destination,
                 deadline, travel_class, budget, stations, trains,
                 reference=[search(origin, destination), search(origin), search(hub, destination),
                            book(leg1, travel_class, group), book(leg2, travel_class, group)],
                 traps={"express over budget": [book(express, travel_class, group)],
                        "express in a cheaper class": [book(express, other_class(express, travel_class), group)],
                        "slow direct, late": [book(slow, travel_class, group)]})


# ---------------------------------------------------------- rebook_hard
def rebook_task(seed):
    """2 travellers hold a booking that now arrives too late: cancel it and rebook via a hub."""
    rng = random.Random(f"rebook_hard:{seed}")
    stations = rng.sample(CITIES, 7)
    origin, hub, destination = stations[0], stations[1], stations[2]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, 2)
    numbers = train_numbers(rng)

    leg1, leg2 = plant_connection(rng, numbers, origin, hub, destination, travel_class, group)
    deadline = deadline_after(leg2["arrive"])
    # the existing booking: a direct train that arrives after the deadline
    old = late_train(rng, next(numbers), origin, destination, deadline)
    existing = {"id": "B1", "train": old["number"], "class": travel_class, "travellers": list(group),
                "status": "active", "price": fare(old, travel_class) * len(group), "fee": 0}
    fee = round(existing["price"] * CANCEL_FEE)
    budget = budget_for(rng, (fare(leg1, travel_class) + fare(leg2, travel_class)) * len(group) + fee, 1.05, 1.15)
    tight = tight_train(rng, next(numbers), leg1, leg2)
    trains = {t["number"]: t for t in (leg1, leg2, old, tight)}
    add_noise(rng, trains, numbers, stations, count=8,
              avoid={(origin, destination), (origin, hub), (hub, destination)})

    cancel = ("cancel", {"booking_id": "B1"})
    return build("rebook_hard", "Fix a booking that arrives too late", "hard", group, origin, destination,
                 deadline, travel_class, budget, stations, trains, bookings=[existing],
                 reference=[("list_bookings", {}), cancel, search(origin), search(hub, destination),
                            book(leg1, travel_class, group), book(leg2, travel_class, group)],
                 traps={"keep the late booking": [],
                        "rebook but keep the old one": [book(leg1, travel_class, group), book(leg2, travel_class, group)],
                        "connection too tight": [cancel, book(leg1, travel_class, group), book(tight, travel_class, group)]})


# --------------------------------------------------- fix_connection_hard
def fix_connection_task(seed):
    """2 travellers hold a connection whose change is too tight. Replace only the second train:
    the budget is too small to cancel both bookings and start over."""
    rng = random.Random(f"fix_connection_hard:{seed}")
    stations = rng.sample(CITIES, 7)
    origin, hub, destination = stations[0], stations[1], stations[2]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, 2)
    numbers = train_numbers(rng)

    leg1, leg2 = plant_connection(rng, numbers, origin, hub, destination, travel_class, group)
    deadline = deadline_after(leg2["arrive"])
    tight = tight_train(rng, next(numbers), leg1, leg2)
    existing = [{"id": f"B{i}", "train": t["number"], "class": travel_class, "travellers": list(group),
                 "status": "active", "price": fare(t, travel_class) * len(group), "fee": 0}
                for i, t in ((1, leg1), (2, tight))]
    keep, drop = existing
    # enough to drop B2 and book leg2, but not to also cancel and rebook B1
    cost = keep["price"] + round(drop["price"] * CANCEL_FEE) + fare(leg2, travel_class) * len(group)
    budget = cost + 10 * int(round(keep["price"] * CANCEL_FEE) * rng.uniform(0.3, 0.7) / 10)
    direct = late_train(rng, next(numbers), origin, destination, deadline)
    trains = {t["number"]: t for t in (leg1, leg2, tight, direct)}
    add_noise(rng, trains, numbers, stations, count=8,
              avoid={(origin, destination), (origin, hub), (hub, destination)})

    return build("fix_connection_hard", "Fix a connection that is too tight", "hard", group, origin,
                 destination, deadline, travel_class, budget, stations, trains, bookings=existing,
                 reference=[("list_bookings", {}), ("cancel", {"booking_id": "B2"}),
                            search(hub, destination), book(leg2, travel_class, group)],
                 traps={"keep both bookings": [],
                        "cancel both and rebook": [("cancel", {"booking_id": "B1"}), ("cancel", {"booking_id": "B2"}),
                                                   book(leg1, travel_class, group), book(leg2, travel_class, group)],
                        "add the good train, keep the tight one": [book(leg2, travel_class, group)],
                        "direct but late": [("cancel", {"booking_id": "B1"}), ("cancel", {"booking_id": "B2"}),
                                            book(direct, travel_class, group)]})


# -------------------------------------------------------- long_trip_hard
def long_trip_task(seed):
    """3 travellers, only a trip with two changes works. Every shortcut breaks a rule:
    the direct train is late, one route has 2 seats, the other has a change that is too tight."""
    rng = random.Random(f"long_trip_hard:{seed}")
    stations = rng.sample(CITIES, 8)
    origin, hub1, hub2, destination = stations[0], stations[1], stations[2], stations[3]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, 3)
    numbers = train_numbers(rng)

    seats = lambda: {travel_class: len(group) + rng.randint(0, 3)}
    leg1 = make_train(rng, next(numbers), origin, hub1, rng.randrange(300, 480, 5),
                      rng.randrange(90, 180, 5), list(CLASSES), seats=seats())
    leg2 = make_train(rng, next(numbers), hub1, hub2, leg1["arrive"] + rng.randrange(40, 90, 5),
                      rng.randrange(90, 180, 5), list(CLASSES), seats=seats())
    leg3 = make_train(rng, next(numbers), hub2, destination, leg2["arrive"] + rng.randrange(40, 90, 5),
                      rng.randrange(90, 180, 5), list(CLASSES), seats=seats())
    deadline = deadline_after(leg3["arrive"])
    budget = budget_for(rng, sum(fare(t, travel_class) for t in (leg1, leg2, leg3)) * len(group))

    direct = late_train(rng, next(numbers), origin, destination, deadline)
    # shortcut via hub1: on time, but only 2 seats in the class for a group of 3
    depart = leg1["arrive"] + rng.randrange(40, 90, 5)
    short = make_train(rng, next(numbers), hub1, destination, depart,
                       min(rng.randrange(120, 240, 5), deadline - 10 - depart), list(CLASSES),
                       seats={travel_class: len(group) - 1})
    # shortcut via hub2: arrives 5-20 minutes before leg3 leaves, too tight to change
    arrive = leg3["depart"] - rng.randrange(5, 25, 5)
    duration = min(rng.randrange(150, 300, 5), arrive - 300)
    tight = make_train(rng, next(numbers), origin, hub2, arrive - duration, duration, list(CLASSES))
    trains = {t["number"]: t for t in (leg1, leg2, leg3, direct, short, tight)}
    add_noise(rng, trains, numbers, stations, count=8,
              avoid={(origin, destination), (origin, hub1), (hub1, hub2), (hub2, destination),
                     (hub1, destination), (origin, hub2)})

    return build("long_trip_hard", "Two changes for a group of three", "hard", group, origin, destination,
                 deadline, travel_class, budget, stations, trains,
                 reference=[search(origin), search(hub1), search(hub2, destination),
                            book(leg1, travel_class, group), book(leg2, travel_class, group),
                            book(leg3, travel_class, group)],
                 traps={"direct but late": [book(direct, travel_class, group)],
                        "two of three on the 2-seat train": [book(leg1, travel_class, group),
                                                            book(short, travel_class, group[:2])],
                        "change too tight at the second hub": [book(tight, travel_class, group),
                                                               book(leg3, travel_class, group)],
                        "first two trains only": [book(leg1, travel_class, group), book(leg2, travel_class, group)]})


# --------------------------------------------------- facilities_medium
def facilities_task(seed):
    """1-2 travellers need a pantry car and few stops. Four types of train run the same route:
    the cheap Passenger has no pantry and many stops, the Express stops too often, the Rajdhani
    is over budget, and the Superfast fits."""
    rng = random.Random(f"facilities_medium:{seed}")
    stations = rng.sample(CITIES, 6)
    origin, destination = stations[0], stations[1]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, rng.randint(1, 2))
    numbers = train_numbers(rng)
    base, max_stops = rng.randrange(150, 240, 5), rng.choice([6, 7, 8])
    seats = lambda: {travel_class: len(group) + rng.randint(1, 6)}

    good = kind_train(rng, next(numbers), origin, destination, rng.randrange(420, 600, 5), base, "Superfast", seats())
    good["stops"] = min(good["stops"], max_stops)
    slow = kind_train(rng, next(numbers), origin, destination, rng.randrange(300, 360, 5), base, "Passenger", seats())
    many = kind_train(rng, next(numbers), origin, destination, rng.randrange(330, 480, 5), base, "Express", seats())
    many["pantry"], many["stops"] = True, max(many["stops"], max_stops + 2)   # only the stops are wrong
    fast = kind_train(rng, next(numbers), origin, destination, rng.randrange(420, 600, 5), base, "Rajdhani", seats())
    deadline = deadline_after(max(t["arrive"] for t in (good, slow, many, fast)))
    budget = budget_for(rng, fare(good, travel_class) * len(group), 1.05, 1.15)
    if fare(fast, travel_class) * len(group) <= budget:                       # the Rajdhani must not fit
        fast["classes"][travel_class]["price"] = 5 * math.ceil(budget * rng.uniform(1.15, 1.35) / len(group) / 5)
    late = late_train(rng, next(numbers), origin, destination, deadline)
    trains = {t["number"]: t for t in (good, slow, many, fast, late)}
    add_noise(rng, trains, numbers, stations, avoid={(origin, destination)}, count=8)

    return build("facilities_medium", "Pantry car and few stops", "medium", group, origin, destination,
                 deadline, travel_class, budget, stations, trains, needs_pantry=True, max_stops=max_stops,
                 reference=[search(origin, destination), book(good, travel_class, group)],
                 traps={"Passenger: no pantry, many stops": [book(slow, travel_class, group)],
                        "Express: too many stops": [book(many, travel_class, group)],
                        "Rajdhani: over budget": [book(fast, travel_class, group)],
                        "late train": [book(late, travel_class, group)]})


# ----------------------------------------------------- cheapest_medium
def cheapest_task(seed):
    """1-2 travellers. A direct train and a connection both meet every rule: book the cheaper one
    (either can be cheaper, 50/50). A cheaper direct train arrives late; a cheaper change is too tight."""
    rng = random.Random(f"cheapest_medium:{seed}")
    stations = rng.sample(CITIES, 7)
    origin, hub, destination = stations[0], stations[1], stations[2]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, rng.randint(1, 2))
    numbers = train_numbers(rng)
    seats = lambda: {travel_class: len(group) + rng.randint(1, 5)}

    direct = kind_train(rng, next(numbers), origin, destination, rng.randrange(420, 600, 5),
                        rng.randrange(180, 270, 5), rng.choice(["Rajdhani", "Superfast"]), seats())
    leg1 = kind_train(rng, next(numbers), origin, hub, rng.randrange(330, 450, 5), rng.randrange(90, 150, 5),
                      "Express", seats())
    leg2 = kind_train(rng, next(numbers), hub, destination, leg1["arrive"] + rng.randrange(40, 100, 5),
                      rng.randrange(90, 150, 5), rng.choice(["Express", "Passenger"]), seats())
    connection = fare(leg1, travel_class) + fare(leg2, travel_class)
    if rng.random() < 0.5:                                                    # the direct train is clearly dearer
        direct["classes"][travel_class]["price"] = 5 * math.ceil(connection * rng.uniform(1.25, 1.5) / 5)
    else:                                                                     # ... or clearly cheaper
        direct["classes"][travel_class]["price"] = 5 * math.floor(connection * rng.uniform(0.6, 0.8) / 5)
    cheaper, dearer = sorted((fare(direct, travel_class), connection))
    deadline = deadline_after(max(direct["arrive"], leg2["arrive"]))
    budget = budget_for(rng, dearer * len(group), 1.1, 1.3)                   # both options fit
    late = late_train(rng, next(numbers), origin, destination, deadline)
    late["classes"][travel_class]["price"] = 5 * round(cheaper * 0.6 / 5)    # cheapest of all, but late
    tight = tight_train(rng, next(numbers), leg1, leg2)
    trains = {t["number"]: t for t in (direct, leg1, leg2, late, tight)}
    add_noise(rng, trains, numbers, stations, count=8,
              avoid={(origin, destination), (origin, hub), (hub, destination)})

    rules = {"objective": "cheapest"}
    top = best_trip({"trains": trains}, make_request(group, origin, destination, deadline, travel_class, budget, **rules))
    other = [book(leg1, travel_class, group), book(leg2, travel_class, group)] if top[0] == (direct["number"],) \
        else [book(direct, travel_class, group)]
    return build("cheapest_medium", "The cheapest valid trip", "medium", group, origin, destination,
                 deadline, travel_class, budget, stations, trains, **rules,
                 reference=[search(origin, destination), search(origin), search(hub, destination)]
                           + [book(trains[n], travel_class, group) for n in top[0]],
                 traps={"the other valid option (dearer)": other,
                        "cheapest direct, but late": [book(late, travel_class, group)],
                        "cheaper change, but too tight": [book(leg1, travel_class, group), book(tight, travel_class, group)]})


# ---------------------------------------------------------- value_hard
def value_task(seed):
    """2 travellers. A fast direct Rajdhani and a slower, cheaper connection both meet every rule.
    The request puts a price on time, so the agent has to work out which one is better value;
    the generator sets that price so either can win, by a clear margin."""
    rng = random.Random(f"value_hard:{seed}")
    stations = rng.sample(CITIES, 7)
    origin, hub, destination = stations[0], stations[1], stations[2]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, 2)
    numbers = train_numbers(rng)
    seats = lambda: {travel_class: len(group) + rng.randint(1, 5)}

    direct = kind_train(rng, next(numbers), origin, destination, rng.randrange(420, 600, 5),
                        rng.randrange(180, 240, 5), "Rajdhani", seats())
    leg1 = kind_train(rng, next(numbers), origin, hub, rng.randrange(330, 420, 5), rng.randrange(120, 170, 5),
                      "Express", seats())
    leg2 = kind_train(rng, next(numbers), hub, destination, leg1["arrive"] + rng.randrange(40, 100, 5),
                      rng.randrange(120, 170, 5), "Express", seats())
    slow_fare = fare(leg1, travel_class) + fare(leg2, travel_class)
    if fare(direct, travel_class) < slow_fare * 1.3:
        direct["classes"][travel_class]["price"] = 5 * math.ceil(slow_fare * rng.uniform(1.3, 1.7) / 5)
    fast_hours = (direct["arrive"] - direct["depart"]) / 60
    slow_hours = (leg2["arrive"] - leg1["depart"]) / 60
    breakeven = (fare(direct, travel_class) - slow_fare) / (slow_hours - fast_hours)   # Rs per hour
    direct_wins = rng.random() < 0.5
    time_value = max(10, 10 * round(breakeven * (rng.uniform(1.6, 2.2) if direct_wins else rng.uniform(0.3, 0.6)) / 10))
    deadline = deadline_after(max(direct["arrive"], leg2["arrive"]))
    budget = budget_for(rng, fare(direct, travel_class) * len(group), 1.1, 1.3)   # both options fit
    late = late_train(rng, next(numbers), origin, destination, deadline)
    late["classes"][travel_class]["price"] = 5 * round(slow_fare * 0.6 / 5)
    tight = tight_train(rng, next(numbers), leg1, leg2)
    trains = {t["number"]: t for t in (direct, leg1, leg2, late, tight)}
    add_noise(rng, trains, numbers, stations, count=8,
              avoid={(origin, destination), (origin, hub), (hub, destination)})

    rules = {"objective": "value", "time_value": time_value}
    top = best_trip({"trains": trains}, make_request(group, origin, destination, deadline, travel_class, budget, **rules))
    other = [book(leg1, travel_class, group), book(leg2, travel_class, group)] if top[0] == (direct["number"],) \
        else [book(direct, travel_class, group)]
    return build("value_hard", "Price versus time", "hard", group, origin, destination,
                 deadline, travel_class, budget, stations, trains, **rules,
                 reference=[search(origin, destination), search(origin), search(hub, destination)]
                           + [book(trains[n], travel_class, group) for n in top[0]],
                 traps={"the other valid option (worse value)": other,
                        "cheapest direct, but late": [book(late, travel_class, group)],
                        "cheaper change, but too tight": [book(leg1, travel_class, group), book(tight, travel_class, group)]})


# -------------------------------------------------------- disruption_hard
REASONS = ["signal failure", "flooding near the track", "a landslide on the line", "a locomotive breakdown"]


def disruption_task(seed):
    """3 travellers. Right after the first booking, the railway cancels the obvious train: a cheap
    direct one that is on time. The agent has to notice the alert and rebook a connection; every
    other shortcut breaks a rule (late, a change that is too tight, over budget)."""
    rng = random.Random(f"disruption_hard:{seed}")
    stations = rng.sample(CITIES, 7)
    origin, hub, destination = stations[0], stations[1], stations[2]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, 3)
    numbers = train_numbers(rng)

    leg1, leg2 = plant_connection(rng, numbers, origin, hub, destination, travel_class, group)
    deadline = deadline_after(leg2["arrive"])
    slow_fare = fare(leg1, travel_class) + fare(leg2, travel_class)
    budget = budget_for(rng, slow_fare * len(group), 1.05, 1.2)

    def direct(kind, seats=None):
        """A direct train that arrives on time."""
        arrive = deadline - rng.randrange(15, 60, 5)
        duration = min(rng.randrange(180, 300, 5), arrive - 300)
        return make_train(rng, next(numbers), origin, destination, arrive - duration, duration, list(CLASSES),
                          seats=seats, kind=kind)
    doomed = direct("Superfast", {travel_class: len(group) + rng.randint(2, 6)})   # cancelled after booking
    doomed["classes"][travel_class]["price"] = 5 * round(slow_fare * rng.uniform(0.6, 0.85) / 5)
    pricey = direct("Rajdhani", {travel_class: len(group) + rng.randint(2, 6)})    # on time, over budget
    pricey["classes"][travel_class]["price"] = 5 * math.ceil(budget * rng.uniform(1.15, 1.4) / len(group) / 5)
    late = late_train(rng, next(numbers), origin, destination, deadline)
    late["classes"][travel_class]["price"] = 5 * round(slow_fare * 0.5 / 5)
    tight = tight_train(rng, next(numbers), leg1, leg2)
    trains = {t["number"]: t for t in (leg1, leg2, doomed, pricey, late, tight)}
    add_noise(rng, trains, numbers, stations, count=8,
              avoid={(origin, destination), (origin, hub), (hub, destination)})

    task = build("disruption_hard", "A train cancelled after booking", "hard", group, origin, destination,
                 deadline, travel_class, budget, stations, trains, disruptions=True,
                 reference=[search(origin, destination), book(doomed, travel_class, group), ("list_bookings", {}),
                            search(origin, hub), search(hub, destination),
                            book(leg1, travel_class, group), book(leg2, travel_class, group)],
                 traps={"book the direct train and finish": [book(doomed, travel_class, group)],
                        "direct but late": [book(late, travel_class, group)],
                        "change too tight": [book(leg1, travel_class, group), book(tight, travel_class, group)],
                        "fast direct, over budget": [book(pricey, travel_class, group)]})
    task.world["disruption"] = {"train": doomed["number"], "reason": rng.choice(REASONS)}
    return task


# -------------------------------------------------------- join_group_hard
def join_group_task(seed):
    """Two travellers hold the last seats on a first train; a third joins them. The group must
    travel together, so all three move to trains with seats for three. The budget pays for the
    cancellation fee, but not for keeping the old booking as well."""
    rng = random.Random(f"join_group_hard:{seed}")
    stations = rng.sample(CITIES, 7)
    origin, hub, destination = stations[0], stations[1], stations[2]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, 3)
    pair = group[:2]
    numbers = train_numbers(rng)
    seats = lambda: {travel_class: len(group) + rng.randint(0, 3)}

    held = make_train(rng, next(numbers), origin, hub, rng.randrange(330, 450, 5), rng.randrange(120, 200, 5),
                      list(CLASSES), seats={travel_class: 0})          # the pair hold its last seats
    move = make_train(rng, next(numbers), origin, hub, held["depart"] + rng.randrange(30, 120, 5),
                      rng.randrange(120, 200, 5), list(CLASSES), seats=seats())
    leg2 = make_train(rng, next(numbers), hub, destination,
                      max(held["arrive"], move["arrive"]) + rng.randrange(40, 100, 5),
                      rng.randrange(120, 240, 5), list(CLASSES), seats=seats())
    deadline = deadline_after(leg2["arrive"])
    # a second train that only connects with the held one, with 2 seats: enough for the pair alone
    depart = held["arrive"] + rng.randrange(40, 80, 5)
    small = make_train(rng, next(numbers), hub, destination, depart,
                       min(rng.randrange(120, 240, 5), deadline - 10 - depart), list(CLASSES), seats={travel_class: 2})
    tight = tight_train(rng, next(numbers), move, leg2)
    late = late_train(rng, next(numbers), origin, destination, deadline)
    existing = {"id": "B1", "train": held["number"], "class": travel_class, "travellers": list(pair),
                "status": "active", "price": fare(held, travel_class) * len(pair), "fee": 0}
    fee = round(existing["price"] * CANCEL_FEE)
    budget = budget_for(rng, fee + (fare(move, travel_class) + fare(leg2, travel_class)) * len(group), 1.05, 1.15)
    trains = {t["number"]: t for t in (held, move, leg2, small, tight, late)}
    add_noise(rng, trains, numbers, stations, count=8,
              avoid={(origin, destination), (origin, hub), (hub, destination)})

    cancel = ("cancel", {"booking_id": "B1"})
    return build("join_group_hard", "A third traveller joins", "hard", group, origin, destination,
                 deadline, travel_class, budget, stations, trains, bookings=[existing],
                 reference=[("list_bookings", {}), search(origin, hub),
                            ("check_seats", {"train": held["number"], "class": travel_class}), cancel,
                            book(move, travel_class, group), search(hub, destination), book(leg2, travel_class, group)],
                 traps={"keep the pair, newcomer on another first train": [book(move, travel_class, group[2:]),
                                                                          book(leg2, travel_class, group)],
                        "the pair on a 2-seat second train, newcomer left out": [book(small, travel_class, pair)],
                        "move everyone but keep the old booking": [book(move, travel_class, group),
                                                                   book(leg2, travel_class, group)],
                        "change too tight": [cancel, book(move, travel_class, group), book(tight, travel_class, group)],
                        "direct but late": [cancel, book(late, travel_class, group)]})


# ----------------------------------------------------------- network_hard
def network_task(seed):
    """A big timetable: 14 stations, about 50 trains, 4 travellers. Only trips with two changes
    work: the direct trains are late or over budget, and every one-change trip is too tight or
    short of seats for four. Noise trains that would open a shortcut are removed."""
    rng = random.Random(f"network_hard:{seed}")
    stations = rng.sample(CITIES, 14)
    origin, hub1, hub2, destination = stations[:4]
    travel_class = rng.choice(list(CLASSES))
    group = rng.sample(NAMES, 4)
    numbers = train_numbers(rng, 80)
    seats = lambda: {travel_class: len(group) + rng.randint(0, 3)}

    leg1 = make_train(rng, next(numbers), origin, hub1, rng.randrange(300, 480, 5),
                      rng.randrange(90, 180, 5), list(CLASSES), seats=seats())
    leg2 = make_train(rng, next(numbers), hub1, hub2, leg1["arrive"] + rng.randrange(40, 90, 5),
                      rng.randrange(90, 180, 5), list(CLASSES), seats=seats())
    leg3 = make_train(rng, next(numbers), hub2, destination, leg2["arrive"] + rng.randrange(40, 90, 5),
                      rng.randrange(90, 180, 5), list(CLASSES), seats=seats())
    deadline = deadline_after(leg3["arrive"])
    budget = budget_for(rng, sum(fare(t, travel_class) for t in (leg1, leg2, leg3)) * len(group), 1.1, 1.25)

    late = late_train(rng, next(numbers), origin, destination, deadline)
    arrive = deadline - rng.randrange(10, 60, 5)
    duration = min(rng.randrange(180, 300, 5), arrive - 300)
    pricey = make_train(rng, next(numbers), origin, destination, arrive - duration, duration, list(CLASSES),
                        seats={travel_class: len(group) + rng.randint(1, 4)}, kind="Rajdhani")
    pricey["classes"][travel_class]["price"] = 5 * math.ceil(budget * rng.uniform(1.15, 1.4) / len(group) / 5)
    # via hub1: on time, but one seat short for the group
    depart = leg1["arrive"] + rng.randrange(40, 90, 5)
    short = make_train(rng, next(numbers), hub1, destination, depart,
                       min(rng.randrange(120, 240, 5), deadline - 10 - depart), list(CLASSES),
                       seats={travel_class: len(group) - 1})
    # via hub2: arrives 5-20 minutes before leg3 leaves, too tight to change
    arrive = leg3["depart"] - rng.randrange(5, 25, 5)
    duration = min(rng.randrange(150, 300, 5), arrive - 300)
    tight = make_train(rng, next(numbers), origin, hub2, arrive - duration, duration, list(CLASSES))
    trains = {t["number"]: t for t in (leg1, leg2, leg3, late, pricey, short, tight)}
    planted = set(trains)
    add_noise(rng, trains, numbers, stations, count=42,
              avoid={(origin, destination), (origin, hub1), (hub1, hub2), (hub2, destination),
                     (hub1, destination), (origin, hub2)})
    request = make_request(group, origin, destination, deadline, travel_class, budget)
    while (shortcut := next((t for t in valid_trips({"trains": trains}, request) if len(t) < 3), None)):
        noise = [number for number in shortcut if number not in planted]
        assert noise, f"planted trains alone make the shortcut {shortcut}"
        del trains[noise[0]]

    return build("network_hard", "A big timetable for a group of four", "hard", group, origin, destination,
                 deadline, travel_class, budget, stations, trains,
                 reference=[search(origin), search(hub1), search(hub2, destination),
                            book(leg1, travel_class, group), book(leg2, travel_class, group),
                            book(leg3, travel_class, group)],
                 traps={"direct but late": [book(late, travel_class, group)],
                        "fast direct, over budget": [book(pricey, travel_class, group)],
                        "three of four on the 3-seat train": [book(leg1, travel_class, group),
                                                             book(short, travel_class, group[:3])],
                        "change too tight at the second hub": [book(tight, travel_class, group),
                                                               book(leg3, travel_class, group)],
                        "first two trains only": [book(leg1, travel_class, group), book(leg2, travel_class, group)]})


TASKS = {
    "direct_easy": direct_task,
    "soldout_easy": soldout_task,
    "connection_medium": connection_task,
    "budget_medium": budget_task,
    "rebook_hard": rebook_task,
    "fix_connection_hard": fix_connection_task,
    "long_trip_hard": long_trip_task,
    "facilities_medium": facilities_task,
    "cheapest_medium": cheapest_task,
    "value_hard": value_task,
    "disruption_hard": disruption_task,
    "join_group_hard": join_group_task,
    "network_hard": network_task,
}
DEMO_TASK = "direct_easy"


def build_task(task_id, seed):
    if task_id not in TASKS:
        raise KeyError(f"unknown task '{task_id}'. Available: {', '.join(TASKS)}")
    return TASKS[task_id](seed)
