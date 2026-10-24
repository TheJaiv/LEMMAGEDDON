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

CITIES = ["Mumbai", "Pune", "Nashik", "Surat", "Vadodara", "Ahmedabad", "Nagpur", "Indore",
          "Bhopal", "Jaipur", "Delhi", "Agra", "Lucknow", "Hyderabad", "Bengaluru", "Chennai"]
NAMES = ["Jaiv", "Shrey", "Sahil", "Kabir", "Priya", "Arjun", "Tanisha", "Vikram", "Isha", "Rohan"]
CLASSES = {"SL": 1.6, "3A": 4.0, "2A": 5.8}   # class -> rupees per minute of travel, roughly
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


def train_numbers(rng):
    return iter(str(n) for n in rng.sample(range(11001, 22999), 20))


def make_train(rng, number, origin, destination, depart, duration, classes, seats=None):
    """One train: a single leg from origin to destination. `seats` fixes chosen seat counts."""
    fares = {}
    for name in CLASSES:                          # keep a fixed class order: SL, 3A, 2A
        if name in classes:
            price = 5 * round(duration * CLASSES[name] * rng.uniform(0.9, 1.1) / 5)
            fares[name] = {"price": price, "seats": rng.randint(4, 30)}
    for name, count in (seats or {}).items():
        fares[name]["seats"] = count
    return {"number": number, "origin": origin, "destination": destination,
            "depart": depart, "arrive": depart + duration, "classes": fares}


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
            f"Every traveller needs a seat on every train of the trip. Leave no extra bookings.\n"
            f"{current}\n"
            f"Use the tools to search trains, check seats, book and cancel. When the bookings are done, call `finish`.")


def build(task_id, title, difficulty, group, origin, destination, deadline, travel_class, budget,
          stations, trains, reference, traps, bookings=()):
    request = {"travellers": list(group), "origin": origin, "destination": destination,
               "deadline": deadline, "min_connection": MIN_CONNECTION,
               "travel_class": travel_class, "budget": budget}
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


TASKS = {
    "direct_easy": direct_task,
    "soldout_easy": soldout_task,
    "connection_medium": connection_task,
    "budget_medium": budget_task,
    "rebook_hard": rebook_task,
}
DEMO_TASK = "direct_easy"


def build_task(task_id, seed):
    if task_id not in TASKS:
        raise KeyError(f"unknown task '{task_id}'. Available: {', '.join(TASKS)}")
    return TASKS[task_id](seed)
