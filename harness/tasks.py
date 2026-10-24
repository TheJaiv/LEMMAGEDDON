"""The tasks: seeded train-booking trips.

Every task is built from a seed, so the same (task_id, seed) always gives the same
timetable, trip request and reference solution. The generator first plants one trip
that meets every rule (this becomes the reference solution), then adds distractor
trains that each break one rule. The verifier checks the rules, not the planted trip,
so any valid set of bookings scores.
"""
import math
import random
from dataclasses import dataclass

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
    world: dict      # timetable, travellers and fees; a copy is sent to the sandbox
    request: dict    # the trip rules; the verifier checks the bookings against these
    reference: list  # (tool, args) actions that solve the task; used by the oracle


def hhmm(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


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


def describe(request, cancel_fee):
    """The trip request in plain words: the first observation the agent gets."""
    return (f"Book train tickets for this trip.\n"
            f"Travellers: {', '.join(request['travellers'])} (all travel together on the same trains).\n"
            f"From: {request['origin']}   To: {request['destination']}\n"
            f"Arrive by: {hhmm(request['deadline'])} at the latest. All trains run on the same day.\n"
            f"Class: {request['travel_class']} on every train.\n"
            f"Budget: Rs {request['budget']} in total for all travellers, including any cancellation fees "
            f"(cancelling a booking costs {round(cancel_fee * 100)}% of its price).\n"
            f"If you change trains, the next train must leave from the same station at least "
            f"{request['min_connection']} minutes after the previous one arrives.\n"
            f"Every traveller needs a seat on every train of the trip. Leave no extra bookings.\n"
            f"Current bookings: none.\n"
            f"Use the tools to search trains, check seats and book. When the bookings are done, call `finish`.")


# ---------------------------------------------------------- direct_easy
def direct_task(seed):
    rng = random.Random(f"direct_easy:{seed}")
    stations = rng.sample(CITIES, 6)
    origin, destination = stations[0], stations[1]
    travel_class = rng.choice(list(CLASSES))
    traveller = rng.choice(NAMES)
    numbers = iter(str(n) for n in rng.sample(range(11001, 22999), 20))

    # the planted trip: one direct train that meets every rule
    good = make_train(rng, next(numbers), origin, destination, rng.randrange(360, 720, 5),
                      rng.randrange(150, 360, 5), list(CLASSES), seats={travel_class: rng.randint(2, 9)})
    deadline = 30 * math.ceil((good["arrive"] + 20) / 30)
    price = good["classes"][travel_class]["price"]
    budget = 50 * math.ceil(price * rng.uniform(1.15, 1.4) / 50)

    # distractors on the same route, each breaking exactly one rule
    duration = max(120, good["arrive"] - good["depart"] - rng.randrange(0, 60, 5))
    late = make_train(rng, next(numbers), origin, destination,                # arrives too late
                      deadline - duration + rng.randrange(30, 120, 5), duration, list(CLASSES))
    depart = max(300, good["depart"] - rng.randrange(30, 180, 5))
    wrong = make_train(rng, next(numbers), origin, destination, depart,       # lacks the required class
                       min(rng.randrange(150, 360, 5), deadline - 10 - depart),
                       [c for c in CLASSES if c != travel_class])
    trains = {t["number"]: t for t in (good, late, wrong)}
    add_noise(rng, trains, numbers, stations, avoid={(origin, destination)}, count=8)

    request = {"travellers": [traveller], "origin": origin, "destination": destination,
               "deadline": deadline, "min_connection": MIN_CONNECTION,
               "travel_class": travel_class, "budget": budget}
    world = {"stations": sorted(stations), "trains": trains, "travellers": [traveller],
             "cancel_fee": CANCEL_FEE, "bookings": []}
    return TaskInstance(
        task_id="direct_easy", title="Book a direct train", difficulty="easy",
        instructions=describe(request, CANCEL_FEE), world=world, request=request,
        reference=[("search_trains", {"origin": origin, "destination": destination}),
                   ("book", {"train": good["number"], "class": travel_class, "travellers": [traveller]})],
    )


TASKS = {
    "direct_easy": direct_task,
}
DEMO_TASK = "direct_easy"


def build_task(task_id, seed):
    if task_id not in TASKS:
        raise KeyError(f"unknown task '{task_id}'. Available: {', '.join(TASKS)}")
    return TASKS[task_id](seed)
