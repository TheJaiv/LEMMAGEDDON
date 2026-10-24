"""Sandbox service: the railway booking system the agent's tools talk to.

This container sits on an internal Docker network with no internet access. It holds
the timetable, the seats and the bookings for one episode, and the tools that change
them. There are no code-running or file tools. It never sees the reference solution
or the reward code: those live in the harness container.
"""
import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8000"))
MAX_OUTPUT = 4000          # characters returned to the agent per tool call

STATE = {"world": None, "bookings": []}
LOCK = threading.Lock()    # one request at a time reads or changes the state


def hhmm(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n[output cut at {MAX_OUTPUT} characters]"


def world():
    if STATE["world"] is None:
        raise RuntimeError("no episode has been started")
    return STATE["world"]


def find_station(name):
    for station in world()["stations"]:
        if station.lower() == name.strip().lower():
            return station
    raise ValueError(f"unknown station '{name}'. Stations: {', '.join(world()['stations'])}")


def find_train(number):
    train = world()["trains"].get(number.strip())
    if train is None:
        raise ValueError(f"unknown train '{number}'. Use search_trains to find train numbers")
    return train


def find_class(train, name):
    key = name.strip().upper()
    if key not in train["classes"]:
        raise ValueError(f"train {train['number']} has no class '{name}'. "
                         f"Classes on this train: {', '.join(train['classes'])}")
    return key, train["classes"][key]


def find_traveller(name):
    for traveller in world()["travellers"]:
        if traveller.lower() == name.strip().lower():
            return traveller
    raise ValueError(f"'{name}' is not on this trip. Travellers: {', '.join(world()['travellers'])}")


def leg(train):
    return (f"{train['number']}  {train['origin']} {hhmm(train['depart'])} -> "
            f"{train['destination']} {hhmm(train['arrive'])}")


def spent():
    return sum(b["fee"] if b["status"] == "cancelled" else b["price"] for b in STATE["bookings"])


# ----------------------------------------------------------------------------
# tools the agent can call
# ----------------------------------------------------------------------------
def tool_search_trains(args):
    origin = find_station(args["origin"])
    destination = find_station(args["destination"]) if args.get("destination") else None
    found = sorted((t for t in world()["trains"].values()
                    if t["origin"] == origin and destination in (None, t["destination"])),
                   key=lambda t: (t["depart"], t["number"]))
    where = f"from {origin} to {destination}" if destination else f"from {origin}"
    if not found:
        return f"No trains {where}."
    lines = [leg(t) + "   " + " | ".join(f"{c} Rs {f['price']}" for c, f in t["classes"].items())
             for t in found]
    return f"Trains {where} (price is per traveller):\n" + "\n".join(lines)


def tool_check_seats(args):
    train = find_train(args["train"])
    name, fare = find_class(train, args["class"])
    return f"Train {train['number']} class {name}: {fare['seats']} seats left, Rs {fare['price']} per traveller."


def tool_book(args):
    train = find_train(args["train"])
    name, fare = find_class(train, args["class"])
    if not args["travellers"]:
        raise ValueError("travellers must list at least one name")
    names = [find_traveller(n) for n in args["travellers"]]
    if len(set(names)) != len(names):
        raise ValueError("a traveller is listed more than once")
    if fare["seats"] < len(names):
        raise ValueError(f"only {fare['seats']} seat{'' if fare['seats'] == 1 else 's'} left in class {name} "
                         f"on train {train['number']}")
    fare["seats"] -= len(names)
    booking = {"id": f"B{len(STATE['bookings']) + 1}", "train": train["number"], "class": name,
               "travellers": names, "status": "active", "price": fare["price"] * len(names), "fee": 0}
    STATE["bookings"].append(booking)
    return (f"Booked {booking['id']}: train {leg(train)}, class {name}, for {', '.join(names)}. "
            f"Price Rs {booking['price']}. Total spent so far Rs {spent()}.")


def tool_cancel(args):
    wanted = args["booking_id"].strip().upper()
    booking = next((b for b in STATE["bookings"] if b["id"] == wanted), None)
    if booking is None:
        raise ValueError(f"unknown booking '{args['booking_id']}'")
    if booking["status"] == "cancelled":
        raise ValueError(f"booking {booking['id']} is already cancelled")
    booking["status"] = "cancelled"
    booking["fee"] = round(booking["price"] * world()["cancel_fee"])
    world()["trains"][booking["train"]]["classes"][booking["class"]]["seats"] += len(booking["travellers"])
    return (f"Cancelled {booking['id']}. Fee Rs {booking['fee']}, refund Rs {booking['price'] - booking['fee']}. "
            f"Total spent so far Rs {spent()}.")


def tool_list_bookings(args):
    lines = []
    for b in STATE["bookings"]:
        cost = f"fee Rs {b['fee']}" if b["status"] == "cancelled" else f"Rs {b['price']}"
        lines.append(f"{b['id']}  {b['status']:<9}  train {leg(world()['trains'][b['train']])}  "
                     f"class {b['class']}  {', '.join(b['travellers'])}  {cost}")
    return ("\n".join(lines) if lines else "No bookings.") + f"\nTotal spent: Rs {spent()}"


TOOLS = {
    "search_trains": tool_search_trains,
    "check_seats": tool_check_seats,
    "book": tool_book,
    "cancel": tool_cancel,
    "list_bookings": tool_list_bookings,
}


# ----------------------------------------------------------------------------
# endpoints used by the harness
# ----------------------------------------------------------------------------
def do_reset(body):
    STATE["world"] = copy.deepcopy(body["world"])
    STATE["bookings"] = copy.deepcopy(body["world"].get("bookings", []))
    return {"ok": True}


def do_tool(body):
    name, args = body["name"], body.get("args", {})
    if name not in TOOLS:
        return {"output": f"error: unknown tool '{name}'", "is_error": True}
    try:
        return {"output": clip(TOOLS[name](args)), "is_error": False}
    except Exception as exc:  # tool-level problems go back to the agent, not the harness
        return {"output": f"error: {exc}", "is_error": True}


def do_state(_body=None):
    if STATE["world"] is None:
        return {"bookings": [], "seats": {}}
    seats = {number: {c: f["seats"] for c, f in t["classes"].items()}
             for number, t in world()["trains"].items()}
    return {"bookings": STATE["bookings"], "seats": seats}


ROUTES_POST = {"/reset": do_reset, "/tool": do_tool}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        elif self.path == "/state":
            with LOCK:
                self._send(200, do_state())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        handler = ROUTES_POST.get(self.path)
        if handler is None:
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            with LOCK:
                result = handler(body)
            self._send(200, result)
        except Exception as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *args):  # keep the container log quiet
        pass


if __name__ == "__main__":
    print(f"sandbox listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
