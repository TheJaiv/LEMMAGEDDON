"""Tool definitions shown to the model, plus argument checking."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_trains",
        "description": ("List trains leaving a station, with departure and arrival times and the price "
                        "per traveller in each class. Give a destination to see only trains to that station."),
        "parameters": {"type": "object", "required": ["origin"], "properties": {
            "origin": {"type": "string", "description": "Station the trains leave from, e.g. 'Pune'."},
            "destination": {"type": "string", "description": "Optional. Station the trains go to."}}}}},
    {"type": "function", "function": {
        "name": "check_seats", "description": "Show how many seats are left in one class of one train.",
        "parameters": {"type": "object", "required": ["train", "class"], "properties": {
            "train": {"type": "string", "description": "Train number, e.g. '12041'."},
            "class": {"type": "string", "description": "Travel class: 'SL', '3A' or '2A'."}}}}},
    {"type": "function", "function": {
        "name": "book", "description": "Book seats on one train in one class for one or more travellers.",
        "parameters": {"type": "object", "required": ["train", "class", "travellers"], "properties": {
            "train": {"type": "string", "description": "Train number, e.g. '12041'."},
            "class": {"type": "string", "description": "Travel class: 'SL', '3A' or '2A'."},
            "travellers": {"type": "array", "items": {"type": "string"},
                           "description": "Names of the travellers who need a seat, e.g. ['Asha', 'Ravi']."}}}}},
    {"type": "function", "function": {
        "name": "cancel", "description": "Cancel a booking. Part of the price is kept as a cancellation fee.",
        "parameters": {"type": "object", "required": ["booking_id"], "properties": {
            "booking_id": {"type": "string", "description": "Booking id, e.g. 'B1'."}}}}},
    {"type": "function", "function": {
        "name": "list_bookings", "description": "List every booking made so far and the total spent.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "finish", "description": "Call this when the bookings are complete.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

SPECS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
TOOL_NAMES = list(SPECS)


def check_args(name, args):
    """Return an error message if `args` don't match the tool's schema, else None."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    spec = SPECS[name]
    for key in spec.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    for key, value in args.items():
        if key not in spec["properties"]:
            return f"unexpected argument '{key}'"
        kind = spec["properties"][key]["type"]
        if kind == "string" and not isinstance(value, str):
            return f"argument '{key}' must be a string"
        if kind == "array" and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            return f"argument '{key}' must be a list of strings"
    return None
