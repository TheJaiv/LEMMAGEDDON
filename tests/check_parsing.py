"""Feeds differently shaped model replies through the harness's reply parser, without a network.

    python3 tests/check_parsing.py

Each case is a chat-completions response shaped the way some provider or model sends it.
The parser must find the right tool calls, or none, and never crash.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.agents import reply_from_response   # noqa: E402

SEARCH = {"origin": "Pune", "destination": "Agra"}


def response(message, usage=True):
    return {"choices": [{"message": message}], **({"usage": {"total_tokens": 42}} if usage else {})}


def native(arguments, call_id="call_a", name="search_trains"):
    return response({"content": None, "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": arguments}}]})


CASES = [
    ("standard: arguments as a JSON string", native(json.dumps(SEARCH)), [("search_trains", SEARCH, "native")]),
    ("arguments as an object, not a string", native(SEARCH), [("search_trains", SEARCH, "native")]),
    ("no call id, spaces around the name", native(json.dumps(SEARCH), call_id=None, name=" search_trains "),
     [("search_trains", SEARCH, "native")]),
    ("older single function_call field", response({"content": None, "function_call": {"name": "list_bookings", "arguments": "{}"}}),
     [("list_bookings", {}, "native")]),
    ("objects instead of dicts (like the openai library)",
     NS(choices=[NS(message=NS(content=None, tool_calls=[NS(id="x", function=NS(name="finish", arguments="{}"))]))],
        usage=NS(total_tokens=7)), [("finish", {}, "native")]),
    ("<tool_call> tags in the text (Hermes / Qwen style)",
     response({"content": 'Let me search.\n<tool_call>{"name": "search_trains", "arguments": {"origin": "Pune", "destination": "Agra"}}</tool_call>'}),
     [("search_trains", SEARCH, "text")]),
    ("a ```json block in the text",
     response({"content": 'I will book now:\n```json\n{"name": "book", "arguments": {"train": "15834", "class": "2A", "travellers": ["Priya"]}}\n```'}),
     [("book", {"train": "15834", "class": "2A", "travellers": ["Priya"]}, "text")]),
    ("a bare JSON list with tool/args keys",
     response({"content": '[{"tool": "check_seats", "args": {"train": "15834", "class": "2A"}}, {"tool": "finish", "args": {}}]'}),
     [("check_seats", {"train": "15834", "class": "2A"}, "text"), ("finish", {}, "text")]),
    ("nested {\"function\": {...}} in the text",
     response({"content": '{"function": {"name": "cancel", "arguments": "{\\"booking_id\\": \\"B1\\"}"}}'}),
     [("cancel", {"booking_id": "B1"}, "text")]),
    ("content as a list of parts", response({"content": [{"type": "text", "text": "<tool_call>{\"name\": \"finish\", \"arguments\": {}}</tool_call>"}]}),
     [("finish", {}, "text")]),
    ("plain prose, no tool call", response({"content": "I think train 15834 is the best option."}), []),
    ("broken JSON inside <tool_call>", response({"content": "<tool_call>{name: book, arguments: }</tool_call>"}), []),
    ("no choices at all (provider error body)", {"error": {"message": "overloaded"}}, []),
    ("message is null", {"choices": [{"message": None}]}, []),
]


def main():
    problems = 0
    for label, resp, expected in CASES:
        try:
            reply = reply_from_response(resp)
            got = [(c.name, c.parse()[0], c.source) for c in reply.tool_calls]
            ok = got == expected
        except Exception as exc:     # the parser must never crash
            got, ok = f"CRASH {type(exc).__name__}: {exc}", False
        problems += not ok
        print(f"{'ok ' if ok else 'BAD'}  {label}" + ("" if ok else f"\n     expected {expected}\n     got      {got}"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
