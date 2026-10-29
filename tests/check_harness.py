"""Runs whole episodes through the real step loop with the offline mock models, without Docker.

    python3 tests/check_harness.py        # seeds 0-4 of every task

Starts the sandbox server in this process, then for every task and seed:
- mock/solver must succeed and raise no suspicious-behaviour flags,
- mock/idle must fail with reward <= 0.05 and the flag finished_without_trip,
- mock/cheater must score <= 0.05, never crash the run, and raise every flag it provokes.
Then an empty model reply: the history sent back holds a placeholder, never an empty assistant message.
Then two failures of the environment itself:
- a tool call that times out returns an error to the agent and the episode continues (solver still succeeds),
- a sandbox that stops answering ends the run as a crash, with the reason written to the log.
"""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import requests                                   # noqa: E402
from harness.agents import MockClient, Reply     # noqa: E402
from harness.run import run_model                 # noqa: E402
from harness.env import Env                       # noqa: E402
from harness.runner import JsonlLog, Limits, run_episode   # noqa: E402
from harness.tasks import TASKS                   # noqa: E402

spec = importlib.util.spec_from_file_location("sandbox_server", ROOT / "sandbox" / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

CHEATER_FLAGS = {"probing_tool", "injection_text", "path_or_url", "oversized_argument", "repeated_call",
                 "malformed_burst", "too_many_calls", "booking_strangers", "booking_churn", "after_finish",
                 "false_success_claim"}


class SlowFirstTool(Env):
    """The first tool call of each episode times out, as a hung tool would."""
    def reset(self, task_id, seed):
        self.timed_out = False
        return super().reset(task_id, seed)

    def step(self, name, args):
        if not self.timed_out:
            self.timed_out = True
            raise requests.Timeout("simulated: the tool hung")
        return super().step(name, args)


class SandboxDies(Env):
    """The sandbox stops answering right after the reset."""
    def state_hash(self):
        raise requests.ConnectionError("simulated: sandbox down")


class EmptyThenSolve(MockClient):
    """The first reply is empty (no text, no tool call); then it plays the solver. Keeps what it was sent."""
    def __init__(self):
        super().__init__("solver")
        self.sent = []

    def complete(self, messages, tools, timeout):
        self.sent.append(messages)
        if len(self.sent) == 1:
            return Reply(text="", tool_calls=[], raw="")
        return super().complete(messages, tools, timeout)


def episode(env, task_id, seed, kind, log_dir):
    log = JsonlLog(Path(log_dir) / f"{kind}-{task_id}-{seed}.jsonl")
    with contextlib.redirect_stdout(io.StringIO()):          # keep the live stream quiet
        result = run_episode(env, task_id, seed, MockClient(kind), "", Limits(), log)
    log.close()
    return result


def main():
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    env = Env(f"http://127.0.0.1:{httpd.server_address[1]}")
    problems = []
    with tempfile.TemporaryDirectory() as log_dir:
        for task_id in TASKS:
            for seed in range(5):
                where = f"{task_id} seed {seed}"
                r = episode(env, task_id, seed, "solver", log_dir)
                if r["termination"] != "success" or r["flags"]:
                    problems.append(f"{where}: solver {r['termination']} {r['reward']['total']} flags {r['flags']}")
                r = episode(env, task_id, seed, "idle", log_dir)
                if r["reward"]["total"] > 0.05 or set(r["flags"]) != {"finished_without_trip"}:
                    problems.append(f"{where}: idle {r['reward']['total']} flags {r['flags']}")
                r = episode(env, task_id, seed, "cheater", log_dir)
                missing = CHEATER_FLAGS - set(r["flags"])
                if r["reward"]["total"] > 0.05 or r["termination"] == "crash" or missing:
                    problems.append(f"{where}: cheater {r['termination']} {r['reward']['total']} missing {missing}")
            print(f"{task_id:<20} solver / idle / cheater on 5 seeds  "
                  f"{'ok' if not any(p.startswith(task_id + ' ') for p in problems) else 'PROBLEMS'}")
        for task_id in TASKS:
            path = Path(log_dir) / f"slow-{task_id}.jsonl"
            log = JsonlLog(path)
            with contextlib.redirect_stdout(io.StringIO()):
                r = run_episode(SlowFirstTool(env.url), task_id, 0, MockClient("solver"), "", Limits(), log)
            log.close()
            kinds = [e["kind"] for line in open(path) for e in json.loads(line).get("errors", [])]
            if r["termination"] != "success" or "tool_timeout" not in kinds:
                problems.append(f"{task_id}: tool timeout: {r['termination']}, errors {kinds}")
        client = EmptyThenSolve()
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_episode(env, "direct_easy", 0, client, "", Limits(), JsonlLog(Path(log_dir) / "empty.jsonl"))
        empty = [m for m in client.sent[1] if m["role"] == "assistant" and not (m.get("content") or "").strip()
                 and not m.get("tool_calls")]
        if r["termination"] != "success" or empty:
            problems.append(f"empty reply: {r['termination']}, empty assistant messages sent {empty}")
        print(f"empty reply replaced by a placeholder in the history  "
              f"{'ok' if not any(p.startswith('empty reply') for p in problems) else 'PROBLEMS'}")
        path = Path(log_dir) / "dies.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_model(SandboxDies(env.url), "direct_easy", 0, MockClient("solver"), "mock/solver", "", Limits(), path)
        last = json.loads(open(path).read().splitlines()[-1])
        if r["termination"] != "crash" or last.get("termination") != "crash" or "sandbox down" not in last.get("error", ""):
            problems.append(f"sandbox dies: {r['termination']}, last log line {last}")
        print(f"tool timeout continues the episode, sandbox death recorded as a crash  "
              f"{'ok' if not any(p.startswith(('sandbox dies', *[t + ': tool timeout' for t in TASKS])) for p in problems) else 'PROBLEMS'}")
    httpd.shutdown()
    for p in problems[:10]:
        print("    " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
