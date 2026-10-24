"""Runs whole episodes through the real step loop with the offline mock models, without Docker.

    python3 tests/check_harness.py        # seeds 0-4 of every task

Starts the sandbox server in this process, then for every task and seed:
- mock/solver must succeed and raise no suspicious-behaviour flags,
- mock/idle must fail with reward <= 0.05 and the flag finished_without_trip,
- mock/cheater must score <= 0.05, never crash the run, and raise every flag it provokes.
"""
import contextlib
import importlib.util
import io
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.agents import MockClient            # noqa: E402
from harness.env import Env                       # noqa: E402
from harness.runner import JsonlLog, Limits, run_episode   # noqa: E402
from harness.tasks import TASKS                   # noqa: E402

spec = importlib.util.spec_from_file_location("sandbox_server", ROOT / "sandbox" / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

CHEATER_FLAGS = {"probing_tool", "injection_text", "path_or_url", "oversized_argument", "repeated_call",
                 "malformed_burst", "too_many_calls", "booking_strangers", "booking_churn", "after_finish",
                 "false_success_claim"}


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
    httpd.shutdown()
    for p in problems[:10]:
        print("    " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
