"""Checks every task on many seeds, without Docker or a model.

    python3 tests/check_tasks.py          # 200 seeds per task
    python3 tests/check_tasks.py 1000     # more seeds

For every task and seed: the same seed builds the same task, the reference solution runs
without tool errors and scores >= 0.95, doing nothing scores <= 0.05, every trap the task
lists scores < 0.95, and every train runs within one day. The reference solution scored by a
verifier that has run out of time scores 0. Actions go through the real
sandbox tool code (imported directly, no HTTP) and the real verifier.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.tasks import TASKS, build_task   # noqa: E402
from harness.verifier import score_bookings   # noqa: E402

spec = importlib.util.spec_from_file_location("sandbox_server", ROOT / "sandbox" / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def play(task, actions):
    """Run actions through the sandbox tools, return (reward, tool errors)."""
    server.do_reset({"world": json.loads(json.dumps(task.world))})   # a copy, as over HTTP
    errors = []
    for name, args in actions:
        result = server.do_tool({"name": name, "args": args})
        if result["is_error"]:
            errors.append(f"{name}: {result['output']}")
    return score_bookings(task, json.loads(json.dumps(server.do_state()))), errors


def check(task_id, seed):
    problems = []
    task = build_task(task_id, seed)
    if json.dumps(task.__dict__, sort_keys=True) != json.dumps(build_task(task_id, seed).__dict__, sort_keys=True):
        problems.append("the same seed built a different task")
    reward, errors = play(task, task.reference)
    if reward["total"] < 0.95 or errors:
        problems.append(f"reference scored {reward['total']} {errors or reward['notes']}")
    late = score_bookings(task, json.loads(json.dumps(server.do_state())), seconds=0)
    if late["total"] != 0 or "verifier_timeout" not in " ".join(late["notes"]):
        problems.append(f"the reference scored {late['total']} by a verifier out of time")
    reward, _ = play(task, [])
    if reward["total"] > 0.05:
        problems.append(f"no-op scored {reward['total']}")
    for name, actions in task.traps.items():
        reward, _ = play(task, actions)
        if reward["total"] >= 0.95:
            problems.append(f"trap '{name}' scored {reward['total']}")
    if not all(300 <= t["depart"] < t["arrive"] <= 23 * 60 + 55 for t in task.world["trains"].values()):
        problems.append("a train runs outside one day")
    return task, problems


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    failed = 0
    for task_id in TASKS:
        problems, texts, traps = [], set(), 0
        for seed in range(seeds):
            task, found = check(task_id, seed)
            problems += [f"seed {seed}: {p}" for p in found]
            texts.add(task.instructions)
            traps += len(task.traps)
        status = "ok" if not problems else f"{len(problems)} PROBLEMS"
        print(f"{task_id:<18} {task.difficulty:<6} {seeds} seeds  {len(texts)} distinct  "
              f"{traps} traps played  {status}")
        for p in problems[:5]:
            print("    " + p)
        failed += len(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
