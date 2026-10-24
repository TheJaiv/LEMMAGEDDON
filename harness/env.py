"""The environment: reset / step / score, backed by the sandbox service."""
import hashlib
import json

import requests

from .tasks import build_task
from .verifier import score_bookings

PASS_SCORE = 0.95   # a run counts as a success at or above this score


class Env:
    def __init__(self, sandbox_url, tool_timeout=60):
        self.url = sandbox_url.rstrip("/")
        self.tool_timeout = tool_timeout
        self.task = None

    def _post(self, path, payload, timeout):
        response = requests.post(self.url + path, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    # ---------------------------------------------------------------- reset
    def reset(self, task_id, seed):
        """Start a clean episode. The same (task_id, seed) always gives the same start."""
        self.task = build_task(task_id, seed)
        self._post("/reset", {"world": self.task.world}, self.tool_timeout)
        return self.task.instructions

    # ----------------------------------------------------------------- step
    def step(self, name, args):
        """Run one tool call. Returns (text, is_error)."""
        result = self._post("/tool", {"name": name, "args": args}, self.tool_timeout)
        return result["output"], result["is_error"]

    def state(self):
        """The bookings and seat counts as the sandbox holds them right now."""
        response = requests.get(self.url + "/state", timeout=self.tool_timeout)
        response.raise_for_status()
        return response.json()

    def state_hash(self):
        return hashlib.sha256(json.dumps(self.state(), sort_keys=True).encode()).hexdigest()[:8]

    # ---------------------------------------------------------------- score
    def score(self):
        """Programmatic verifier (see verifier.py). Returns the reward components and their total."""
        return score_bookings(self.task, self.state())
