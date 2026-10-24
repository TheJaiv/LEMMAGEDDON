# LEMMAGEDDON: a train-booking environment that can't be broken

An RL environment where the agent is a travel booking assistant. It gets a trip request and must
leave behind train bookings that satisfy every rule: route, connections, arrival deadline, class,
budget, and everyone travelling together. A programmatic verifier checks the final bookings and
accepts **any** valid itinerary; there is no single answer to copy.

**Demo task: `direct_easy`** (the default `TASK_ID`), seed 0.

## Quick start

You need Docker with Docker Compose v2. Nothing else.

```bash
cp .env.example .env        # set MODEL, API_KEY and BASE_URL (leave MODEL=mock/solver for an offline demo)
docker compose up --build --abort-on-container-exit --exit-code-from harness
```

One command builds both images, runs the reference solution and the no-op agent on the task
(both must pass before any model call), then runs the model. Logs go to `logs/<run-id>/`.

## The task

What the agent sees at the start (`rebook_hard`, seed 0):

```
Book train tickets for this trip.
Travellers: Vikram, Shrey (all travel together on the same trains).
From: Ahmedabad   To: Delhi
Arrive by: 13:30 at the latest. All trains run on the same day.
Class: 2A on every train.
Budget: Rs 4250 in total for all travellers. Everything counts: the tickets you keep (including existing bookings) and cancellation fees (20% of a booking's price).
If you change trains, the next train must leave from the same station at least 30 minutes after the previous one arrives.
Every traveller needs a seat on every train of the trip. Leave no extra bookings.
Current bookings (already paid):
  B1: train 16346 Ahmedabad 10:15 -> Delhi 14:20, class 2A, for Vikram, Shrey, Rs 2770
```

**Success** = reward ≥ 0.95, which requires every rule to hold.

### Tools

Domain tools only: no code execution, no file access.

| Tool | Arguments | Returns |
|---|---|---|
| `search_trains` | `origin`, `destination` (optional) | trains from a station: number, times, price per class |
| `check_seats` | `train`, `class` | seats left in that class |
| `book` | `train`, `class`, `travellers` (list of names) | booking id and price, or an error |
| `cancel` | `booking_id` | fee and refund (20% of the price is kept) |
| `list_bookings` | none | every booking and the total spent |
| `finish` | none | ends the episode |

State persists within an episode: booking takes seats, cancelling returns them.

### Reward

```
reward = covered × budget_score × tidy
```

| Part | Meaning |
|---|---|
| `covered` | share of travellers on one shared valid trip: origin to destination, each change at the same station ≥ 30 min later, the required class on every train, arriving by the deadline. If travellers are split across trains, only the biggest group counts |
| `budget_score` | tickets kept + cancellation fees within budget: 1; up to 10% over: 0.5; more: 0 |
| `tidy` | 1 − 0.1 per leftover or duplicate booking, never below 0.5 |

Booking nothing scores 0. Booking everything goes far over budget and scores 0. An over-budget
trip can never reach 0.95. The verifier reads only *which* bookings exist; it recomputes every
price, fee, time and seat count from its own copy of the timetable, so numbers reported by the
sandbox are never trusted. Every score comes with plain-language notes, for example
`"Shrey: on different trains from the rest of the group"`.

## Task set

Every task is a seeded generator: each seed gives a different timetable and request, and the same
seed always gives the same one. The generator plants one valid trip (the reference solution), then
trap trains that each break exactly one rule.

| Task | Level | What makes it hard |
|---|---|---|
| `direct_easy` | easy | a cheaper train arrives after the deadline |
| `soldout_easy` | easy | the earliest train is sold out in the class; a group must not be split |
| `connection_medium` | medium | no direct train fits; change at a hub with at least 30 minutes |
| `budget_medium` | medium | the fast direct express is over budget; a cheaper connection fits |
| `rebook_hard` | hard | an existing booking arrives late: cancel it and rebook via a hub |
| `fix_connection_hard` | hard | an existing connection is too tight; the budget only allows replacing the second train |
| `long_trip_hard` | hard | 3 travellers, only a two-change trip works; every shortcut breaks a rule |

## Configuration (`.env`)

| Variable | Meaning |
|---|---|
| `MODEL`, `API_KEY`, `BASE_URL` | any OpenAI-compatible endpoint. `MODEL=mock/solver` or `mock/idle` runs offline |
| `TASK_ID` | one of the tasks above (default `direct_easy`) |
| `SEED` | seed for the task generator (default `0`) |
| `MODE` | `full` (default): checks, then the model on `TASK_ID`. `all`: checks for every task, then the model on each. `baselines`: checks for every task, no model call |
| `APPEND_FILE` | a text file in `inputs/` added to the end of the system prompt exactly as written (default `inputs/append.txt` if present). A named file that is missing, empty or not UTF-8 stops the run |
| `MAX_TOKENS` | optional cap on each model reply |

Examples:
```bash
MODE=baselines docker compose up --build --abort-on-container-exit --exit-code-from harness
TASK_ID=long_trip_hard SEED=7 docker compose up --build --abort-on-container-exit --exit-code-from harness
```

## Outputs and exit codes

`logs/<run-id>/` (run ids are UTC timestamps):
- `gate.jsonl`, `gate.json`: reference solution and no-op results per task
- `episode.jsonl` (`episode-<task>.jsonl` with `MODE=all`): one line per model reply with the
  timestamp, step, seed, full model input, raw model output, parsed actions, tool results, errors,
  state hash, reward components and total, tokens and latency; a final line with the termination
  reason and the verifier's result
- `summary.json`

Exit codes: `0` finished, `1` crash, `2` a baseline check failed, `3` bad configuration.

## Limits in a model run

| Limit | Value |
|---|---|
| Steps | 40 (one step = one model reply) |
| Time | 10 minutes from the first model request |
| Tool calls per reply | 10 (extra calls are not run and get an error back) |
| Tool output | 4,000 characters |
| Context | old tool outputs trimmed to keep the conversation under about 48K tokens |

Malformed replies (no tool call, invalid JSON, unknown tool, wrong argument types) get a clear
error back and use up a step; the run continues. Termination reasons: `success`, `failure`,
`step_cap`, `time_cap`, `crash`.

## Isolation

- **harness** container: task generator, reference solutions, verifier, model client. It is the
  only container with internet access, used for the model API.
- **sandbox** container: the booking system the tools talk to. It sits on an internal Docker
  network with no internet, runs as a non-root user with all Linux capabilities dropped, and its
  image contains none of the harness code.
- No tool runs code or touches files, so the agent can only act through the six booking tools.

## Tests

```bash
python3 tests/check_tasks.py 300
```

For every task and 300 seeds: the same seed builds the same task; the reference solution runs
without tool errors and scores ≥ 0.95; doing nothing scores ≤ 0.05; every planted trap scores
< 0.95; every train runs within one day. It uses the real sandbox tool code and the real verifier
and needs only Python 3.

## Layout

```
docker-compose.yml        two services: sandbox (no internet) and harness
Dockerfile.harness        image for the harness
Dockerfile.sandbox        image for the sandbox
sandbox/server.py         the booking system and its tools
harness/run.py            entry point: baseline checks, then the model
harness/runner.py         step loop, limits, context trimming, logging
harness/env.py            reset / step / state / score
harness/tasks.py          seeded task generators, reference solutions and traps
harness/verifier.py       the rules and the reward
harness/tools.py          tool schemas shown to the model, argument checks
harness/agents.py         OpenAI-compatible client, offline mock models, baselines
tests/check_tasks.py      checks every task on many seeds
check_model.py            one-message check of your model credentials
```

Built on the hackathon template https://github.com/ayushmothiya1/RL-ENV-TEMPLATE (Docker setup,
step loop, logging, baseline gate). The train world, tools, verifier, tasks and tests were written
during the hackathon.
