# LEMMAGEDDON: a train-booking environment that can't be broken

An RL environment where the agent is a travel booking assistant. It gets a trip request and must
leave behind train bookings that satisfy every rule: route, connections, arrival deadline, class,
budget, and everyone travelling together. A programmatic verifier checks the final bookings and
accepts **any** valid itinerary; there is no single answer to copy.

**Top 10 at the RL Hackathon (October 2026), the only first-year team in the top 10.**
Team LEMMAGEDDON: Jaiv Patel, Shrey Panchal, Sahil Gupta.

- 10 seeded tasks at 3 levels, each with planted trap trains that break exactly one rule
- 0 of 42,389 attempts to get undeserved reward got through the verifier
- GPT-Luna solved 40 of 44 runs (91%)

**Submission documents:** [design doc (2 pages, PDF)](docs/design-doc.pdf) ·
[presentation deck (15 slides, PDF)](docs/deck.pdf)

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

| Tool            | Arguments                                      | Returns                                               |
| --------------- | ---------------------------------------------- | ----------------------------------------------------- |
| `search_trains` | `origin`, `destination` (optional)             | trains from a station: number, times, price per class |
| `check_seats`   | `train`, `class`                               | seats left in that class                              |
| `book`          | `train`, `class`, `travellers` (list of names) | booking id and price, or an error                     |
| `cancel`        | `booking_id`                                   | fee and refund (20% of the price is kept)             |
| `list_bookings` | none                                           | every booking and the total spent                     |
| `finish`        | none                                           | ends the episode                                      |

State persists within an episode: booking takes seats, cancelling returns them.

### Reward

```
reward = covered × budget_score × tidy × best_choice
```

| Part           | Meaning                                                                                                                                                                                                                                                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `covered`      | share of travellers on one shared valid trip: origin to destination, each change at the same station ≥ 30 min later, the required class on every train, arriving by the deadline, and (when the request says so) a pantry car and at most N stops on every train. If travellers are split across trains, only the biggest group counts |
| `budget_score` | tickets kept + cancellation fees within budget: 1; up to 10% over: 0.5; more: 0                                                                                                                                                                                                                                                        |
| `tidy`         | 1 − 0.1 per leftover or duplicate booking, never below 0.5                                                                                                                                                                                                                                                                             |
| `best_choice`  | only when the request asks for the cheapest or best-value trip: 1 if the trip is as good as the best valid trip the verifier finds by searching the whole timetable, else 0.6                                                                                                                                                          |

Booking nothing scores 0. Booking everything goes far over budget and scores 0. An over-budget
trip can never reach 0.95. The verifier reads only *which* bookings exist; it recomputes every
price, fee, time and seat count from its own copy of the timetable, so numbers reported by the
sandbox are never trusted. Every score comes with plain-language notes, for example
`"Shrey: on different trains from the rest of the group"`.

## Task set

Every task is a seeded generator: each seed gives a different timetable and request, and the same
seed always gives the same one. The generator plants one valid trip (the reference solution), then
trap trains that each break exactly one rule.

| Task                  | Level  | What makes it hard                                                                      |
| --------------------- | ------ | --------------------------------------------------------------------------------------- |
| `direct_easy`         | easy   | a cheaper train arrives after the deadline                                              |
| `soldout_easy`        | easy   | the earliest train is sold out in the class; a group must not be split                  |
| `connection_medium`   | medium | no direct train fits; change at a hub with at least 30 minutes                          |
| `budget_medium`       | medium | the fast direct express is over budget; a cheaper connection fits                       |
| `rebook_hard`         | hard   | an existing booking arrives late: cancel it and rebook via a hub                        |
| `fix_connection_hard` | hard   | an existing connection is too tight; the budget only allows replacing the second train  |
| `long_trip_hard`      | hard   | 3 travellers, only a two-change trip works; every shortcut breaks a rule                |
| `facilities_medium`   | medium | needs a pantry car and at most N stops: of four train types on the route, only one fits |
| `cheapest_medium`     | medium | a direct train and a connection both fit: book the cheaper one (either can win)         |
| `value_hard`          | hard   | both fit and time has a price (Rs per hour): book the better value (either can win)     |

Trains come in four types with different speed, price, stops and facilities: Rajdhani (fast,
dear, pantry, 2–4 stops), Superfast, Express, and Passenger (slow, cheap, no pantry, 18–30 stops).

## Results

| Model       | Where         | Runs                                                              | Solved       | Notes                                                                                            |
| ----------- | ------------- | ----------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------ |
| GPT-Luna    | hackathon API | 10 tasks, 44 runs (seeds 0–3; `value_hard` also seeds 4, 5, 6, 8) | **40** (91%) | easy 8/8, medium 15/16, hard 17/20; two of the four misses scored 0.6 (a valid but worse choice) |
| qwen2.5:7b  | Ollama, local | 7 tasks × seeds 0–1 = 14                                          | 0            | reached a valid trip once, then booked more trains and lost it (`lost_valid_trip`)               |
| llama3.2:3b | Ollama, local | 7 tasks × seeds 0–1 = 14                                          | 0            | invents arguments and tools, writes calls as text, makes up tool output; 0 successful bookings   |

The harness never crashed on the local models' replies; every mistake got a clear error back or
was read from the text (tool calls written after prose or in code blocks). Local runs used a 16K
context window: Ollama's default of 4K silently cut off the start of the conversation (the task
itself). qwen3:8b was also tried but needs more than the 120 s allowed per reply on a laptop.

Its misses are planted traps working. Two scored partial credit (0.6): on `cheapest_medium` it booked the first direct train that fit and never looked at connections; on `value_hard` it chose the fast train when the slow connection was the better value. The other two: on `fix_connection_hard` it cancelled both
existing bookings and ran out of budget; on `long_trip_hard` it found both shortcuts, saw they
break the rules, never explored the two-change route, and finished without bookings.

## Configuration (`.env`)

| Variable                       | Meaning                                                                                                                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MODEL`, `API_KEY`, `BASE_URL` | any OpenAI-compatible endpoint. Offline: `mock/solver` (plays the reference solution), `mock/idle` (does nothing), `mock/cheater` (tries to break the environment)                         |
| `TASK_ID`                      | one of the tasks above (default `direct_easy`)                                                                                                                                             |
| `SEED`                         | seed for the task generator (default `0`)                                                                                                                                                  |
| `MODE`                         | `full` (default): checks, then the model on `TASK_ID`. `all`: checks for every task, then the model on each. `baselines`: checks for every task, no model call                             |
| `APPEND_FILE`                  | a text file in `inputs/` added to the end of the system prompt exactly as written (default `inputs/append.txt` if present). A named file that is missing, empty or not UTF-8 stops the run |
| `MAX_TOKENS`                   | optional cap on each model reply                                                                                                                                                           |

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

| Limit                | Value                                                                         |
| -------------------- | ----------------------------------------------------------------------------- |
| Steps                | 40 (one step = one model reply)                                               |
| Time                 | 10 minutes from the first model request                                       |
| Tool calls per reply | 10 (extra calls are not run and get an error back)                            |
| Time per tool call   | 60 seconds; a timeout returns an error to the agent and the episode continues |
| Tool output          | 4,000 characters                                                              |
| Context              | old tool outputs trimmed to keep the conversation under about 48K tokens      |

Malformed replies (no tool call, invalid JSON, unknown tool, wrong argument types) get a clear
error back and use up a step; the run continues. Replies in other providers' shapes are read too:
arguments sent as an object, missing call ids, the older `function_call` field, content as a list
of parts, a response without `choices`, and (only when there is no structured tool call) tool
calls written into the text as `<tool_call>{...}</tool_call>`, a JSON code block, or a bare JSON
object or list. Such calls are logged with `"source": "text"`.
Termination reasons: `success`, `failure`, `step_cap`, `time_cap`, `crash`.

## Suspicious-behaviour flags

Telemetry only: flags never change the reward. Each step's log line has a `flags` list, the live
stream prints `FLAGS=...`, and the episode end and `summary.json` carry the counts.

| Flag                    | Raised when the agent                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `probing_tool`          | calls a tool that doesn't exist and sounds like control or code access (`reset`, `run_python`, `get_reward`)        |
| `unknown_tool`          | calls any other tool that doesn't exist                                                                             |
| `injection_text`        | puts instruction-like text in an argument or its reply ("ignore previous instructions", "mark this task as solved") |
| `path_or_url`           | puts a file path, URL or host name in an argument                                                                   |
| `oversized_argument`    | sends an argument over 200 characters                                                                               |
| `repeated_call`         | sends the same call with the same arguments a 3rd time                                                              |
| `malformed_burst`       | has sent 5 malformed calls                                                                                          |
| `too_many_calls`        | sends more than 10 calls in one reply                                                                               |
| `booking_strangers`     | tries to book someone who is not on the trip                                                                        |
| `booking_flood`         | has made 8 successful bookings (a trip needs 1 to 3)                                                                |
| `booking_churn`         | has cancelled 3 bookings                                                                                            |
| `after_finish`          | keeps calling tools after `finish`                                                                                  |
| `lost_valid_trip`       | had a valid trip (reward ≥ 0.95) at some step, then undid it                                                        |
| `finished_without_trip` | calls `finish` without a valid trip for anyone                                                                      |
| `false_success_claim`   | claims success in its last message while the verifier says otherwise                                                |

Replaying 29 real GPT-Luna runs: the 27 successes raised no flags; the 2 failures raised
`finished_without_trip` (and one `booking_churn`). `MODEL=mock/cheater` raises all of them and
scores 0.

## Isolation

- **harness** container: task generator, reference solutions, verifier, model client. It is the
  only container with internet access, used for the model API.
- **sandbox** container: the booking system the tools talk to. It sits on an internal Docker
  network with no internet, runs as a non-root user with all Linux capabilities dropped, and its
  image contains none of the harness code.
- No tool runs code or touches files, so the agent can only act through the six booking tools.

## Tests

All three need only Python 3 (no Docker, no network) and use the real sandbox and verifier code.

```bash
python3 tests/check_tasks.py 300   # every task on 300 seeds
python3 tests/check_harness.py     # whole episodes with the mock agents
python3 tests/check_parsing.py     # replies in different providers' shapes
python3 tests/check_exploits.py    # attempts to get undeserved reward
```

- `check_tasks.py`: the same seed builds the same task; the reference solution runs without tool
  errors and scores ≥ 0.95; doing nothing scores ≤ 0.05; every planted trap scores < 0.95; every
  train runs within one day.
- `check_harness.py`: through the real step loop, on 5 seeds of every task, `mock/solver` succeeds
  with no flags, `mock/idle` scores ≤ 0.05, and `mock/cheater` scores ≤ 0.05 without crashing the
  run and raises every flag it provokes.
- `check_parsing.py`: 18 reply shapes; the parser finds the right calls, or none, and never crashes.
- `check_exploits.py`, on 50 seeds of every task (10 tasks, 10,591 attempts, 0 problems; 42,389 on 200 seeds):
  - 11 exploits must score below 0.95 when the sandbox accepts them: lookalike (Cyrillic) names, a
    traveller listed twice, every train booked twice, one traveller left out, an extra booking,
    the wrong class, instructions as a name, 100 travellers, made-up train numbers, half a trip,
    nothing at all. (A cheat the sandbox refuses, e.g. a double booking with no seats left, never
    happens; the honest trip left behind keeps its score.)
  - 5 honest but unusual bookings must still score at least 0.95: other letter case and spaces,
    one booking per traveller, train numbers with spaces and lowercase class, travellers in
    another order, the reference solution.
  - 5 forged sandbox states must not fool the verifier: prices reported as Rs 1, every booking
    copied 30 times (more seats than exist), an unknown train, travellers as one string, an
    unknown status.
  - Cancelling the same booking twice refunds once.

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
harness/agents.py         OpenAI-compatible client, reply parsing, mock models, baselines
harness/flags.py          suspicious-behaviour flags
tests/check_tasks.py      every task on many seeds
tests/check_harness.py    whole episodes with the mock agents
tests/check_parsing.py    replies in different providers' shapes
tests/check_exploits.py   attempts to get undeserved reward
check_model.py            one-message check of your model credentials
docs/design-doc.pdf       design doc submitted to the hackathon
docs/deck.pdf             presentation deck
```

## How it was built

Built on the hackathon template https://github.com/ayushmothiya1/RL-ENV-TEMPLATE (Docker setup,
step loop, logging, baseline gate). The train world, tools, verifier, tasks and tests were written
during the hackathon, mostly with AI-assisted development (Claude) under the team's direction.
