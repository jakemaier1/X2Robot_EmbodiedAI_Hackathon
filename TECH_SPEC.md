# Tech Spec: Orange-Item Harness (state machine over the WALLX relay)

Hackathon: X Square Robot BRIGHT 2026 Embodied Agent Hackathon. Task: move a fixed set of orange items into a basket.

This spec says **what** to build and **why**. Where it says "open", choose a simple default, note it in a code comment, and keep it easy to change. Do not invent requirements beyond this file.

## 1. Context

- A robot client (provided, runs on the robot) sends the harness a request every cycle: 3 JPEG camera images, both arm poses, and an `instruction` string.
- The harness forwards it to WALLX (a VLA model, WebSocket + MessagePack), and returns WALLX's reply to the client **unchanged**. The client executes the motion.
- The harness's only lever is the `instruction` string it forwards. It never sends motion commands.
- Only the WebUI can stop the robot. The harness cannot.
- Base code: the organizer's demo `harness.py` (reading it first is required). `harness_sm.py` is an earlier prototype with a different design (LLM-discovered targets, background checks). It may be mined for ideas but is superseded by this spec.

## 2. The task

A configured list of required items, in order:

1. carrot
2. carton
3. tangerine

For each item: **Grab → Verify → Drop → Verify**, then the next item. When the list is finished: **Done**.

```
 for each item in the queue:
   GRAB ──► VERIFY_GRAB ──pass──► DROP ──► VERIFY_DROP ──pass──► next item
     ▲            │fail                        │fail
     └────────────┴────────────────────────────┘   (retry, counts as an attempt)
 queue empty ──► DONE (keep sending a "go home" instruction; WebUI ends the session)
```

The item list, per-item wording and prompts must be **data/config**, not hard-coded in the machine logic.

## 3. Decisions already made

| Topic | Decision |
|---|---|
| Drop | One instruction to WALLX, e.g. "Put the carrot in the basket." No separate move/release steps. |
| Attempts | An item that fails verification **3 times** is skipped. |
| Skipped items | Move to a skipped list. Once the main queue is empty, if time remains, try each skipped item once more (open: exact policy), then Done. |
| Time | A global budget (default ~420 s; the hard case limit is 480 s). When exceeded, go to Done regardless of state. |
| Done | Keep forwarding a harmless instruction (e.g. "Return both arms to the home position."). Never rely on the harness to stop the robot. |

## 4. Verification behavior (decision with rationale)

**Default: blocking verification.** When the machine decides it is time to verify, the relay holds the current client request, asks the vision LLM, and only then forwards the next request to WALLX. While it waits, the client gets no new actions, so the arm effectively pauses.

Why: it matches the demo's behavior (the demo already awaits its agent call inside the request), it keeps the user's preference of not sending new instructions to WALLX mid-verify, and the LLM sees a settled scene. Cost is a pause of roughly 1–3 s per check, which is small against the 8-minute limit (about 6–10 checks per run).

Unknown, so do not assume: whether pausing mid-motion hurts WALLX's next chunk, and the client's waiting budget for a reply (see the participant guide: the demo uses a 10 s agent timeout and a 60 s WALLX timeout). Keep the verify call behind one function so a non-blocking variant could be swapped in later. Do not build the non-blocking variant now.

Cadence (open): how many forwarded requests to wait after an instruction changes before the first verify, and how often to re-verify while the state is unresolved. Make this a configurable number with a simple default.

## 5. Architecture / refactor guidance

Goal: the state machine must be testable **without network, asyncio or robot**.

Suggested split (structure is a suggestion; the boundaries are the requirement):

1. **Protocol helpers** (pack/unpack/validate_request/etc.): keep from the demo unchanged.
2. **Vision LLM client:** one generic call: images + a question in, a parsed JSON dict out. Chat-Completions-compatible endpoint with image inputs, configured by `AGENT_CHAT_URL`, `AGENT_MODEL`, `AGENT_API_KEY`. Tolerate JSON wrapped in prose or code fences.
3. **State machine (pure):** current instruction, current verify question, and `on_verdict(dict)` that updates state, attempt counts and the queue. Also exposes enough state for logging. No I/O.
4. **Relay:** the demo's WebSocket relay structure, with the demo's agent hook replaced by "ask the machine for the instruction, run a verify when the machine asks for one, feed the verdict back."

Remove from the demo: `VisionAgent.decide()`, goal/previous/history bookkeeping, `--agent-every`, and the rule that an empty agent instruction kills the session.

Keep from the demo: the handshake (read upstream metadata, forward it to the client), validation, frame packing, error handling, the timeouts, `WALLX_ENDPOINT` / `WALLX_HEADERS_FILE`, `--passthrough`, `--host`, `--port`.

Session handling: the demo resets its state when the client's `instruction` changes (a new case selected in the WebUI). Keep an equivalent: a new session gets a fresh machine. The client's own instruction string is **not** forwarded while the machine is active.

## 6. Verify questions and verdicts (suggested, tune freely)

Grab: "Is the {item} held in a gripper, lifted off the table?" → `{"holding": bool, "is_target": bool}`
Drop: "Is the {item} inside the basket and both grippers empty?" → `{"in_basket": bool, "gripper_empty": bool}`

Shared system prompt should say: three cameras (front, left_wrist, right_wrist); the robot's **right** arm appears on the **left** of the front image; only the named item counts; answer with one JSON object.

Failure handling: a failed LLM call (timeout, bad JSON, HTTP error) must **not** kill the session. Treat it as "no verdict" and decide a sensible behavior (open; the demo's behavior of closing the session is explicitly not wanted here).

## 7. Out of scope / do not decide on your own

- Recovery states (e.g. putting back a wrong object, retrieving an item that missed the basket). v1 only retries Grab. Log when it happens; do not build recovery.
- Any motion, interpolation or action filtering. The harness does not touch actions.
- Stopping the robot or homing through any interface other than forwarding an instruction.
- Changing the task JSON, the model config or the WebUI setup.
- Prompt wording beyond what is above; the user will tune prompts against the real robot.

## 8. Logging

One JSON line per event, flushed: request index, state, instruction forwarded, verify start/result (the verdict dict), transitions with a reason, attempt counts, skipped items, elapsed time. Never log images, API keys or full observations.

## 9. Testing (offline only)

Fake WALLX server + fake robot client + scripted LLM. No robot, no real endpoints, no API keys in files or tests. Cover at least:

1. Reply forwarded unchanged; handshake forwarded; extra client fields (e.g. `control_context`) preserved.
2. Happy path: all three items placed in order, then Done.
3. An item fails 3 times and is skipped; the others complete; skipped item is retried per the policy chosen in §3.
4. LLM failure (exception/garbage) does not close the session.
5. Time budget exceeded forces Done.
6. The client's vague goal string is never forwarded to WALLX while the machine is active.
7. Machine unit tests that run with no asyncio at all.

## 10. Acceptance

- `--passthrough` still behaves like the demo.
- All offline tests pass.
- A short README comment at the top of the main file listing: env vars, how to run, and every "open" decision you made and the default you picked.
