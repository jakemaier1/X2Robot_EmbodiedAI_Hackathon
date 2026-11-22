#!/usr/bin/env python3
"""Flowers-in-a-vase harness: robot client (WebUI) -> planner + state machine -> WALLX.

The only lever is the `instruction` string forwarded to WALLX; metadata and
action replies are forwarded byte-for-byte. Only WebUI can stop the robot.
Protocol helpers come from the demo (harness.py); everything else is here.

Files
  flower_harness.py        relay + Claude client (this file)
  flower_machine.py        pure state machine, plan checks, keyword fallback
  flower_task.json         colors, states, {color} instructions, lengths, prompts
  test_flower_harness.py   offline tests (no robot, no network, no keys)

Settings (local_settings.py next to this file; git-ignored, chmod 600)
  WALLX_ENDPOINT, WALLX_HEADERS        WALLX upstream
  ANTHROPIC_API_KEY                    optional; else the SDK reads the
                                       ANTHROPIC_API_KEY env var or an
                                       `ant auth login` profile
  ANTHROPIC_MODEL  = "claude-sonnet-5-5" (default)
  ANTHROPIC_EFFORT = "low" (default; low | medium | high | xhigh | max)

Run
  python -m unittest -v test_flower_harness.py
  python flower_harness.py --host YOUR_LAN_IP --port 8001 [--task PATH] [--agent-timeout S]
  For a passthrough connectivity check use the demo: python harness.py --passthrough

Decisions (1-3 are tunable in flower_task.json)
  1. Plan: on a session's first request, Claude gets the client's instruction
     and the three camera images and returns {"actions": [color, ...]}, one per
     flower, in order (at most max_actions). The held request waits. On a
     failed call or an unusable plan: one retry, then a keyword parse of the
     instruction text (flower_machine.fallback_plan). An empty plan goes home.
  2. Act: each action runs the steps with {color} filled in; each step
     forwards its instruction for `requests` client requests (~1.3 s each).
  2b. Verify: with a grasp_check, after GRAB_FLOWER Claude names the color the
     gripper holds (VERIFY_GRASP). Wrong color still needed -> swap the queue
     order; wrong color not needed -> PUT_BACK, grab again; nothing -> grab
     again (regrabs_per_action), then give the action up. Unclear -> carry on.
  3. Check: after an action, if it can still change something (a retry is
     left, or another action of that color is queued), Claude counts the
     flowers per color in the vase. No increase for the action's color -> the
     action goes back on the front of the queue (retries_per_action times).
     A failed check counts as success.
  4. DONE forwards done_instruction for done_requests chunks (3), then the
     relay closes the client connection and the process exits.
  5. A new client connection, or a change of the client's instruction string,
     starts a fresh plan and machine. The client's own instruction is never
     forwarded.
  6. No time budget; the WebUI's case timeout ends the session.
  7. Each Claude call times out after --agent-timeout=10 s; the SDK does not
     retry (the planner's one retry is ours). Server-side refusal fallback
     (beta server-side-fallback-2026-07-01, fallbacks="default") is on.
"""

"""
    {"state": "GRAB_BOTTLE",  "requests": 10, "instruction": "Grab the yellow and red bottle and hold it in place"},
    {"state": "GRAB_FLOWER",  "requests": 10, "instruction": "While holding the bottle in place, grab a blue flower by its stem, right below the flower petals."},
    {"state": "INSERT_STEM",  "requests": 10, "instruction": "While holding the bottle in place, insert the end of the flower stem into the top of the bottle."},
    {"state": "PLACE_BOTTLE", "requests": 10, "instruction": "While holding the bottle in place, lower the flower stem into the bottle."}
"""

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import runpy
import stat
import time

import anthropic
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from harness import MAX_FRAME_BYTES, instruction_text, pack, unpack
from harness import validate_request as demo_validate_request
from flower_machine import (INSTRUCTION, NOTHING, VERIFY, Machine, TaskConfig, fallback_plan, validate_counts,
                            validate_plan)

DEFAULT_TASK = Path(__file__).with_name("flower_task.json")
SETTINGS_FILE = Path(__file__).with_name("local_settings.py")
CAMERAS = ("front", "left_wrist", "right_wrist")
EMBODIMENT = "teletwin_desktop/gripper-g"  # the only embodiment this station uses
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "low"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
PLAN_ATTEMPTS = 2  # Claude, one retry, then the keyword fallback


def validate_request(request):
    """The demo's validation, restricted to EMBODIMENT (harness.py stays unchanged)."""
    if not isinstance(request, dict) or request.get("embodiment") != EMBODIMENT:
        raise ValueError("Unsupported embodiment")
    # The demo validator only knows its own labels, so show it one it accepts.
    demo_validate_request({**request, "embodiment": "quanta_x1_pro/gripper-g"})


def load_settings(path=SETTINGS_FILE):
    """Read the constants from local_settings.py; errors name the setting, never its value.

    runpy (not import): no .pyc copy of the key lands in __pycache__, and no
    same-named module elsewhere on sys.path can shadow it.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Missing {path.name}: create it next to flower_harness.py "
                         "(see the Settings list in the module docstring).")
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(f"Warning: {path.name} is readable by other users; run: chmod 600 {path.name}",
              flush=True)
    values = runpy.run_path(str(path))
    settings = {}
    endpoint = values.get("WALLX_ENDPOINT")
    if not isinstance(endpoint, str) or not endpoint.strip() or endpoint.startswith("PASTE_"):
        raise SystemExit(f"Set WALLX_ENDPOINT in {path.name}")
    settings["WALLX_ENDPOINT"] = endpoint.strip()
    headers = values.get("WALLX_HEADERS", {})
    if not isinstance(headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
        raise SystemExit(f"WALLX_HEADERS in {path.name} must be a dict of strings")
    settings["WALLX_HEADERS"] = dict(headers)
    key = values.get("ANTHROPIC_API_KEY")
    if key is not None and not isinstance(key, str):
        raise SystemExit(f"ANTHROPIC_API_KEY in {path.name} must be a string")
    key = (key or "").strip()
    settings["ANTHROPIC_API_KEY"] = key if key and not key.startswith("PASTE_") else None
    model = values.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    if not isinstance(model, str) or not model.strip():
        raise SystemExit(f"ANTHROPIC_MODEL in {path.name} must be a non-empty string")
    settings["ANTHROPIC_MODEL"] = model.strip()
    effort = values.get("ANTHROPIC_EFFORT", DEFAULT_EFFORT)
    if effort not in EFFORTS:
        raise SystemExit(f"ANTHROPIC_EFFORT in {path.name} must be one of {', '.join(EFFORTS)}")
    settings["ANTHROPIC_EFFORT"] = effort
    return settings


def load_task(path):
    return TaskConfig.from_dict(json.loads(Path(path).read_text()))


def log_json(record):
    print(json.dumps(record, default=str), flush=True)


def elapsed_ms(started):
    return round((time.monotonic() - started) * 1000, 1)


def error_fields(error):
    """Safe log fields for a failed Claude call: never keys, images or bodies."""
    record = {"error": type(error).__name__}
    if isinstance(error, anthropic.APIStatusError):
        record.update(http_status=error.status_code, message=str(error.message)[:200])
    elif isinstance(error, (ValueError, TypeError)):  # our checks; SDK "no credentials"
        record["message"] = str(error)[:200]
    return record


def gripper_fields(observation, response):
    """Observed gripper per arm, and the min/max gripper WALLX commands for the next
    chunk, for calibrating release detection. Logging only: never raises."""
    fields = {}
    try:
        state = observation["observation"]["state"]
        fields["gripper"] = {arm: round(float(state[f"{arm}_ee"]["gripper"]), 3)
                             for arm in ("left", "right")}
        actions = unpack(response)["actions"]
        fields["gripper_cmd"] = {arm: [round(float(min(g)), 3), round(float(max(g)), 3)]
                                 for arm in ("left", "right")
                                 for g in [list(actions[f"{arm}_ee"]["gripper"])]}
    except Exception:
        pass
    return fields


# ---- Claude: planner and vase checker ------------------------------------------

def image_blocks(images):
    """The three camera JPEGs as Messages API image blocks, each after a label."""
    blocks = []
    for name in CAMERAS:
        image, media_type = images.get(name), "image/jpeg"
        if isinstance(image, (bytes, bytearray)):
            image = base64.b64encode(image).decode("ascii")
        elif isinstance(image, str) and image.startswith("data:"):
            header, _, image = image.partition(",")
            media_type = header[5:].split(";")[0] or media_type
        if not isinstance(image, str) or not image:
            raise ValueError(f"Missing JPEG observation: {name}")
        blocks += [{"type": "text", "text": f"{name} camera:"},
                   {"type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image}}]
    return blocks


def plan_schema(colors):
    return {"type": "object",
            "properties": {"actions": {"type": "array",
                                       "items": {"type": "string", "enum": list(colors)}}},
            "required": ["actions"], "additionalProperties": False}


UNCLEAR = "unclear"


def held_schema(colors):
    return {"type": "object",
            "properties": {"scene": {"type": "string"},
                           "held": {"type": "string", "enum": [*colors, NOTHING, UNCLEAR]}},
            "required": ["scene", "held"], "additionalProperties": False}


def counts_schema(colors):
    # "scene" comes first so the model describes what it sees before it counts.
    return {"type": "object",
            "properties": {"scene": {"type": "string"},
                           **{color: {"type": "integer"} for color in colors}},
            "required": ["scene", *colors], "additionalProperties": False}


class ClaudeVision:
    """One Messages API call per question: three images + text in, schema-valid JSON
    out (structured outputs). Blocking; the relay runs it in a thread."""

    def __init__(self, client, config, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT):
        self.client, self.config, self.model, self.effort = client, config, model, effort

    def ask(self, system, question, images, schema):
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,  # thinking counts toward it; the JSON itself is tiny
            betas=[FALLBACK_BETA],
            fallbacks="default",  # a safety decline re-runs on the fallback model, same call
            system=system,
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user",
                       "content": image_blocks(images) + [{"type": "text", "text": question}]}],
        )
        if response.stop_reason != "end_turn":  # refusal or max_tokens: JSON not guaranteed
            raise ValueError(f"Claude stopped with {response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise ValueError("No text block in Claude reply")
        return json.loads(text)

    def plan(self, images, instruction):
        question = self.config.planner_question.replace(INSTRUCTION, instruction)
        return self.ask(self.config.planner_system_prompt, question, images,
                        plan_schema(self.config.colors))

    def held_color(self, images):
        gc = self.config.grasp_check
        return self.ask(gc.system_prompt, gc.question, images, held_schema(self.config.colors))

    def vase_counts(self, images):
        return self.ask(self.config.check_system_prompt, self.config.check_question, images,
                        counts_schema(self.config.colors))


# ---- relay -------------------------------------------------------------------


class FlowerRelay:
    """vision.plan(images, instruction) and vision.vase_counts(images) return dicts
    and may raise; so does vision.held_color(images) when the task has a grasp_check.
    They block, so they run in a thread."""

    def __init__(self, upstream_url, headers, config, vision, timeout=60, log=log_json):
        self.upstream_url = upstream_url
        self.headers = headers
        self.config = config
        self.vision = vision
        self.timeout = timeout
        self.log = log
        self.finished = asyncio.Event()  # set after a session's last DONE chunk; main() exits

    async def make_plan(self, observation, goal):
        """Blocking: the held client request waits. Claude, one retry, then keywords."""
        images = observation["observation"]["images"]
        for attempt in range(1, PLAN_ATTEMPTS + 1):
            started = time.monotonic()
            record = {"event": "plan", "source": "claude", "attempt": attempt}
            try:
                reply = await asyncio.to_thread(self.vision.plan, images, goal)
                plan = validate_plan(reply, self.config)
            except Exception as error:
                self.log({**record, **error_fields(error), "ms": elapsed_ms(started)})
                continue
            self.log({**record, "plan": plan, "ms": elapsed_ms(started)})
            return plan
        plan = fallback_plan(goal, self.config)
        self.log({"event": "plan", "source": "keyword_fallback", "plan": plan})
        return plan

    async def held_color(self, observation, index):
        """Blocking: the held client request waits. (color, NOTHING or None, log record)."""
        started = time.monotonic()
        record = {"event": "grasp_check", "index": index, "state": VERIFY}
        held = None
        try:
            reply = await asyncio.to_thread(self.vision.held_color,
                                            observation["observation"]["images"])
            answer = reply.get("held") if isinstance(reply, dict) else None
            if answer not in (*self.config.colors, NOTHING, UNCLEAR):
                raise ValueError("held must be a color, none or unclear")
            held = None if answer == UNCLEAR else answer
            record["held"] = answer
            if isinstance(reply.get("scene"), str):
                record["scene"] = reply["scene"][:300]
        except Exception as error:
            record.update(error_fields(error))
        record["ms"] = elapsed_ms(started)
        return held, record

    async def vase_counts(self, observation, index):
        """Blocking: the held client request waits. ({color: int} or None, log record)."""
        started = time.monotonic()
        record = {"event": "vase_check", "index": index}
        try:
            reply = await asyncio.to_thread(self.vision.vase_counts,
                                            observation["observation"]["images"])
            counts = validate_counts(reply, self.config)
            if isinstance(reply.get("scene"), str):
                record["scene"] = reply["scene"][:300]
        except Exception as error:
            counts = None
            record.update(error_fields(error))
        record.update(counts=counts, ms=elapsed_ms(started))
        return counts, record

    async def handle(self, downstream):
        machine = goal = position = None
        index = 0
        embodiment = None  # last label seen; short and non-secret, for diagnostics
        self.log({"event": "client_connected", "upstream": self.upstream_url.split("?")[0]})
        try:
            async with connect(
                self.upstream_url,
                additional_headers=self.headers,
                proxy=None,
                ping_interval=None,
                compression=None,
                open_timeout=10,
                close_timeout=2,
                max_size=MAX_FRAME_BYTES,
            ) as upstream:
                metadata = await asyncio.wait_for(upstream.recv(), self.timeout)
                if not isinstance(unpack(metadata), dict):
                    raise ValueError("Expected WALLX metadata object")
                await downstream.send(metadata)
                self.log({"event": "wallx_connected"})
                async for frame in downstream:
                    started = time.monotonic()
                    observation = unpack(frame)
                    if isinstance(observation, dict):
                        embodiment = observation.get("embodiment")
                    validate_request(observation)
                    client_goal = instruction_text(observation.get("instruction"))
                    if machine is None or client_goal != goal:
                        goal, index, position = client_goal, 0, None
                        self.log({"event": "session_start", "client_goal": goal})
                        machine = Machine(self.config, await self.make_plan(observation, goal))
                    if machine.awaiting_verify:
                        wanted = machine.action
                        held, record = await self.held_color(observation, index)
                        outcome = machine.on_held(held)
                        self.log({**record, "wanted": wanted.color, "outcome": outcome,
                                  "now_placing": machine.action.color if machine.action else None,
                                  "queued": machine.queued})
                    if machine.awaiting_check:
                        action = machine.action
                        counts, record = await self.vase_counts(observation, index)
                        outcome = machine.on_vase_counts(counts)
                        self.log({**record, "color": action.color, "attempt": action.attempt,
                                  "outcome": outcome, "in_vase": dict(machine.in_vase)})
                    instruction = machine.next_instruction()
                    if (machine.state, machine.started) != position:
                        position = (machine.state, machine.started)
                        action = machine.action
                        self.log({"event": "state", "index": index, "state": machine.state,
                                  "action": machine.started,
                                  "color": action.color if action else None,
                                  "attempt": action.attempt if action else None,
                                  "queued": machine.queued, "instruction": instruction})
                    observation["instruction"] = instruction
                    await asyncio.wait_for(upstream.send(pack(observation)), self.timeout)
                    # Keep the complete reply, including error/timing fields, unchanged.
                    response = await asyncio.wait_for(upstream.recv(), self.timeout)
                    await asyncio.wait_for(downstream.send(response), self.timeout)
                    self.log({"event": "chunk", "index": index, "state": machine.state,
                              "ms": elapsed_ms(started), **gripper_fields(observation, response)})
                    index += 1
                    if machine.finished:
                        self.log({"event": "finished", "index": index,
                                  "message": "Task done; closing. Confirm the session is stopped in WebUI."})
                        await downstream.close(code=1000, reason="Task done")
                        self.finished.set()
                        return
        except ConnectionClosed:
            self.log({"event": "closed", "message": "Connection closed; end/confirm the session in WebUI."})
        except Exception as error:
            # Never log HTTP bodies, credentials, images or full observations.
            record = {"event": "harness_failed", "error": type(error).__name__,
                      "message": "Stop the session in WebUI."}
            if type(error) is ValueError:  # our own fixed validation messages; no data in them
                record["detail"] = str(error)[:200]
                record["embodiment_received"] = repr(embodiment)[:80]
            # A rejected WALLX handshake (e.g. 401/403/404) carries a safe status code.
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status is not None:
                record["upstream_http_status"] = status
            self.log(record)
            await downstream.close(code=1011, reason="Harness failure; stop in WebUI")


async def main():
    parser = argparse.ArgumentParser(description="Flowers-in-a-vase harness (see module docstring).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--task", default=str(DEFAULT_TASK))
    parser.add_argument("--agent-timeout", type=float, default=10, help="Timeout per Claude call (s)")
    args = parser.parse_args()
    settings = load_settings()
    config = load_task(args.task)
    wallx_url, headers = settings["WALLX_ENDPOINT"], settings["WALLX_HEADERS"]
    client = anthropic.Anthropic(api_key=settings["ANTHROPIC_API_KEY"],
                                 timeout=args.agent_timeout, max_retries=0)
    vision = ClaudeVision(client, config, settings["ANTHROPIC_MODEL"], settings["ANTHROPIC_EFFORT"])
    relay = FlowerRelay(wallx_url, headers, config, vision)
    async with serve(
        relay.handle, args.host, args.port,
        ping_interval=None, compression=None, max_size=MAX_FRAME_BYTES,
    ):
        per_action = sum(s.requests for s in config.steps)
        print(f"Flower harness listening at ws://{args.host}:{args.port}", flush=True)
        print("Per action: "
              + " -> ".join(f"{s.state}({s.requests})" for s in config.steps)
              + f" = {per_action} requests (~{per_action * 1.3:.0f} s); "
              f"up to {config.max_actions} actions, {config.retries_per_action} retry each; "
              f"colors: {', '.join(config.colors)}", flush=True)
        if settings["ANTHROPIC_API_KEY"]:
            source = "key from local_settings.py"
        elif os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            source = "key from the environment"
        else:
            source = "no key set: the SDK will try an `ant auth login` profile"
        print(f"Planner/checker: {settings['ANTHROPIC_MODEL']} (effort {settings['ANTHROPIC_EFFORT']}), "
              f"{source}; timeout {args.agent_timeout:g} s", flush=True)
        # Names only, never values.
        print(f"WALLX upstream: {wallx_url.split('?')[0]}; "
              f"headers sent: {sorted(headers) or 'none'}", flush=True)
        await relay.finished.wait()
        print("Task done after the DONE chunks; harness exiting. "
              "Confirm the robot session is stopped in WebUI.", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Harness exited. Confirm the robot session is stopped in WebUI.")
