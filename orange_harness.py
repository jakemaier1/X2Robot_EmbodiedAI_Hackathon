#!/usr/bin/env python3
"""Orange-item harness: robot client (WebUI) -> state machine -> WALLX relay.

Implements TECH_SPEC.md. The relay structure, protocol helpers, handshake,
timeouts and error handling come from the organizer demo (harness.py, imported
unchanged). The only lever is the `instruction` string forwarded to WALLX;
action replies are forwarded byte-for-byte. Only WebUI can stop the robot.

Files
  orange_harness.py   relay + vision LLM client (this file)
  orange_machine.py   pure state machine (no I/O, no asyncio)
  orange_task.json    item list, per-item wording, prompts, cadence, budget
  test_orange_harness.py  offline tests

Settings (local_settings.py next to this file; git-ignored, see the .example)
  WALLX_ENDPOINT      WALLX WebSocket URL, e.g. wss://HOST:PORT
  WALLX_HEADERS       dict of handshake headers; {} for a local WALLX
  AGENT_CHAT_URL      full Chat Completions URL of a vision-capable model
  AGENT_MODEL         model name for that endpoint
  AGENT_API_KEY       key for that endpoint (separate from WALLX)
  Setting values are never logged. Env var ORANGE_TASK_FILE (optional): task JSON.

Run
  python -m unittest -v test_orange_harness.py
  python orange_harness.py --host YOUR_LAN_IP --port 8001 --passthrough   # demo behaviour
  python orange_harness.py --host YOUR_LAN_IP --port 8001                 # state machine
  Options: --task PATH, --agent-timeout SECONDS (default 10, as in the demo).

Task JSON
  "items" entries are names, or {"name": ..., <any template key>: ...} to
  override wording for one item. "{item}" in templates is replaced by the name.
  A verdict passes when every key in grab_pass / drop_pass is exactly true.

Open decisions (defaults; all except the last two are in orange_task.json)
  1. Skipped items: after the main queue, one retry round in the original
     order; each skipped item gets retry_max_attempts=1 failed verify before it
     is marked failed. retry_max_attempts=0 disables the retry round. The time
     budget still applies, so the round only runs if time remains.
  2. Cadence: first verify after verify_after=8 forwarded requests once the
     instruction changes; after a failed grab verdict or a no-verdict, verify
     again after reverify_every=4 requests. Untuned; tune on the robot.
  3. No verdict (LLM timeout, HTTP error, unparsable JSON, missing/non-bool
     keys): not an attempt and does not close the session. The current
     instruction keeps being forwarded and the verify repeats after
     reverify_every requests. The time budget bounds this.
  4. Failed drop verify: counts as an attempt and returns to GRAB of the same
     item (per the spec diagram); waits verify_after before the next verify.
  5. Time budget time_budget_s=420 is measured from the first request of the
     session (the harness cannot see the WebUI's own case timer). It is checked
     before each verify and after each verdict, so a verdict that arrives late
     is still applied before going to DONE.
  6. Session: a new client connection, or a change of the client's instruction
     string, starts a fresh machine (the demo's reset rule).
  7. Verify input: the three images of the held request plus the question.
     No pose state or history is sent.
  9. Embodiment: only teletwin_desktop/gripper-g is accepted (EMBODIMENT); anything
     else is rejected. The request is forwarded to WALLX with it untouched.
  8. A vision call times out after --agent-timeout=10 s (the demo's value).
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
import urllib.error
import urllib.request

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from harness import MAX_FRAME_BYTES, instruction_text, pack, unpack
from harness import validate_request as demo_validate_request
from orange_machine import Machine, TaskConfig

CAMERAS = ("front", "left_wrist", "right_wrist")
# The only embodiment this station uses; anything else is rejected.
EMBODIMENT = "teletwin_desktop/gripper-g"


def validate_request(request):
    """The demo's validation, restricted to EMBODIMENT (harness.py is unchanged)."""
    if not isinstance(request, dict) or request.get("embodiment") != EMBODIMENT:
        raise ValueError("Unsupported embodiment")
    # The demo validator only knows its own labels, so show it one it accepts.
    demo_validate_request({**request, "embodiment": "quanta_x1_pro/gripper-g"})
DEFAULT_TASK = Path(__file__).with_name("orange_task.json")
SETTINGS_FILE = Path(__file__).with_name("local_settings.py")
_PLACEHOLDER = "PASTE_"


def load_settings(path=SETTINGS_FILE, need_agent=True):
    """Read the constants from local_settings.py by explicit path.

    Run with runpy (not import), so no .pyc copy of the key lands in
    __pycache__ and no same-named module elsewhere on sys.path can shadow it.
    Errors name the setting, never its value.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Missing {path.name}: copy local_settings.example.py to "
                         f"{path.name} next to orange_harness.py and fill it in.")
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(f"Warning: {path.name} is readable by other users; run: chmod 600 {path.name}",
              flush=True)
    values = runpy.run_path(str(path))
    required = ["WALLX_ENDPOINT"] + (["AGENT_CHAT_URL", "AGENT_MODEL", "AGENT_API_KEY"]
                                     if need_agent else [])
    settings = {}
    for name in required:
        value = values.get(name)
        if not isinstance(value, str) or not value.strip() or value.startswith(_PLACEHOLDER):
            raise SystemExit(f"Set {name} in {path.name}")
        settings[name] = value.strip()
    headers = values.get("WALLX_HEADERS", {})
    if not isinstance(headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
        raise SystemExit(f"WALLX_HEADERS in {path.name} must be a dict of strings")
    settings["WALLX_HEADERS"] = dict(headers)
    effort = values.get("AGENT_REASONING_EFFORT")
    if effort is not None and (not isinstance(effort, str) or not effort.strip()):
        raise SystemExit(f"AGENT_REASONING_EFFORT in {path.name} must be a string or None")
    settings["AGENT_REASONING_EFFORT"] = effort.strip() if effort else None
    return settings


def load_task(path):
    return TaskConfig.from_dict(json.loads(Path(path).read_text()))


def log_json(record):
    print(json.dumps(record, default=str), flush=True)


# ---- vision LLM client -------------------------------------------------------

def parse_json_object(text):
    """Return the first JSON object in text, tolerating prose and code fences."""
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
        start = text.find("{", start + 1)
    raise ValueError("No JSON object in agent reply")


def message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some endpoints return content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    raise ValueError("Unexpected agent message content")


def _http_error_message(error):
    """The `error.message` field of an API error body (Gemini wraps it in a list), else ''."""
    try:
        body = json.loads(error.read(4096))
        if isinstance(body, list) and body:
            body = body[0]
        return str(body["error"]["message"])[:200]
    except Exception:
        return ""


class VisionClient:
    """One generic call: three camera images + a question in, a JSON dict out."""

    def __init__(self, url, model, key="", timeout=10, reasoning_effort=None):
        self.url, self.model, self.key, self.timeout = url, model, key, timeout
        self.reasoning_effort = reasoning_effort  # e.g. "minimal"; None = model default

    def ask(self, images, question, system_prompt):
        content = [{"type": "text", "text": question}]
        for name in CAMERAS:
            image = images.get(name)
            if isinstance(image, bytes):
                image = base64.b64encode(image).decode("ascii")
            if not isinstance(image, str) or not image:
                raise ValueError(f"Missing JPEG observation: {name}")
            if not image.startswith("data:image/"):
                image = "data:image/jpeg;base64," + image
            content += [{"type": "text", "text": name},
                        {"type": "image_url", "image_url": {"url": image}}]
        body = {"model": self.model, "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.url, json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        return parse_json_object(message_text(result["choices"][0]["message"]["content"]))


# ---- relay -------------------------------------------------------------------

class OrangeRelay:
    """config=None means passthrough (the demo's --passthrough behaviour).

    verifier(images, question, system_prompt) -> dict; may raise. Runs in a thread.
    """

    def __init__(self, upstream_url, headers=None, config=None, verifier=None,
                 timeout=60, clock=time.monotonic, log=log_json):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if config is not None and verifier is None:
            raise ValueError("A verifier is required unless running passthrough")
        self.upstream_url = upstream_url
        self.headers = headers or {}
        self.config = config
        self.verifier = verifier
        self.timeout = timeout
        self.clock = clock
        self.log = log

    async def verify(self, observation, question):
        """Blocking verification: the held client request waits for this.

        Returns the verdict dict, or None on any failure. A non-blocking variant
        would replace only this method.
        """
        try:
            verdict = await asyncio.to_thread(
                self.verifier, observation["observation"]["images"], question,
                self.config.system_prompt)
        except Exception as error:
            # Never log keys, images or raw bodies. For HTTP errors, add the status
            # and the provider's short error message (it carries neither).
            record = {"event": "verify_error", "error": type(error).__name__}
            if isinstance(error, urllib.error.HTTPError):
                record["http_status"] = error.code
                record["message"] = _http_error_message(error)
            self.log(record)
            return None
        if not isinstance(verdict, dict):
            self.log({"event": "verify_error", "error": "non-dict verdict"})
            return None
        return verdict

    def _log_events(self, machine, index):
        now = self.clock()
        for record in machine.drain_events():
            self.log({**record, "index": index, "elapsed_s": round(machine.elapsed(now), 2)})

    async def _steer(self, observation, machine, index):
        """Ask the machine for the instruction, verifying first if it asks to."""
        question = machine.tick(self.clock())
        self._log_events(machine, index)
        if question is not None:
            self.log({"event": "verify_start", "index": index, "state": machine.state,
                      "item": machine.item.name, "question": question})
            started = time.monotonic()
            verdict = await self.verify(observation, question)
            self.log({"event": "verify_result", "index": index, "item": machine.item.name,
                      "verdict": verdict,
                      "ms": round((time.monotonic() - started) * 1000, 1)})
            machine.on_verdict(verdict, self.clock())
            self._log_events(machine, index)
        # The client's own instruction is never forwarded while the machine runs.
        observation["instruction"] = machine.instruction
        return pack(observation)

    async def handle(self, downstream):
        machine = client_goal = None
        index = 0
        embodiment = None  # last embodiment label seen; short, non-secret, for diagnostics
        # First sign that the robot client reached THIS process.
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
                    if self.config is not None:
                        goal = instruction_text(observation.get("instruction"))
                        if machine is None or goal != client_goal:
                            client_goal, index = goal, 0
                            machine = Machine(self.config, self.clock())
                            self.log({"event": "session_start", "client_goal": goal})
                        frame = await self._steer(observation, machine, index)
                    steered = time.monotonic()
                    await asyncio.wait_for(upstream.send(frame), self.timeout)
                    # Keep the complete reply, including error/timing fields, unchanged.
                    response = await asyncio.wait_for(upstream.recv(), self.timeout)
                    await asyncio.wait_for(downstream.send(response), self.timeout)
                    done = time.monotonic()
                    record = {"event": "chunk", "index": index,
                              "elapsed_ms": round((done - started) * 1000, 1),
                              # harness side (incl. any verify) vs WALLX round trip
                              "steer_ms": round((steered - started) * 1000, 1),
                              "wallx_ms": round((done - steered) * 1000, 1)}
                    if machine is not None:
                        machine.forwarded()
                        record.update(instruction=machine.instruction,
                                      **machine.snapshot(self.clock()))
                    self.log(record)
                    index += 1
        except ConnectionClosed:
            self.log({"event": "closed", "message": "Connection closed; end/confirm the session in WebUI."})
        except Exception as error:
            # Don't log HTTP bodies, credentials, images, or full observations.
            record = {"event": "harness_failed", "error": type(error).__name__,
                      "message": "Stop the session in WebUI."}
            # ValueErrors here are our own fixed validation messages (no data in them),
            # so they are safe to show; other exception types stay type-only.
            if type(error) is ValueError:
                record["detail"] = str(error)[:200]
                if "embodiment" in record["detail"]:
                    record["embodiment_received"] = repr(embodiment)[:80]
            # A rejected WALLX handshake (e.g. 401/403/404) carries a safe status code.
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status is not None:
                record["upstream_http_status"] = status
            self.log(record)
            await downstream.close(code=1011, reason="Harness failure; stop in WebUI")


async def main():
    parser = argparse.ArgumentParser(description="Orange-item harness (see module docstring).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--passthrough", action="store_true", help="Relay without the state machine")
    parser.add_argument("--task", default=os.environ.get("ORANGE_TASK_FILE", str(DEFAULT_TASK)))
    parser.add_argument("--agent-timeout", type=float, default=10, help="Vision call timeout (s)")
    args = parser.parse_args()
    settings = load_settings(need_agent=not args.passthrough)
    wallx_url, headers = settings["WALLX_ENDPOINT"], settings["WALLX_HEADERS"]
    config = verifier = None
    if not args.passthrough:
        config = load_task(args.task)
        verifier = VisionClient(settings["AGENT_CHAT_URL"], settings["AGENT_MODEL"],
                                settings["AGENT_API_KEY"], timeout=args.agent_timeout,
                                reasoning_effort=settings["AGENT_REASONING_EFFORT"]).ask
        print(f"Vision model: {settings['AGENT_MODEL']} at "
              f"{settings['AGENT_CHAT_URL'].split('?')[0]}; reasoning_effort="
              f"{settings['AGENT_REASONING_EFFORT'] or 'model default'}; "
              f"timeout {args.agent_timeout:g} s", flush=True)
    relay = OrangeRelay(wallx_url, headers, config=config, verifier=verifier)
    async with serve(
        relay.handle, args.host, args.port,
        ping_interval=None, compression=None, max_size=MAX_FRAME_BYTES,
    ):
        print(f"Orange harness listening at ws://{args.host}:{args.port}"
              f" ({'passthrough' if config is None else 'state machine'})", flush=True)
        # Names only, never values. No headers file => no auth headers are sent.
        print(f"WALLX upstream: {wallx_url.split('?')[0]}; "
              f"headers sent: {sorted(headers) or 'none'}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Harness exited. Confirm the robot session is stopped in WebUI.")
