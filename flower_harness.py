#!/usr/bin/env python3
"""Flowers-in-a-bottle harness (MVP): robot client (WebUI) -> state machine -> WALLX.

The only lever is the `instruction` string forwarded to WALLX; metadata and
action replies are forwarded byte-for-byte. Only WebUI can stop the robot.
Protocol helpers come from the demo (harness.py); settings loading and
embodiment validation are reused from orange_harness.py.

Files
  flower_harness.py        relay (this file)
  flower_machine.py        pure state machine (no I/O, no asyncio)
  flower_task.json         flower count, states, instructions, step lengths
  test_flower_harness.py   offline tests

Settings (local_settings.py, shared with orange_harness.py)
  WALLX_ENDPOINT, WALLX_HEADERS, AGENT_CHAT_URL, AGENT_MODEL, AGENT_API_KEY,
  AGENT_REASONING_EFFORT (see local_settings.example.py).

Run
  python -m unittest -v test_flower_harness.py
  python flower_harness.py --host YOUR_LAN_IP --port 8001 [--task PATH] [--agent-timeout S]
  For a passthrough connectivity check use: python orange_harness.py --passthrough

Decisions (MVP defaults; 1-2 are tunable in flower_task.json)
  1. Each state forwards its instruction for `requests` client requests
     (~1.3 s of motion each, measured in server_logs.txt), then moves on. No
     verification of the actions, no retries. Lengths are untuned.
  2. "All blue flowers": after each PLACE_BOTTLE, one blocking vision call asks
     flowers_left_question about the images of the held request. Only
     {"blue_flowers_left": false} ends the task. true, a failed call or an
     unusable answer runs another cycle (the WebUI can always stop it).
  3. DONE keeps forwarding done_instruction; the WebUI ends the session.
  4. A new client connection, or a change of the client's instruction string,
     starts a fresh machine (the demo's reset rule). The client's own
     instruction is never forwarded.
  5. No time budget; the WebUI's case timeout ends the session.
  6. A vision call times out after --agent-timeout=10 s (the demo's value).
"""

import argparse
import asyncio
import json
from pathlib import Path
import time
import urllib.error

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from harness import MAX_FRAME_BYTES, instruction_text, pack, unpack
from orange_harness import VisionClient, _http_error_message, load_settings, log_json, validate_request
from flower_machine import Machine, TaskConfig

DEFAULT_TASK = Path(__file__).with_name("flower_task.json")


def load_task(path):
    return TaskConfig.from_dict(json.loads(Path(path).read_text()))


class FlowerRelay:
    """ask(images, question, system_prompt) -> dict; may raise. Runs in a thread."""

    def __init__(self, upstream_url, headers, config, ask, timeout=60, log=log_json):
        self.upstream_url = upstream_url
        self.headers = headers
        self.config = config
        self.ask = ask
        self.timeout = timeout
        self.log = log

    async def flowers_left(self, observation, index):
        """Blocking: the held client request waits. True/False, or None on any failure."""
        started = time.monotonic()
        record = {"event": "flower_check", "index": index}
        try:
            verdict = await asyncio.to_thread(
                self.ask, observation["observation"]["images"],
                self.config.flowers_left_question, self.config.system_prompt)
            left = verdict.get("blue_flowers_left") if isinstance(verdict, dict) else None
        except Exception as error:
            # Never log keys, images or raw bodies; HTTP status and short message are safe.
            left, record["error"] = None, type(error).__name__
            if isinstance(error, urllib.error.HTTPError):
                record.update(http_status=error.code, message=_http_error_message(error))
        left = left if isinstance(left, bool) else None
        self.log({**record, "blue_flowers_left": left,
                  "ms": round((time.monotonic() - started) * 1000, 1)})
        return left

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
                        machine = Machine(self.config)
                        self.log({"event": "session_start", "client_goal": goal})
                    if machine.cycle_complete:
                        machine.on_flowers_left(await self.flowers_left(observation, index))
                    instruction = machine.next_instruction()
                    if (machine.state, machine.flowers_done) != position:
                        position = (machine.state, machine.flowers_done)
                        self.log({"event": "state", "index": index, "state": machine.state,
                                  "flowers_done": machine.flowers_done,
                                  "instruction": instruction})
                    observation["instruction"] = instruction
                    await asyncio.wait_for(upstream.send(pack(observation)), self.timeout)
                    # Keep the complete reply, including error/timing fields, unchanged.
                    response = await asyncio.wait_for(upstream.recv(), self.timeout)
                    await asyncio.wait_for(downstream.send(response), self.timeout)
                    self.log({"event": "chunk", "index": index, "state": machine.state,
                              "ms": round((time.monotonic() - started) * 1000, 1)})
                    index += 1
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
    parser = argparse.ArgumentParser(description="Flowers-in-a-bottle harness (see module docstring).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--task", default=str(DEFAULT_TASK))
    parser.add_argument("--agent-timeout", type=float, default=10, help="Vision call timeout (s)")
    args = parser.parse_args()
    settings = load_settings()
    config = load_task(args.task)
    wallx_url, headers = settings["WALLX_ENDPOINT"], settings["WALLX_HEADERS"]
    ask = VisionClient(settings["AGENT_CHAT_URL"], settings["AGENT_MODEL"], settings["AGENT_API_KEY"],
                       timeout=args.agent_timeout,
                       reasoning_effort=settings["AGENT_REASONING_EFFORT"]).ask
    relay = FlowerRelay(wallx_url, headers, config, ask)
    async with serve(
        relay.handle, args.host, args.port,
        ping_interval=None, compression=None, max_size=MAX_FRAME_BYTES,
    ):
        per_flower = sum(s.requests for s in config.steps)
        print(f"Flower harness listening at ws://{args.host}:{args.port}", flush=True)
        print("Cycle per blue flower: "
              + " -> ".join(f"{s.state}({s.requests})" for s in config.steps)
              + f" = {per_flower} requests (~{per_flower * 1.3:.0f} s), then ask if any are left",
              flush=True)
        print(f"Vision model: {settings['AGENT_MODEL']} at {settings['AGENT_CHAT_URL'].split('?')[0]}; "
              f"timeout {args.agent_timeout:g} s", flush=True)
        # Names only, never values.
        print(f"WALLX upstream: {wallx_url.split('?')[0]}; "
              f"headers sent: {sorted(headers) or 'none'}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Harness exited. Confirm the robot session is stopped in WebUI.")
