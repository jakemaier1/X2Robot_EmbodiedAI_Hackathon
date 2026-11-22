"""Offline tests for the orange-item harness. No robot, no real endpoints, no keys."""

import asyncio
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import sys
import threading
import unittest

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError

from harness import pack, unpack
from orange_harness import DEFAULT_TASK, OrangeRelay, VisionClient, load_task, parse_json_object
from orange_machine import DONE, DROP, GRAB, Machine, TaskConfig

GOAL = "Put the orange things in the basket."
CONFIG = load_task(DEFAULT_TASK)
FAST = replace(CONFIG, verify_after=1, reverify_every=1)  # verify on every request after the first

GRAB_OK = {"holding": True, "is_target": True}
GRAB_NO = {"holding": False, "is_target": False}
DROP_OK = {"in_basket": True, "gripper_empty": True}
DROP_NO = {"in_basket": False, "gripper_empty": True}


def observation(goal=GOAL):
    return {
        "embodiment": "teletwin_desktop/gripper-g",
        "normalization": "provided", "instruction": goal,
        "observation": {
            "state": {name: {"position": [0.1, 0.2, 0.3], "rotation_rpy": [0, 0, 0], "gripper": 0.5}
                      for name in ("left_ee", "right_ee")},
            "images": {name: "aW1hZ2U=" for name in ("front", "left_wrist", "right_wrist")},
        },
        "control_context": {"left": [None] * 7, "right": [None] * 7},
    }


def kind_of(question):
    return "grab" if "held in a gripper" in question else "drop"


def item_of(question):
    return next(i.name for i in CONFIG.items if f" {i.name} " in question)


class Script:
    """Scripted vision LLM: answers[(item, kind)] is a list consumed in order;
    when it runs out (or is absent) the step passes."""

    def __init__(self, answers=None):
        self.answers = {k: list(v) for k, v in (answers or {}).items()}
        self.questions = []

    def __call__(self, images, question, system_prompt):
        self.questions.append(question)
        key = (item_of(question), kind_of(question))
        queue = self.answers.get(key)
        answer = queue.pop(0) if queue else (GRAB_OK if key[1] == "grab" else DROP_OK)
        if isinstance(answer, Exception):
            raise answer
        return answer


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def drive(machine, script, now=0.0, steps=200):
    """Drive the machine exactly like the relay does; return forwarded instructions."""
    forwarded = []
    for _ in range(steps):
        question = machine.tick(now)
        if question is not None:
            try:
                verdict = script(None, question, machine.config.system_prompt)
            except Exception:
                verdict = None
            machine.on_verdict(verdict, now)
        forwarded.append(machine.instruction)
        machine.forwarded()
        if machine.state == DONE:
            break
    return forwarded


# ---- machine: no asyncio, no network -------------------------------------------

class MachineTests(unittest.TestCase):
    def test_machine_module_imports_no_asyncio_or_network(self):
        code = ("import sys, orange_machine; "
                "print(sorted(m for m in ('asyncio', 'websockets', 'urllib.request', 'socket') "
                "if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True, cwd=str(DEFAULT_TASK.parent)).stdout.strip()
        self.assertEqual(out, "[]")

    def test_config_is_data(self):
        self.assertEqual([i.name for i in CONFIG.items], ["carrot", "carton", "tangerine"])
        self.assertEqual(CONFIG.items[0].drop_instruction, "Put the carrot in the basket.")
        self.assertIn('{"holding": bool, "is_target": bool}', CONFIG.items[0].grab_question)
        self.assertEqual((CONFIG.max_attempts, CONFIG.time_budget_s), (3, 420))
        custom = TaskConfig.from_dict({**json.loads(DEFAULT_TASK.read_text()), "items": [
            "pear", {"name": "plum", "grab_instruction": "Pinch the {item} gently."}]})
        self.assertEqual(custom.items[1].grab_instruction, "Pinch the plum gently.")
        self.assertEqual(custom.items[0].grab_instruction, "Pick up the pear.")
        with self.assertRaises(ValueError):
            TaskConfig.from_dict({**json.loads(DEFAULT_TASK.read_text()), "items": []})

    def test_happy_path_in_order_then_done(self):
        machine = Machine(FAST, 0)
        forwarded = drive(machine, Script())
        self.assertEqual(forwarded, [
            "Pick up the carrot.", "Put the carrot in the basket.",
            "Pick up the carton.", "Put the carton in the basket.",
            "Pick up the tangerine.", "Put the tangerine in the basket.",
            "Return both arms to the home position.",
        ])
        self.assertEqual(machine.placed, ["carrot", "carton", "tangerine"])
        self.assertEqual(machine.done_reason, "queue_complete")
        # DONE keeps forwarding home and never verifies again.
        for _ in range(5):
            self.assertIsNone(machine.tick(1))
            self.assertEqual(machine.instruction, "Return both arms to the home position.")
            machine.forwarded()

    def test_cadence_waits_verify_after_then_reverify_every(self):
        config = replace(CONFIG, verify_after=3, reverify_every=2)
        machine = Machine(config, 0)
        script = Script({("carrot", "grab"): [GRAB_NO]})
        questions_at = []
        for step in range(10):
            if machine.tick(0):
                questions_at.append(step)
                machine.on_verdict(script(None, machine.verify_question, ""), 0)
            machine.forwarded()
        # The held (verified) request is forwarded too, so it counts toward the wait:
        # 3 requests, verify fails at #3, re-verify 2 later at #5 (pass -> DROP),
        # first drop verify 3 later at #8.
        self.assertEqual(questions_at, [3, 5, 8])

    def test_item_skipped_after_three_failures_then_retried_once_and_placed(self):
        machine = Machine(FAST, 0)
        script = Script({("carton", "grab"): [GRAB_NO, GRAB_NO, GRAB_NO]})
        drive(machine, script)
        self.assertEqual(machine.placed, ["carrot", "tangerine", "carton"])
        self.assertEqual(machine.failed, [])
        self.assertEqual(machine.done_reason, "queue_complete")

    def test_skipped_item_that_fails_retry_is_failed(self):
        machine = Machine(FAST, 0)
        drive(machine, Script({("carton", "grab"): [GRAB_NO] * 4}))
        self.assertEqual(machine.placed, ["carrot", "tangerine"])
        self.assertEqual(machine.failed, ["carton"])
        events = [e for e in machine.drain_events() if e.get("reason") == "skipped"]
        self.assertEqual(events[0]["skipped"], ["carton"])

    def test_retry_round_can_be_disabled(self):
        machine = Machine(replace(FAST, retry_max_attempts=0), 0)
        drive(machine, Script({("carton", "grab"): [GRAB_NO] * 3}))
        self.assertEqual(machine.skipped, ["carton"])
        self.assertEqual(machine.placed, ["carrot", "tangerine"])

    def test_drop_failure_counts_and_returns_to_grab(self):
        machine = Machine(FAST, 0)
        script = Script({("carrot", "drop"): [DROP_NO]})
        for _ in range(3):  # grab, verify grab ok, verify drop fails
            question = machine.tick(0)
            if question:
                machine.on_verdict(script(None, question, ""), 0)
            machine.forwarded()
        self.assertEqual((machine.state, machine.item.name, machine.attempts), (GRAB, "carrot", 1))
        self.assertEqual(machine.instruction, "Pick up the carrot.")
        self.assertTrue(any(e["event"] == "recovery_not_built" for e in machine.drain_events()))

    def test_mixed_failures_share_one_attempt_budget(self):
        machine = Machine(FAST, 0)
        drive(machine, Script({("carrot", "grab"): [GRAB_NO], ("carrot", "drop"): [DROP_NO, DROP_NO]}))
        self.assertEqual(machine.placed, ["carton", "tangerine", "carrot"])

    def test_no_verdict_is_not_an_attempt(self):
        machine = Machine(FAST, 0)
        bad = [RuntimeError("timeout"), "not a dict", {"holding": "yes", "is_target": True},
               {"is_target": True}, None]
        script = Script({("carrot", "grab"): bad * 2})
        for _ in range(20):
            question = machine.tick(0)
            if question:
                try:
                    verdict = script(None, question, "")
                except RuntimeError:
                    verdict = None
                machine.on_verdict(verdict, 0)
            machine.forwarded()
            if machine.state == DROP:
                break
        self.assertEqual((machine.state, machine.item.name, machine.attempts), (DROP, "carrot", 0))
        self.assertEqual(len(script.questions), 11)

    def test_time_budget_forces_done_from_any_state(self):
        machine = Machine(FAST, 100)
        self.assertIsNone(machine.tick(100))
        machine.forwarded()
        self.assertIsNotNone(machine.tick(200))  # VERIFY_GRAB
        machine.on_verdict(GRAB_OK, 100 + 420)  # verdict applied, then budget hit
        self.assertEqual((machine.state, machine.done_reason), (DONE, "time_budget"))
        self.assertEqual(machine.instruction, "Return both arms to the home position.")
        machine = Machine(FAST, 0)
        self.assertIsNone(machine.tick(419.9))
        machine.tick(420)
        self.assertEqual(machine.state, DONE)


# ---- relay: fake WALLX + fake client + scripted LLM ----------------------------

ACTION = pack({"embodiment": "quanta_x1_pro/gripper-g",
               "actions": {a: {"position": np.ones((32, 3)), "rotation_rpy": np.zeros((32, 3)),
                               "gripper": np.zeros(32)} for a in ("left_ee", "right_ee")},
               "server_timing": {"infer_ms": 25}})
METADATA = pack({"features": {}, "model": "test-vla"})


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def run_session(self, relay_kwargs, goals, headers=None):
        """Fake client sends one observation per goal entry; returns what WALLX saw."""
        received, logs = [], []

        async def wallx(ws):
            await ws.send(METADATA)
            async for frame in ws:
                received.append(frame)
                await ws.send(ACTION)

        async with serve(wallx, "127.0.0.1", 0) as upstream:
            uri = f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}"
            relay = OrangeRelay(uri, headers or {}, log=logs.append, **relay_kwargs)
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                local = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                for connection in goals:
                    async with connect(local) as client:
                        self.assertEqual(await client.recv(), METADATA)
                        for goal in connection:
                            await client.send(pack(observation(goal)))
                            self.assertEqual(await asyncio.wait_for(client.recv(), 5), ACTION)
        return received, logs

    async def test_embodiment_forwarded_untouched(self):
        obs = observation()
        raw = pack(obs)
        received, logs = [], []

        async def wallx(ws):
            await ws.send(METADATA)
            async for frame in ws:
                received.append(unpack(frame)); await ws.send(ACTION)

        async with serve(wallx, "127.0.0.1", 0) as up:
            relay = OrangeRelay(f"ws://127.0.0.1:{up.sockets[0].getsockname()[1]}", {},
                                config=FAST, verifier=Script(), log=logs.append)
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                async with connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as c:
                    await c.recv(); await c.send(raw)
                    self.assertEqual(await asyncio.wait_for(c.recv(), 5), ACTION)
        self.assertEqual(received[0]["embodiment"], "teletwin_desktop/gripper-g")
        self.assertFalse(any(r["event"] == "harness_failed" for r in logs))

    async def test_other_embodiments_are_rejected(self):
        for bad in ("quanta_x1_pro/gripper-g", "quanta_x1_pro", None):
            obs = observation(); obs["embodiment"] = bad
            logs = []
            relay = OrangeRelay("ws://127.0.0.1:1", {}, log=logs.append)
            with self.assertRaises(ValueError):
                from orange_harness import validate_request
                validate_request(obs)

    async def test_passthrough_is_byte_exact(self):
        raw = pack(observation())
        received, _ = await self.run_session({}, [[GOAL, GOAL]])
        self.assertEqual(received, [raw, raw])

    async def test_happy_path_forwards_unchanged_and_never_leaks_goal(self):
        script = Script()
        received, logs = await self.run_session(
            {"config": FAST, "verifier": script}, [[GOAL] * 9])
        sent = [unpack(f) for f in received]
        self.assertEqual([o["instruction"] for o in sent], [
            "Pick up the carrot.", "Put the carrot in the basket.",
            "Pick up the carton.", "Put the carton in the basket.",
            "Pick up the tangerine.", "Put the tangerine in the basket.",
        ] + ["Return both arms to the home position."] * 3)
        for obs in sent:
            self.assertNotEqual(obs["instruction"], GOAL)
            self.assertEqual(obs["observation"], observation()["observation"])
            self.assertEqual(obs["control_context"], observation()["control_context"])
        self.assertEqual(len(script.questions), 6)
        events = [r["event"] for r in logs]
        self.assertIn("verify_start", events)
        self.assertIn("verify_result", events)
        done = [r for r in logs if r.get("to") == DONE]
        self.assertEqual(done[0]["reason"], "drop_verified")
        last = [r for r in logs if r["event"] == "chunk"][-1]
        self.assertEqual(last["placed"], ["carrot", "carton", "tangerine"])
        for record in logs:
            json.dumps(record)  # every log record is one JSON line
            self.assertNotIn("aW1hZ2U=", json.dumps(record))

    async def test_skip_and_retry_over_the_relay(self):
        script = Script({("carton", "grab"): [GRAB_NO] * 3})
        received, logs = await self.run_session(
            {"config": FAST, "verifier": script}, [[GOAL] * 16])
        sent = [unpack(f)["instruction"] for f in received]
        self.assertEqual(sent[-1], "Return both arms to the home position.")
        self.assertEqual(sent.count("Put the carton in the basket."), 1)
        self.assertTrue(any(r.get("event") == "retry_skipped" for r in logs))

    async def test_llm_failures_do_not_close_the_session(self):
        script = Script({("carrot", "grab"): [RuntimeError("boom"), "garbage", {"holding": 1}]})
        received, logs = await self.run_session(
            {"config": FAST, "verifier": script}, [[GOAL] * 8])
        sent = [unpack(f)["instruction"] for f in received]
        self.assertEqual(len(sent), 8)
        self.assertEqual(sent[:4], ["Pick up the carrot."] * 4)
        self.assertEqual(sent[4], "Put the carrot in the basket.")
        self.assertEqual([r["error"] for r in logs if r["event"] == "verify_error"],
                         ["RuntimeError", "non-dict verdict"])
        self.assertFalse(any(r["event"] == "harness_failed" for r in logs))

    async def test_time_budget_forces_done(self):
        clock = Clock()
        script = Script()

        def slow_verifier(*args):
            clock.now += 500  # the budget runs out while the verify is pending
            return script(*args)

        received, logs = await self.run_session(
            {"config": FAST, "verifier": slow_verifier, "clock": clock}, [[GOAL] * 4])
        sent = [unpack(f)["instruction"] for f in received]
        self.assertEqual(sent, ["Pick up the carrot."] + ["Return both arms to the home position."] * 3)
        self.assertEqual([r["reason"] for r in logs if r.get("to") == DONE], ["time_budget"])

    async def test_new_goal_or_connection_starts_a_fresh_machine(self):
        received, logs = await self.run_session(
            {"config": FAST, "verifier": Script()},
            [[GOAL, GOAL, "Another case."], [GOAL]])
        sent = [unpack(f)["instruction"] for f in received]
        self.assertEqual(sent, ["Pick up the carrot.", "Put the carrot in the basket.",
                                "Pick up the carrot.", "Pick up the carrot."])
        self.assertEqual([r["client_goal"] for r in logs if r["event"] == "session_start"],
                         [GOAL, "Another case.", GOAL])
        self.assertNotIn("Another case.", sent)

    async def test_upstream_timeout_closes_instead_of_replaying(self):
        release = asyncio.Event()

        async def wallx(ws):
            await ws.send(METADATA)
            await ws.recv()
            await release.wait()

        async with serve(wallx, "127.0.0.1", 0) as upstream:
            uri = f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}"
            relay = OrangeRelay(uri, {}, config=FAST, verifier=Script(), timeout=0.1, log=lambda r: None)
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                async with connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as client:
                    await client.recv()
                    await client.send(pack(observation()))
                    with self.assertRaises(ConnectionClosedError) as caught:
                        await asyncio.wait_for(client.recv(), 2)
                    self.assertEqual(caught.exception.rcvd.code, 1011)
                release.set()


# ---- vision client ---------------------------------------------------------------

class VisionClientTests(unittest.TestCase):
    def test_parse_tolerates_prose_and_fences(self):
        self.assertEqual(parse_json_object('{"holding": true}'), {"holding": True})
        self.assertEqual(parse_json_object('Sure!\n```json\n{"holding": false, "x": {"y": 1}}\n```'),
                         {"holding": False, "x": {"y": 1}})
        self.assertEqual(parse_json_object('I see {stuff}. {"in_basket": true}'), {"in_basket": True})
        for bad in ("", "no json here", "[1, 2]", "{broken"):
            with self.assertRaises(ValueError):
                parse_json_object(bad)

    def test_http_contract(self):
        captured, replies = [], ['Here you go:\n```json\n{"holding": true, "is_target": true}\n```',
                                 "I cannot tell."]

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append((self.headers.get("Authorization"),
                                 json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                body = json.dumps({"choices": [{"message": {"content": replies.pop(0)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = VisionClient(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                                  "test-vision", "agent-test-key")
            images = observation()["observation"]["images"]
            question = CONFIG.items[0].grab_question
            self.assertEqual(client.ask(images, question, CONFIG.system_prompt), GRAB_OK)
            auth, body = captured[0]
            self.assertEqual(auth, "Bearer agent-test-key")
            self.assertEqual(body["model"], "test-vision")
            self.assertEqual(body["messages"][0], {"role": "system", "content": CONFIG.system_prompt})
            content = body["messages"][1]["content"]
            self.assertEqual(content[0], {"type": "text", "text": question})
            urls = [p["image_url"]["url"] for p in content if p["type"] == "image_url"]
            self.assertEqual(urls, ["data:image/jpeg;base64,aW1hZ2U="] * 3)
            with self.assertRaises(ValueError):
                client.ask(images, question, CONFIG.system_prompt)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
