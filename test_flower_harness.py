"""Offline tests for the flower harness. No robot, no real endpoints, no keys."""

import asyncio
from dataclasses import replace
import json
import subprocess
import sys
import unittest

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError

from harness import pack, unpack
from flower_harness import DEFAULT_TASK, FlowerRelay, load_task, parse_json_object
from flower_machine import DONE, Machine, TaskConfig

GOAL = "Put the blue flowers in the bottle."
CONFIG = load_task(DEFAULT_TASK)
BOTTLE, FLOWER, INSERT, PLACE = (s.instruction for s in CONFIG.steps)
HOME = CONFIG.done_instruction
# GRAB_BOTTLE lasts 2 requests, every other step 1.
TINY = replace(CONFIG, steps=tuple(replace(s, requests=2 if i == 0 else 1)
                                   for i, s in enumerate(CONFIG.steps)))
CYCLE = [BOTTLE, BOTTLE, FLOWER, INSERT, PLACE]


class Script:
    """Scripted vision model: answers are consumed in order; an Exception is raised."""

    def __init__(self, *answers):
        self.answers, self.questions = list(answers), []

    def __call__(self, images, question, system_prompt):
        self.questions.append(question)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def drive(machine, answers, requests):
    """Drive the machine like the relay does; return the forwarded instructions."""
    sent, answers = [], list(answers)
    for _ in range(requests):
        if machine.cycle_complete:
            machine.on_flowers_left(answers.pop(0))
        sent.append(machine.next_instruction())
    return sent


def observation(goal=GOAL, embodiment="teletwin_desktop/gripper-g"):
    return {
        "embodiment": embodiment,
        "normalization": "provided", "instruction": goal,
        "observation": {
            "state": {name: {"position": [0.1, 0.2, 0.3], "rotation_rpy": [0, 0, 0], "gripper": 0.5}
                      for name in ("left_ee", "right_ee")},
            "images": {name: "aW1hZ2U=" for name in ("front", "left_wrist", "right_wrist")},
        },
        "control_context": {"left": [None] * 7, "right": [None] * 7},
    }


# ---- machine: no asyncio, no network -------------------------------------------

class MachineTests(unittest.TestCase):
    def test_machine_module_imports_no_asyncio_or_network(self):
        code = ("import sys, flower_machine; "
                "print(sorted(m for m in ('asyncio', 'websockets', 'urllib.request', 'socket') "
                "if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True, cwd=str(DEFAULT_TASK.parent)).stdout.strip()
        self.assertEqual(out, "[]")

    def test_config_is_data(self):
        self.assertEqual([s.state for s in CONFIG.steps],
                         ["GRAB_BOTTLE", "GRAB_FLOWER", "INSERT_STEM", "PLACE_BOTTLE"])
        self.assertIn("blue flower by its stem", FLOWER)
        self.assertIn('"blue_flowers_left"', CONFIG.flowers_left_question)

    def test_config_rejects_bad_values(self):
        good = json.loads(DEFAULT_TASK.read_text())
        step = good["steps"][0]
        for bad in ({"steps": []}, {"flowers_left_question": ""},
                    {"steps": [{**step, "requests": 0}]},
                    {"steps": [{**step, "requests": True}]},
                    {"steps": [step, step]},
                    {"steps": [{**step, "state": DONE}]},
                    {"steps": [{**step, "instruction": " "}]},
                    {"done_instruction": ""}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                TaskConfig.from_dict({**good, **bad})

    def test_starts_with_the_bottle_and_cycles_until_no_flowers_left(self):
        machine = Machine(TINY)
        self.assertEqual(machine.state, "GRAB_BOTTLE")
        states = []
        for _ in range(5):
            machine.next_instruction()
            states.append(machine.state)  # `state` names the instruction just returned
        self.assertEqual(states, ["GRAB_BOTTLE", "GRAB_BOTTLE", "GRAB_FLOWER",
                                  "INSERT_STEM", "PLACE_BOTTLE"])
        self.assertTrue(machine.cycle_complete)
        with self.assertRaises(RuntimeError):
            machine.next_instruction()  # the check must come first
        machine = Machine(TINY)
        self.assertEqual(drive(machine, [True, True, False], 18), CYCLE * 3 + [HOME] * 3)
        self.assertEqual((machine.state, machine.flowers_done), (DONE, 3))

    def test_no_answer_runs_another_cycle(self):
        machine = Machine(TINY)
        self.assertEqual(drive(machine, [None, False], 11), CYCLE * 2 + [HOME])

    def test_default_config_lengths(self):
        per_flower = sum(s.requests for s in CONFIG.steps)
        sent = drive(Machine(CONFIG), [False], per_flower + 1)
        self.assertEqual(sent[:CONFIG.steps[0].requests], [BOTTLE] * CONFIG.steps[0].requests)
        self.assertEqual(sent.count(FLOWER), CONFIG.steps[1].requests)
        self.assertEqual(sent[-2:], [PLACE, HOME])


# ---- relay: fake WALLX + fake client --------------------------------------------

ACTION = pack({"embodiment": "teletwin_desktop/gripper-g",
               "actions": {a: {"position": np.ones((32, 3)), "rotation_rpy": np.zeros((32, 3)),
                               "gripper": np.zeros(32)} for a in ("left_ee", "right_ee")},
               "server_timing": {"infer_ms": 25}})
METADATA = pack({"features": {}, "model": "test-vla"})


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def run_session(self, connections, ask, config=TINY, embodiment="teletwin_desktop/gripper-g"):
        """Fake client sends one observation per goal in each connection; returns
        (instructions WALLX saw, observations WALLX saw, log records)."""
        received, logs = [], []

        async def wallx(ws):
            await ws.send(METADATA)
            async for frame in ws:
                received.append(unpack(frame))
                await ws.send(ACTION)

        async with serve(wallx, "127.0.0.1", 0) as upstream:
            uri = f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}"
            relay = FlowerRelay(uri, {}, config, ask, log=logs.append)
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                local = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                for goals in connections:
                    async with connect(local) as client:
                        self.assertEqual(await client.recv(), METADATA)
                        for goal in goals:
                            await client.send(pack(observation(goal, embodiment)))
                            self.assertEqual(await asyncio.wait_for(client.recv(), 5), ACTION)
        return [o["instruction"] for o in received], received, logs

    async def test_full_run_forwards_only_the_instruction_change(self):
        ask = Script({"blue_flowers_left": True}, {"blue_flowers_left": False})
        sent, received, logs = await self.run_session([[GOAL] * 12], ask)
        self.assertEqual(sent, CYCLE * 2 + [HOME] * 2)
        self.assertEqual(ask.questions, [CONFIG.flowers_left_question] * 2)
        for obs in received:
            self.assertEqual(obs["observation"], observation()["observation"])
            self.assertEqual(obs["control_context"], observation()["control_context"])
        states = [(r["state"], r["flowers_done"]) for r in logs if r["event"] == "state"]
        self.assertEqual(states, [("GRAB_BOTTLE", 0), ("GRAB_FLOWER", 0), ("INSERT_STEM", 0),
                                  ("PLACE_BOTTLE", 0), ("GRAB_BOTTLE", 1), ("GRAB_FLOWER", 1),
                                  ("INSERT_STEM", 1), ("PLACE_BOTTLE", 1), (DONE, 2)])
        for record in logs:
            self.assertNotIn("aW1hZ2U=", json.dumps(record))  # one JSON line, no images

    async def test_vision_failures_do_not_close_the_session(self):
        ask = Script(RuntimeError("boom"), "garbage", {"blue_flowers_left": "no"},
                     {"blue_flowers_left": False})
        sent, _, logs = await self.run_session([[GOAL] * 21], ask)
        self.assertEqual(sent, CYCLE * 4 + [HOME])
        checks = [r for r in logs if r["event"] == "flower_check"]
        self.assertEqual([c["blue_flowers_left"] for c in checks], [None, None, None, False])
        self.assertEqual(checks[0]["error"], "RuntimeError")
        self.assertFalse(any(r["event"] == "harness_failed" for r in logs))

    async def test_new_goal_or_connection_starts_a_fresh_machine(self):
        sent, _, logs = await self.run_session([[GOAL, GOAL, GOAL, "Another case."], [GOAL]], Script())
        self.assertEqual(sent, [BOTTLE, BOTTLE, FLOWER, BOTTLE, BOTTLE])
        self.assertEqual([r["client_goal"] for r in logs if r["event"] == "session_start"],
                         [GOAL, "Another case.", GOAL])

    async def test_wrong_embodiment_closes_with_diagnostics(self):
        logs = []

        async def wallx(ws):
            await ws.send(METADATA)
            await ws.wait_closed()

        async with serve(wallx, "127.0.0.1", 0) as upstream:
            relay = FlowerRelay(f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}", {},
                                TINY, Script(), log=logs.append)
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                async with connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as client:
                    await client.recv()
                    await client.send(pack(observation(embodiment="quanta_x1_pro/gripper-g")))
                    with self.assertRaises(ConnectionClosedError) as caught:
                        await asyncio.wait_for(client.recv(), 2)
                    self.assertEqual(caught.exception.rcvd.code, 1011)
        failed = [r for r in logs if r["event"] == "harness_failed"]
        self.assertEqual(failed[0]["embodiment_received"], "'quanta_x1_pro/gripper-g'")

    async def test_upstream_timeout_closes_instead_of_replaying(self):
        release = asyncio.Event()

        async def wallx(ws):
            await ws.send(METADATA)
            await ws.recv()
            await release.wait()

        async with serve(wallx, "127.0.0.1", 0) as upstream:
            relay = FlowerRelay(f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}", {},
                                TINY, Script(), timeout=0.1, log=lambda r: None)
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                async with connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as client:
                    await client.recv()
                    await client.send(pack(observation()))
                    with self.assertRaises(ConnectionClosedError) as caught:
                        await asyncio.wait_for(client.recv(), 2)
                    self.assertEqual(caught.exception.rcvd.code, 1011)
                release.set()


class VisionClientTests(unittest.TestCase):
    def test_parse_tolerates_prose_and_fences(self):
        self.assertEqual(parse_json_object('{"blue_flowers_left": true}'), {"blue_flowers_left": True})
        self.assertEqual(parse_json_object('Sure!\n```json\n{"blue_flowers_left": false}\n```'),
                         {"blue_flowers_left": False})
        self.assertEqual(parse_json_object('I see {stuff}. {"a": {"b": 1}}'), {"a": {"b": 1}})
        for bad in ("", "no json here", "[1, 2]", "{broken"):
            with self.assertRaises(ValueError):
                parse_json_object(bad)


if __name__ == "__main__":
    unittest.main()
