"""Offline tests for the flower harness. No robot, no real endpoints, no keys."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from harness import pack, unpack
from flower_harness import (DEFAULT_TASK, FALLBACK_BETA, ClaudeVision, FlowerRelay, image_blocks,
                            load_settings, load_task)
from flower_machine import DONE, Machine, TaskConfig, fallback_plan, validate_counts, validate_plan

GOAL = "Put both blue flowers in the vase"
CONFIG = load_task(DEFAULT_TASK)
HOME = CONFIG.done_instruction
# GRAB_FLOWER lasts 2 requests, INSERT_STEM 1.
TINY = replace(CONFIG, steps=tuple(replace(s, requests=2 if i == 0 else 1)
                                   for i, s in enumerate(CONFIG.steps)))
NO_RETRY = replace(TINY, retries_per_action=0)


def cycle(color, config=TINY):
    """The instructions one action forwards."""
    return [config.instruction(s, color) for s in config.steps for _ in range(s.requests)]


def counts(blue=0, white=0):
    return {"blue": blue, "white": white}


def drive(machine, answers, requests):
    """Drive the machine like the relay does; return the forwarded instructions."""
    sent, answers = [], list(answers)
    for _ in range(requests):
        if machine.awaiting_check:
            machine.on_vase_counts(answers.pop(0))
        sent.append(machine.next_instruction())
    return sent


class FakeVision:
    """Scripted planner and checker: answers are consumed in order; an Exception is raised."""

    def __init__(self, plans=(), checks=()):
        self.plans, self.checks = list(plans), list(checks)
        self.planned = []  # instructions the planner saw

    def plan(self, images, instruction):
        self.planned.append(instruction)
        return self._next(self.plans)

    def vase_counts(self, images):
        return self._next(self.checks)

    @staticmethod
    def _next(answers):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


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
                "print(sorted(m for m in ('asyncio', 'websockets', 'urllib.request', 'socket', "
                "'anthropic') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True, cwd=str(DEFAULT_TASK.parent)).stdout.strip()
        self.assertEqual(out, "[]")

    def test_config_is_data(self):
        self.assertEqual([s.state for s in CONFIG.steps], ["GRAB_FLOWER", "INSERT_STEM"])
        self.assertEqual(CONFIG.colors, ("blue", "white"))
        self.assertTrue(all("{color}" in s.instruction for s in CONFIG.steps))
        self.assertIn("{instruction}", CONFIG.planner_question)

    def test_config_rejects_bad_values(self):
        good = json.loads(DEFAULT_TASK.read_text())
        step = good["steps"][0]
        for bad in ({"steps": []}, {"check_question": ""}, {"planner_system_prompt": " "},
                    {"steps": [{**step, "requests": 0}]},
                    {"steps": [{**step, "requests": True}]},
                    {"steps": [step, step]},
                    {"steps": [{**step, "state": DONE}]},
                    {"steps": [{**step, "instruction": " "}]},
                    {"done_instruction": ""}, {"done_requests": 0}, {"done_requests": 1.5},
                    {"colors": []}, {"colors": ["blue", "blue"]}, {"colors": ["blue", ""]},
                    {"max_actions": 0}, {"max_actions": 2.0},
                    {"retries_per_action": -1}, {"retries_per_action": True},
                    {"planner_question": "No placeholder"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                TaskConfig.from_dict({**good, **bad})

    def test_runs_the_plan_in_order_with_the_color_filled_in(self):
        machine = Machine(TINY, ["blue", "white"])
        self.assertEqual((machine.state, machine.queued), ("GRAB_FLOWER", ["white"]))
        sent = drive(machine, [counts(blue=1), counts(blue=1, white=1)], 8)
        self.assertEqual(sent, cycle("blue") + cycle("white") + [HOME] * 2)
        self.assertTrue(all("{color}" not in s for s in sent))
        self.assertIn("Grab the blue flower", sent[0])
        self.assertIn("white flower stem", sent[5])
        self.assertEqual((machine.state, machine.started, machine.in_vase),
                         (DONE, 2, counts(blue=1, white=1)))

    def test_empty_plan_goes_home(self):
        machine = Machine(TINY, [])
        self.assertEqual(machine.state, DONE)
        self.assertEqual([machine.next_instruction() for _ in range(2)], [HOME, HOME])

    def test_finished_after_done_requests_home_chunks(self):
        self.assertEqual(CONFIG.done_requests, 3)
        machine = Machine(NO_RETRY, ["blue"])
        finished = []
        for _ in range(6):
            machine.next_instruction()
            finished.append(machine.finished)
        self.assertEqual(finished, [False] * 5 + [True])  # 3 action chunks, then 3 home

    def test_failed_placement_is_retried_once(self):
        machine = Machine(TINY, ["blue"])
        drive(machine, [], 3)
        self.assertTrue(machine.awaiting_check)
        self.assertEqual(machine.on_vase_counts(counts()), "retry")
        self.assertEqual((machine.action.color, machine.action.attempt), ("blue", 1))
        # The retry is the last chance and nothing else is blue: no check, then home.
        self.assertEqual(drive(machine, [], 4), cycle("blue") + [HOME])

    def test_gives_up_after_the_retry(self):
        machine = Machine(TINY, ["blue", "blue"])
        outcomes = []
        for _ in range(10):  # three 3-request actions; the third check is due on the 10th
            if machine.awaiting_check:
                outcomes.append(machine.on_vase_counts(counts(blue=1 if len(outcomes) == 2 else 0)))
            machine.next_instruction()
        self.assertEqual(outcomes, ["retry", "failed", "placed"])
        self.assertEqual((machine.next_instruction(), machine.started), (HOME, 3))

    def test_success_means_the_count_went_up(self):
        machine = Machine(TINY, ["blue", "blue"])
        sent = drive(machine, [counts(blue=1), counts(blue=1)], 10)  # 2nd blue not in -> retry
        self.assertEqual(sent, cycle("blue") * 3 + [HOME])

    def test_unusable_counts_count_as_success(self):
        for bad in (None, "garbage", {"white": "2"}, {"white": True}, {"white": -1}, {}):
            with self.subTest(bad=bad):
                machine = Machine(TINY, ["white", "white"])
                drive(machine, [], 3)
                self.assertEqual(machine.on_vase_counts(bad), "assumed")
                self.assertEqual(machine.in_vase["white"], 1)
                drive(machine, [], 3)
                self.assertEqual(machine.on_vase_counts(counts(white=2)), "placed")

    def test_no_check_when_it_could_not_change_anything(self):
        machine = Machine(NO_RETRY, ["blue", "white"])
        self.assertEqual(drive(machine, [], 7), cycle("blue") + cycle("white") + [HOME])

    def test_check_must_come_first_and_only_when_due(self):
        machine = Machine(TINY, ["blue"])
        with self.assertRaises(RuntimeError):
            machine.on_vase_counts(counts())
        drive(machine, [], 3)
        with self.assertRaises(RuntimeError):
            machine.next_instruction()

    def test_plan_is_capped_and_colors_checked(self):
        self.assertEqual(len(Machine(TINY, ["blue"] * 6).queued), TINY.max_actions - 1)
        with self.assertRaises(ValueError):
            Machine(TINY, ["red"])

    def test_default_config_lengths(self):
        sent = drive(Machine(CONFIG, ["white"]), [counts(white=1)], sum(s.requests for s in CONFIG.steps) + 1)
        grab, insert = (CONFIG.instruction(s, "white") for s in CONFIG.steps)
        self.assertEqual(sent[:CONFIG.steps[0].requests], [grab] * CONFIG.steps[0].requests)
        self.assertEqual(sent.count(insert), CONFIG.steps[1].requests)
        self.assertEqual(sent[-2:], [insert, HOME])


class PlanTests(unittest.TestCase):
    def test_validate_plan(self):
        self.assertEqual(validate_plan({"actions": ["blue", "white"]}, TINY), ["blue", "white"])
        self.assertEqual(validate_plan({"actions": []}, TINY), [])
        self.assertEqual(validate_plan({"actions": ["blue"] * 6}, TINY), ["blue"] * 4)
        for bad in (None, "garbage", [], {}, {"actions": "blue"}, {"actions": ["red"]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_plan(bad, TINY)

    def test_validate_counts(self):
        self.assertEqual(validate_counts({"blue": 1, "white": 0, "extra": 3}, TINY), counts(blue=1))
        for bad in (None, {"blue": 1}, {"blue": -1, "white": 0}, {"blue": True, "white": 0},
                    {"blue": "1", "white": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_counts(bad, TINY)

    def test_keyword_fallback_on_the_test_cases(self):
        cases = {
            "Put the blue flower in the vase": ["blue"],
            "Put both white flowers in the vase; leave the blue ones alone": ["white", "white"],
            "Put both blue flowers and one white flower in the vase": ["blue", "blue", "white"],
            "Put all the flowers in the vase": ["blue", "blue", "white", "white"],
            "Put both blue flowers in the vase": ["blue", "blue"],
            "Put the blue flowers in the vase; leave everything else alone": ["blue", "blue"],
            "Put the roses in the vase": ["blue", "blue"],
            "Put one peony in the vase": ["white"],
            "Put the white flowers but not the blue ones in the vase": ["white", "white"],
            "Put two flowers in the vase": ["blue", "blue"],
            "Do a little dance": ["blue"],
        }
        for instruction, plan in cases.items():
            with self.subTest(instruction=instruction):
                self.assertEqual(fallback_plan(instruction, TINY), plan)


# ---- relay: fake WALLX + fake client --------------------------------------------

ACTION = pack({"embodiment": "teletwin_desktop/gripper-g",
               "actions": {a: {"position": np.ones((32, 3)), "rotation_rpy": np.zeros((32, 3)),
                               "gripper": np.zeros(32)} for a in ("left_ee", "right_ee")},
               "server_timing": {"infer_ms": 25}})
METADATA = pack({"features": {}, "model": "test-vla"})


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def run_session(self, connections, vision, config=TINY, embodiment="teletwin_desktop/gripper-g"):
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
            relay = FlowerRelay(uri, {}, config, vision, log=logs.append)
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                local = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                for goals in connections:
                    async with connect(local) as client:
                        self.assertEqual(await client.recv(), METADATA)
                        for goal in goals:
                            await client.send(pack(observation(goal, embodiment)))
                            self.assertEqual(await asyncio.wait_for(client.recv(), 5), ACTION)
        return [o["instruction"] for o in received], received, logs

    async def test_full_run_plans_then_forwards_only_the_instruction_change(self):
        goal = "Put both white flowers in the vase; leave the blue ones alone"
        vision = FakeVision([{"actions": ["white", "white"]}], [counts(white=1), counts(white=2)])
        sent, received, logs = await self.run_session([[goal] * 8], vision)
        self.assertEqual(sent, cycle("white") * 2 + [HOME] * 2)
        self.assertEqual(vision.planned, [goal])
        for obs in received:
            self.assertEqual(obs["observation"], observation()["observation"])
            self.assertEqual(obs["control_context"], observation()["control_context"])
        plans = [r for r in logs if r["event"] == "plan"]
        self.assertEqual([(p["source"], p["plan"]) for p in plans], [("claude", ["white", "white"])])
        self.assertEqual([r["outcome"] for r in logs if r["event"] == "vase_check"],
                         ["placed", "placed"])
        states = [(r["state"], r["action"], r["color"]) for r in logs if r["event"] == "state"]
        self.assertEqual(states, [("GRAB_FLOWER", 1, "white"), ("INSERT_STEM", 1, "white"),
                                  ("GRAB_FLOWER", 2, "white"), ("INSERT_STEM", 2, "white"),
                                  (DONE, 2, None)])
        for record in logs:
            self.assertNotIn("aW1hZ2U=", json.dumps(record))  # one JSON line, no images

    async def test_failed_placement_runs_the_action_again(self):
        vision = FakeVision([{"actions": ["blue"]}], [counts()])
        sent, _, logs = await self.run_session([[GOAL] * 7], vision)
        self.assertEqual(sent, cycle("blue") * 2 + [HOME])
        check = next(r for r in logs if r["event"] == "vase_check")
        self.assertEqual((check["outcome"], check["color"], check["attempt"]), ("retry", "blue", 0))

    async def test_planner_failures_fall_back_to_keywords(self):
        vision = FakeVision([RuntimeError("boom"), {"actions": ["red"]}],
                            [counts(blue=1), counts(blue=2)])
        sent, _, logs = await self.run_session([[GOAL] * 7], vision)
        self.assertEqual(sent, cycle("blue") * 2 + [HOME])
        plans = [r for r in logs if r["event"] == "plan"]
        self.assertEqual([(p["source"], p.get("attempt"), p.get("error")) for p in plans],
                         [("claude", 1, "RuntimeError"), ("claude", 2, "ValueError"),
                          ("keyword_fallback", None, None)])
        self.assertEqual(plans[-1]["plan"], ["blue", "blue"])

    async def test_planner_retry_can_succeed_and_an_empty_plan_goes_home(self):
        vision = FakeVision([TimeoutError(), {"actions": []}])
        sent, _, _ = await self.run_session([[GOAL] * 2], vision)
        self.assertEqual(sent, [HOME, HOME])

    async def test_vision_check_failures_do_not_close_the_session(self):
        vision = FakeVision([{"actions": ["blue"]}], [RuntimeError("boom")])
        sent, _, logs = await self.run_session([[GOAL] * 4], vision)
        self.assertEqual(sent, cycle("blue") + [HOME])
        check = next(r for r in logs if r["event"] == "vase_check")
        self.assertEqual((check["error"], check["counts"], check["outcome"]),
                         ("RuntimeError", None, "assumed"))
        self.assertFalse(any(r["event"] == "harness_failed" for r in logs))

    async def test_new_goal_or_connection_starts_a_fresh_plan(self):
        vision = FakeVision([{"actions": ["blue"]}, {"actions": ["white"]}, {"actions": ["blue"]}])
        sent, _, logs = await self.run_session([[GOAL, GOAL, GOAL, "Another case."], [GOAL]], vision)
        grab = TINY.steps[0]
        self.assertEqual(sent, cycle("blue") + [TINY.instruction(grab, "white"),
                                                TINY.instruction(grab, "blue")])
        self.assertEqual(vision.planned, [GOAL, "Another case.", GOAL])
        self.assertEqual([r["client_goal"] for r in logs if r["event"] == "session_start"],
                         [GOAL, "Another case.", GOAL])

    async def test_closes_and_signals_exit_after_the_done_chunks(self):
        logs = []

        async def wallx(ws):
            await ws.send(METADATA)
            async for _ in ws:
                await ws.send(ACTION)

        async with serve(wallx, "127.0.0.1", 0) as upstream:
            relay = FlowerRelay(f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}", {},
                                NO_RETRY, FakeVision([{"actions": ["blue"]}]), log=logs.append)
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                async with connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as client:
                    await client.recv()
                    for _ in range(6):  # 3 action chunks + 3 DONE chunks
                        self.assertFalse(relay.finished.is_set())
                        await client.send(pack(observation()))
                        self.assertEqual(await asyncio.wait_for(client.recv(), 5), ACTION)
                    with self.assertRaises(ConnectionClosedOK) as caught:
                        await asyncio.wait_for(client.recv(), 2)
                    self.assertEqual(caught.exception.rcvd.code, 1000)
        self.assertTrue(relay.finished.is_set())
        self.assertEqual(logs[-1]["event"], "finished")
        chunks = [r for r in logs if r["event"] == "chunk"]
        self.assertEqual([c["state"] for c in chunks], ["GRAB_FLOWER"] * 2 + ["INSERT_STEM"] + [DONE] * 3)
        self.assertEqual((chunks[0]["gripper"], chunks[0]["gripper_cmd"]),
                         ({"left": 0.5, "right": 0.5}, {"left": [0.0, 0.0], "right": [0.0, 0.0]}))

    async def test_wrong_embodiment_closes_with_diagnostics(self):
        logs = []

        async def wallx(ws):
            await ws.send(METADATA)
            await ws.wait_closed()

        async with serve(wallx, "127.0.0.1", 0) as upstream:
            relay = FlowerRelay(f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}", {},
                                TINY, FakeVision(), log=logs.append)
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
                                TINY, FakeVision([{"actions": ["blue"]}]), timeout=0.1,
                                log=lambda r: None)
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                async with connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as client:
                    await client.recv()
                    await client.send(pack(observation()))
                    with self.assertRaises(ConnectionClosedError) as caught:
                        await asyncio.wait_for(client.recv(), 2)
                    self.assertEqual(caught.exception.rcvd.code, 1011)
                release.set()


# ---- Claude client: request shape, no network -----------------------------------

class FakeClient:
    """Stands in for anthropic.Anthropic: records beta.messages.create kwargs."""

    def __init__(self, text='{"actions": ["blue"]}', stop_reason="end_turn"):
        content = [SimpleNamespace(type="thinking", thinking="")]
        if text is not None:
            content.append(SimpleNamespace(type="text", text=text))
        self.response = SimpleNamespace(stop_reason=stop_reason, content=content)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class ClaudeVisionTests(unittest.TestCase):
    IMAGES = observation()["observation"]["images"]

    def test_plan_request(self):
        client = FakeClient()
        vision = ClaudeVision(client, CONFIG)
        self.assertEqual(vision.plan(self.IMAGES, GOAL), {"actions": ["blue"]})
        (call,) = client.calls
        self.assertEqual(call["model"], "claude-sonnet-5-5")
        self.assertEqual((call["betas"], call["fallbacks"]), ([FALLBACK_BETA], "default"))
        self.assertEqual(call["system"], CONFIG.planner_system_prompt)
        self.assertEqual(call["output_config"]["effort"], "low")
        schema = call["output_config"]["format"]["schema"]
        self.assertEqual(schema["properties"]["actions"]["items"]["enum"], ["blue", "white"])
        content = call["messages"][0]["content"]
        self.assertEqual([b["source"]["data"] for b in content if b["type"] == "image"],
                         ["aW1hZ2U="] * 3)
        self.assertEqual(content[-1], {"type": "text", "text": f"Operator instruction: {GOAL}"})

    def test_vase_counts_request(self):
        client = FakeClient('{"blue": 1, "white": 0}')
        self.assertEqual(ClaudeVision(client, CONFIG, effort="medium").vase_counts(self.IMAGES),
                         counts(blue=1))
        (call,) = client.calls
        self.assertEqual(call["system"], CONFIG.check_system_prompt)
        self.assertEqual(call["output_config"]["effort"], "medium")
        self.assertEqual(call["output_config"]["format"]["schema"]["required"], ["blue", "white"])

    def test_refusal_or_missing_text_raises(self):
        for client in (FakeClient(stop_reason="refusal"), FakeClient(stop_reason="max_tokens"),
                       FakeClient(text=None)):
            with self.assertRaises(ValueError):
                ClaudeVision(client, CONFIG).plan(self.IMAGES, GOAL)

    def test_image_blocks_accept_base64_data_urls_and_bytes(self):
        blocks = image_blocks({"front": "data:image/png;base64,QUJD", "left_wrist": b"ABC",
                               "right_wrist": "QUJD"})
        sources = [b["source"] for b in blocks if b["type"] == "image"]
        self.assertEqual([(s["media_type"], s["data"]) for s in sources],
                         [("image/png", "QUJD"), ("image/jpeg", "QUJD"), ("image/jpeg", "QUJD")])
        self.assertEqual([b["text"] for b in blocks if b["type"] == "text"],
                         ["front camera:", "left_wrist camera:", "right_wrist camera:"])
        with self.assertRaises(ValueError):
            image_blocks({"front": "QUJD", "left_wrist": ""})


class SettingsTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "local_settings.py")
            path.write_text(text)
            path.chmod(0o600)
            return load_settings(path)

    def test_defaults_and_placeholder_key(self):
        settings = self.load('WALLX_ENDPOINT = "ws://x"\nANTHROPIC_API_KEY = "PASTE_KEY"\n')
        self.assertEqual((settings["ANTHROPIC_API_KEY"], settings["ANTHROPIC_MODEL"],
                          settings["ANTHROPIC_EFFORT"]), (None, "claude-sonnet-5-5", "low"))

    def test_bad_settings_exit(self):
        for text in ('', 'WALLX_ENDPOINT = "ws://x"\nANTHROPIC_EFFORT = "minimal"\n',
                     'WALLX_ENDPOINT = "ws://x"\nANTHROPIC_API_KEY = 5\n'):
            with self.subTest(text=text), self.assertRaises(SystemExit):
                self.load(text)


if __name__ == "__main__":
    unittest.main()
