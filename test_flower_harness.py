"""Offline tests for the flower harness. No robot, no real endpoints, no keys."""

import asyncio
import io
import json
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError

from harness import pack, unpack
from flower_harness import CAMERAS, CLASSIFY_CAMERAS, DEFAULT_TASK, FlowerRelay, VisionClient, load_task, parse_json_object
from flower_machine import DONE, NONE, Machine, TaskConfig

GOAL = "Put the blue flowers in the bottle."
CONFIG = load_task(DEFAULT_TASK)  # the real file; only checked to load, so tuning it never breaks tests
# A fixed test task: FIRST lasts 2 requests, the others 1; re-classify every 2;
# no match -> OPEN (1 request), then back one step.
TINY = TaskConfig.from_dict({
    "steps": [{"state": state, "instruction": f"Do {state.lower()}.", "requests": 2 if i == 0 else 1}
              for i, state in enumerate(["FIRST", "SECOND", "THIRD"])],
    "scenes": [{"id": "S_END", "description": "Finished.", "next": DONE},
               {"id": "S_3", "description": "Needs third.", "next": "THIRD"},
               {"id": "S_2", "description": "Needs second.", "next": "SECOND"},
               {"id": "S_1", "description": "Needs first.", "next": "FIRST"}],
    "unclear": {"state": "OPEN", "instruction": "Open.", "requests": 1},
    "reclassify_every": 2,
    "done_instruction": "Go home.",
    "system_prompt": "Three cameras. Answer with one JSON object.",
})
A, B, C = (s.instruction for s in TINY.steps)
OPEN = TINY.unclear.instruction
HOME = TINY.done_instruction


def scene(scene_id, reason="test"):
    return {"reason": reason, "scene": scene_id}


class Script:
    """Scripted vision model: answers are consumed in order (an Exception is raised);
    once they run out every scene is S_END."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, images, question, system_prompt, cameras):
        self.calls.append((question, system_prompt, tuple(cameras)))
        answer = self.answers.pop(0) if self.answers else scene("S_END")
        if isinstance(answer, Exception):
            raise answer
        return answer


def drive(machine, requests, scenes=()):
    """Drive the machine like the relay does; return the forwarded instructions.
    Every classification after the given scenes run out is S_END."""
    sent, scenes = [], list(scenes)
    for _ in range(requests):
        if machine.classify_due:
            machine.on_scene(scenes.pop(0) if scenes else "S_END")
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

    def test_real_task_file_loads(self):
        self.assertTrue(CONFIG.steps)
        self.assertIn(DONE, [s.next for s in CONFIG.scenes])  # some scene must end the task
        for s in CONFIG.scenes:
            self.assertIn(f"- {s.id}: {s.description}", CONFIG.classify_question)

    def test_classify_question_lists_scenes_in_priority_order(self):
        question = TINY.classify_question
        positions = [question.index(f"- {s}:") for s in ("S_END", "S_3", "S_2", "S_1", NONE)]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('"scene"', question)

    def test_config_rejects_bad_values(self):
        good = json.loads(DEFAULT_TASK.read_text())
        step, scene_ = good["steps"][0], good["scenes"][0]
        for bad in ({"steps": []}, {"scenes": []},
                    {"unclear": {**good["unclear"], "requests": 0}},
                    {"unclear": {**good["unclear"], "state": step["state"]}},
                    {"unclear": {**good["unclear"], "instruction": ""}},
                    {"scenes": [{**scene_, "next": good["unclear"]["state"]}]},
                    {"reclassify_every": 0}, {"system_prompt": ""},
                    {"scenes": [{**scene_, "next": "NOWHERE"}]},
                    {"scenes": [{**scene_, "id": NONE}]},
                    {"scenes": [scene_, scene_]},
                    {"scenes": [{**scene_, "description": " "}]},
                    {"steps": [{**step, "requests": 0}]},
                    {"steps": [{**step, "back": "NOWHERE"}]},
                    {"steps": [{**step, "requests": True}]},
                    {"steps": [step, step]},
                    {"steps": [{**step, "state": DONE}]},
                    {"done_instruction": ""}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                TaskConfig.from_dict({**good, **bad})

    def test_scene_picks_the_next_state_in_any_order(self):
        machine = Machine(TINY)
        self.assertEqual((machine.state, machine.classify_due), ("FIRST", False))
        machine.next_instruction(), machine.next_instruction()
        self.assertTrue(machine.classify_due)
        with self.assertRaises(RuntimeError):
            machine.next_instruction()  # the classification must come first
        # FIRST -> SECOND -> THIRD -> back to SECOND (e.g. a dropped flower) -> skip to DONE.
        sent = drive(Machine(TINY), 8, ["S_2", "S_3", "S_2", "S_END"])
        self.assertEqual(sent, [A, A, B, C, B, HOME, HOME, HOME])

    def test_same_scene_keeps_the_step_and_reclassifies_sooner(self):
        sent = drive(Machine(TINY), 6, ["S_1", "S_2"])
        self.assertEqual(sent, [A, A, A, A, B, HOME])  # 2 requests, S_1: 2 more (reclassify_every)

    def test_no_match_opens_grippers_then_steps_back(self):
        for answer in (None, NONE, "MADE_UP"):
            with self.subTest(answer=answer):
                # Unknown in SECOND: open, then FIRST with a full budget, then classify.
                self.assertEqual(drive(Machine(TINY), 7, ["S_2", answer]), [A, A, B, OPEN, A, A, HOME])
        # Unknown in THIRD steps back to SECOND; unknown in FIRST stays on FIRST.
        self.assertEqual(drive(Machine(TINY), 6, ["S_3", None]), [A, A, C, OPEN, B, HOME])
        self.assertEqual(drive(Machine(TINY), 6, [None]), [A, A, OPEN, A, A, HOME])

    def test_step_back_target_overrides_the_previous_step(self):
        steps = [{"state": t.state, "instruction": t.instruction, "requests": t.requests}
                 for t in TINY.steps]
        steps[2]["back"] = "FIRST"
        config = TaskConfig.from_dict({
            "steps": steps, "unclear": {"state": "OPEN", "instruction": OPEN, "requests": 1},
            "scenes": [{"id": s.id, "description": s.description, "next": s.next} for s in TINY.scenes],
            "done_instruction": HOME, "system_prompt": TINY.system_prompt})
        # Unknown in THIRD now goes back to FIRST, not SECOND.
        self.assertEqual(drive(Machine(config), 6, ["S_3", None]), [A, A, C, OPEN, A, A])

    def test_no_classification_while_opening(self):
        machine = Machine(TINY)
        drive(machine, 2)
        machine.on_scene(None)
        self.assertEqual((machine.state, machine.classify_due), ("OPEN", False))
        self.assertEqual(machine.next_instruction(), OPEN)
        self.assertFalse(machine.classify_due)  # straight back to FIRST, no classify in between
        self.assertEqual((machine.next_instruction(), machine.state), (A, "FIRST"))

    def test_real_task_starts_with_the_first_step_for_its_requests(self):
        first = CONFIG.steps[0]
        machine = Machine(CONFIG)
        self.assertEqual([machine.next_instruction() for _ in range(first.requests)],
                         [first.instruction] * first.requests)
        self.assertTrue(machine.classify_due)


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
        ask = Script(scene("S_2"), scene("S_3", "Flower held above the bottle."), scene("S_END"))
        sent, received, logs = await self.run_session([[GOAL] * 6], ask)
        self.assertEqual(sent, [A, A, B, C, HOME, HOME])
        for obs in received:
            self.assertEqual(obs["observation"], observation()["observation"])
            self.assertEqual(obs["control_context"], observation()["control_context"])
        # Every classification asks the scene question about the front camera only.
        self.assertEqual(CLASSIFY_CAMERAS, ("front",))
        self.assertEqual(ask.calls, [(TINY.classify_question, TINY.system_prompt, ("front",))] * 3)
        classified = [(r["from"], r["scene"], r["to"]) for r in logs if r["event"] == "classify"]
        self.assertEqual(classified, [("FIRST", "S_2", "SECOND"), ("SECOND", "S_3", "THIRD"),
                                      ("THIRD", "S_END", DONE)])
        self.assertEqual([r["reason"] for r in logs if r["event"] == "classify"][1],
                         "Flower held above the bottle.")
        self.assertEqual([r["state"] for r in logs if r["event"] == "state"],
                         ["FIRST", "SECOND", "THIRD", DONE])
        for record in logs:
            self.assertNotIn("aW1hZ2U=", json.dumps(record))  # one JSON line, no images

    async def test_vision_failures_open_grippers_without_closing(self):
        ask = Script(scene("S_2"), RuntimeError("boom"), "garbage", {"scene": 3}, scene(NONE))
        sent, _, logs = await self.run_session([[GOAL] * 16], ask)
        # Each failure opens the grippers, then steps back to FIRST with a fresh budget.
        self.assertEqual(sent, [A, A, B] + [OPEN, A, A] * 4 + [HOME])
        classified = [r for r in logs if r["event"] == "classify"]
        self.assertEqual([(r["scene"], r["to"]) for r in classified],
                         [("S_2", "SECOND"), (None, "OPEN"), (None, "OPEN"), (None, "OPEN"),
                          (NONE, "OPEN"), ("S_END", DONE)])
        self.assertEqual(classified[1]["error"], "RuntimeError")
        self.assertFalse(any(r["event"] == "harness_failed" for r in logs))

    async def test_new_goal_or_connection_starts_a_fresh_machine(self):
        ask = Script(scene("S_2"))
        sent, _, logs = await self.run_session([[GOAL, GOAL, GOAL, "Another case."], [GOAL]], ask)
        self.assertEqual(sent, [A, A, B, A, A])
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
    def test_ask_sends_only_the_requested_cameras(self):
        bodies = []

        def urlopen(request, timeout):
            bodies.append(json.loads(request.data))
            reply = {"choices": [{"message": {"content": '```json\n{"completed": true}\n```'}}]}
            return io.BytesIO(json.dumps(reply).encode())

        client = VisionClient("http://vision.invalid/v1/chat/completions", "test-vision")
        images = observation()["observation"]["images"]
        with mock.patch("flower_harness.urllib.request.urlopen", urlopen):
            self.assertEqual(client.ask(images, "Done?", "Front camera.", ("front",)), {"completed": True})
            client.ask(images, "Left?", "Three cameras.")
        names = [[p["text"] for p in b["messages"][1]["content"][1:] if p["type"] == "text"]
                 for b in bodies]
        self.assertEqual(names, [["front"], list(CAMERAS)])

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
