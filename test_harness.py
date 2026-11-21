"""Offline protocol checks. No connection to a robot or the real WALLX API."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError

from harness import Harness, VisionAgent, pack, unpack, validate_request


def observation(goal="Put the spoon in the bowl."):
    return {
        "embodiment": "quanta_x1_pro/gripper-g",
        "normalization": "provided", "instruction": goal,
        "observation": {
            "state": {name: {"position": [0.1, 0.2, 0.3], "rotation_rpy": [0, 0, 0], "gripper": 0.5}
                      for name in ("left_ee", "right_ee")},
            "images": {name: "aW1hZ2U=" for name in ("front", "left_wrist", "right_wrist")},
        },
        "control_context": {"left": [None] * 7, "right": [None] * 7},
    }


class Planner:
    def __init__(self):
        self.calls = []

    def decide(self, obs, goal, previous, history):
        self.calls.append((goal, list(history)))
        return f"next-step-{len(self.calls)}"


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_replans_preserves_observation_and_forwards_reply_bytes(self):
        received = []
        metadata = pack({"features": {}, "model": "test-vla"})
        action = pack({"embodiment": "quanta_x1_pro/gripper-g", "actions": {a: {"position": np.ones((32, 3)), "rotation_rpy": np.zeros((32, 3)), "gripper": np.zeros(32)} for a in ("left_ee", "right_ee")}, "server_timing": {"infer_ms": 25}})

        async def backend(ws):
            self.assertEqual(ws.request.headers["Authorization"], "Bearer wallx-test-key")
            await ws.send(metadata)
            async for frame in ws:
                received.append(unpack(frame))
                await ws.send(action)

        planner = Planner()
        async with serve(backend, "127.0.0.1", 0) as upstream:
            uri = f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}"
            relay = Harness(uri, {"Authorization": "Bearer wallx-test-key"}, planner, agent_every=2)
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                local = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                async with connect(local) as client:
                    self.assertEqual(await client.recv(), metadata)
                    for _ in range(3):
                        await client.send(pack(observation()))
                        self.assertEqual(await client.recv(), action)
                    await client.send(pack(observation("Move the bowl.")))
                    self.assertEqual(await client.recv(), action)
                # A new connection must not inherit the previous case's history.
                async with connect(local) as client:
                    await client.recv()
                    await client.send(pack(observation()))
                    await client.recv()
        self.assertEqual([x["instruction"] for x in received], ["next-step-1", "next-step-1", "next-step-2", "next-step-3", "next-step-4"])
        self.assertEqual(planner.calls[2][1], [])
        self.assertEqual(planner.calls[3][1], [])
        for obs in received:
            self.assertEqual(obs["observation"], observation()["observation"])
            self.assertEqual(obs["control_context"], observation()["control_context"])
            self.assertIsInstance(obs["instruction"], str)

    async def test_passthrough_and_server_error_are_byte_exact(self):
        raw = pack(observation())
        error = pack({"error": "test-model-error"})

        async def backend(ws):
            await ws.send(pack({"model": "mock"}))
            self.assertEqual(await ws.recv(), raw)
            await ws.send(error)

        async with serve(backend, "127.0.0.1", 0) as upstream:
            uri = f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}"
            relay = Harness(uri, {})
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                async with connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as client:
                    await client.recv()
                    await client.send(raw)
                    self.assertEqual(await client.recv(), error)

    async def test_timeout_closes_instead_of_replaying_an_action(self):
        release = asyncio.Event()

        async def backend(ws):
            await ws.send(pack({"model": "mock"}))
            await ws.recv()
            await release.wait()

        async with serve(backend, "127.0.0.1", 0) as upstream:
            uri = f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}"
            relay = Harness(uri, {}, timeout=0.1)
            async with serve(relay.handle, "127.0.0.1", 0) as server:
                async with connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as client:
                    await client.recv()
                    await client.send(pack(observation()))
                    with self.assertRaises(ConnectionClosedError) as caught:
                        await asyncio.wait_for(client.recv(), 2)
                    self.assertEqual(caught.exception.rcvd.code, 1011)
                release.set()


class AgentTests(unittest.TestCase):
    def test_rejects_wrong_schema_rtc_and_nonfinite_state(self):
        for change in (lambda r: r.update(instruction=["goal"]),
                       lambda r: r.update(use_rtc=True),
                       lambda r: r["observation"]["images"].pop("front"),
                       lambda r: r["observation"]["state"]["left_ee"].update(gripper=float("nan"))):
            r = observation(); change(r)
            with self.assertRaises((ValueError, KeyError)):
                validate_request(r)

    def test_vision_http_contract_and_invalid_decision(self):
        captured = []
        decision = {"instruction": "Grasp the spoon.", "reason": "The spoon is on the table."}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append((self.headers.get("Authorization"), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                body = json.dumps({"choices": [{"message": {"content": json.dumps(decision)}}]}).encode()
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
            agent = VisionAgent(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "test-vision", "agent-test-key")
            self.assertEqual(agent.decide(observation(), "goal", "previous", []), "Grasp the spoon.")
            auth, body = captured[0]
            self.assertEqual(auth, "Bearer agent-test-key")
            self.assertEqual(body["model"], "test-vision")
            images = [part for part in body["messages"][1]["content"] if part["type"] == "image_url"]
            self.assertEqual(len(images), 3)
            self.assertTrue(all(x["image_url"]["url"].startswith("data:image/jpeg;base64,") for x in images))
            decision["instruction"] = ""
            with self.assertRaises(ValueError):
                agent.decide(observation(), "goal", "previous", [])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
