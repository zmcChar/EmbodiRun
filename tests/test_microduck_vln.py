"""CPU tests for the real HTTP boundary and bounded episode execution."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from embodirun.model_services.backends.vvla.http import VvlaHttpClient, VvlaHttpError
from embodirun.model_services.contracts import ImagePayload, PolicyObservation, Session

# Optional integration sources are not installed with EmbodiRun's core.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "integrations/microduck_vln/src"))

from embodirun_microduck.assets import md5, verify_manifest  # noqa: E402
from embodirun_microduck.process import managed_service  # noqa: E402
from embodirun_microduck.protocol import ACTION_SPACE, IMAGE_FIELD, decode_actions  # noqa: E402
from embodirun_microduck.service import ActiveVLNAdapter  # noqa: E402

spec = importlib.util.spec_from_file_location("microduck_example", REPO / "examples/microduck_vln/run_demo.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def chunk(rows):
    import torch

    return SimpleNamespace(
        actions=torch.tensor(rows),
        latency_ms=1.0,
        timing={},
        policy_version=1,
        trace=SimpleNamespace(text="stop", stop_reason="eos", parsed_actions=SimpleNamespace(valid=True)),
    )


class FakeCore:
    def __init__(self):
        self.calls = []
        self.resets = []
        self.policy = SimpleNamespace(collate=lambda obs, ids: obs)

    def execute(self, batch, session_ids):
        self.calls.append((batch, session_ids))
        return [chunk([[0, 0], [-1, 0], [-1, 0]])]

    def reset_sessions(self, keys):
        self.resets.extend(keys)


class HttpBoundaryTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("vvla") is None:
            self.skipTest("Optional EmbodiInfer/Torch dependencies unavailable")
        from vvla.engine.serve.http_server import PolicyHttpService, create_http_server

        self.core = FakeCore()
        service = PolicyHttpService(ActiveVLNAdapter(self.core), token="test-token", max_body_bytes=100000)
        self.server = create_http_server(service, host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.client = VvlaHttpClient(self.url, token="test-token")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def observation(self, session, request_id="request-1", step=0):
        encoded = io.BytesIO()
        Image.fromarray(np.full((2, 2, 3), 127, dtype=np.uint8)).save(encoded, format="PNG")
        return PolicyObservation(
            session.session_id,
            request_id,
            step,
            "Go to the bin",
            {},
            (ImagePayload(IMAGE_FIELD, "image/png", encoded.getvalue()),),
        )

    def test_real_http_recurrence_idempotency_reset_and_close(self):
        self.assertEqual(self.client.capabilities()["adapter"]["action_space"], ACTION_SPACE)
        session = self.client.open_session(robot_id="microduck-sim", action_space=ACTION_SPACE)
        obs = self.observation(session)
        first = self.client.step(obs)
        self.assertEqual(first, self.client.step(obs))
        self.assertEqual(len(self.core.calls), 1)  # Retry must not advance recurrent state.
        pixels = self.core.calls[0][0][0].images
        self.assertAlmostEqual(float(pixels[0, 0, 0, 0]), 127 / 255, places=6)
        self.client.step(self.observation(session, "request-2", 1))
        self.assertEqual(self.core.calls[0][1], self.core.calls[1][1])
        self.assertEqual(decode_actions(first.actions[0].values["rows"]), [(0, 0)])
        reset = self.client.reset(session.session_id, request_id="reset-1")
        self.assertGreater(reset.revision, session.revision)
        self.client.step(self.observation(reset, "request-3"))
        self.client.close(session.session_id)
        with self.assertRaises(VvlaHttpError):
            self.client.step(self.observation(session, "request-4", 1))
        second = self.client.open_session(robot_id="microduck-sim", action_space=ACTION_SPACE)
        self.client.step(self.observation(second, "request-5"))
        self.assertNotEqual(self.core.calls[0][1], self.core.calls[-1][1])
        self.client.close(second.session_id)
        self.assertGreaterEqual(len(self.core.resets), 3)

    def test_unauthorized_request_and_step_gap_never_infer(self):
        with self.assertRaises(VvlaHttpError):
            VvlaHttpClient(self.url).capabilities()
        session = self.client.open_session(robot_id="microduck-sim", action_space=ACTION_SPACE)
        with self.assertRaises(VvlaHttpError):
            self.client.step(self.observation(session, step=3))
        self.assertEqual(self.core.calls, [])
        self.client.close(session.session_id)


class FakeSimulation:
    callback = None
    robot = SimpleNamespace(data=SimpleNamespace(time=0.0))
    mpc_ms = []

    def reset(self, start):
        self.executed = []
        self.path_length = 0.0
        self.physics_steps = 0

    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def execute(self, action):
        self.executed.append(action)
        self.physics_steps += 1
        return {"pose": self.pose()}

    def pose(self):
        return [0.0, 0.0, 0.0]

    def distance(self, goal):
        return 0.5


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.closed = []

    def open_session(self, **kwargs):
        return Session("session-1", 0)

    def step(self, obs):
        return SimpleNamespace(
            action_space=ACTION_SPACE,
            actions=[SimpleNamespace(kind="discrete_chunk", values={"rows": self.rows, "valid": True})],
            timing={"policy_ms": 1.0},
            request_id=obs.request_id,
            session_id=obs.session_id,
            session_revision=0,
        )

    def close(self, session_id):
        self.closed.append(session_id)


class ExecutionTests(unittest.TestCase):
    def run_episode(self, rows, limit=60, hold=1, fail=False):
        args = SimpleNamespace(seed=7, no_video=True, max_steps=limit, success_radius=1.0, success_hold_steps=hold)
        record = {"id": "../untrusted-id", "instruction": "Go to the bin", "start": [0, 0, 0], "goal": [0.5, 0]}
        sim = FakeSimulation()
        client = FakeClient(rows)
        if fail:
            client.step = lambda obs: (_ for _ in ()).throw(TimeoutError("transport timeout"))
        with tempfile.TemporaryDirectory() as folder:
            if fail:
                with self.assertRaises(TimeoutError):
                    runner.run_episode(args, sim, client, record, 0, Path(folder), "")
            else:
                runner.run_episode(args, sim, client, record, 0, Path(folder), "")
            result = json.loads((Path(folder) / "episode_000.json").read_text())
        self.assertEqual(client.closed, ["session-1"])
        return result, sim

    def test_stop_discards_later_actions(self):
        result, sim = self.run_episode([[0, 0], [1, 75], [-1, 0]])
        self.assertEqual(sim.executed, [(0, 0)])
        self.assertTrue(result["success"])

    def test_success_requires_hold_endpoints(self):
        result, _ = self.run_episode([[0, 0], [-1, 0], [-1, 0]], hold=3)
        self.assertFalse(result["success"])
        result, _ = self.run_episode([[1, 25], [1, 25], [0, 0]], hold=3)
        self.assertTrue(result["success"])

    def test_action_cap_truncates_chunk(self):
        result, sim = self.run_episode([[1, 25], [1, 50], [1, 75]], limit=2)
        self.assertEqual(sim.executed, [(1, 25), (1, 50)])
        self.assertEqual(result["termination"], "max_steps")

    def test_invalid_rows_never_move(self):
        for rows in (
            [[1, 999], [-1, 0], [-1, 0]],
            [[-1, 0], [-1, 0], [-1, 0]],
            [[float("nan"), 0], [-1, 0], [-1, 0]],
            [[1.5, 25], [-1, 0], [-1, 0]],
        ):
            with self.assertRaises(ValueError):
                decode_actions(rows)
        result, sim = self.run_episode([[-1, 0], [-1, 0], [-1, 0]])
        self.assertEqual(sim.executed, [])
        self.assertEqual(result["termination"], "invalid_model_output")

    def test_transport_failure_persists_error_and_closes_session(self):
        result, sim = self.run_episode([[0, 0], [-1, 0], [-1, 0]], fail=True)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["session_cleared"])
        self.assertEqual(sim.executed, [])

    def test_integrity_detects_same_size_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            asset = root / "asset"
            asset.write_bytes(b"original")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": {"project/asset": {"md5": md5(asset), "bytes": 8}}}))
            with patch("embodirun_microduck.assets.asset_files", return_value={"project/asset": asset}):
                self.assertEqual(verify_manifest(manifest, {})["files_verified"], 1)
                asset.write_bytes(b"modified")
                with self.assertRaisesRegex(RuntimeError, "MD5 mismatch"):
                    verify_manifest(manifest, {})


class ProcessLifecycleTests(unittest.TestCase):
    def launch_failure(self, code, timeout):
        children = []
        original = subprocess.Popen

        def launch(command, **kwargs):
            child = original([sys.executable, "-c", code], **kwargs)
            children.append(child)
            return child

        args = SimpleNamespace(
            inference_root=Path("."),
            checkpoint=Path("."),
            seed=7,
            max_new_tokens=8,
            max_context=1024,
            sample=False,
            startup_timeout=timeout,
            request_timeout=1.0,
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("embodirun_microduck.process.subprocess.Popen", side_effect=launch),
        ):
            with self.assertRaises((RuntimeError, TimeoutError)), managed_service(args, Path(folder)):
                self.fail("A failed service must not become ready")
            self.assertFalse((Path(folder) / "service.ready.json").exists())
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    def test_startup_failure_reaps_child(self):
        self.launch_failure("raise SystemExit(3)", 5.0)

    def test_startup_timeout_terminates_child(self):
        self.launch_failure("import time; time.sleep(30)", 0.05)


class PublicEntrypointTests(unittest.TestCase):
    def test_public_example_and_service_help_without_gpu_dependencies(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO / "integrations/microduck_vln/src")])
        commands = [
            [str(REPO / "examples/microduck_vln/run_demo.py"), "--help"],
            ["-m", "embodirun_microduck.service", "--help"],
        ]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, *command], env=env, text=True, capture_output=True, timeout=15, check=False
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--checkpoint", result.stdout)

    def test_inference_path_override_and_legacy_alias(self):
        with patch.dict(os.environ, {"MICRODUCK_INFERENCE_ROOT": "public-checkout", "MICRODUCK_VVLA_ROOT": "legacy"}):
            with patch.object(sys, "argv", ["run_demo.py"]):
                self.assertEqual(runner.parse_args().inference_root, Path("public-checkout"))
            for flag in ["--inference-root", "--vvla-root"]:
                with self.subTest(flag=flag), patch.object(sys, "argv", ["run_demo.py", flag, "explicit-checkout"]):
                    self.assertEqual(runner.parse_args().inference_root, Path("explicit-checkout"))


if __name__ == "__main__":
    unittest.main()
