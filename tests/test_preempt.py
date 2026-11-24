"""Preemption: a terminated worker hands its task back at once, and the controller tells preempted workers from failed ones."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tts_retake import backends
from tts_retake.config import load
from tts_retake.daemon import Board, Daemon
from tts_retake.scheduler import Controller

ROOT = Path(__file__).resolve().parents[1]


def job(tmp, extra=""):
    tmp = Path(tmp)
    row = {"id": "d0", "speakers": {"host": {"voice": "a.wav", "text": "Hello."}, "guest": {"voice": "b.wav", "text": "Hi."}},
           "turns": [{"speaker": "host", "text": "one two"}, {"speaker": "guest", "text": "three four"}]}
    (tmp / "in.jsonl").write_text(json.dumps(row) + "\n")
    env = '{PYTHONPATH = "' + str(ROOT) + '"}'
    (tmp / "job.toml").write_text(f"""[job]
input = "in.jsonl"
output = "out"
[check]
class = "tests.fakes:FakeSlow"
[scheduler]
host = "127.0.0.1"
{extra}
[engines.check]
env = {env}
[local]
gpus = []
cpu_workers = 0
""")
    return load(tmp / "job.toml")


def wait_for(cond, seconds=60):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.2)
    return False


class WorkerTermTest(unittest.TestCase):
    def terminate(self, whole_group):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = job(tmp)
            run = Path(cfg["job"]["output"]) / ".run"
            run.mkdir(parents=True)
            board = Board(cfg)
            daemon = Daemon(board, cfg, run)
            try:
                board.put("d0", "check", 0)
                env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}", "TTS_RETAKE_HANDLE": "h1"}
                p = subprocess.Popen([sys.executable, "-m", "tts_retake", "worker", "--daemon", str(run / "daemon.json"), "--kind", "cpu"],
                                     env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.assertTrue(wait_for(lambda: board.busy("d0") == "working"))
                time.sleep(3)
                started = time.time()
                if whole_group:
                    os.killpg(p.pid, signal.SIGTERM)
                else:
                    p.send_signal(signal.SIGTERM)
                p.wait(timeout=30)
                self.assertTrue(wait_for(lambda: board.busy("d0") == "queued", 15))
                self.assertLess(time.time() - started, 15)
                results = board.drain()
                self.assertTrue(any("released" in r for r in results))
                self.assertFalse(any(r.get("ok") is False for r in results))
                self.assertEqual(board.snapshot()["workers"], {})
            finally:
                daemon.close()

    def test_worker_hands_back_task(self):
        self.terminate(whole_group=False)

    def test_job_terminated_with_engine(self):
        self.terminate(whole_group=True)


class AccountTest(unittest.TestCase):
    def test_preempted_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Controller(job(tmp, "max_worker_failures = 2"))
            try:
                c.board.put("d0", "check", 0)
                c.board.claim({"id": "w1", "kind": "cpu", "handle": "j1"})
                c.account([{"handle": "j1", "kind": "cpu", "outcome": "preempted", "detail": ""}])
                self.assertEqual(c.board.busy("d0"), "queued")
                self.assertEqual(c.exits, {"cpu": {"preempted": 1}})
                self.assertEqual(c.errors, {})
                c.account([{"handle": "j2", "kind": "gpu", "outcome": "failed", "detail": ""}])
                with self.assertRaises(RuntimeError):
                    c.account([{"handle": "j3", "kind": "gpu", "outcome": "failed", "detail": ""}])
                self.assertEqual(c.exits["gpu"], {"failed": 2})
            finally:
                c.daemon.close()


class DemandTest(unittest.TestCase):
    def test_checks_spill_to_gpus_when_cpu_jobs_are_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Controller(job(tmp, 'backend = "slurm"\ntarget_minutes = 60'))
            try:
                for i in range(10):
                    c.board.put(f"x{i}", "check", 0)
                self.assertEqual(c.demand(), {"gpu": 0, "cpu": 1})
                for i in range(10, 400):
                    c.board.put(f"x{i}", "check", 0)
                self.assertEqual(c.demand(), {"gpu": 6, "cpu": 4})
            finally:
                c.daemon.close()


class SlurmTest(unittest.TestCase):
    def test_sacct_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = job(tmp)
            pool = backends.Slurm(cfg, Path(tmp))
            self.assertIn("#SBATCH --no-requeue", pool.script("gpu"))
            pool.gone = {"1": {"kind": "gpu"}, "2": {"kind": "gpu"}, "3": {"kind": "cpu"}, "4": {"kind": "cpu"}}
            out = "1|PREEMPTED|0:15|n1\n2|FAILED|1:0|n2\n3|CANCELLED by 1|0:15|n3\n4|COMPLETING|0:0|n4\n"
            with mock.patch.object(backends.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, out, "")):
                ended = {e["handle"]: e["outcome"] for e in pool.ended()}
            self.assertEqual(ended, {"1": "preempted", "2": "failed", "3": "cancelled"})
            self.assertEqual(list(pool.gone), ["4"])


if __name__ == "__main__":
    unittest.main()
