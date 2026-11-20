"""End to end on this machine: daemon, scheduler, local workers and the repair policy, with stand-in engines."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from tts_retake.config import load
from tts_retake.dialogue import Folder
from tts_retake.scheduler import Controller

ROOT = Path(__file__).resolve().parents[1]


class SchedulerTest(unittest.TestCase):
    def test_failed_and_suspect_chunks_are_retaken(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            turns = [{"speaker": "host" if i % 2 == 0 else "guest", "text": "one two three four five"} for i in range(8)]
            rows = [{"id": f"d{i}", "speakers": {"host": {"voice": "a.wav", "text": "Hello."}, "guest": {"voice": "b.wav", "text": "Hi."}},
                     "turns": turns} for i in range(2)]
            (tmp / "in.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            env = '{PYTHONPATH = "' + str(ROOT) + '"}'
            (tmp / "job.toml").write_text(f"""[job]
input = "in.jsonl"
output = "out"
[synth]
class = "tests.fakes:FakeSynth"
max_chunk_words = 10
min_chunk_words = 10
[check]
class = "tests.fakes:FakeCheck"
[scheduler]
interval_seconds = 1
idle_minutes = 1
host = "127.0.0.1"
[engines.synth]
env = {env}
[engines.check]
env = {env}
[local]
gpus = ["0"]
cpu_workers = 1
""")
            cfg = load(tmp / "job.toml")
            status = Controller(cfg).run_forever()
            self.assertEqual(set(status.values()), {"done"})
            f = Folder(tmp / "out" / "d0")
            self.assertIn([2, 3], f.plan["chunks"])
            self.assertEqual(f.state["tried"], ["0-2"])
            report = [json.loads(line) for line in (tmp / "out" / "report.jsonl").read_text().splitlines()]
            self.assertEqual(len(report), 2)


if __name__ == "__main__":
    unittest.main()
