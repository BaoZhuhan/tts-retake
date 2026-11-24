"""Benchmark: defects are cut into the middle chunk with the right labels, a run folder is prepared, and verdicts are scored."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

from tts_retake import bench
from tts_retake.dialogue import read_json, write_json

SR = 16000


def make_source(root, items=4):
    voice = root / "voice.wav"
    sf.write(str(voice), np.random.default_rng(0).normal(0, 0.05, 2 * SR).astype(np.float32), SR)
    for n in range(items):
        d = root / "source" / f"item{n}"
        d.mkdir(parents=True)
        turns, chunks, spans = [], [], []
        for k in range(3):
            start, times, audio = len(turns), [], []
            for i in range(12):
                times.append({"turn": len(turns), "s": i * 1.5 + 0.1, "e": i * 1.5 + 1.3})
                turns.append({"speaker": "ab"[i % 2], "text": "one two three four"})
                audio.append(np.sin(2 * np.pi * (200 + 30 * (i % 2)) * np.arange(int(1.5 * SR)) / SR).astype(np.float32) * 0.1)
            chunks.append([start, len(turns)])
            spans.append(times)
            sf.write(str(d / f"chunk_{k:03d}.wav"), np.concatenate(audio), SR)
        write_json(d / "plan.json", {"id": f"item{n}", "priority": 0.0, "order": ["a", "b"], "pauses": [0.5] * len(turns), "chunks": chunks,
                                     "speakers": {s: {"voice": str(voice), "text": "Hello there."} for s in "ab"}, "turns": turns})
        write_json(d / "spans.json", {"chunks": spans})
    return root / "source"


@unittest.skipIf(sf is None, "soundfile is not installed")
class BenchTest(unittest.TestCase):
    def test_build_prepare_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = make_source(tmp)
            manifest = bench.build(source, tmp / "bench", per_kind=2, clean=2, seed=1,
                                   kinds=["drop_middle", "drop_opening", "drop_ending", "insert_speech", "leak", "silence", "tone"])
            self.assertEqual(manifest["defects"], {"clean": 2, **{k: 2 for k in ("drop_middle", "drop_opening", "drop_ending", "insert_speech",
                                                                               "leak", "silence", "tone")}})
            rows = [json.loads(line) for line in (tmp / "bench" / "input.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 16)
            for row in rows:
                item = tmp / "bench" / row["id"]
                labels = read_json(item / "label.json")["chunks"]
                kind = row["id"].split("__")[1]
                self.assertEqual([c["defect"] for c in labels], [None, None, None] if kind == "clean" else [None, kind, None])
                src = source / row["id"].split("__")[0]
                self.assertEqual((item / "chunk_000.wav").stat().st_ino, (src / "chunk_000.wav").stat().st_ino)
                before, after = sf.info(str(src / "chunk_001.wav")).duration, sf.info(str(item / "chunk_001.wav")).duration
                if kind.startswith("drop"):
                    self.assertLess(after, before - 2)
                elif kind == "silence":
                    self.assertAlmostEqual(after - before, 9.0, places=2)
                elif kind != "clean":
                    self.assertGreater(after, before)
            job = bench.prepare(tmp / "bench", tmp / "run")
            self.assertIn("max_redos = 0", job.read_text())
            for item in (tmp / "run" / "output").iterdir():
                kind = item.name.split("__")[1]
                status = ["pass", "pass" if kind in ("clean", "tone") else "suspect" if kind == "silence" else "fail", "pass"]
                reasons = {"drop_middle": ["missing words 40%"], "silence": ["silence 9 s"]}.get(kind, ["added words 30%"])
                write_json(item / "verdict.json", {"chunks": [{"k": k, "status": s, "fail": reasons if s == "fail" else [],
                                                                "suspect": reasons if s == "suspect" else []} for k, s in enumerate(status)]})
            table = bench.score(tmp / "run")
            self.assertEqual(table["clean"]["n"], 2 * 3 + 7 * 2 * 2)
            self.assertEqual(table["clean"]["fail"], 0.0)
            self.assertEqual(table["drop_middle"]["right_reason"], 1.0)
            self.assertEqual(table["drop_opening"]["detected"], 1.0)
            self.assertEqual(table["drop_opening"]["right_reason"], 0.0)
            self.assertEqual(table["silence"]["detected"], 1.0)
            self.assertEqual(table["tone"]["detected"], 0.0)
            self.assertTrue((tmp / "run" / "bench_report.md").exists())


if __name__ == "__main__":
    unittest.main()
