"""Chunk decisions from transcript metrics and speaker embeddings."""
import unittest

import numpy as np

from tts_retake.check import judge
from tts_retake.config import DEFAULTS

PLAN = {"order": ["a", "b"], "chunks": [[0, 20]]}
METRICS = {"missing": 0.02, "added": 0.02, "opening_dropped": False, "ending_dropped": False, "leak": [], "wpm": 160,
           "tone_runs": [], "longest_silence_s": 1.0}
A, B = np.eye(4)[0], np.eye(4)[1]


def turns(swapped=()):
    rows = [{"turn": i, "speaker": "ab"[i % 2], "s": 2.0 * i, "e": 2.0 * i + 2, "ei": i, "f0": 120.0} for i in range(20)]
    emb = np.array([(B if r["speaker"] == "a" else A) if r["turn"] in swapped else (A if r["speaker"] == "a" else B) for r in rows])
    return rows, emb


class JudgeTest(unittest.TestCase):
    def decide(self, metrics=METRICS, swapped=()):
        rows, emb = turns(swapped)
        return judge(PLAN, [{"metrics": metrics, "turns": rows}], [emb], {"a": (A, 120.0), "b": (B, 125.0)}, DEFAULTS["check"])["chunks"][0]

    def test_clean_chunk_passes(self):
        self.assertEqual(self.decide()["status"], "pass")

    def test_word_errors(self):
        self.assertEqual(self.decide({**METRICS, "missing": 0.3})["status"], "fail")
        self.assertEqual(self.decide({**METRICS, "added": 0.09})["status"], "suspect")

    def test_voice_swap_needs_a_run_of_turns(self):
        self.assertEqual(self.decide(swapped=range(5, 10))["status"], "fail")
        self.assertEqual(self.decide(swapped=[7])["status"], "pass")


if __name__ == "__main__":
    unittest.main()
