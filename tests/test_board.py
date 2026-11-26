"""Task hand-out on the controller board."""
import unittest

from tts_retake.config import DEFAULTS
from tts_retake.daemon import Board


class RetryElsewhereTest(unittest.TestCase):
    def test_failed_task_goes_to_another_worker(self):
        board = Board(DEFAULTS)
        a, b = {"id": "a", "kind": "cpu"}, {"id": "b", "kind": "cpu"}
        board.put("d1", "check", 0)
        self.assertEqual(board.claim(a)["task"]["dialogue"], "d1")
        board.claim(b)
        board.done("a", "d1", {"ok": False, "error": "crash"})
        board.put("d1", "check", 0)
        self.assertIn("wait", board.claim(a))
        self.assertEqual(board.claim(b)["task"]["dialogue"], "d1")

    def test_only_worker_still_retries(self):
        board = Board(DEFAULTS)
        a = {"id": "a", "kind": "cpu"}
        board.put("d1", "check", 0)
        board.claim(a)
        board.done("a", "d1", {"ok": False, "error": "crash"})
        board.put("d1", "check", 0)
        self.assertEqual(board.claim(a)["task"]["dialogue"], "d1")


class GroupClaimTest(unittest.TestCase):
    def test_gpu_worker_can_claim_only_synth(self):
        board = Board(DEFAULTS)
        g = {"id": "g", "kind": "gpu"}
        board.put("c1", "check", 0)
        board.put("s1", "synth", 0)
        board.put("s2", "synth", 0)
        self.assertEqual(board.claim(g)["task"]["dialogue"], "s1")
        self.assertEqual(board.claim(g, ["synth"])["task"]["dialogue"], "s2")
        self.assertIn("wait", board.claim(g, ["synth"]))
        board.release("g", "test")
        self.assertEqual(board.snapshot()["queued"], {"synth": 2, "check": 1})


if __name__ == "__main__":
    unittest.main()
