"""Input validation, pauses, chunk plans and chunk splitting."""
import json
import tempfile
import unittest
from pathlib import Path

from tts_retake.dialogue import Folder, chunk_turns, load_dialogues, pauses

SYNTH = {"max_chunk_words": 12, "min_chunk_words": 4, "cut_pause": 5.0}


def dialogue(n=10):
    return {"id": "d1", "speakers": {"host": {"voice": "a.wav", "text": "Hello there."}, "guest": {"voice": "b.wav", "text": "Hi."}},
            "turns": [{"speaker": "host" if i % 2 == 0 else "guest", "text": "one two three four"} for i in range(n)]}


class DialogueTest(unittest.TestCase):
    def test_input_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.jsonl"
            path.write_text(json.dumps(dialogue()) + "\n")
            d = load_dialogues(path)[0]
            self.assertEqual(d["speakers"]["host"]["voice"], str(Path(tmp, "a.wav").resolve()))
            bad = dialogue()
            bad["turns"][0]["speaker"] = "nobody"
            path.write_text(json.dumps(bad) + "\n")
            with self.assertRaises(ValueError):
                load_dialogues(path)

    def test_pauses_are_seeded_and_can_be_given(self):
        turns = dialogue()["turns"]
        self.assertEqual(pauses(turns, "x"), pauses(turns, "x"))
        turns[0]["pause"] = 7
        self.assertEqual(pauses(turns, "x")[0], 7.0)

    def test_chunks_cover_all_turns(self):
        turns = dialogue()["turns"]
        chunks = chunk_turns(turns, [1.0] * len(turns), 12, 4, 5.0)
        self.assertEqual([c for pair in chunks for c in pair][0], 0)
        self.assertEqual(chunks[-1][1], len(turns))
        self.assertTrue(all(a[1] == b[0] for a, b in zip(chunks, chunks[1:])))

    def test_split_moves_later_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Folder(Path(tmp) / "d1")
            plan = f.create(dialogue(), SYNTH)
            for k in range(len(plan["chunks"])):
                f.chunk(k).write_text(str(k))
            n = len(plan["chunks"])
            self.assertTrue(f.split(0))
            self.assertEqual(len(f.plan["chunks"]), n + 1)
            self.assertFalse(f.chunk(0).exists() or f.chunk(1).exists())
            self.assertEqual(f.chunk(2).read_text(), "1")


if __name__ == "__main__":
    unittest.main()
