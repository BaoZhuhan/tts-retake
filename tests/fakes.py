"""Stand-in engines for tests: synthesis writes placeholder chunks, checks fail chunks by a fixed script."""
from tts_retake.dialogue import Folder, key, read_json, write_json


class FakeSynth:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, folder):
        f = Folder(folder)
        counts = read_json(f.path("synth_counts.json"), {})
        made = []
        for k, chunk in enumerate(f.plan["chunks"]):
            if not f.chunk(k).exists():
                counts[key(chunk)] = counts.get(key(chunk), 0) + 1
                f.chunk(k).write_bytes(f"{key(chunk)} take {counts[key(chunk)]}".encode())
                made.append(k)
        write_json(f.path("synth_counts.json"), counts)
        return {"made": made}


class FakeCheck:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, folder):
        f = Folder(folder)
        plan, counts = f.plan, read_json(f.path("synth_counts.json"), {})
        chunks = []
        for k, chunk in enumerate(plan["chunks"]):
            take = counts.get(key(chunk), 1)
            status = "fail" if chunk[0] == 2 and take < 4 and chunk[1] - chunk[0] > 1 else "suspect" if chunk[0] == 0 and take == 1 else "pass"
            chunks.append({"k": k, "turns": chunk, "status": status, "fail": [], "suspect": []})
        write_json(f.path("verdict.json"), {"sig": [f.sig(k) for k in range(len(plan["chunks"]))], "chunks": chunks})
        return {"status": [c["status"] for c in chunks]}
