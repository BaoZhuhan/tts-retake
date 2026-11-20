"""Dialogue input, chunk plans, and the files of one dialogue folder."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

CHUNK_FILES = ("wav", "check.json", "emb.npy", "words.json", "attempts.json")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load_dialogues(path) -> list[dict]:
    path = Path(path)
    dialogues, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        d, where = json.loads(line), f"{path.name} line {n}"
        speakers = d.get("speakers") or {}
        if len(speakers) != 2:
            raise ValueError(f"{where}: give exactly two speakers")
        for name, s in speakers.items():
            if not s.get("voice") or not s.get("text"):
                raise ValueError(f"{where}: speaker {name!r} needs 'voice' and 'text'")
        turns = d.get("turns") or []
        if not turns or any(t.get("speaker") not in speakers or not str(t.get("text", "")).strip() for t in turns):
            raise ValueError(f"{where}: every turn needs a known speaker and some text")
        did = str(d.get("id", ""))
        if did in seen or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", did):
            raise ValueError(f"{where}: ids must be unique and use letters, digits, '.', '_' or '-'")
        seen.add(did)
        dialogues.append({
            "id": did, "priority": float(d.get("priority", 0)),
            "speakers": {k: {"voice": str((path.parent / v["voice"]).resolve()), "text": v["text"]} for k, v in speakers.items()},
            "turns": [{k: t[k] for k in ("speaker", "text", "pause") if k in t} for t in turns],
        })
    return dialogues


def pauses(turns, seed: str) -> list[float]:
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16))
    drawn = [min(3.0, max(0.25, rng.lognormvariate(0.0, 0.5))) for _ in turns]
    return [round(float(t.get("pause", d)), 2) for t, d in zip(turns, drawn)]


def chunk_turns(turns, gaps, max_words: int, min_words: int, cut_pause: float) -> list[list[int]]:
    chunks, start, words = [], 0, 0
    for i, turn in enumerate(turns):
        words += len(turn["text"].split())
        last = i == len(turns) - 1
        following = 0 if last else len(turns[i + 1]["text"].split())
        if last or words + following > max_words or (words >= min_words and gaps[i] >= cut_pause):
            chunks.append([start, i + 1])
            start, words = i + 1, 0
    return chunks


def key(chunk) -> str:
    return f"{chunk[0]}-{chunk[1]}"


class Folder:
    def __init__(self, path):
        self.dir = Path(path)

    def path(self, name: str) -> Path:
        return self.dir / name

    def chunk(self, k: int, kind: str = "wav") -> Path:
        return self.dir / f"chunk_{k:03d}.{kind}"

    @property
    def plan(self):
        return read_json(self.path("plan.json"))

    @property
    def state(self):
        return read_json(self.path("state.json"), {"redos": {}, "parked": [], "tried": []})

    def save_state(self, state) -> None:
        write_json(self.path("state.json"), state)

    def create(self, dialogue, synth) -> dict:
        self.dir.mkdir(parents=True, exist_ok=True)
        gaps = pauses(dialogue["turns"], dialogue["id"])
        chunks = chunk_turns(dialogue["turns"], gaps, synth["max_chunk_words"], synth["min_chunk_words"], synth["cut_pause"])
        plan = {**dialogue, "order": list(dialogue["speakers"]), "pauses": gaps, "chunks": chunks}
        write_json(self.path("plan.json"), plan)
        return plan

    def sig(self, k: int):
        try:
            st = self.chunk(k).stat()
        except FileNotFoundError:
            return None
        return [st.st_size, st.st_mtime_ns]

    def drop(self, k: int, keep: Path | None = None) -> None:
        for kind in CHUNK_FILES:
            p = self.chunk(k, kind)
            if p.exists():
                if keep is None:
                    p.unlink()
                else:
                    keep.mkdir(parents=True, exist_ok=True)
                    os.replace(p, keep / p.name)

    def restore(self, k: int, keep: Path) -> None:
        self.drop(k)
        for p in keep.glob(f"chunk_{k:03d}.*"):
            os.replace(p, self.dir / p.name)
        shutil.rmtree(keep, ignore_errors=True)

    def split(self, k: int) -> bool:
        plan = self.plan
        a, b = plan["chunks"][k]
        if b - a < 2:
            return False
        words = [len(t["text"].split()) for t in plan["turns"][a:b]]
        sums = [sum(words[:i]) for i in range(1, len(words))]
        cut = a + 1 + min(range(len(sums)), key=lambda i: abs(sums[i] - sum(words) / 2))
        for j in range(len(plan["chunks"]) - 1, k, -1):
            for kind in CHUNK_FILES:
                if self.chunk(j, kind).exists():
                    os.replace(self.chunk(j, kind), self.chunk(j + 1, kind))
        self.drop(k)
        plan["chunks"][k:k + 1] = [[a, cut], [cut, b]]
        write_json(self.path("plan.json"), plan)
        return True

    def assemble(self) -> dict:
        import numpy as np
        import soundfile as sf

        plan, pieces, meta, sr, at = self.plan, [], [], None, 0
        for k, (a, b) in enumerate(plan["chunks"]):
            audio, rate = sf.read(str(self.chunk(k)), dtype="float32")
            audio = audio.mean(1) if audio.ndim > 1 else audio
            sr = sr or rate
            gap = plan["pauses"][b - 1] if k < len(plan["chunks"]) - 1 else 0.0
            meta.append({"turns": [a, b], "start_s": round(at / sr, 2), "seconds": round(len(audio) / sr, 2)})
            pieces += [audio, np.zeros(int(gap * sr), dtype=np.float32)]
            at += len(audio) + int(gap * sr)
        full = np.concatenate(pieces)
        peak = float(np.abs(full).max()) or 1.0
        tmp = self.path(".dialogue.tmp.wav")
        sf.write(str(tmp), 0.95 * full / peak if peak > 1 else full, sr)
        os.replace(tmp, self.path("dialogue.wav"))
        return {"seconds": round(len(full) / sr, 2), "chunks": meta}
