"""Benchmark for the checks: clean dialogue items, defects with known labels cut into their audio, and scores of a check-only run."""
from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from .dialogue import read_json, write_json

KINDS = {"drop_middle": "fail", "drop_opening": "fail", "drop_ending": "fail", "insert_speech": "fail", "leak": "fail",
         "fast": "fail", "slow": "fail", "silence": "flag", "tone": "fail"}
REASONS = {"missing words": "missing", "added words": "added", "opening dropped": "opening", "ending dropped": "ending",
           "reference text spoken": "leak", "speaking rate": "rate", "sustained tone": "tone", "silence": "silence",
           "voice swap": "swap", "swap-like": "swap", "voice drift": "drift", "possible drift": "drift", "voice change": "drift"}
RIGHT = {"drop_middle": {"missing"}, "drop_opening": {"opening", "missing"}, "drop_ending": {"ending", "missing"},
         "insert_speech": {"added"}, "leak": {"leak", "trimmed"}, "fast": {"rate"}, "slow": {"rate"}, "silence": {"silence"},
         "tone": {"tone"}, "third_voice": {"drift", "swap"}, "swap": {"swap"}}


def load(path):
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32")
    return (y.mean(1) if y.ndim > 1 else y), sr


def save(path, y, sr):
    import soundfile as sf

    sf.write(str(path), np.asarray(y, dtype=np.float32), sr)


def resample(y, sr, to):
    if sr == to:
        return y
    import librosa

    return librosa.resample(y, orig_sr=sr, target_sr=to)


def layout(spans, seconds):
    s = sorted((t for t in spans if t.get("s") is not None and t.get("e") is not None), key=lambda t: t["turn"])
    cuts = [0.0] + [round((a["e"] + b["s"]) / 2, 3) for a, b in zip(s, s[1:])] + [seconds]
    return s, cuts


def words_of(plan, turn):
    return len(plan["turns"][turn]["text"].split())


class Defects:
    def __init__(self, plan, k, y, sr, spans, rng, donors):
        self.plan, self.k, self.y, self.sr, self.rng, self.donors = plan, k, y, sr, rng, donors
        self.spans, self.cuts = layout(spans, len(y) / sr)
        a, b = plan["chunks"][k]
        self.total = sum(words_of(plan, i) for i in range(a, b))
        self.spoken = [j for j, t in enumerate(self.spans) if words_of(plan, t["turn"]) >= 3]

    def at(self, seconds):
        return int(round(seconds * self.sr))

    def middle(self):
        n = len(self.spans)
        return self.rng.randint(max(1, n // 3), max(1, 2 * n // 3)) if n >= 3 else None

    def insert(self, piece, detail):
        j = self.middle()
        if j is None:
            return None
        cut = self.at(self.cuts[j])
        return np.concatenate([self.y[:cut], piece, self.y[cut:]]), {**detail, "at_s": round(self.cuts[j], 2)}

    def drop_middle(self):
        n, wanted = len(self.spans), 0.3 * self.total
        if len(self.spoken) < 6:
            return None
        lo, hi = self.spoken[2], self.spoken[-3]
        for _ in range(20):
            i = self.rng.randint(lo, hi)
            removed, j = 0, i
            while j < hi and removed < wanted:
                removed += words_of(self.plan, self.spans[j]["turn"])
                j += 1
            if removed >= wanted and j <= n:
                a, b = self.at(self.cuts[i]), self.at(self.cuts[j])
                return np.concatenate([self.y[:a], self.y[b:]]), {"turns": [self.spans[i]["turn"], self.spans[j - 1]["turn"]],
                                                                  "removed_words": removed, "share": round(removed / self.total, 2)}
        return None

    def drop_opening(self):
        if len(self.spoken) < 5:
            return None
        j = self.spoken[1]
        return self.y[self.at(self.cuts[j + 1]):], {"cut_s": round(self.cuts[j + 1], 2), "through_turn": self.spans[j]["turn"]}

    def drop_ending(self):
        if len(self.spoken) < 5:
            return None
        j = self.spoken[-2]
        return self.y[:self.at(self.cuts[j])], {"cut_s": round(self.cuts[j], 2), "from_turn": self.spans[j]["turn"]}

    def insert_speech(self):
        wanted = 0.35 * self.total
        for _ in range(10):
            plan, k, y, sr, spans = self.rng.choice(self.donors)
            s, cuts = layout(spans, len(y) / sr)
            if len(s) < 3:
                continue
            i = self.rng.randrange(len(s))
            got, j = 0, i
            while j < len(s) and got < wanted:
                got += words_of(plan, s[j]["turn"])
                j += 1
            if got >= wanted:
                piece = resample(y[int(cuts[i] * sr):int(cuts[j] * sr)], sr, self.sr)
                return self.insert(piece, {"donor": plan["id"], "donor_turns": [s[i]["turn"], s[j - 1]["turn"]],
                                           "added_words": got, "share": round(got / self.total, 2)})
        return None

    def leak(self):
        a, _ = self.plan["chunks"][self.k]
        speaker = self.plan["turns"][a]["speaker"]
        ref, sr = load(self.plan["speakers"][speaker]["voice"])
        piece = np.concatenate([resample(ref, sr, self.sr), np.zeros(self.at(0.4), dtype=np.float32)])
        return np.concatenate([piece, self.y]), {"speaker": speaker, "seconds": round(len(piece) / self.sr, 2)}

    def stretch(self, target):
        import librosa

        wpm = self.total / (len(self.y) / self.sr) * 60
        rate = target / wpm
        if 0.95 < rate < 1.05:
            return None
        return librosa.effects.time_stretch(self.y, rate=rate), {"wpm_before": round(wpm), "wpm_after": target, "rate": round(rate, 3)}

    def fast(self):
        return self.stretch(250)

    def slow(self):
        return self.stretch(85)

    def silence(self):
        return self.insert(np.zeros(self.at(9.0), dtype=np.float32), {"seconds": 9.0})

    def tone(self):
        t = np.arange(self.at(3.0)) / self.sr
        f0 = self.rng.uniform(150, 320)
        wave = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 7))
        level = float(np.sqrt(np.mean(self.y ** 2))) or 0.05
        piece = (wave / np.sqrt(np.mean(wave ** 2)) * level).astype(np.float32)
        return self.insert(piece, {"seconds": 3.0, "f0": round(f0, 1)})


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def items_in(folder):
    return sorted(p for p in Path(folder).iterdir() if (p / "plan.json").exists())


def labels_of(item):
    plan = read_json(item / "plan.json")
    return read_json(item / "label.json", {"chunks": [{"k": k, "defect": None} for k in range(len(plan["chunks"]))]})


def emit(src, out, name, labels, replaced=None):
    d = out / name
    d.mkdir(parents=True)
    plan = read_json(src / "plan.json")
    plan["id"] = name
    write_json(d / "plan.json", plan)
    for k in range(len(plan["chunks"])):
        target = d / f"chunk_{k:03d}.wav"
        if replaced and k in replaced:
            save(target, replaced[k], replaced["sr"])
        else:
            link(src / f"chunk_{k:03d}.wav", target)
    write_json(d / "label.json", labels)
    return {"id": name, "speakers": {s: {"voice": v["voice"], "text": v["text"]} for s, v in plan["speakers"].items()},
            "turns": plan["turns"]}


def build(source, out, per_kind=30, clean=60, seed=0, kinds=None) -> dict:
    source, out = Path(source), Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; build into a new folder")
    out.mkdir(parents=True)
    rng = random.Random(seed)
    items = items_in(source)
    given = [p for p in items if any(c.get("defect") for c in labels_of(p)["chunks"])]
    base = [p for p in items if p not in given]
    rng.shuffle(base)
    rows, made = [], Counter()
    for p in base[:clean]:
        rows.append(emit(p, out, f"{p.name}__clean", labels_of(p)))
        made["clean"] += 1
    donors = []
    for p in base[: min(len(base), 40)]:
        plan, spans = read_json(p / "plan.json"), read_json(p / "spans.json")
        for k in range(len(plan["chunks"])):
            y, sr = load(p / f"chunk_{k:03d}.wav")
            donors.append((plan, k, y, sr, spans["chunks"][k]))
    for kind in kinds or KINDS:
        pool = base[:]
        rng.shuffle(pool)
        for p in pool:
            if made[kind] >= per_kind:
                break
            plan, spans = read_json(p / "plan.json"), read_json(p / "spans.json")
            k = len(plan["chunks"]) // 2
            y, sr = load(p / f"chunk_{k:03d}.wav")
            others = [d for d in donors if d[0]["id"] != plan["id"]]
            result = getattr(Defects(plan, k, y, sr, spans["chunks"][k], rng, others), kind)()
            if result is None:
                continue
            audio, detail = result
            labels = {"chunks": [{"k": j, "defect": kind if j == k else None, **({"expect": KINDS[kind], "detail": detail} if j == k else {})}
                                 for j in range(len(plan["chunks"]))]}
            rows.append(emit(p, out, f"{p.name}__{kind}", labels, {k: audio, "sr": sr}))
            made[kind] += 1
    for p in given:
        rows.append(emit(p, out, p.name, labels_of(p)))
        for c in labels_of(p)["chunks"]:
            if c.get("defect"):
                made[c["defect"]] += 1
    with open(out / "input.jsonl", "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    manifest = {"source": str(source), "seed": seed, "items": len(rows), "defects": dict(made)}
    write_json(out / "manifest.json", manifest)
    return manifest


def prepare(bench, run) -> Path:
    bench, run = Path(bench).resolve(), Path(run).resolve()
    output = run / "output"
    output.mkdir(parents=True, exist_ok=False)
    for item in items_in(bench):
        d = output / item.name
        d.mkdir()
        shutil.copy(item / "plan.json", d / "plan.json")
        shutil.copy(item / "label.json", d / "label.json")
        for wav in item.glob("chunk_*.wav"):
            link(wav, d / wav.name)
    job = run / "job.toml"
    job.write_text(f'[job]\ninput = "{bench / "input.jsonl"}"\noutput = "output"\n\n[check]\nassemble = false\n\n[repair]\nmax_redos = 0\nretry_suspect = false\n')
    return job


def category(reason):
    return next((c for prefix, c in REASONS.items() if reason.startswith(prefix)), "other")


def score(run) -> dict:
    run = Path(run)
    output = run / "output" if (run / "output").exists() else run
    rows = []
    for item in items_in(output):
        verdict, labels = read_json(item / "verdict.json"), read_json(item / "label.json")
        if not verdict or not labels:
            continue
        for c, lab in zip(verdict["chunks"], labels["chunks"]):
            check = read_json(item / f"chunk_{c['k']:03d}.check.json", {})
            reasons = {category(r) for r in c["fail"] + c["suspect"]}
            if check.get("metrics", {}).get("trimmed_s"):
                reasons.add("trimmed")
            kind = lab.get("defect") or "clean"
            expect = lab.get("expect", "pass")
            hit = c["status"] == "fail" or (expect == "flag" and c["status"] == "suspect") or (kind == "leak" and "trimmed" in reasons)
            rows.append({"item": item.name, "k": c["k"], "kind": kind, "expect": expect, "status": c["status"],
                         "detected": hit if kind != "clean" else None, "right_reason": bool(reasons & RIGHT.get(kind, set())) if kind != "clean" else None,
                         "reasons": c["fail"] + c["suspect"]})
    table = {}
    for kind in sorted({r["kind"] for r in rows}, key=lambda k: (k != "clean", k)):
        rs = [r for r in rows if r["kind"] == kind]
        n = len(rs)
        status = Counter(r["status"] for r in rs)
        entry = {"n": n, **{s: round(status[s] / n, 3) for s in ("pass", "suspect", "fail")}}
        if kind == "clean":
            entry["reasons"] = dict(Counter(category(x) for r in rs for x in r["reasons"]).most_common())
        else:
            entry["detected"] = round(sum(r["detected"] for r in rs) / n, 3)
            entry["right_reason"] = round(sum(r["right_reason"] for r in rs) / n, 3)
        table[kind] = entry
    with open(run / "bench_rows.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    lines = ["| kind | n | detected | right reason | pass | suspect | fail |", "|---|---|---|---|---|---|---|"]
    for kind, e in table.items():
        lines.append(f"| {kind} | {e['n']} | {e.get('detected', '')} | {e.get('right_reason', '')} | {e['pass']} | {e['suspect']} | {e['fail']} |")
    if "clean" in table:
        lines += ["", "clean chunks flagged by: " + ", ".join(f"{k} {v}" for k, v in table["clean"]["reasons"].items())]
    (run / "bench_report.md").write_text("\n".join(lines) + "\n")
    return table
