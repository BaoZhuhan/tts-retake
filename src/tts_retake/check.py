"""Automatic checks of synthesized chunks: ASR fidelity (Whisper or Parakeet), speaking rate, reference leak, tones, silences and ECAPA speaker similarity."""
from __future__ import annotations

import difflib
import hashlib
import os
import re
from pathlib import Path

import numpy as np

from .dialogue import Folder, read_json, write_json

TOKEN = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list[str]:
    return [t.strip("'") for t in TOKEN.findall(text.lower().replace("’", "'")) if t.strip("'")]


def f0_of(y):
    import librosa

    if len(y) < 4000:
        return None
    f0 = librosa.yin(y, fmin=65, fmax=420, sr=16000, frame_length=1024, hop_length=320)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=320)[0][:len(f0)]
    f0 = f0[:len(rms)]
    voiced = f0[(rms >= np.percentile(rms, 50)) & (f0 > 66) & (f0 < 415)]
    return round(float(np.median(voiced)), 1) if len(voiced) >= 10 else None


def centroid(vectors, fallback):
    if len(vectors) < 3:
        return fallback
    v = np.array(vectors)
    c = v.mean(0)
    for _ in range(2):
        c = v[(v @ c) >= np.percentile(v @ c, 20)].mean(0)
    return c / np.linalg.norm(c)


def longest_run(flags):
    run, best = [], []
    for f in flags + [None]:
        if f is not None and f["swap"]:
            run.append(f)
        else:
            best, run = max(best, run, key=len), []
    return best


def judge(plan, checks, embeddings, refs, th) -> dict:
    a, b = plan["order"]
    other = {a: b, b: a}
    f0 = {s: refs[s][1] for s in (a, b)}
    cross = bool(f0[a] and f0[b] and abs(np.log2(f0[a] / f0[b])) > 0.5)
    gender = {s: ("male" if f0[s] < 165 else "female") if f0[s] else None for s in (a, b)}
    cen = {s: centroid([embeddings[k][t["ei"]] for k, q in enumerate(checks) for t in q["turns"] if "ei" in t and t["speaker"] == s], refs[s][0])
           for s in (a, b)}
    chunks = []
    for k, q in enumerate(checks):
        m, fail, suspect = q["metrics"], [], []
        for name, value in (("missing", m["missing"]), ("added", m["added"])):
            if value > th[f"{name}_fail"]:
                fail.append(f"{name} words {value:.0%}")
            elif value > th[f"{name}_suspect"]:
                suspect.append(f"{name} words {value:.0%}")
        fail += [x for x, bad in (("opening dropped", m["opening_dropped"]), ("ending dropped", m["ending_dropped"])) if bad]
        if len(m["leak"]) >= 2:
            fail.append("reference text spoken: " + " ".join(m["leak"][:5]))
        if not th["wpm_fail"][0] <= m["wpm"] <= th["wpm_fail"][1]:
            fail.append(f"speaking rate {m['wpm']:.0f} wpm")
        elif not th["wpm_suspect"][0] <= m["wpm"] <= th["wpm_suspect"][1]:
            suspect.append(f"speaking rate {m['wpm']:.0f} wpm")
        if m["tone_runs"]:
            fail.append(f"sustained tone at {m['tone_runs'][0][0]} s")
        if m["longest_silence_s"] > th["silence_suspect"]:
            suspect.append(f"silence {m['longest_silence_s']:.0f} s")
        flags = []
        for t in q["turns"]:
            if "ei" not in t:
                continue
            e, own, rival = embeddings[k][t["ei"]], t["speaker"], other[t["speaker"]]
            oc, xc, orf, xrf = (float(e @ cen[own]), float(e @ cen[rival]), float(e @ refs[own][0]), float(e @ refs[rival][0]))
            g, p = gender[own], t.get("f0")
            wrong_pitch = p is not None and g is not None and (p < th["female_f0_min"] if g == "female" else p > th["male_f0_max"])
            swap = xc > oc + th["swap_margin"] and xrf > orf
            flags.append({"turn": t["turn"], "seconds": round(t["e"] - t["s"], 1), "own": round(oc, 3), "other": round(xc, 3),
                          "own_ref": round(orf, 3), "other_ref": round(xrf, 3), "f0": p, "swap": swap, "wrong_pitch": wrong_pitch,
                          "drift": oc < th["drift_own"] and orf < th["drift_reference"] and not swap})
        run = longest_run(flags)
        if len(run) >= (th["swap_run_cross_gender"] if cross else th["swap_run_same_gender"]):
            agree = sum(f["wrong_pitch"] for f in run) >= len(run) / 2
            (fail if agree or not cross else suspect).append(f"voice swap turns {run[0]['turn']}-{run[-1]['turn']}")
        elif len(run) >= 2:
            suspect.append(f"swap-like turns {run[0]['turn']}-{run[-1]['turn']}")
        drift = [f for f in flags if f["drift"]]
        hard = [f for f in drift if f["seconds"] >= 2 and (f["own"] < th["drift_hard"] or f["wrong_pitch"])]
        if hard:
            fail.append("voice drift turns " + ",".join(str(f["turn"]) for f in hard[:6]))
        elif drift:
            suspect.append("possible drift turns " + ",".join(str(f["turn"]) for f in drift[:6]))
        chunks.append({"k": k, "turns": plan["chunks"][k], "status": "fail" if fail else "suspect" if suspect else "pass",
                       "fail": fail, "suspect": suspect, "speaker": flags})
    return {"cross_gender": cross, "chunks": chunks}


class Whisper:
    def __init__(self, name, device):
        import whisper

        self.model, self.fp16 = whisper.load_model(name, device=device), device == "cuda"

    def words(self, y16):
        r = self.model.transcribe(y16, language="en", fp16=self.fp16, condition_on_previous_text=False, word_timestamps=True)
        return [[w["word"], round(w["start"], 2), round(w["end"], 2)] for s in r["segments"] for w in s.get("words", [])]


class Parakeet:
    def __init__(self, name, device):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(name, map_location=device).eval()

    def words(self, y16):
        out = self.model.transcribe([np.ascontiguousarray(y16, dtype=np.float32)], timestamps=True, batch_size=1, verbose=False)
        hyp = (out[0] if isinstance(out, tuple) else out)[0]
        return [[w["word"], round(float(w["start"]), 2), round(float(w["end"]), 2)] for w in hyp.timestamp["word"]]


def asr_model(name, device):
    return (Parakeet if "parakeet" in name.lower() else Whisper)(name, device)


class Checker:
    def __init__(self, cfg):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.cfg, self.torch = cfg["check"], torch
        self.device = os.environ.get("TTS_RETAKE_DEVICE", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.cache = Path(cfg["job"]["output"]) / ".cache"
        self.asr = asr_model(self.cfg["asr"], self.device)
        self.encoder = EncoderClassifier.from_hparams(source=self.cfg["speaker_model"], savedir=str(self.cache / "speaker_model"),
                                                      run_opts={"device": self.device})

    def embed(self, y):
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None].to(self.device)
            e = self.encoder.encode_batch(x).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    def load16k(self, path):
        import librosa
        import soundfile as sf

        y, sr = sf.read(str(path), dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        return y, sr, librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y

    def reference(self, voice):
        st = os.stat(voice)
        name = hashlib.sha1(f"{voice}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
        path = self.cache / "references" / f"{name}.npz"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            y = self.load16k(voice)[2]
            np.savez(path, emb=self.embed(y), f0=np.array(f0_of(y) or 0.0))
        z = np.load(path)
        return z["emb"], float(z["f0"]) or None

    def transcribe(self, f, k, y16):
        cached = read_json(f.chunk(k, "words.json"))
        if cached and cached.get("sig") == f.sig(k) and cached.get("asr") == self.cfg["asr"]:
            return cached["words"]
        words = self.asr.words(y16)
        write_json(f.chunk(k, "words.json"), {"sig": f.sig(k), "asr": self.cfg["asr"], "words": words})
        return words

    def check_chunk(self, f, plan, k, trimmed=0.0):
        import librosa
        import soundfile as sf

        from .sustained import runs, sustained_windows

        lo, hi = plan["chunks"][k]
        turns = plan["turns"]
        y, sr, y16 = self.load16k(f.chunk(k))
        hyp, times = [], []
        for w, s, e in self.transcribe(f, k, y16):
            for t in tokens(w):
                hyp.append(t)
                times.append((s, e))
        ref, owner = [], []
        for i in range(lo, hi):
            for t in tokens(turns[i]["text"]):
                ref.append(t)
                owner.append(i)
        sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
        found, matched = [None] * len(ref), set()
        for a, b, n in sm.get_matching_blocks():
            for j in range(n):
                found[a + j] = times[b + j]
                matched.add(b + j)
        vocab = {t for turn in turns for t in tokens(turn["text"])}
        leak_words = {t for s in plan["order"] for t in tokens(plan["speakers"][s]["text"]) if len(t) >= 5} - vocab
        leaked = [j for j in range(len(hyp)) if j not in matched and hyp[j] in leak_words]
        blocks = [(a, b) for a, b, n in sm.get_matching_blocks() if n >= 4]
        if leaked and not trimmed and blocks and blocks[0][0] <= 3 and max(leaked) < blocks[0][1]:
            cut = max(0.0, times[blocks[0][1]][0] - 0.25)
            if cut >= 1.0:
                tmp = f.path(f".chunk_{k:03d}.trim.wav")
                sf.write(str(tmp), y[int(cut * sr):], sr)
                os.replace(tmp, f.chunk(k))
                return self.check_chunk(f, plan, k, trimmed=round(cut, 2))
        per = {}
        for i, t in zip(owner, found):
            d = per.setdefault(i, {"n": 0, "found": 0, "s": None, "e": None})
            d["n"] += 1
            if t is not None:
                d["found"] += 1
                d["s"] = t[0] if d["s"] is None else min(d["s"], t[0])
                d["e"] = t[1] if d["e"] is None else max(d["e"], t[1])
        rows, embeddings = [], []
        for i in range(lo, hi):
            d = per.get(i, {"n": 0, "found": 0, "s": None, "e": None})
            row = {"turn": i, "speaker": turns[i]["speaker"], "words": d["n"], "found": round(d["found"] / max(1, d["n"]), 2)}
            if d["s"] is not None:
                row.update(s=round(d["s"], 2), e=round(d["e"], 2))
                segment = y16[int(d["s"] * 16000):int(min(d["e"], d["s"] + 10) * 16000)]
                if d["e"] - d["s"] >= 1.5 and row["found"] >= 0.5 and len(segment) >= 16000:
                    row.update(ei=len(embeddings), f0=f0_of(segment))
                    embeddings.append(self.embed(segment))
            rows.append(row)
        seconds = len(y16) / 16000
        rms = librosa.feature.rms(y=y16, frame_length=1024, hop_length=320)[0]
        db = 20 * np.log10(rms + 1e-9)
        longest = current = 0
        for quiet in db < max(-50, np.percentile(db, 95) - 45):
            current = current + 1 if quiet else 0
            longest = max(longest, current)
        spoken = [r for r in rows if r["words"] >= 3]
        metrics = {"seconds": round(seconds, 2), "words": len(ref),
                   "missing": round(1 - sum(x is not None for x in found) / max(1, len(ref)), 3),
                   "added": round(1 - len(matched) / max(1, len(hyp)), 3), "wpm": round(len(ref) / max(seconds, 1e-6) * 60, 1),
                   "leak": sorted({hyp[j] for j in leaked}), "trimmed_s": trimmed,
                   "tone_runs": [[round(a, 1), round(b, 1)] for a, b in (runs(sustained_windows(y, sr)) if len(y) > 3 * sr else [])],
                   "longest_silence_s": round(longest * 320 / 16000, 2),
                   "opening_dropped": bool(spoken) and all(r["found"] < 0.3 for r in spoken[:2]),
                   "ending_dropped": bool(spoken) and all(r["found"] < 0.3 for r in spoken[-2:])}
        np.save(f.chunk(k, "emb.npy"), np.array(embeddings) if embeddings else np.zeros((0, 192), dtype=np.float32))
        write_json(f.chunk(k, "check.json"), {"sig": f.sig(k), "turns_range": [lo, hi], "metrics": metrics, "turns": rows})

    def run(self, folder) -> dict:
        f = Folder(folder)
        plan = f.plan
        n = len(plan["chunks"])
        refs = {s: self.reference(plan["speakers"][s]["voice"]) for s in plan["order"]}
        checked = 0
        for k in range(n):
            q = read_json(f.chunk(k, "check.json"))
            if not (q and q["sig"] == f.sig(k) and q["turns_range"] == plan["chunks"][k]):
                self.check_chunk(f, plan, k)
                checked += 1
        checks = [read_json(f.chunk(k, "check.json")) for k in range(n)]
        verdict = judge(plan, checks, [np.load(f.chunk(k, "emb.npy")) for k in range(n)], refs, self.cfg)
        verdict["sig"] = [f.sig(k) for k in range(n)]
        if self.cfg["assemble"]:
            verdict["audio"] = f.assemble()
        write_json(f.path("verdict.json"), verdict)
        return {"checked": checked, "status": [c["status"] for c in verdict["chunks"]]}
