"""MOSS-TTS (v1.5) synthesis: each turn is cloned from its own speaker's reference, turns of a chunk are generated as one batch
trimmed of edge silence and joined with the planned pauses."""
from __future__ import annotations

import time

import numpy as np

from .dialogue import Folder, write_json
from .language import LANGS, units, wpm


class SynthTTS:
    def __init__(self, cfg):
        import soundfile
        import torch
        from transformers import AutoModel, AutoProcessor

        c = self.cfg = cfg["synth"]
        self.sf, self.torch = soundfile, torch
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.proc = AutoProcessor.from_pretrained(c["model"], trust_remote_code=True)
        self.proc.audio_tokenizer = self.proc.audio_tokenizer.to("cuda")
        self.model = AutoModel.from_pretrained(c["model"], trust_remote_code=True, attn_implementation="sdpa",
                                               dtype=torch.bfloat16).to("cuda").eval()
        self.sr = int(self.proc.model_config.sampling_rate)
        self.codes = {}

    def reference(self, path):
        if path not in self.codes:
            wav, rate = self.sf.read(path, dtype="float32", always_2d=True)
            wav = self.torch.from_numpy(wav.mean(1, keepdims=True).T.copy())
            self.codes[path] = self.proc.encode_audios_from_wav([wav], sampling_rate=rate)[0]
        return self.codes[path]

    def generate(self, items):
        convs = [[self.proc.build_user_message(text=t, reference=[self.reference(v)], language=lang)] for t, v, lang in items]
        batch = self.proc(convs, mode="generation")
        longest = max(units(t) for t, _, _ in items)
        with self.torch.no_grad():
            out = self.model.generate(input_ids=batch["input_ids"].to("cuda"), attention_mask=batch["attention_mask"].to("cuda"),
                                      max_new_tokens=int(longest / 2.0 * 12.5 * 2) + 150)
        return [self.trim(m.audio_codes_list[0].detach().float().cpu().numpy().reshape(-1)) for m in self.proc.decode(out)]

    def trim(self, audio, top_db=40, pad=0.05):
        import librosa

        if len(audio) < self.sr // 10:
            return audio
        _, (a, b) = librosa.effects.trim(audio, top_db=top_db)
        p = int(pad * self.sr)
        return audio[max(0, a - p):b + p]

    def turns(self, plan, idx):
        lang = plan.get("language", "en")
        name, low, high = LANGS[lang][0], *self.cfg["rate_wpm"]
        best, todo = {}, list(idx)
        for attempt in range(1, self.cfg["attempts"] + 1):
            for s in range(0, len(todo), self.cfg["batch"]):
                part = todo[s:s + self.cfg["batch"]]
                items = [(plan["turns"][i]["text"], plan["speakers"][plan["turns"][i]["speaker"]]["voice"], name) for i in part]
                for i, audio in zip(part, self.generate(items)):
                    rate = wpm(units(plan["turns"][i]["text"]), len(audio) / self.sr, lang)
                    score = abs(rate - self.cfg["target_wpm"])
                    if i not in best or score < best[i][0]:
                        best[i] = (score, audio, attempt, round(rate, 1))
            todo = [i for i in todo if not low <= best[i][3] <= high]
            if not todo:
                break
        return best

    def run(self, folder) -> dict:
        f = Folder(folder)
        plan = f.plan
        made = []
        for k, (a, b) in enumerate(plan["chunks"]):
            if f.chunk(k).exists():
                continue
            started = time.time()
            best = self.turns(plan, range(a, b))
            parts = []
            for i in range(a, b):
                parts.append(best[i][1])
                if i < b - 1:
                    parts.append(np.zeros(int(plan["pauses"][i] * self.sr), dtype=np.float32))
            tmp = f.path(f".chunk_{k:03d}.tmp.wav")
            self.sf.write(str(tmp), np.concatenate(parts), self.sr)
            tmp.replace(f.chunk(k))
            write_json(f.chunk(k, "attempts.json"), {"attempts": max(best[i][2] for i in best), "wpm": [best[i][3] for i in range(a, b)],
                                                     "seconds": round(time.time() - started, 1)})
            made.append(k)
        return {"made": made}
