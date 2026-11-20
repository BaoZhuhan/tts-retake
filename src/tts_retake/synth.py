"""MOSS-TTSD synthesis of the missing chunks of a dialogue folder."""
from __future__ import annotations

import sys
import time

from .dialogue import Folder, write_json


class Synth:
    def __init__(self, cfg):
        c = self.cfg = cfg["synth"]
        if c["moss_cli_dir"]:
            sys.path.insert(0, c["moss_cli_dir"])
        import soundfile
        import torch
        import torchaudio
        from moss_ttsd_app import _merge_consecutive_speaker_tags, build_conversation, load_backend, normalize_text

        self.sf, self.torch, self.ta = soundfile, torch, torchaudio
        self.merge, self.build, self.norm = _merge_consecutive_speaker_tags, build_conversation, normalize_text
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.low_vram = torch.cuda.get_device_properties(0).total_memory / 2**30 < c["low_vram_below_gb"]
        if self.low_vram:
            from transformers import AutoModel, AutoProcessor

            self.proc = AutoProcessor.from_pretrained(c["model"], trust_remote_code=True, codec_path=c["codec"])
            self.proc.audio_tokenizer.eval()
            self.model = AutoModel.from_pretrained(c["model"], trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                   attn_implementation="sdpa").to("cuda").eval()
            self.device = torch.device("cuda")
            self.sr = int(getattr(self.proc.model_config, "sampling_rate", 24000))
        else:
            self.model, self.proc, self.device, self.sr = load_backend(c["model"], c["codec"], "cuda", "auto")
        self.decoding = {"audio_temperature": c["temperature"], "audio_top_p": c["top_p"], "audio_top_k": c["top_k"],
                         "audio_repetition_penalty": c["repetition_penalty"]}

    def decode(self, out):
        if not self.low_vram:
            return self.proc.decode(out)
        self.model.to("cpu")
        self.torch.cuda.empty_cache()
        self.proc.audio_tokenizer.to("cuda")
        try:
            return self.proc.decode(out)
        finally:
            self.proc.audio_tokenizer.to("cpu")
            self.torch.cuda.empty_cache()
            self.model.to("cuda")

    def reference(self, path):
        wav, rate = self.sf.read(path, dtype="float32", always_2d=True)
        wav = self.torch.from_numpy(wav.T.copy()).mean(0, keepdim=True)
        return self.ta.functional.resample(wav, rate, self.sr)

    def generate(self, text, codes, prompt, words):
        conv, mode, _ = self.build(text, codes, prompt, self.proc)
        batch = self.proc(conv, mode=mode)
        max_new = int(words / 2.5 * 12.5 * 1.8) + 300
        low, high = self.cfg["rate_wpm"]
        best = None
        for attempt in range(1, self.cfg["attempts"] + 1):
            with self.torch.no_grad():
                out = self.model.generate(input_ids=batch["input_ids"].to(self.device),
                                          attention_mask=batch["attention_mask"].to(self.device),
                                          max_new_tokens=max_new, **self.decoding)
            audio = self.decode(out)[0].audio_codes_list[0].detach().float().cpu().numpy().reshape(-1)
            seconds = len(audio) / self.sr
            wpm = words / max(seconds, 1e-6) * 60
            capped = seconds >= (max_new - 300) / 12.5 * 0.98
            score = abs(wpm - self.cfg["target_wpm"]) + (1000 if capped else 0)
            if best is None or score < best[0]:
                best = (score, audio, attempt, round(wpm, 1))
            if low <= wpm <= high and not capped:
                break
        return best

    def run(self, folder) -> dict:
        f = Folder(folder)
        plan = f.plan
        first, second = plan["order"]
        voices = {s: plan["speakers"][s] for s in plan["order"]}
        refs = [self.reference(voices[first]["voice"]), self.reference(voices[second]["voice"])]
        codes = self.proc.encode_audios_from_wav(refs, sampling_rate=self.sr)
        prompts = {first: (f"[S1] {voices[first]['text']}[S2] {voices[second]['text']}",
                           self.proc.encode_audios_from_wav([self.torch.cat(refs, dim=-1)], sampling_rate=self.sr)[0]),
                   second: (f"[S2] {voices[second]['text']}[S1] {voices[first]['text']}",
                            self.proc.encode_audios_from_wav([self.torch.cat(refs[::-1], dim=-1)], sampling_rate=self.sr)[0])}
        made = []
        for k, (a, b) in enumerate(plan["chunks"]):
            if f.chunk(k).exists():
                continue
            turns = plan["turns"][a:b]
            words = sum(len(t["text"].split()) for t in turns)
            body = "".join(f"[S{1 if t['speaker'] == first else 2}] {t['text']} " for t in turns)
            prefix, prompt = prompts[turns[0]["speaker"]]
            started = time.time()
            _, audio, attempts, wpm = self.generate(self.norm(self.merge(prefix + body)), codes, prompt, words)
            tmp = f.path(f".chunk_{k:03d}.tmp.wav")
            self.sf.write(str(tmp), audio, self.sr)
            tmp.replace(f.chunk(k))
            write_json(f.chunk(k, "attempts.json"), {"attempts": attempts, "wpm": wpm, "seconds": round(time.time() - started, 1)})
            made.append(k)
        return {"made": made}
