"""MOSS-TTSD synthesis of the missing chunks of dialogue folders. With synth.fast the chunks of several dialogues (and their retries)
are generated together as one batch through FastGen; with synth.sglang.command, on GPUs of at least synth.sglang.ttsd_min_gb,
they are concurrent requests to SGLang-Omni (the same continuation prompts, built here; on 32 GB cards its KV cache holds too
few requests to beat FastGen); otherwise one at a time through the model's own generate."""
from __future__ import annotations

import sys
import time

from .dialogue import Folder, write_json
from .language import units


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
        self.decoding = {"audio_temperature": c["temperature"], "audio_top_p": c["top_p"], "audio_top_k": c["top_k"],
                         "audio_repetition_penalty": c["repetition_penalty"]}
        self.server = None
        if c["sglang"]["command"] and torch.cuda.get_device_properties(0).total_memory / 2**30 >= c["sglang"]["ttsd_min_gb"]:
            from transformers import AutoProcessor

            from .sglang import Server
            self.server = Server(cfg, c["model"])
            self.proc = AutoProcessor.from_pretrained(c["model"], trust_remote_code=True, codec_path=c["codec"])
            self.proc.audio_tokenizer = self.proc.audio_tokenizer.to("cuda").eval()
            self.device = torch.device("cpu")
            self.sr = int(getattr(self.proc.model_config, "sampling_rate", 24000))
            return
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
        self.gen = self.model.generate
        if c["fast"]:
            from .fastgen import FastGen
            self.fast = FastGen(self.model, slots=c["batch"])

    def decode(self, out):
        """Codes to audio. FastGen's static buffers are freed first: on long chunks they leave the codec too little memory, and
        an OOM inside the codec leaves it stuck in streaming mode."""
        if self.cfg["fast"]:
            self.fast.release()
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

    def generate(self, fn, jobs):
        """fn(jobs) with FastGen's slots halved and retried on running out of GPU memory (outside the except block, so the
        failed attempt's tensors are freed first)."""
        try:
            return fn(jobs)
        except self.torch.cuda.OutOfMemoryError:
            if not self.cfg["fast"] or self.fast.slots == 1:
                raise
        self.fast.want = max(1, self.fast.slots // 2)
        self.fast.release()
        return self.generate(fn, jobs)

    def inputs(self, jobs):
        convs = [conv for text, prompt, _, codes in jobs for conv in self.build(text, codes, prompt, self.proc)[0]]
        batch = self.proc(convs, mode="continuation")
        return {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}

    def batch(self, jobs):
        """Audio for each (text, prompt, words, codes) job, one at a time through the model's own generate."""
        out = []
        for job in jobs:
            with self.torch.no_grad():
                res = self.gen(**self.inputs([job]), max_new_tokens=self.cap(job[2]), **self.decoding)
            out += self.audio(res)
        return out

    def audio(self, res):
        return [m.audio_codes_list[0].detach().float().cpu().numpy().reshape(-1) for m in self.decode(res)]

    def streamed(self, jobs):
        """Best take per job through FastGen in one call: takes are judged by frame count as they finish and a retake goes
        straight into the freed slot (unless another take of that job is still running), so no round waits for its slowest
        take and only each job's best take is decoded. Idle slots start with extra takes of the same jobs. Returns (audio, takes)
        per job."""
        low, high = self.cfg["rate_wpm"]
        n, extra = divmod(self.fast.slots, len(jobs))
        order = [i for i in range(len(jobs)) for _ in range(min(max(1, n + (i < extra)), self.cfg["attempts"]))]
        issued, flying, best = dict.fromkeys(range(len(jobs)), 0), dict.fromkeys(range(len(jobs)), 0), {}
        for i in order:
            issued[i] += 1
            flying[i] += 1

        def retake(p, frames, capped):
            i = order[p]
            flying[i] -= 1
            wpm = jobs[i][2] / max(frames / 12.5, 1e-6) * 60
            score = abs(wpm - self.cfg["target_wpm"]) + (1000 if capped else 0)
            if i not in best or score < best[i][0]:
                best[i] = (score, p, wpm, capped)
            if (low <= best[i][2] <= high and not best[i][3]) or flying[i] or issued[i] >= self.cfg["attempts"]:
                return False
            order.append(i)
            issued[i] += 1
            flying[i] += 1
            return True

        takes = [jobs[i] for i in order]
        res = self.fast.generate(**self.inputs(takes), max_new_tokens=[self.cap(job[2]) for job in takes], retake=retake,
                                 **self.decoding)
        return [(a, issued[i]) for i, a in enumerate(self.audio([res[best[i][1]] for i in range(len(jobs))]))]

    def served(self, jobs, judge):
        """Takes as concurrent SGLang requests: idle slots start with extra takes of the same jobs, a finished take that leaves its
        job out of the rate window (or cut at the cap) queues a retake once no other take of that job is in flight, and each job
        is yielded as it settles."""
        from concurrent.futures import FIRST_COMPLETED, wait

        low, high = self.cfg["rate_wpm"]
        grids = [self.inputs([job])["input_ids"][0].tolist() for job in jobs]
        n, extra = divmod(self.server.slots, len(jobs))
        issued, best, flying = dict.fromkeys(range(len(jobs)), 0), {}, {}

        def submit(i):
            issued[i] += 1
            body = {"input": "-", "prompt_rows": grids[i], "max_new_tokens": self.cap(jobs[i][2]), "moss_sampling": self.decoding}
            flying[self.server.pool.submit(self.server.speak, body)] = i

        for i in range(len(jobs)):
            for _ in range(min(max(1, n + (i < extra)), self.cfg["attempts"])):
                submit(i)
        while flying:
            done, _ = wait(flying, return_when=FIRST_COMPLETED)
            for fut in done:
                i = flying.pop(fut)
                audio, _ = fut.result()
                score, wpm, capped = judge(i, audio)
                if i not in best or score < best[i][0]:
                    best[i] = (score, audio, wpm, capped)
                if i in flying.values():
                    continue
                if not (low <= best[i][2] <= high and not best[i][3]) and issued[i] < self.cfg["attempts"]:
                    submit(i)
                else:
                    yield i, (best[i][0], best[i][1], issued[i], best[i][2])

    @staticmethod
    def cap(words):
        return int(words / 2.5 * 12.5 * 1.8) + 300

    def chunks(self, jobs):
        """Yields (job index, (score, audio, takes, wpm)) as each job settles: in the rate window and not cut at the cap, or
        out of attempts."""
        low, high = self.cfg["rate_wpm"]

        def judge(i, audio):
            seconds = len(audio) / self.sr
            wpm = jobs[i][2] / max(seconds, 1e-6) * 60
            capped = seconds >= (self.cap(jobs[i][2]) - 300) / 12.5 * 0.98
            return abs(wpm - self.cfg["target_wpm"]) + (1000 if capped else 0), round(wpm, 1), capped

        if self.server:
            yield from self.served(jobs, judge)
            return
        if self.cfg["fast"]:
            for i, (audio, takes) in enumerate(self.generate(self.streamed, jobs)):
                score, wpm, _ = judge(i, audio)
                yield i, (score, audio, takes, wpm)
            return
        best, used, todo = {}, dict.fromkeys(range(len(jobs)), 0), list(range(len(jobs)))
        while todo:
            for i, audio in zip(todo, self.generate(self.batch, [jobs[i] for i in todo])):
                used[i] += 1
                score, wpm, capped = judge(i, audio)
                if i not in best or score < best[i][0]:
                    best[i] = (score, audio, used[i], wpm, capped)
            good = {i for i in todo if low <= best[i][3] <= high and not best[i][4]}
            for i in todo:
                if i in good or used[i] >= self.cfg["attempts"]:
                    yield i, best[i][:4]
            todo = [i for i in todo if i not in good and used[i] < self.cfg["attempts"]]

    def run(self, folder) -> dict:
        return self.run_many([folder])[0]

    def jobs(self, folder):
        """(folder, chunk indexes, jobs) for the missing chunks of one dialogue; a job is (text, prompt, words, voice codes) and
        its prompt starts with the voice of the chunk's first speaker."""
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
        todo = [k for k in range(len(plan["chunks"])) if not f.chunk(k).exists()]
        jobs = []
        for k in todo:
            a, b = plan["chunks"][k]
            turns = plan["turns"][a:b]
            body = "".join(f"[S{1 if t['speaker'] == first else 2}] {t['text']} " for t in turns)
            prefix, prompt = prompts[turns[0]["speaker"]]
            jobs.append((self.norm(self.merge(prefix + body)), prompt, sum(units(t["text"]) for t in turns), codes))
        return f, todo, jobs

    def run_many(self, folders) -> list:
        """Missing chunks of several dialogues in one take loop, so short redo lists of many dialogues share FastGen's slots."""
        work = [self.jobs(folder) for folder in folders]
        where = [(f, k) for f, todo, _ in work for k in todo]
        started = time.time()
        for i, (_, audio, attempts, wpm) in self.chunks([job for *_, jobs in work for job in jobs]):
            (f, k), seconds = where[i], round(time.time() - started, 1)
            tmp = f.path(f".chunk_{k:03d}.tmp.wav")
            self.sf.write(str(tmp), audio, self.sr)
            tmp.replace(f.chunk(k))
            write_json(f.chunk(k, "attempts.json"), {"attempts": attempts, "wpm": wpm, "seconds": seconds})
        return [{"made": todo} for _, todo, _ in work]
