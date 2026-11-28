"""Worker: claims tasks from the daemon, runs them in engine processes that keep their models loaded, and reports back."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .daemon import call

SRC = str(Path(__file__).resolve().parents[1])


def gpu_memory_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return float(out.stdout.split()[0]) / 1024 if out.returncode == 0 else None
    except (IndexError, ValueError):
        return None


class Interrupted(RuntimeError):
    pass


def with_src(env):
    return {**env, "PYTHONPATH": os.pathsep.join(p for p in (SRC, env.get("PYTHONPATH", "")) if p)}


class Engine:
    def __init__(self, spec, cfg, device):
        env = with_src({**os.environ, **spec["env"], "TTS_RETAKE_CONFIG": json.dumps(cfg), "TTS_RETAKE_DEVICE": device})
        self.proc = subprocess.Popen([spec["python"], "-m", "tts_retake.engine"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, env=env)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def run(self, kind, folder) -> dict:
        return self.ask({"type": kind, "folder": folder})

    def run_many(self, kind, folders) -> list:
        """One result per folder from a single engine call; an engine error is every folder's result."""
        r = self.ask({"type": kind, "folders": folders})
        return [{"ok": True, **x} for x in r["results"]] if r.get("ok") else [r] * len(folders)

    def ask(self, req) -> dict:
        kind = req["type"]
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            code = self.proc.wait()
            if code == -signal.SIGTERM:
                raise Interrupted(f"{kind} engine was terminated")
            raise RuntimeError(f"{kind} engine exited with code {code}")
        return json.loads(line)

    def close(self):
        if self.alive():
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class Worker:
    def __init__(self, info, kind, only=None, overrides=()):
        self.info, self.cfg, self.only = info, call(info, "/config"), only
        for o in overrides:
            key, value = o.split("=", 1)
            *path, last = key.split(".")
            d = self.cfg
            for k in path:
                d = d[k]
            try:
                d[last] = json.loads(value)
            except ValueError:
                d[last] = value
        memory = gpu_memory_gb() if kind == "gpu" else None
        self.kind = "cpu" if memory is not None and memory < self.cfg["scheduler"]["min_gpu_gb"] else kind
        self.one_engine = self.kind == "gpu" and memory is not None and memory < self.cfg["synth"]["low_vram_below_gb"]
        self.id = f"{socket.gethostname().split('.')[0]}-{os.getpid()}-{uuid.uuid4().hex[:4]}"
        self.engines, self.tasks = {}, []
        self.handle = os.environ.get("TTS_RETAKE_HANDLE") or os.environ.get("SLURM_JOB_ID", "")
        signal.signal(signal.SIGTERM, self.stop)

    def engine(self, kind) -> Engine:
        """Engine for a task type, started on first use; on a small GPU the other engines are closed first so their models fit."""
        spec = self.cfg["engines"][kind]
        key = (spec["python"], json.dumps(spec["env"], sort_keys=True))
        if key not in self.engines or not self.engines[key].alive():
            if self.one_engine:
                for e in self.engines.values():
                    e.close()
            self.engines[key] = Engine(spec, self.cfg, "cuda" if self.kind == "gpu" else "cpu")
        return self.engines[key]

    def release(self, reason):
        try:
            call(self.info, "/release", {"worker": self.id, "reason": reason}, timeout=10)
        except OSError:
            pass

    def stop(self, signum, _frame):
        signal.signal(signum, signal.SIG_DFL)
        self.release(signal.Signals(signum).name)
        for e in self.engines.values():
            if e.alive():
                e.proc.kill()
        os._exit(128 + signum)

    def heartbeat(self):
        while True:
            time.sleep(30)
            try:
                for dialogue in list(self.tasks):
                    call(self.info, "/beat", {"worker": self.id, "dialogue": dialogue})
            except OSError:
                pass

    def loop(self):
        threading.Thread(target=self.heartbeat, daemon=True).start()
        last_work, misses = time.time(), 0
        me = {"id": self.id, "kind": self.kind, "host": socket.gethostname(), "handle": self.handle}
        while time.time() - last_work < 60 * self.cfg["scheduler"]["idle_minutes"] and misses < 40:
            try:
                reply = call(self.info, "/claim", {"worker": me, "only": self.only})
                misses = 0
            except OSError:
                misses += 1
                time.sleep(15)
                continue
            if reply.get("stop"):
                break
            if "task" not in reply:
                time.sleep(reply["wait"])
                continue
            tasks = [reply["task"]]
            group = self.cfg["synth"]["group"] if self.cfg["synth"]["fast"] else 1
            while tasks[0]["type"] == "synth" and len(tasks) < group:
                try:
                    more = call(self.info, "/claim", {"worker": me, "only": ["synth"]})
                except OSError:
                    break
                if "task" not in more:
                    break
                tasks.append(more["task"])
            self.tasks = [t["dialogue"] for t in tasks]
            folders = [str(Path(self.cfg["job"]["output"]) / d) for d in self.tasks]
            kind = tasks[0]["type"]
            try:
                engine = self.engine(kind)
                results = engine.run_many(kind, folders) if len(tasks) > 1 else [engine.run(kind, folders[0])]
            except Interrupted:
                time.sleep(5)
                self.release("engine terminated")
                self.tasks = []
                continue
            except Exception as e:
                results = [{"ok": False, "error": repr(e)[-3000:]}] * len(tasks)
            self.tasks, last_work = [], time.time()
            for t, result in zip(tasks, results):
                try:
                    call(self.info, "/done", {"worker": self.id, "dialogue": t["dialogue"], "result": result})
                except OSError:
                    pass
        for e in self.engines.values():
            e.close()


def main(info, kind, only=None, overrides=()):
    Worker(info, kind, only, overrides).loop()
    sys.exit(0)
