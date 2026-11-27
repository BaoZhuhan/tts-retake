"""SGLang-Omni backend: a private `sgl-omni serve` on this worker's GPU, started on first use and stopped with the engine, and a
thread pool that keeps synth.sglang.concurrency requests in flight against its /v1/audio/speech."""
from __future__ import annotations

import atexit
import io
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class Server:
    def __init__(self, cfg, model):
        import requests
        import soundfile

        c = cfg["synth"]["sglang"]
        self.requests, self.sf, self.model = requests, soundfile, model
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        cmd = [c["command"], "serve", "--model-path", model, "--port", str(port), "--host", "127.0.0.1"]
        if c["config"]:
            cmd += ["--config", c["config"]]
        log = Path(cfg["job"]["output"]) / ".run" / f"sglang-{socket.gethostname()}-{os.getpid()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        atexit.register(self.stop)
        deadline = time.time() + c["start_minutes"] * 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"sgl-omni exited with {self.proc.returncode}; see {log}")
            try:
                if requests.get(f"{self.url}/health", timeout=5).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            self.stop()
            raise RuntimeError(f"sgl-omni did not come up in {c['start_minutes']} min; see {log}")
        self.pool = ThreadPoolExecutor(c["concurrency"])
        self.slots = c["concurrency"]

    def speak(self, body):
        """One request; returns mono float32 audio and its sample rate."""
        r = self.requests.post(f"{self.url}/v1/audio/speech", json={"model": self.model, "voice": "default",
                                                                   "response_format": "wav", **body}, timeout=3600)
        if not r.ok:
            raise RuntimeError(f"sgl-omni {r.status_code}: {r.text[:500]}")
        audio, sr = self.sf.read(io.BytesIO(r.content), dtype="float32", always_2d=True)
        return audio.mean(1), sr

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, 15)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 9)
