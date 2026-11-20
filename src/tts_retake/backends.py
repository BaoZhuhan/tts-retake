"""Worker pools: processes on this machine, SLURM jobs, or workers started by hand on any machine."""
from __future__ import annotations

import getpass
import os
import re
import shlex
import subprocess
import sys

from .dialogue import read_json, write_json
from .worker import SRC, with_src


def worker_command(cfg, run, kind) -> list[str]:
    return [cfg["worker"]["python"], "-m", "tts_retake", "worker", "--daemon", str(run / "daemon.json"), "--kind", kind]


def make(cfg, run):
    return {"local": Local, "slurm": Slurm, "manual": Manual}[cfg["scheduler"]["backend"]](cfg, run)


class Manual:
    def __init__(self, cfg, run):
        self.cfg, self.run = cfg, run
        (run / "logs").mkdir(parents=True, exist_ok=True)

    def scale(self, demand):
        pass

    def shutdown(self):
        pass


class Local(Manual):
    def __init__(self, cfg, run):
        super().__init__(cfg, run)
        gpus = cfg["local"]["gpus"]
        self.gpus = [str(g) for g in (self.detect() if gpus == "auto" else gpus)]
        self.procs = {}

    @staticmethod
    def detect():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return []
        return out.stdout.split() if out.returncode == 0 else []

    def scale(self, demand):
        self.procs = {slot: p for slot, p in self.procs.items() if p.poll() is None}
        wanted = [("gpu", g) for g in self.gpus[:demand["gpu"]]]
        wanted += [("cpu", str(i)) for i in range(min(self.cfg["local"]["cpu_workers"], demand["cpu"]))]
        for kind, slot in wanted:
            if f"{kind}{slot}" not in self.procs:
                env = with_src({**os.environ, **self.cfg["worker"]["env"], "CUDA_VISIBLE_DEVICES": slot if kind == "gpu" else ""})
                with open(self.run / "logs" / f"worker-{kind}{slot}.log", "a") as log:
                    self.procs[f"{kind}{slot}"] = subprocess.Popen(worker_command(self.cfg, self.run, kind), env=env, stdout=log, stderr=log)

    def shutdown(self):
        for p in self.procs.values():
            p.terminate()
        for p in self.procs.values():
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()


def gigabytes(mem) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)B?", str(mem).strip().upper())
    if not m:
        return 0.0
    return float(m.group(1)) * {"K": 1 / 2**20, "M": 1 / 1024, "": 1 / 1024, "G": 1, "T": 1024}[m.group(2)]


class Slurm(Manual):
    def __init__(self, cfg, run):
        super().__init__(cfg, run)
        self.ledger = run / "slurm_jobs.json"
        self.jobs = read_json(self.ledger, {})

    def script(self, kind) -> str:
        s = self.cfg["slurm"]
        name = f"{s['job_name']}-{kind}"
        lines = ["#!/bin/bash", f"#SBATCH --job-name={name}", f"#SBATCH --output={self.run / 'logs'}/{name}-%j.out"]
        lines += [f"#SBATCH --{k}={v}" for k, v in s[kind]["options"].items() if v not in ("", None)]
        lines += list(s["setup"])
        lines += [f"export {k}={shlex.quote(str(v))}" for k, v in self.cfg["worker"]["env"].items()]
        lines += [f"export PYTHONPATH={shlex.quote(SRC)}${{PYTHONPATH:+:$PYTHONPATH}}", "exec " + shlex.join(worker_command(self.cfg, self.run, kind))]
        return "\n".join(lines) + "\n"

    def live(self) -> dict:
        out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i %T"], capture_output=True, text=True, timeout=120)
        if out.returncode:
            return {j: dict(v, state="UNKNOWN") for j, v in self.jobs.items()}
        states = dict(line.split() for line in out.stdout.splitlines() if line.strip())
        self.jobs = {j: dict(v, state=states[j]) for j, v in self.jobs.items() if j in states}
        write_json(self.ledger, self.jobs)
        return self.jobs

    def fits(self, kind) -> bool:
        s = self.cfg["slurm"]
        jobs = [s[v["kind"]]["options"] for v in self.jobs.values()] + [s[kind]["options"]]
        cpus = sum(int(o.get("cpus-per-task", 1)) for o in jobs)
        mem = sum(gigabytes(o.get("mem", 0)) for o in jobs)
        return (not s["max_cpus"] or cpus <= s["max_cpus"]) and (not s["max_mem_gb"] or mem <= s["max_mem_gb"])

    def submit(self, kind):
        out = subprocess.run(["sbatch", "--parsable"], input=self.script(kind), capture_output=True, text=True, timeout=120)
        if out.returncode:
            print(f"sbatch failed: {out.stderr.strip()}", file=sys.stderr, flush=True)
            return False
        self.jobs[out.stdout.strip().split(";")[0]] = {"kind": kind, "state": "PENDING"}
        write_json(self.ledger, self.jobs)
        return True

    def scale(self, demand):
        jobs = self.live()
        for kind in ("gpu", "cpu"):
            s = self.cfg["slurm"][kind]
            mine = [v for v in jobs.values() if v["kind"] == kind]
            pending = sum(v["state"] == "PENDING" for v in mine)
            for _ in range(max(0, min(demand[kind], s["max_jobs"]) - len(mine))):
                if pending >= s["max_pending"] or not self.fits(kind) or not self.submit(kind):
                    break
                pending += 1

    def shutdown(self):
        if self.jobs:
            subprocess.run(["scancel", *self.jobs], capture_output=True, timeout=120)
            self.jobs = {}
            write_json(self.ledger, self.jobs)
