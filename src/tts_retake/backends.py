"""Worker pools: processes on this machine, SLURM jobs, or workers started by hand on any machine."""
from __future__ import annotations

import getpass
import os
import re
import shlex
import signal
import subprocess
import sys

from .dialogue import read_json, write_json
from .worker import SRC, with_src


def worker_command(cfg, run, kind) -> list[str]:
    return [cfg["worker"]["python"], "-m", "tts_retake", "worker", "--daemon", str(run / "daemon.json"), "--kind", kind]


FAILED_STATES = {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL"}
OUTCOMES = {"PREEMPTED": "preempted", "TIMEOUT": "timeout", "DEADLINE": "timeout", "CANCELLED": "cancelled", "COMPLETED": "finished",
            **{s: "failed" for s in FAILED_STATES}}


def make(cfg, run):
    return {"local": Local, "slurm": Slurm, "manual": Manual}[cfg["scheduler"]["backend"]](cfg, run)


class Manual:
    def __init__(self, cfg, run):
        self.cfg, self.run = cfg, run
        (run / "logs").mkdir(parents=True, exist_ok=True)

    def scale(self, demand):
        pass

    def ended(self) -> list:
        return []

    def shutdown(self):
        pass


class Local(Manual):
    def __init__(self, cfg, run):
        super().__init__(cfg, run)
        gpus = cfg["local"]["gpus"]
        self.gpus = [str(g) for g in (self.detect() if gpus == "auto" else gpus)]
        self.procs, self.started = {}, 0

    @staticmethod
    def detect():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return []
        return out.stdout.split() if out.returncode == 0 else []

    def ended(self) -> list:
        out = []
        for slot, (kind, handle, p) in list(self.procs.items()):
            code = p.poll()
            if code is None:
                continue
            del self.procs[slot]
            outcome = "finished" if code == 0 else "cancelled" if code in (-signal.SIGTERM, 128 + signal.SIGTERM) else "failed"
            out.append({"handle": handle, "kind": kind, "outcome": outcome, "detail": f"exit {code}"})
        return out

    def scale(self, demand):
        wanted = [("gpu", g) for g in self.gpus[:demand["gpu"]]]
        wanted += [("cpu", str(i)) for i in range(min(self.cfg["local"]["cpu_workers"], demand["cpu"]))]
        for kind, slot in wanted:
            if f"{kind}{slot}" not in self.procs:
                self.started += 1
                handle = f"local-{kind}{slot}-{self.started}"
                env = with_src({**os.environ, **self.cfg["worker"]["env"], "CUDA_VISIBLE_DEVICES": slot if kind == "gpu" else "",
                                "TTS_RETAKE_HANDLE": handle})
                with open(self.run / "logs" / f"worker-{kind}{slot}.log", "a") as log:
                    p = subprocess.Popen(worker_command(self.cfg, self.run, kind), env=env, stdout=log, stderr=log)
                self.procs[f"{kind}{slot}"] = (kind, handle, p)

    def shutdown(self):
        for _, _, p in self.procs.values():
            p.terminate()
        for _, _, p in self.procs.values():
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
        self.gone = {}

    def script(self, kind) -> str:
        s = self.cfg["slurm"]
        name = f"{s['job_name']}-{kind}"
        lines = ["#!/bin/bash", f"#SBATCH --job-name={name}", f"#SBATCH --output={self.run / 'logs'}/{name}-%j.out"]
        lines += [f"#SBATCH --{k}={v}" for k, v in s[kind]["options"].items() if v not in ("", None)]
        if not {"requeue", "no-requeue"} & set(s[kind]["options"]):
            lines.append("#SBATCH --no-requeue")
        lines += list(s["setup"])
        lines += [f"export {k}={shlex.quote(str(v))}" for k, v in self.cfg["worker"]["env"].items()]
        lines += [f"export PYTHONPATH={shlex.quote(SRC)}${{PYTHONPATH:+:$PYTHONPATH}}", "exec " + shlex.join(worker_command(self.cfg, self.run, kind))]
        return "\n".join(lines) + "\n"

    def live(self) -> dict:
        out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i %T"], capture_output=True, text=True, timeout=120)
        if out.returncode:
            return {j: dict(v, state="UNKNOWN") for j, v in self.jobs.items()}
        states = dict(line.split() for line in out.stdout.splitlines() if line.strip())
        self.gone.update({j: v for j, v in self.jobs.items() if j not in states})
        self.jobs = {j: dict(v, state=states[j]) for j, v in self.jobs.items() if j in states}
        write_json(self.ledger, self.jobs)
        return self.jobs

    def ended(self) -> list:
        if not self.gone:
            return []
        out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(self.gone), "-o", "JobID,State,ExitCode,NodeList"],
                             capture_output=True, text=True, timeout=120)
        final = {}
        for line in out.stdout.splitlines():
            parts = line.split("|")
            if len(parts) == 4:
                final[parts[0]] = parts[1:]
        done = []
        for j, v in list(self.gone.items()):
            state, code, node = final.get(j, ["", "", ""])
            word = state.split()[0] if state else ""
            if word in ("RUNNING", "PENDING", "REQUEUED", "COMPLETING", "SUSPENDED"):
                continue
            if not word and out.returncode == 0 and v.setdefault("tries", 0) < 3:
                v["tries"] += 1
                continue
            del self.gone[j]
            done.append({"handle": j, "kind": v["kind"], "outcome": OUTCOMES.get(word, "unknown"), "detail": f"{state or 'no record'} exit {code} {node}".strip()})
        return done

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
