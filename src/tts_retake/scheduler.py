"""Controller: reads every dialogue folder, applies the repair policy, queues tasks and sizes the worker pool to the remaining work."""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

from . import backends
from .daemon import Board, Daemon
from .dialogue import Folder, key, load_dialogues, read_json, write_json

FINAL = ("done", "review", "error")


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out = Path(cfg["job"]["output"])
        self.run = self.out / ".run"
        self.run.mkdir(parents=True, exist_ok=True)
        dialogues = load_dialogues(cfg["job"]["input"])
        self.dialogues = {d["id"]: d for d in dialogues[: cfg["job"]["limit"] or None]}
        self.errors = {k: v for k, v in read_json(self.run / "errors.json", {}).items() if isinstance(v, list)}
        self.status = {}
        self.exits = read_json(self.run / "worker_exits.json", {})
        self.failed_in_row = {"gpu": 0, "cpu": 0}
        self.board = Board(cfg)
        self.daemon = Daemon(self.board, cfg, self.run)
        self.backend = backends.make(cfg, self.run)

    def advance(self, did, d) -> str:
        """Next step of one dialogue; it errors out after failing on max_task_errors nodes (or 3x that many times anywhere)."""
        hosts, limit = self.errors.get(did, []), self.cfg["repair"]["max_task_errors"]
        if len(set(hosts)) >= limit or len(hosts) >= 3 * limit:
            return "error"
        f = Folder(self.out / did)
        plan = f.plan or f.create(d, self.cfg["synth"])
        n = len(plan["chunks"])
        state = f.state
        chronic = max(state["redos"].values(), default=0) >= self.cfg["repair"]["chronic_after"]
        priority = d["priority"] - (1000 if chronic else 0)
        if any(f.sig(k) is None for k in range(n)):
            self.board.put(did, "synth", priority)
            return "synth"
        verdict = read_json(f.path("verdict.json"))
        if not verdict or verdict.get("sig") != [f.sig(k) for k in range(n)]:
            self.board.put(did, "check", priority)
            return "check"
        if self.repair(f, plan, verdict, state):
            return self.advance(did, d)
        return "review" if state["parked"] else "done"

    def repair(self, f, plan, verdict, state) -> bool:
        rp, changed = self.cfg["repair"], False
        for c in verdict["chunks"]:
            k, name = c["k"], key(plan["chunks"][c["k"]])
            held = f.path("previous") / name
            if held.exists():
                if c["status"] == "pass":
                    shutil.rmtree(held)
                else:
                    f.restore(k, held)
                    changed = True
                continue
            if name in state["parked"]:
                continue
            if c["status"] == "fail":
                redos = state["redos"].get(name, 0)
                if redos >= rp["max_redos"]:
                    state["parked"].append(name)
                elif redos >= rp["split_after"] and f.split(k):
                    for part in f.plan["chunks"][k:k + 2]:
                        state["redos"][key(part)] = redos + 1
                    changed = True
                    break
                else:
                    f.drop(k)
                    state["redos"][name] = redos + 1
                    changed = True
            elif c["status"] == "suspect" and rp["retry_suspect"] and name not in state["tried"]:
                state["tried"].append(name)
                f.drop(k, keep=held)
                changed = True
        f.save_state(state)
        return changed

    def demand(self) -> dict:
        """Workers wanted: GPUs for the synthesis backlog within target_minutes; CPU check workers for the checks that running
        synthesis keeps producing plus the queued ones, with idle GPUs taking the rest."""
        snap = self.board.snapshot()
        est = snap["seconds_per_task"]
        horizon = 60 * self.cfg["scheduler"]["target_minutes"]
        synth = snap["queued"]["synth"] + snap["running"]["synth"]
        check = snap["queued"]["check"] + snap["running"]["check"]
        cpu_possible = self.cfg["scheduler"]["backend"] != "local" or self.cfg["local"]["cpu_workers"] > 0
        gpu_seconds = synth * est["synth"] + (0 if cpu_possible else check * est["check_gpu"])
        gpu = min(synth + (0 if cpu_possible else check), math.ceil(gpu_seconds / horizon) if gpu_seconds else 0)
        idle_gpus = sum(w["kind"] == "gpu" and not w.get("task") for w in snap["workers"].values())
        spare = max(0, snap["queued"]["check"] - idle_gpus)
        cap = {"slurm": self.cfg["slurm"]["cpu"]["max_jobs"], "local": self.cfg["local"]["cpu_workers"]}.get(self.cfg["scheduler"]["backend"], spare)
        stream = snap["running"]["synth"] * est["check_cpu"] / est["synth"]
        need = math.ceil(stream + spare * est["check_cpu"] / horizon)
        cpu = min(max(need, 1 if spare else 0), cap) if cpu_possible else 0
        uncovered = max(0.0, spare - cpu * horizon / est["check_cpu"]) if cpu_possible else 0.0
        gpu += min(math.ceil(uncovered), math.ceil(uncovered * est["check_gpu"] / horizon)) if uncovered else 0
        return {"gpu": max(gpu, 1 if synth else 0), "cpu": cpu}

    def log(self, name, entry):
        with open(self.run / name, "a") as log:
            log.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}) + "\n")

    def account(self, ended):
        for e in ended:
            self.board.release_handle(e["handle"], f"worker {e['outcome']}")
            counts = self.exits.setdefault(e["kind"], {})
            counts[e["outcome"]] = counts.get(e["outcome"], 0) + 1
            self.log("workers.log", e)
            if e["outcome"] == "failed":
                self.failed_in_row[e["kind"]] += 1
        write_json(self.run / "worker_exits.json", self.exits)
        limit = self.cfg["scheduler"]["max_worker_failures"]
        for kind, n in self.failed_in_row.items():
            if limit and n >= limit:
                raise RuntimeError(f"{n} {kind} workers failed in a row; see {self.run / 'workers.log'} and {self.run / 'logs'}")

    def step(self) -> bool:
        self.board.expire()
        results = self.board.drain()
        for r in results:
            if r.get("ok") and r.get("kind") in self.failed_in_row:
                self.failed_in_row[r["kind"]] = 0
        self.account(self.backend.ended())
        for r in results:
            if "released" in r:
                self.log("released.log", r)
            elif not r.get("ok"):
                self.errors.setdefault(r["dialogue"], []).append(str(r.get("worker", "")).rsplit("-", 2)[0])
                with open(self.run / "errors.log", "a") as log:
                    log.write(json.dumps(r) + "\n")
        write_json(self.run / "errors.json", self.errors)
        for did, d in self.dialogues.items():
            self.status[did] = self.board.busy(did) or self.advance(did, d)
        demand = self.demand()
        self.backend.scale(demand)
        counts = {s: sum(v == s for v in self.status.values()) for s in ("synth", "check", "queued", "working", *FINAL)}
        write_json(self.run / "status.json", {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "dialogues": len(self.status), "counts": counts,
                                              "demand": demand, "board": self.board.snapshot(), "worker_exits": self.exits})
        return all(v in FINAL for v in self.status.values())

    def report(self):
        with open(self.out / "report.jsonl", "w") as fh:
            for did, status in self.status.items():
                f = Folder(self.out / did)
                verdict = read_json(f.path("verdict.json"), {})
                chunks = [{"turns": c["turns"], "status": c["status"], "fail": c["fail"], "suspect": c["suspect"]} for c in verdict.get("chunks", [])]
                fh.write(json.dumps({"id": did, "status": status, "audio": str(f.path("dialogue.wav")), "redos": f.state["redos"],
                                     "parked": f.state["parked"], "chunks": chunks}) + "\n")

    def run_forever(self):
        try:
            while not self.step():
                time.sleep(self.cfg["scheduler"]["interval_seconds"])
            self.board.stopping = True
            deadline = time.time() + 120
            while self.board.snapshot()["workers"] and time.time() < deadline:
                self.board.expire()
                time.sleep(5)
        finally:
            self.backend.shutdown()
            self.report()
            self.daemon.close()
        return self.status
