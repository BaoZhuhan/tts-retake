"""Coordinator daemon: an in-memory task board served over HTTP, so workers on any machine claim work without shared-file locks."""
from __future__ import annotations

import json
import os
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request

from .dialogue import write_json


class Board:
    def __init__(self, cfg):
        self.lease = 60 * cfg["scheduler"]["lease_minutes"]
        self.wait = min(10, cfg["scheduler"]["interval_seconds"])
        self.lock = threading.Lock()
        self.tasks, self.workers, self.results = {}, {}, []
        self.speed = dict(cfg["scheduler"]["estimates"])
        self.stopping = False

    def put(self, dialogue, kind, priority):
        with self.lock:
            self.tasks.setdefault(dialogue, {"dialogue": dialogue, "type": kind, "priority": priority, "worker": None,
                                             "queued": time.time()})

    def busy(self, dialogue):
        with self.lock:
            t = self.tasks.get(dialogue)
            return t and ("working" if t["worker"] else "queued")

    def claim(self, worker):
        with self.lock:
            now = time.time()
            self.workers[worker["id"]] = {**self.workers.get(worker["id"], {}), **worker, "seen": now}
            if self.stopping:
                self.workers.pop(worker["id"], None)
                return {"stop": True}
            types = ["synth", "check"] if worker["kind"] == "gpu" else ["check"]
            free = [t for t in self.tasks.values() if t["worker"] is None and t["type"] in types]
            if not free:
                return {"wait": self.wait}
            t = min(free, key=lambda t: (types.index(t["type"]), -t["priority"], t["queued"]))
            t.update(worker=worker["id"], lease=now + self.lease, started=now)
            self.workers[worker["id"]]["task"] = t["dialogue"]
            return {"task": {"dialogue": t["dialogue"], "type": t["type"]}}

    def beat(self, worker, dialogue):
        with self.lock:
            if worker in self.workers:
                self.workers[worker]["seen"] = time.time()
            t = self.tasks.get(dialogue)
            if t and t["worker"] == worker:
                t["lease"] = time.time() + self.lease
            return {"ok": bool(t and t["worker"] == worker)}

    def done(self, worker, dialogue, result):
        with self.lock:
            t = self.tasks.get(dialogue)
            if not t or t["worker"] != worker:
                return {"ok": False}
            del self.tasks[dialogue]
            w = self.workers.get(worker, {})
            w.update(task=None, done=w.get("done", 0) + 1)
            seconds = time.time() - t["started"]
            name = "synth" if t["type"] == "synth" else f"check_{w.get('kind', 'cpu')}"
            if result.get("ok"):
                self.speed[name] = round(0.7 * self.speed[name] + 0.3 * seconds, 1)
            self.results.append({**result, "dialogue": dialogue, "type": t["type"], "seconds": round(seconds, 1), "worker": worker,
                                 "kind": w.get("kind")})
            return {"ok": True}

    def free(self, worker, reason):
        w = self.workers.pop(worker, None) or {}
        for t in self.tasks.values():
            if t["worker"] == worker:
                t["worker"] = None
                self.results.append({"released": reason, "dialogue": t["dialogue"], "type": t["type"], "worker": worker,
                                     "seconds": round(time.time() - t["started"], 1)})
        return w

    def release(self, worker, reason):
        with self.lock:
            self.free(worker, reason)
            return {"ok": True}

    def release_handle(self, handle, reason):
        with self.lock:
            for worker in [k for k, w in self.workers.items() if handle and w.get("handle") == handle]:
                self.free(worker, reason)

    def expire(self):
        with self.lock:
            now = time.time()
            for t in self.tasks.values():
                if t["worker"] and t["lease"] < now:
                    t["worker"] = None
            self.workers = {k: w for k, w in self.workers.items() if now - w["seen"] < self.lease}

    def drain(self) -> list:
        with self.lock:
            out, self.results = self.results, []
            return out

    def snapshot(self) -> dict:
        with self.lock:
            count = lambda kind, running: sum(t["type"] == kind and bool(t["worker"]) == running for t in self.tasks.values())
            return {"queued": {k: count(k, False) for k in ("synth", "check")}, "running": {k: count(k, True) for k in ("synth", "check")},
                    "workers": {k: dict(w) for k, w in self.workers.items()}, "seconds_per_task": dict(self.speed)}


class Daemon:
    def __init__(self, board, cfg, run_dir):
        token = secrets.token_hex(16)
        routes = {"/claim": lambda b: board.claim(b["worker"]), "/beat": lambda b: board.beat(b["worker"], b.get("dialogue")),
                  "/done": lambda b: board.done(b["worker"], b["dialogue"], b["result"]), "/config": lambda b: cfg,
                  "/release": lambda b: board.release(b["worker"], b.get("reason", "released")),
                  "/status": lambda b: board.snapshot()}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                route = routes.get(self.path)
                if self.headers.get("X-Token") != token or route is None:
                    return self.reply(403, {"error": "forbidden"})
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                self.reply(200, route(body))

            def reply(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("0.0.0.0", cfg["scheduler"]["port"]), Handler)
        host = cfg["scheduler"]["host"] or socket.getfqdn()
        self.info = {"url": f"http://{host}:{self.server.server_address[1]}", "token": token}
        path = run_dir / "daemon.json"
        write_json(path, self.info)
        os.chmod(path, 0o600)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def call(info, path, body=None, timeout=60):
    data = json.dumps(body or {}).encode()
    req = request.Request(info["url"] + path, data=data, headers={"X-Token": info["token"], "Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
