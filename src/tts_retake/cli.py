"""Command line: run a job, start a worker, show progress, or check a configuration before running."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def interrupt(*_):
    raise KeyboardInterrupt


def run(a):
    from .config import load
    from .scheduler import Controller

    cfg = load(a.job, a.cluster)
    if a.limit:
        cfg["job"]["limit"] = a.limit
    signal.signal(signal.SIGTERM, interrupt)
    try:
        status = Controller(cfg).run_forever()
    except KeyboardInterrupt:
        print("stopped; run the same command again to continue")
        return 130
    except RuntimeError as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 1
    print(json.dumps({s: sum(v == s for v in status.values()) for s in set(status.values())}))
    return 0 if all(v == "done" for v in status.values()) else 1


def worker(a):
    from .worker import main

    info = json.loads(Path(a.daemon).read_text()) if a.daemon else {"url": a.url, "token": a.token}
    main(info, a.kind)


def status(a):
    from .config import load

    path = Path(load(a.job)["job"]["output"]) / ".run" / "status.json"
    if not path.exists():
        print("no status yet")
        return 1
    s = json.loads(path.read_text())
    board = s["board"]
    print(f"{s['at']}  dialogues {s['dialogues']}  " + "  ".join(f"{k} {v}" for k, v in s["counts"].items() if v))
    print("queued " + json.dumps(board["queued"]) + "  running " + json.dumps(board["running"]) + "  wanted workers " + json.dumps(s["demand"]))
    print("seconds per task " + json.dumps(board["seconds_per_task"]))
    for kind, counts in s.get("worker_exits", {}).items():
        print(f"{kind} workers ended: " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for w, info in board["workers"].items():
        print(f"  {w}  {info['kind']}  {info.get('task') or 'idle'}  done {info.get('done', 0)}")
    return 0


def check_config(a):
    from . import backends
    from .config import load
    from .dialogue import load_dialogues

    cfg = load(a.job, a.cluster)
    dialogues = load_dialogues(cfg["job"]["input"])
    voices = {s["voice"] for d in dialogues for s in d["speakers"].values()}
    missing = sorted(v for v in voices if not Path(v).exists())
    print(f"dialogues {len(dialogues)}, turns {sum(len(d['turns']) for d in dialogues)}, voices {len(voices)}, missing voices {len(missing)}")
    for v in missing[:10]:
        print("  missing", v)
    print(f"backend {cfg['scheduler']['backend']}, output {cfg['job']['output']}")
    problems = len(missing)
    for name, spec in (("worker", cfg["worker"]), *cfg["engines"].items()):
        probe = {"worker": "import tts_retake", "synth": "import torch, torchaudio, soundfile, transformers",
                 "check": "import torch, speechbrain, librosa, soundfile, " + ("nemo.collections.asr" if "parakeet" in cfg["check"]["asr"].lower() else "faster_whisper" if cfg["check"]["asr"].startswith("fw:") else "whisper")}[name]
        env = {**spec["env"], "PYTHONPATH": ":".join(p for p in (str(Path(__file__).resolve().parents[1]), spec["env"].get("PYTHONPATH", "")) if p)}
        r = subprocess.run([spec["python"], "-c", probe], capture_output=True, text=True, env={**os.environ, **env})
        problems += r.returncode != 0
        print(f"{name:7s} {spec['python']}: {'ok' if r.returncode == 0 else r.stderr.strip().splitlines()[-1]}")
    if cfg["scheduler"]["backend"] == "slurm":
        pool = backends.Slurm(cfg, Path(cfg["job"]["output"]) / ".run")
        for kind in ("gpu", "cpu"):
            script = pool.script(kind)
            print(f"--- {kind} job script ---\n{script}", end="")
            r = subprocess.run(["sbatch", "--test-only"], input=script, capture_output=True, text=True)
            problems += r.returncode != 0
            print(f"sbatch --test-only: {(r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else r.returncode}")
    print("ok" if not problems else f"{problems} problem(s)")
    return 1 if problems else 0


def bench(a):
    from . import bench as b

    if a.action == "build":
        print(json.dumps(b.build(a.paths[0], a.paths[1], a.per_kind, a.clean, a.seed, a.kinds.split(",") if a.kinds else None)))
    elif a.action == "prepare":
        job = b.prepare(a.paths[0], a.paths[1])
        print(f"run the checks with: tts-retake run {job} --cluster <cluster file>")
    else:
        b.score(a.paths[0])
        print((Path(a.paths[0]) / "bench_report.md").read_text(), end="")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tts-retake", description="Quality-gated two-speaker dialogue synthesis with MOSS-TTSD.")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="synthesize, check and repair every dialogue of a job")
    p.add_argument("job")
    p.add_argument("--cluster")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=run)
    p = sub.add_parser("worker", help="serve a running job (started by the scheduler, or by hand on another machine)")
    p.add_argument("--daemon", help="daemon.json written by the controller")
    p.add_argument("--url")
    p.add_argument("--token")
    p.add_argument("--kind", choices=["gpu", "cpu"], default="gpu")
    p.set_defaults(fn=worker)
    p = sub.add_parser("status", help="progress of a job")
    p.add_argument("job")
    p.set_defaults(fn=status)
    p = sub.add_parser("check-config", help="validate the input, the Python environments and the job scripts")
    p.add_argument("job")
    p.add_argument("--cluster")
    p.set_defaults(fn=check_config)
    p = sub.add_parser("bench", help="benchmark the checks: build items with known defects, prepare a check-only run, score it")
    p.add_argument("action", choices=["build", "prepare", "score"])
    p.add_argument("paths", nargs="+", help="build SOURCE OUT | prepare BENCH RUN | score RUN")
    p.add_argument("--per-kind", type=int, default=30)
    p.add_argument("--clean", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kinds", default="", help="comma-separated defect kinds (default: all)")
    p.set_defaults(fn=bench)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
