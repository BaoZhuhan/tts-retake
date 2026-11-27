"""Configuration: built-in defaults merged with a job file (what to synthesize) and a cluster file (where to run it)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

DEFAULTS = {
    "job": {"input": "", "output": "", "limit": 0},
    "synth": {"class": "tts_retake.synth:Synth", "moss_cli_dir": "", "model": "OpenMOSS-Team/MOSS-TTSD-v1.0",
              "codec": "OpenMOSS-Team/MOSS-Audio-Tokenizer", "temperature": 1.1, "top_p": 0.9, "top_k": 50,
              "repetition_penalty": 1.1, "attempts": 3, "rate_wpm": [110, 225], "target_wpm": 160, "duration_control": True,
              "max_chunk_words": 700, "min_chunk_words": 250, "cut_pause": 5.0, "low_vram_below_gb": 30, "batch": 8, "fast": True, "group": 4,
              "sglang": {"command": "", "config": "", "concurrency": 8, "start_minutes": 20, "min_gb": 30, "ttsd_min_gb": 80}},
    "check": {"class": "tts_retake.check:Checker", "asr": "fw:small.en", "asr_multilingual": {"de": "fw:large-v3", "es": "fw:large-v3", "fr": "fw:large-v3", "zh": "fw5:large-v3"}, "speaker_model": "speechbrain/spkrec-ecapa-voxceleb",
              "missing_fail": 0.20, "missing_suspect": 0.10, "added_fail": 0.18, "added_suspect": 0.08,
              "wpm_fail": [95, 225], "wpm_suspect": [110, 195], "silence_suspect": 6.0, "lead_speech_fail": 3.0, "swap_margin": 0.05,
              "swap_run_cross_gender": 3, "swap_run_same_gender": 4, "drift_own": 0.30, "drift_reference": 0.25,
              "drift_hard": 0.15, "swap_ref_margin": 0.3, "change_drop": 0.4, "change_min_s": 1.5, "female_f0_min": 145, "male_f0_max": 175, "assemble": True},
    "repair": {"max_redos": 6, "split_after": 2, "chronic_after": 3, "retry_suspect": True, "max_task_errors": 3},
    "scheduler": {"backend": "local", "host": "", "port": 0, "interval_seconds": 15, "lease_minutes": 20,
                  "idle_minutes": 30, "target_minutes": 60, "min_gpu_gb": 20, "max_worker_failures": 10,
                  "estimates": {"synth": 900, "check_gpu": 60, "check_cpu": 180}},
    "engines": {"synth": {"python": "", "env": {}}, "check": {"python": "", "env": {}}},
    "worker": {"python": "", "env": {}},
    "local": {"gpus": "auto", "cpu_workers": 1},
    "slurm": {"job_name": "tts-retake", "setup": [], "max_cpus": 0, "max_mem_gb": 0,
              "gpu": {"max_jobs": 4, "max_pending": 2, "options": {"gres": "gpu:1", "cpus-per-task": 4, "mem": "48G", "time": "24:00:00"}},
              "cpu": {"max_jobs": 4, "max_pending": 2, "options": {"cpus-per-task": 4, "mem": "16G", "time": "12:00:00"}}},
}
FREE = {"env", "options", "estimates", "asr_multilingual"}


def merge(base, over, where=""):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in base and where.rsplit(".", 1)[-1] not in FREE:
            raise ValueError(f"unknown setting {where + '.' if where else ''}{k}")
        out[k] = merge(base.get(k, {}), v, f"{where}.{k}".strip(".")) if isinstance(v, dict) and isinstance(base.get(k, {}), dict) else v
    return out


def load(job, cluster=None) -> dict:
    import tomllib

    job = Path(job).resolve()
    cfg = merge(DEFAULTS, tomllib.loads(job.read_text(encoding="utf-8")))
    if cluster:
        cfg = merge(cfg, tomllib.loads(Path(cluster).read_text(encoding="utf-8")))
    for name in ("input", "output"):
        if not cfg["job"][name]:
            raise ValueError(f"job.{name} is required")
        cfg["job"][name] = str((job.parent / cfg["job"][name]).resolve())
    for spec in (cfg["worker"], *cfg["engines"].values()):
        spec["python"] = spec["python"] or sys.executable
    return cfg
