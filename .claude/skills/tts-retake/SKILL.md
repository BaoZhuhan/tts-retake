---
name: tts-retake
description: Set up and run tts-retake (quality-gated MOSS-TTSD dialogue synthesis) on the user's machine or cluster. Use when the user wants to synthesize two-speaker dialogues with MOSS-TTSD, write a tts-retake cluster file for their environment, or start, monitor or resume a tts-retake job.
---

# tts-retake

Goal: a working cluster file for this environment, a validated job, and a running controller. Ask the user only for what you cannot find.

## 1. Find out where you are

- `nvidia-smi --query-gpu=index,name,memory.total --format=csv` shows local GPUs.
- `command -v sbatch` tells whether this is a SLURM cluster. If it is:
  - `sacctmgr -nP show assoc user=$USER format=account,partition,qos` lists the accounts.
  - `sinfo -o "%P %G %l %c %m"` lists partitions, GPU types, time limits, CPUs and memory.
  - Note GPU types with less than 20 GB (for example 2080, 1080, P100): put their nodes in `exclude`, or leave them in and they will only run checks.
- Find Python environments: an environment that imports `torch`, `torchaudio`, `transformers` and `soundfile` for synthesis, and one that imports
  `torch`, `whisper`, `speechbrain`, `librosa` and `soundfile` for checks. They may be the same. The controller needs Python 3.11 or newer.
- Find the MOSS-TTS repository (the folder with `clis/moss_ttsd_app.py`) and the Hugging Face cache that holds `OpenMOSS-Team/MOSS-TTSD-v1.0`.
  If compute nodes have no internet, set `HF_HUB_OFFLINE = "1"` in the synthesis environment.

## 2. Write the cluster file

Start from `configs/clusters/local.toml` (one machine) or `configs/clusters/slurm.toml` (SLURM) and save it as
`configs/clusters/<site>.local.toml`, which git ignores. Fill in:

- `[synth] moss_cli_dir`, `[engines.synth] python` and `env`, `[engines.check] python` and `env`, `[worker] python`.
- SLURM: `account`, `partition`, `gres`, `time`, `mem` for `[slurm.gpu]` and `[slurm.cpu]`; `max_jobs`, `max_pending`, `max_cpus`,
  `max_mem_gb` within the user's allocation; `setup` lines the site needs (modules, `umask`).
- Local: `gpus` ("auto" or a list of indices) and `cpu_workers`.

Every other setting has a default in `src/tts_retake/config.py`; an unknown key is an error, so typos show up at once.

## 3. Validate

```bash
tts-retake check-config job.toml --cluster configs/clusters/<site>.local.toml
```

Fix every line that is not `ok`. On SLURM it prints both job scripts and runs `sbatch --test-only`; start estimates from `--test-only` are
often pessimistic, so a late estimate is not a reason to change partitions.

## 4. Run

1. Smoke test with one dialogue: `tts-retake run job.toml --cluster <file> --limit 1`. Expect `output/<id>/dialogue.wav` and a `done` line.
2. Full run. The controller must stay up for the whole job: use `tmux`/`nohup` on a machine that allows long processes, or submit the
   controller itself as a 1-CPU SLURM job on clusters that kill long processes on login nodes.
3. Progress: `tts-retake status job.toml`. Worker logs are in `output/.run/logs/`, task errors in `output/.run/errors.log`.
4. After a stop or crash, run the same command again; it continues from the files on disk.

## Notes

- English only. Two speakers per dialogue.
- `output/report.jsonl` lists every dialogue; `review` means some chunk never passed and should be listened to.
- Thresholds were tuned on English clinical dialogues. For a new kind of material, run a small batch first and listen to a few `pass` and
  `fail` chunks before trusting them.
