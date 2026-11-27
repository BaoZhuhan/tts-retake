# tts-retake

Quality-gated two-speaker dialogue synthesis with [MOSS-TTSD](https://github.com/OpenMOSS/MOSS-TTSD) 1.0 or
[MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) 1.5. Long dialogues are synthesized in chunks;
every chunk is checked automatically, and only the chunks that fail are synthesized again.

- **Checks**: Whisper word alignment (missing or added words, dropped opening or ending, reference text read aloud), speaking rate,
  sustained tones, long silences, and ECAPA speaker similarity for swapped or drifting voices.
- **Repair**: a failed chunk is retaken up to 6 times and split in two after 2 retakes; a suspect chunk gets one more take that is kept only
  if it passes; chunks that never pass are listed for human review.
- **Scheduler**: a small HTTP daemon hands out work, so workers on any machine claim tasks without shared-file locks. The pool grows and
  shrinks with the remaining work and the measured speed of each task.
- **Anywhere**: one machine with GPUs, a SLURM cluster, or workers started by hand. Where to run is one TOML file.

English, Chinese, French, Spanish and German.

| Engine | `[synth]` in the job file |
|---|---|
| MOSS-TTSD 1.0 (default): each chunk is generated as one two-speaker passage | nothing to set |
| MOSS-TTS 1.5: each turn is cloned from its own speaker, a chunk's turns are generated as one batch | `class = "tts_retake.synth_tts:SynthTTS"`, `model = "OpenMOSS-Team/MOSS-TTS-v1.5"` |

## Install

```bash
git clone https://github.com/BaoZhuhan/tts-retake && cd tts-retake
pip install -e .
```

The controller needs Python 3.11 and nothing else. Synthesis needs an environment with MOSS-TTSD and the `clis/` folder of the
[MOSS-TTS repository](https://github.com/OpenMOSS/MOSS-TTS); checks need `speechbrain`, `librosa`, `faster-whisper` (English default `asr = "fw:small.en"`, fast enough for CPU workers) and
`openai-whisper`; `fw:` picks faster-whisper for any Whisper model (`fw5:` with beam 5), a bare name openai-whisper (GPU only: slower than real time on CPU). Other languages use `check.asr_multilingual`, one model for
all or one per language (by default faster-whisper large-v3, beam 5 for zh, from a bake-off, so every language checks on CPU workers). Each can live in its own
environment; the cluster file says which Python runs which step.

Optional faster engine: [SGLang-Omni](https://github.com/sgl-project/sglang-omni) at commit `87a11dbe9` with
`patches/sglang-omni-prompt-rows.patch` applied (adds MOSS-TTSD continuation prompts and per-request sampling, and fixes the
prompt audio being returned with the output). Set `[synth.sglang] command = "/path/to/sgl-omni"` (plus `config`, a pipeline YAML,
and `concurrency`); each worker then starts a private server on its GPU. On one H200, with one take per item and the same checks,
MOSS-TTS 1.5 runs at RTF 0.017 (concurrency 8) against FastGen's 0.049, and MOSS-TTSD at 0.058 (concurrency 16) against 0.108;
pass/fail counts match FastGen's within a chunk or two.

## Input

One dialogue per line of a JSONL file. Voice paths are relative to the file; `text` is the sentence spoken in the reference clip.

```json
{"id": "visit-001", "priority": 0, "language": "en",
 "speakers": {"doctor": {"voice": "voices/doctor.wav", "text": "Good morning, how can I help you today?"},
              "patient": {"voice": "voices/patient.wav", "text": "I have been having pain in my knee."}},
 "turns": [{"speaker": "doctor", "text": "Come on in, have a seat."},
           {"speaker": "patient", "text": "Thanks. It's my knee again.", "pause": 1.2}]}
```

`pause` (seconds after a turn) is optional. Higher `priority` dialogues go first. `language` is `en` (default), `zh`, `fr`, `es`
or `de`; English is checked with `check.asr`, other languages with `check.asr_multilingual` (Whisper large-v3).

## Run

```bash
cp configs/job.example.toml job.toml
cp configs/clusters/slurm.toml my-cluster.toml
tts-retake check-config job.toml --cluster my-cluster.toml
tts-retake run job.toml --cluster my-cluster.toml --limit 1
tts-retake run job.toml --cluster my-cluster.toml
tts-retake status job.toml
```

`check-config` validates the input, imports each Python environment and, on SLURM, prints the job scripts and tests them with
`sbatch --test-only`. Run the controller where it can stay up for the whole job (a `tmux` session, or a small SLURM job).

Results go to `output/<id>/dialogue.wav`, with `output/report.jsonl` listing every dialogue, its retakes and any chunk left for review.
A run can be stopped and started again; it continues from the files on disk.

## Where to run

| Cluster file | For |
|---|---|
| `configs/clusters/local.toml` | a workstation or bare-metal server: one worker per GPU plus CPU check workers |
| `configs/clusters/slurm.toml` | a SLURM cluster: account, partitions, job sizes, limits on jobs, CPUs and memory |

With `backend = "manual"`, start workers yourself on any machine that can reach the controller:

```bash
tts-retake worker --url http://controller-host:PORT --token TOKEN --kind gpu
```

The URL and token are in `output/.run/daemon.json`.

A preempted or cancelled worker hands its task back at once and is replaced; the controller stops after `max_worker_failures`
failed workers of one kind in a row.

All settings and their defaults are in [`src/tts_retake/config.py`](src/tts_retake/config.py). The defaults come from synthesizing and
repairing 1,410 English clinical dialogues (about 11,000 chunks) for the SIMAX dataset.

## Benchmark the checks

Start from dialogue folders whose chunks you trust (`plan.json`, `chunk_NNN.wav`, and `spans.json` with each turn's start and end):

```bash
tts-retake bench build trusted/ bench/ --per-kind 30 --clean 60
tts-retake bench prepare bench/ runs/whisper-small
tts-retake run runs/whisper-small/job.toml --cluster my-cluster.toml
tts-retake bench score runs/whisper-small
```

`build` cuts known defects into the middle chunk of some folders (removed or inserted speech, reference clip, rate, silence, tone) and
copies folders that already carry a labelled `label.json`. `score` reports detection per defect and the flag rate of clean chunks.

## Agent skill

[`.claude/skills/tts-retake/SKILL.md`](.claude/skills/tts-retake/SKILL.md) lets a coding agent such as Claude Code inspect the machine
or cluster it runs on, write the cluster file, validate it and start a job.

## License

MIT
