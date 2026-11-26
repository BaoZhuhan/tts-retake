"""Engine process: loads the synthesis and check models once and runs the tasks its worker sends as JSON lines (a request with
"folders" runs several dialogues at once through the engine's run_many); cached GPU memory
is handed back after each task so the synth and check engines of one GPU worker fit side by side."""
import importlib
import json
import os
import sys
import traceback


def main():
    reply = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    cfg = json.loads(os.environ["TTS_RETAKE_CONFIG"])
    loaded = {}
    for line in sys.stdin:
        req = json.loads(line)
        try:
            if req["type"] not in loaded:
                module, name = cfg[req["type"]]["class"].split(":")
                loaded[req["type"]] = getattr(importlib.import_module(module), name)(cfg)
            if "folders" in req:
                result = {"ok": True, "results": loaded[req["type"]].run_many(req["folders"])}
            else:
                result = {"ok": True, **loaded[req["type"]].run(req["folder"])}
        except Exception:
            result = {"ok": False, "error": traceback.format_exc()[-3000:]}
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            sys.modules["torch"].cuda.empty_cache()
        reply.write(json.dumps(result) + "\n")
        reply.flush()


if __name__ == "__main__":
    main()
