"""Engine process: loads the synthesis and check models once and runs the tasks its worker sends as JSON lines."""
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
            result = {"ok": True, **loaded[req["type"]].run(req["folder"])}
        except Exception:
            result = {"ok": False, "error": traceback.format_exc()[-3000:]}
        reply.write(json.dumps(result) + "\n")
        reply.flush()


if __name__ == "__main__":
    main()
