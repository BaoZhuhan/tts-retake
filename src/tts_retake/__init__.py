"""tts-retake: quality-gated dialogue synthesis with MOSS-TTSD or MOSS-TTS. Numba's JIT cache is kept per machine: a cache in a
shared site-packages, written by other nodes, segfaults librosa.pyin."""
import os
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), f"tts-retake-numba-{os.getuid()}"))
__version__ = "0.1.0"
