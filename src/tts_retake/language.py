"""Per-language text handling: tokens for alignment, length units for chunking and speaking rate, and model language names.
LANGS maps a code to (name MOSS-TTS expects, English words per unit of the same text), so every rate reads as English words per minute."""
import re

LANGS = {"en": ("English", 1.0), "zh": ("Chinese", 0.62), "fr": ("French", 0.89), "es": ("Spanish", 0.97), "de": ("German", 0.97), "ja": ("Japanese", 0.45), "ko": ("Korean", 1.47)}
CJK = "\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
TOKEN = re.compile(rf"[{CJK}]|[^\W_{CJK}]+(?:'[^\W_{CJK}]+)*")


def check(lang: str) -> str:
    if lang not in LANGS:
        raise ValueError(f"language {lang!r} is not supported; use one of {', '.join(LANGS)}")
    return lang


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.casefold().replace("’", "'"))


def units(text: str) -> int:
    return len(tokens(text))


def is_cjk(token: str) -> bool:
    return bool(re.fullmatch(f"[{CJK}]", token))


def wpm(n_units: int, seconds: float, lang: str) -> float:
    return n_units / max(seconds, 1e-6) * 60 * LANGS[lang][1]
