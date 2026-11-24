"""Per-language text handling: tokens for alignment, length units for chunking and speaking rate, and model language names."""
import re

# code: (name used by MOSS-TTS, factor that turns units per minute into English-like words per minute)
LANGS = {"en": ("English", 1.0), "zh": ("Chinese", 0.6), "fr": ("French", 0.9), "es": ("Spanish", 0.85), "de": ("German", 1.1)}
CJK = "㐀-䶿一-鿿豈-﫿"
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
