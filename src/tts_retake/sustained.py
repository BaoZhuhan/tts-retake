"""Detect sustained non-speech sound (a collapsed generation) in synthesized audio."""
import numpy as np, librosa


def sustained_windows(y: np.ndarray, sr: int, win_s=1.0, hop_s=0.5, min_db=-35.0, max_cv=0.5, max_mod=0.3, flat_cv=0.1):
    hop = 240 if sr >= 20000 else 160
    rms = librosa.feature.rms(y=y, frame_length=2 * hop, hop_length=hop)[0]
    fps = sr / hop
    n, h = int(win_s * fps), int(hop_s * fps)
    out = []
    for i in range(0, len(rms) - n + 1, h):
        r = rms[i:i + n]
        db = 20 * np.log10(r.mean() + 1e-9)
        cv = r.std() / (r.mean() + 1e-9)
        env = r - r.mean()
        fe = np.abs(np.fft.rfft(env * np.hanning(len(env)))); fr = np.fft.rfftfreq(len(env), 1 / fps)
        mod = fe[(fr >= 2) & (fr <= 8)].sum() / (fe[1:].sum() + 1e-9)
        out.append((i / fps, float(db > min_db and (cv < flat_cv or (cv < max_cv and mod < max_mod)))))
    return np.array(out)


def runs(w, win_s=1.0, min_len_s=2.0):
    res, s = [], None
    for t, f in w:
        if f and s is None: s = t
        if (not f) and s is not None:
            if t - s >= min_len_s: res.append((s, t + win_s - 0.5))
            s = None
    if s is not None and w[-1, 0] + win_s - s >= min_len_s: res.append((s, w[-1, 0] + win_s))
    return res
