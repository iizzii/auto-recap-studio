"""Karaoke-style ASS caption engine aligned to exact script text."""

from __future__ import annotations

import difflib
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Any

from pipeline.models import Beat

log = logging.getLogger(__name__)


def _ass_time(sec: float) -> str:
    """Format seconds as ASS timestamp ``H:MM:SS.cs``."""
    cs_total = max(0, int(round(sec * 100)))
    cs = cs_total % 100
    s_total = cs_total // 100
    h = s_total // 3600
    m = (s_total % 3600) // 60
    s = s_total % 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _hex_to_ass_bgr(hex_color: str) -> str:
    """Convert ``#RRGGBB`` to ASS ``&H00BBGGRR``."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "&H00FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


def _resolve_font(preferred: str, fallback: str) -> str:
    """Verify *preferred* font via ``fc-list``, falling back to *fallback*."""
    try:
        res = subprocess.run(
            ["fc-list", f":family={preferred}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return preferred
    except OSError:
        pass
    return fallback


def _align_words_to_script(
    beats: list[Beat],
    whisper_words: list[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """Align Whisper word timings to exact script words via SequenceMatcher."""
    script_tokens: list[tuple[str, float, float]] = []
    t_cursor = 0.0
    for beat in beats:
        dur = beat.audio_duration_sec or 1.0
        words = beat.narration.split()
        step = dur / max(1, len(words))
        for i, w in enumerate(words):
            script_tokens.append((w, t_cursor + i * step, t_cursor + (i + 1) * step))
        t_cursor += dur

    if not whisper_words:
        return script_tokens

    s_norm = [w.lower().strip(".,!?;:\"'") for w, _, _ in script_tokens]
    w_norm = [w.lower().strip(".,!?;:\"'") for w, _, _ in whisper_words]
    sm = difflib.SequenceMatcher(None, s_norm, w_norm)

    aligned: list[tuple[str, float, float]] = list(script_tokens)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                word = script_tokens[i1 + offset][0]
                _, ws, we = whisper_words[j1 + offset]
                aligned[i1 + offset] = (word, ws, max(ws + 0.05, we))
        elif tag == "replace" and (i2 - i1) > 0 and (j2 - j1) > 0:
            span_start = whisper_words[j1][1]
            span_end = max(span_start + 0.1, whisper_words[j2 - 1][2])
            step = (span_end - span_start) / (i2 - i1)
            for k in range(i2 - i1):
                word = script_tokens[i1 + k][0]
                aligned[i1 + k] = (
                    word,
                    span_start + k * step,
                    span_start + (k + 1) * step,
                )
    return aligned


def _chunk_words(
    words: list[tuple[str, float, float]],
    min_words: int = 2,
    max_words: int = 4,
    max_chars: int = 18,
) -> list[list[tuple[str, float, float]]]:
    """Group word timings into 2-4 word chunks <= *max_chars* without splitting contractions."""
    chunks: list[list[tuple[str, float, float]]] = []
    cur: list[tuple[str, float, float]] = []
    cur_len = 0

    for item in words:
        w = item[0]
        added_len = len(w) + (1 if cur else 0)
        if cur and (len(cur) >= max_words or (len(cur) >= min_words and cur_len + added_len > max_chars)):
            chunks.append(cur)
            cur = [item]
            cur_len = len(w)
        else:
            cur.append(item)
            cur_len += added_len

    if cur:
        chunks.append(cur)
    return chunks


def generate_captions(
    slug: str,
    beats: list[Beat],
    stitched_voice: Path,
    work_dir: Path,
    config: dict[str, Any],
    *,
    mock_tts: bool = False,
) -> tuple[Path, float]:
    """Generate ``05_captions.ass`` and return ``(ass_path, word_match_ratio)``."""
    cap_cfg = config.get("captions", {})
    whisper_words: list[tuple[str, float, float]] = []

    if not mock_tts:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            str(cap_cfg.get("model", "base.en")),
            device=str(cap_cfg.get("device", "cpu")),
            compute_type=str(cap_cfg.get("compute_type", "int8")),
            cpu_threads=int(cap_cfg.get("cpu_threads", 2)),
        )
        segments, _ = model.transcribe(str(stitched_voice), word_timestamps=True)
        for seg in segments:
            for w in getattr(seg, "words", []) or []:
                txt = w.word.strip()
                if txt:
                    whisper_words.append((txt, float(w.start), float(w.end)))

    aligned = _align_words_to_script(beats, whisper_words)
    w_min, w_max = cap_cfg.get("words_per_chunk", [2, 4])
    max_chars = int(cap_cfg.get("max_chars_per_line", 18))
    chunks = _chunk_words(aligned, int(w_min), int(w_max), max_chars)

    font_name = _resolve_font(
        str(cap_cfg.get("font_name", "Montserrat ExtraBold")),
        str(cap_cfg.get("font_fallback", "DejaVu Sans Bold")),
    )
    font_size = int(cap_cfg.get("font_size", 78))
    outline_px = int(cap_cfg.get("outline_px", 5))
    shadow_px = int(cap_cfg.get("shadow_px", 2))
    alignment = int(cap_cfg.get("alignment", 2))
    margin_v = int(cap_cfg.get("margin_v", 420))

    rot = cap_cfg.get("accent_rotation", ["#FFE600"])
    digest = int(hashlib.sha256(slug.encode("utf-8")).hexdigest(), 16)
    active_hex = rot[digest % len(rot)]
    active_ass = _hex_to_ass_bgr(active_hex)
    inactive_ass = _hex_to_ass_bgr(str(cap_cfg.get("inactive_color", "#FFFFFF")))

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{font_size},{active_ass},{inactive_ass},"
        f"&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline_px},{shadow_px},"
        f"{alignment},40,40,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    for chunk in chunks:
        c_start = chunk[0][1]
        c_end = max(c_start + 0.1, chunk[-1][2])
        k_parts: list[str] = []
        for word, ws, we in chunk:
            dur_cs = max(5, int(round((we - ws) * 100)))
            k_parts.append(f"{{\\kf{dur_cs}}}{word}")
        text_line = " ".join(k_parts)
        events.append(
            f"Dialogue: 0,{_ass_time(c_start)},{_ass_time(c_end)},Default,,0,0,0,,{text_line}"
        )

    ass_path = work_dir / "05_captions.ass"
    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")

    # Word match ratio (1.0 in mock or exact alignment)
    match_ratio = 1.0
    if whisper_words:
        s_words = " ".join(w for w, _, _ in aligned).lower().split()
        w_words = " ".join(w for w, _, _ in whisper_words).lower().split()
        match_ratio = difflib.SequenceMatcher(None, s_words, w_words).ratio()
    return ass_path, match_ratio
