"""Karaoke-style ASS caption engine aligned to exact script text."""

from __future__ import annotations

import difflib
import hashlib
import logging
import re
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
    """Convert ``#RRGGBB`` to ASS Style ``&H00BBGGRR``."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "&H00FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


def _hex_to_ass_tag_bgr(hex_color: str) -> str:
    """Convert ``#RRGGBB`` to ASS inline override tag ``&HBBGGRR&``."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "&HFFFFFF&"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{b}{g}{r}&".upper()


def _resolve_font(preferred: str, fallback: str) -> str:
    """Resolve *preferred* font from bundled assets/fonts TTF or system ``fc-list``."""
    fonts_dir = Path(__file__).resolve().parent.parent / "assets" / "fonts"
    pref_key = re.sub(r"[^a-z0-9]", "", preferred.lower())
    if fonts_dir.exists():
        for ttf in sorted(fonts_dir.glob("*.ttf")):
            if pref_key and pref_key[:8] in re.sub(r"[^a-z0-9]", "", ttf.stem.lower()):
                try:
                    from PIL import ImageFont

                    fam, _style = ImageFont.truetype(str(ttf), 24).getname()
                    if fam:
                        return fam
                except Exception:
                    pass
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
        clean_nar = re.sub(r"[—–/]+", " ", beat.narration)
        words = [w.strip("—–- ") for w in clean_nar.split() if w.strip("—–- ")]
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
    caption_mode = str(cap_cfg.get("mode", "single_word"))
    min_word_ms = int(cap_cfg.get("min_word_display_ms", 110))
    w_min, w_max = cap_cfg.get("words_per_chunk", [1, 1])
    max_chars = int(cap_cfg.get("max_chars_per_line", 16))

    if caption_mode == "single_word":
        merged_single: list[list[tuple[str, float, float]]] = []
        i = 0
        while i < len(aligned):
            w, ws, we = aligned[i]
            dur_ms = (we - ws) * 1000.0
            if (
                dur_ms < min_word_ms
                and len(w) <= 3
                and i + 1 < len(aligned)
                and len(w) + 1 + len(aligned[i + 1][0]) <= max_chars
            ):
                nw, _nws, nwe = aligned[i + 1]
                merged_single.append([(f"{w} {nw}", ws, nwe)])
                i += 2
            else:
                merged_single.append([(w, ws, we)])
                i += 1
        chunks = merged_single
    else:
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
    prev_end = 0.0
    for chunk in chunks:
        c_start = max(prev_end, chunk[0][1])
        c_end = max(c_start + 0.08, chunk[-1][2])
        prev_end = c_end
        if caption_mode == "single_word":
            text_line = " ".join(w.strip("—–- ").upper() for w, _, _ in chunk if w.strip("—–- "))
            if not text_line:
                continue
        else:
            k_parts: list[str] = []
            for word, ws, we in chunk:
                dur_cs = max(5, int(round((we - ws) * 100)))
                k_parts.append(f"{{\\k{dur_cs}}}{word.upper()}")
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


_REACTION_GOLD_WORDS: frozenset[str] = frozenset(
    {
        "wait",
        "bro",
        "no",
        "nope",
        "what",
        "whoa",
        "yikes",
        "karma",
        "cooked",
        "trap",
        "run",
        "dead",
        "wild",
        "insane",
        "snapped",
        "boom",
        "nah",
        "bruh",
        "ouch",
        "stop",
        "help",
        "ambush",
        "plot",
        "twist",
    }
)


def _norm_token(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _is_emphasis_token(raw_word: str, emphasis_word: str) -> bool:
    norm = _norm_token(raw_word)
    if not norm:
        return False
    emp_norm = _norm_token(emphasis_word)
    if emp_norm and (norm == emp_norm or (len(emp_norm) >= 4 and emp_norm in norm)):
        return True
    if raw_word.rstrip("\"'").endswith("!"):
        return True
    return norm in _REACTION_GOLD_WORDS


def _align_cue_words(
    cue_text: str,
    cue_start: float,
    cue_end: float,
    whisper_words: list[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """Align words of a single cue to Whisper word timings inside [cue_start, cue_end]."""
    clean = re.sub(r"[—–/]+", " ", cue_text)
    words = [w.strip("—–- ") for w in clean.split() if w.strip("—–- ")]
    if not words:
        return []
    dur = max(0.15, cue_end - cue_start)
    weights = [max(2, len(_norm_token(w)) or 2) for w in words]
    total_w = float(sum(weights))
    cursor = cue_start
    fallback: list[tuple[str, float, float]] = []
    for w, wt in zip(words, weights):
        wdur = dur * (wt / total_w)
        fallback.append((w, cursor, cursor + wdur))
        cursor += wdur

    local_w = [
        (tw, max(cue_start, ws), min(cue_end, we))
        for tw, ws, we in whisper_words
        if we > cue_start - 0.15 and ws < cue_end + 0.15 and we > ws
    ]
    if not local_w:
        return fallback

    s_norm = [_norm_token(w) for w in words]
    w_norm = [_norm_token(w) for w, _, _ in local_w]
    sm = difflib.SequenceMatcher(None, s_norm, w_norm)
    aligned = list(fallback)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                _, ws, we = local_w[j1 + off]
                aligned[i1 + off] = (words[i1 + off], ws, max(ws + 0.06, we))
        elif tag == "replace" and (i2 - i1) > 0 and (j2 - j1) > 0:
            span_s = local_w[j1][1]
            span_e = max(span_s + 0.10, local_w[j2 - 1][2])
            step = (span_e - span_s) / (i2 - i1)
            for k in range(i2 - i1):
                aligned[i1 + k] = (words[i1 + k], span_s + k * step, span_s + (k + 1) * step)
    return aligned


def build_chapter_pop_captions(
    placed_timings: list[dict[str, Any]],
    full_wav: Path,
    ass_path: Path,
    config: dict[str, Any],
    *,
    mock_tts: bool = False,
) -> Path:
    """Build viral 1-2 word pop captions at the movie/booth seam with gold emphasis highlights."""
    cap_cfg = config.get("captions", {})
    font_name = _resolve_font(
        str(cap_cfg.get("font_name", "Montserrat ExtraBold")),
        str(cap_cfg.get("font_fallback", "DejaVu Sans Bold")),
    )
    font_size = int(cap_cfg.get("font_size", 92))
    outline_px = int(cap_cfg.get("outline_px", 7))
    shadow_px = int(cap_cfg.get("shadow_px", 3))
    margin_v = int(cap_cfg.get("chapter_margin_v", 905))
    white_ass = _hex_to_ass_bgr(str(cap_cfg.get("active_color", "#FFFFFF")))
    white_tag = _hex_to_ass_tag_bgr(str(cap_cfg.get("active_color", "#FFFFFF")))
    gold_tag = _hex_to_ass_tag_bgr(str(cap_cfg.get("emphasis_color", "#FFD84D")))

    whisper_words: list[tuple[str, float, float]] = []
    if not mock_tts and full_wav.exists():
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                str(cap_cfg.get("model", "base.en")),
                device=str(cap_cfg.get("device", "cpu")),
                compute_type=str(cap_cfg.get("compute_type", "int8")),
                cpu_threads=int(cap_cfg.get("cpu_threads", 2)),
            )
            segs, _ = model.transcribe(str(full_wav), word_timestamps=True)
            for seg in segs:
                for w in getattr(seg, "words", []) or []:
                    txt = w.word.strip()
                    if txt:
                        whisper_words.append((txt, float(w.start), float(w.end)))
        except Exception as exc:
            log.warning("Whisper caption alignment skipped (%s); using proportional cue timing.", exc)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{font_size},{white_ass},{white_ass},"
        f"&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,{outline_px},{shadow_px},"
        f"2,40,40,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    pop_tag = r"{\fscx92\fscy92\t(0,60,\fscx112\fscy112)\t(60,120,\fscx100\fscy100)}"
    min_word_ms = int(cap_cfg.get("min_word_display_ms", 120))
    max_chunk_chars = 14

    prev_end = 0.0
    for cue in placed_timings:
        c_start = float(cue["start_sec"])
        c_end = float(cue["end_sec"])
        emp = str(cue.get("emphasis_word", ""))
        aligned = _align_cue_words(str(cue.get("text", "")), c_start, c_end, whisper_words)
        if not aligned:
            continue

        # Pack into punchy 1-2 word chunks without crossing cue boundaries
        chunks: list[list[tuple[str, float, float]]] = []
        idx = 0
        while idx < len(aligned):
            w, ws, we = aligned[idx]
            dur_ms = (we - ws) * 1000.0
            if (
                idx + 1 < len(aligned)
                and (len(w) <= 3 or dur_ms < min_word_ms)
                and not _is_emphasis_token(w, emp)
                and len(w) + 1 + len(aligned[idx + 1][0]) <= max_chunk_chars
            ):
                chunks.append([aligned[idx], aligned[idx + 1]])
                idx += 2
            else:
                chunks.append([aligned[idx]])
                idx += 1

        for chunk in chunks:
            s_t = max(prev_end, chunk[0][1])
            e_t = max(s_t + 0.08, min(c_end + 0.04, chunk[-1][2]))
            prev_end = e_t
            styled_parts: list[str] = []
            for raw_w, _, _ in chunk:
                disp = raw_w.strip("—–- ").upper()
                if not disp:
                    continue
                if _is_emphasis_token(raw_w, emp):
                    styled_parts.append(f"{{\\c{gold_tag}}}{disp}{{\\c{white_tag}}}")
                else:
                    styled_parts.append(disp)
            if not styled_parts:
                continue
            text_line = pop_tag + " ".join(styled_parts)
            events.append(
                f"Dialogue: 0,{_ass_time(s_t)},{_ass_time(e_t)},Default,,0,0,0,,{text_line}"
            )

    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return ass_path
