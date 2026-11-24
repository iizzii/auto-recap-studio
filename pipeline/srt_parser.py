"""SRT/VTT subtitle parser, cleaner, chunker, and speech-density calculator."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pysrt

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BRACKET_CAPTION_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_SPEAKER_PREFIX_RE = re.compile(r"^\s*[A-Z][A-Z0-9_ ]{1,24}:\s*")
_WHITESPACE_RE = re.compile(r"\s+")
_SRT_BLOCK_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})[^\n]*\n(.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def _ts_to_sec(ts: str) -> float:
    """Convert HH:MM:SS,mmm or HH:MM:SS.mmm to seconds."""
    ts_norm = ts.replace(",", ".").strip()
    h, m, s = ts_norm.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _clean_text(raw: str) -> str:
    """Strip HTML tags, [sound]/(groans) captions, and SPEAKER: prefixes."""
    text = _HTML_TAG_RE.sub("", raw)
    text = _BRACKET_CAPTION_RE.sub("", text)
    lines = [_SPEAKER_PREFIX_RE.sub("", line) for line in text.splitlines()]
    cleaned = _WHITESPACE_RE.sub(" ", " ".join(lines)).strip()
    return cleaned


def parse_srt(path: Path) -> list[dict[str, Any]]:
    """Parse an .srt or .vtt file into cleaned ``{start, end, text}`` segments."""
    segments: list[dict[str, Any]] = []
    try:
        subs = pysrt.open(str(path), encoding="utf-8")
        for sub in subs:
            cleaned = _clean_text(sub.text)
            if not cleaned:
                continue
            start = sub.start.ordinal / 1000.0
            end = sub.end.ordinal / 1000.0
            if end > start:
                segments.append({"start": start, "end": end, "text": cleaned})
        if segments:
            return segments
    except Exception:
        pass

    raw = path.read_text(encoding="utf-8", errors="replace")
    for match in _SRT_BLOCK_RE.finditer(raw):
        start = _ts_to_sec(match.group(1))
        end = _ts_to_sec(match.group(2))
        cleaned = _clean_text(match.group(3))
        if cleaned and end > start:
            segments.append({"start": start, "end": end, "text": cleaned})
    return segments


def merge_segments(
    segments: list[dict[str, Any]],
    max_gap_ms: int = 500,
) -> list[dict[str, Any]]:
    """Merge consecutive segments whose gap is <= *max_gap_ms*."""
    if not segments:
        return []
    max_gap_sec = max_gap_ms / 1000.0
    merged: list[dict[str, Any]] = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["start"] - prev["end"] <= max_gap_sec:
            prev["end"] = max(prev["end"], seg["end"])
            prev["text"] = f"{prev['text']} {seg['text']}".strip()
        else:
            merged.append(dict(seg))
    return merged


def chunk_segments(
    segments: list[dict[str, Any]],
    chunk_size: int = 40,
    overlap: int = 5,
) -> list[list[dict[str, Any]]]:
    """Split *segments* into overlapping windows of *chunk_size* segments."""
    if not segments:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    step = max(1, chunk_size - max(0, overlap))
    chunks: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < len(segments):
        window = segments[idx : idx + chunk_size]
        chunks.append(window)
        if idx + chunk_size >= len(segments):
            break
        idx += step
    return chunks


def format_timestamp(sec: float) -> str:
    """Format seconds as ``[HH:MM:SS]``."""
    total = max(0, int(sec))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def format_chunks_for_llm(
    segments: list[dict[str, Any]],
    chunk_tokens: int = 2500,
    overlap_tokens: int = 200,
) -> list[str]:
    """Window segments into overlapping timestamped text chunks bounded by approx tokens."""
    if not segments:
        return []
    chunks: list[str] = []
    start_idx = 0
    while start_idx < len(segments):
        lines: list[str] = []
        tok_count = 0
        idx = start_idx
        while idx < len(segments):
            seg = segments[idx]
            line = f"{format_timestamp(seg['start'])} {seg['text']}"
            line_toks = max(1, int(len(line.split()) * 1.3))
            if lines and tok_count + line_toks > chunk_tokens:
                break
            lines.append(line)
            tok_count += line_toks
            idx += 1
        chunks.append("\n".join(lines))
        if idx >= len(segments):
            break
        # Step back by overlap_tokens
        overlap_acc = 0
        next_start = idx
        while next_start > start_idx + 1:
            prev_seg = segments[next_start - 1]
            prev_toks = max(1, int(len(prev_seg["text"].split()) * 1.3))
            if overlap_acc + prev_toks > overlap_tokens:
                break
            overlap_acc += prev_toks
            next_start -= 1
        start_idx = max(start_idx + 1, next_start)
    return chunks


def compute_speech_density(
    segments: list[dict[str, Any]],
    bucket_sec: float = 60.0,
) -> list[float]:
    """Compute words-per-second density across fixed *bucket_sec* time buckets."""
    if not segments:
        return []
    max_end = max(float(s["end"]) for s in segments)
    num_buckets = max(1, math.ceil(max_end / bucket_sec))
    word_counts = [0.0] * num_buckets

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        words = len(str(seg["text"]).split())
        if words == 0 or end <= start:
            continue
        dur = end - start
        first_b = max(0, int(start // bucket_sec))
        last_b = min(num_buckets - 1, int(max(start, end - 1e-6) // bucket_sec))
        for b in range(first_b, last_b + 1):
            b_start = b * bucket_sec
            b_end = (b + 1) * bucket_sec
            overlap = max(0.0, min(end, b_end) - max(start, b_start))
            word_counts[b] += words * (overlap / dur)

    return [count / bucket_sec for count in word_counts]
