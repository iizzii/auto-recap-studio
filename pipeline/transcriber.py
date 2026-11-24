"""Subtitle cascade and faster-whisper auto-transcription fallback."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import requests

from pipeline import ffutil
from pipeline.budget import TimeBudget

log = logging.getLogger(__name__)

_EXIT_CODE_NO_SUBS = 4
_CHUNK_SEC = 600.0  # 10-minute chunks


def _format_srt_ts(sec: float) -> str:
    """Format seconds as ``HH:MM:SS,mmm``."""
    ms_total = max(0, int(round(sec * 1000)))
    ms = ms_total % 1000
    s_total = ms_total // 1000
    h = s_total // 3600
    m = (s_total % 3600) // 60
    s = s_total % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _extract_embedded_subtitles(
    video_path: Path,
    work_dir: Path,
    lang: str = "en",
) -> Optional[Path]:
    """Extract the first matching English text subtitle stream from *video_path*."""
    info = ffutil.probe(video_path)
    text_codecs = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
    chosen_idx: Optional[int] = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") != "subtitle":
            continue
        codec = str(stream.get("codec_name", "")).lower()
        if codec not in text_codecs:
            continue
        tags = stream.get("tags", {}) or {}
        s_lang = str(tags.get("language", "")).lower()
        if s_lang in (lang, "eng", "") or chosen_idx is None:
            chosen_idx = int(stream["index"])
            if s_lang in (lang, "eng"):
                break

    if chosen_idx is None:
        return None

    out_srt = work_dir / "extracted.srt"
    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-map",
            f"0:{chosen_idx}",
            str(out_srt),
        ]
    )
    if out_srt.exists() and out_srt.stat().st_size > 0:
        return out_srt
    return None


def transcribe(
    video_path: Path,
    work_dir: Path,
    config: dict[str, Any],
    budget: TimeBudget,
) -> Path:
    """Extract 16 kHz mono WAV and transcribe in 10-min chunks via faster-whisper."""
    from faster_whisper import WhisperModel

    source_cfg = config.get("source", {})
    model_name = str(source_cfg.get("whisper_model_fallback", "small.en"))
    threads = int(config.get("runtime", {}).get("threads", 2))

    wav_path = work_dir / "transcribe_16k.wav"
    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )

    total_dur = ffutil.duration(wav_path)
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)

    entries: list[tuple[float, float, str]] = []
    offset = 0.0
    chunk_idx = 0
    while offset < total_dur:
        budget.check_or_exit(f"transcribe_chunk_{chunk_idx}")
        chunk_wav = work_dir / f"transcribe_chunk_{chunk_idx}.wav"
        chunk_len = min(_CHUNK_SEC, total_dur - offset)
        ffutil.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{offset:.3f}",
                "-i",
                str(wav_path),
                "-t",
                f"{chunk_len:.3f}",
                "-c",
                "copy",
                str(chunk_wav),
            ]
        )
        segments, _ = model.transcribe(str(chunk_wav), vad_filter=True)
        for seg in segments:
            text = seg.text.strip()
            if text:
                entries.append((offset + seg.start, offset + seg.end, text))
        chunk_wav.unlink(missing_ok=True)
        offset += _CHUNK_SEC
        chunk_idx += 1

    wav_path.unlink(missing_ok=True)
    out_srt = work_dir / "transcribed.srt"
    with out_srt.open("w", encoding="utf-8") as fh:
        for idx, (start, end, text) in enumerate(entries, start=1):
            fh.write(f"{idx}\n{_format_srt_ts(start)} --> {_format_srt_ts(end)}\n{text}\n\n")
    return out_srt


def resolve_subtitles(
    video_path: Path,
    srt_url: Optional[str],
    work_dir: Path,
    config: dict[str, Any],
    budget: TimeBudget,
) -> Path:
    """Resolve subtitles via (1) URL/path, (2) embedded stream, or (3) auto-transcribe."""
    work_dir.mkdir(parents=True, exist_ok=True)
    source_cfg = config.get("source", {})

    if srt_url:
        local = Path(srt_url)
        if local.exists():
            dest = work_dir / "source.srt"
            if local.resolve() != dest.resolve():
                dest.write_bytes(local.read_bytes())
            return dest
        resp = requests.get(srt_url, timeout=60)
        resp.raise_for_status()
        dest = work_dir / "source.srt"
        dest.write_bytes(resp.content)
        return dest

    embedded = _extract_embedded_subtitles(
        video_path, work_dir, str(source_cfg.get("subtitle_language", "en"))
    )
    if embedded is not None:
        return embedded

    if source_cfg.get("auto_transcribe_source", True) and ffutil.has_audio(video_path):
        return transcribe(video_path, work_dir, config, budget)

    log.error(
        "No subtitles found. Provide --srt-source / srt_source_url or enable auto_transcribe_source."
    )
    sys.exit(_EXIT_CODE_NO_SUBS)
