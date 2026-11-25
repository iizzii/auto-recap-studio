"""TTS synthesis engine with frame-accurate silence compression and gap-in-beat math."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

from pipeline import ffutil
from pipeline.models import Beat

log = logging.getLogger(__name__)

_EXIT_CODE_DURATION_EXCEEDED = 6


async def _edge_tts_synth(text: str, voice: str, rate: str, out_mp3: Path) -> None:
    """Synthesize *text* via edge-tts into *out_mp3*."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(str(out_mp3))


def _synthesize_raw_beat(
    beat: Beat,
    raw_path: Path,
    config: dict[str, Any],
    *,
    mock_tts: bool,
) -> Path:
    """Synthesize a single beat's raw audio file."""
    tts_cfg = config.get("tts", {})
    backend = "mock" if mock_tts else str(tts_cfg.get("backend", "edge_tts"))

    if backend == "mock":
        tone_hz = int(tts_cfg.get("mock_tone_hz", 280))
        words = max(3, len(beat.narration.split()))
        dur = max(1.2, words * 0.36)
        wav_out = raw_path.with_suffix(".wav")
        ffutil.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={tone_hz}:sample_rate=48000:duration={dur:.3f}",
                "-af",
                "volume=0.25",
                "-ac",
                "1",
                str(wav_out),
            ]
        )
        return wav_out

    if backend == "edge_tts":
        voice = str(tts_cfg.get("voice", "en-US-AndrewNeural"))
        rate = str(tts_cfg.get("rate", "+8%"))
        retries = int(tts_cfg.get("retries", 4))
        backoff = float(tts_cfg.get("backoff_base_sec", 1.5))
        mp3_out = raw_path.with_suffix(".mp3")

        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                asyncio.run(_edge_tts_synth(beat.narration, voice, rate, mp3_out))
                return mp3_out
            except Exception as exc:
                last_exc = exc
                sleep_s = backoff * (2**attempt)
                log.warning(
                    "edge-tts attempt %d/%d failed for beat %d: %s",
                    attempt + 1,
                    retries,
                    beat.beat_id,
                    exc,
                )
                time.sleep(sleep_s)
        log.warning(
            "edge-tts failed for beat %d (%s); falling back to synthetic tone.",
            beat.beat_id,
            last_exc,
        )
        return _synthesize_raw_beat(beat, raw_path, config, mock_tts=True)

    if backend == "elevenlabs":
        api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set.")
        voice_id = str(tts_cfg.get("voice", "21m00Tcm4TlvDq8ikWAM"))
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        resp = requests.post(
            url,
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={"text": beat.narration, "model_id": "eleven_turbo_v2_5"},
            timeout=60,
        )
        resp.raise_for_status()
        mp3_out = raw_path.with_suffix(".mp3")
        mp3_out.write_bytes(resp.content)
        return mp3_out

    raise ValueError(f"Unsupported TTS backend: {backend}")


def _process_and_pad_beat(
    raw_audio: Path,
    beat_wav: Path,
    config: dict[str, Any],
    *,
    is_last: bool,
    atempo: float = 1.0,
) -> tuple[int, float]:
    """Trim silence, normalize to -16 LUFS, optionally atempo, and pad to exact frames."""
    runtime_cfg = config.get("runtime", {})
    tts_cfg = config.get("tts", {})
    fps = int(runtime_cfg.get("fps", 30))
    trim_db = int(tts_cfg.get("trim_silence_db", -45))
    pause_max_s = float(tts_cfg.get("internal_pause_max_ms", 150)) / 1000.0
    lufs = float(tts_cfg.get("beat_loudness_lufs", -16))
    ending_pad = int(tts_cfg.get("ending_pad_frames", 15))
    gap_frames = ending_pad if is_last else int(tts_cfg.get("beat_gap_frames", 2))

    cleaned_wav = beat_wav.with_name(beat_wav.stem + "_clean.wav")
    af_parts = [
        f"silenceremove=start_periods=1:start_silence=0.02:start_threshold={trim_db}dB"
        f":stop_periods=-1:stop_duration={pause_max_s:.3f}:stop_threshold={trim_db}dB",
    ]
    if abs(atempo - 1.0) > 1e-4:
        af_parts.append(f"atempo={atempo:.4f}")
    af_parts.append(f"loudnorm=I={lufs}:TP=-1.5:LRA=11")

    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_audio),
            "-af",
            ",".join(af_parts),
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(cleaned_wav),
        ]
    )

    dur = ffutil.duration(cleaned_wav)
    speech_frames = max(1, math.ceil(dur * fps))
    total_frames = speech_frames + gap_frames
    target_sec = total_frames / fps

    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(cleaned_wav),
            "-af",
            f"apad=whole_dur={target_sec:.6f}",
            "-t",
            f"{target_sec:.6f}",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(beat_wav),
        ]
    )
    cleaned_wav.unlink(missing_ok=True)
    return total_frames, target_sec


def synthesize_beats(
    beats: list[Beat],
    work_dir: Path,
    config: dict[str, Any],
    *,
    mock_tts: bool = False,
) -> Path:
    """Synthesize all beats, enforce max_duration via bounded atempo, and stitch WAV."""
    runtime_cfg = config.get("runtime", {})
    tts_cfg = config.get("tts", {})
    max_duration_sec = float(runtime_cfg.get("max_duration_sec", 175))
    max_atempo = float(tts_cfg.get("max_global_atempo", 1.18))

    audio_dir = work_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    raw_files: list[Path] = []
    beat_wavs: list[Path] = []
    for idx, beat in enumerate(beats):
        raw_base = audio_dir / f"raw_{beat.beat_id:03d}"
        raw_file = _synthesize_raw_beat(beat, raw_base, config, mock_tts=mock_tts)
        raw_files.append(raw_file)

        beat_wav = audio_dir / f"beat_{beat.beat_id:03d}.wav"
        frames, sec = _process_and_pad_beat(
            raw_file,
            beat_wav,
            config,
            is_last=(idx == len(beats) - 1),
            atempo=1.0,
        )
        beat.audio_frames = frames
        beat.audio_duration_sec = sec
        beat_wavs.append(beat_wav)

    total_sec = sum(b.audio_duration_sec or 0.0 for b in beats)
    if total_sec > max_duration_sec:
        needed_tempo = total_sec / max_duration_sec
        if needed_tempo > max_atempo:
            longest = sorted(
                beats, key=lambda b: b.audio_duration_sec or 0.0, reverse=True
            )[:5]
            summary = ", ".join(
                f"beat {b.beat_id} ({b.audio_duration_sec:.2f}s)" for b in longest
            )
            log.error(
                "Total voiceover %.2fs exceeds max_duration %.2fs even at %.2fx atempo. "
                "Longest beats to trim: %s",
                total_sec,
                max_duration_sec,
                max_atempo,
                summary,
            )
            sys.exit(_EXIT_CODE_DURATION_EXCEEDED)

        log.info(
            "Voiceover %.2fs > %.2fs; applying bounded atempo=%.3fx",
            total_sec,
            max_duration_sec,
            needed_tempo,
        )
        for idx, (beat, raw_file, beat_wav) in enumerate(zip(beats, raw_files, beat_wavs)):
            frames, sec = _process_and_pad_beat(
                raw_file,
                beat_wav,
                config,
                is_last=(idx == len(beats) - 1),
                atempo=needed_tempo,
            )
            beat.audio_frames = frames
            beat.audio_duration_sec = sec

    concat_list = audio_dir / "voice_concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{w.resolve()}'" for w in beat_wavs) + "\n",
        encoding="utf-8",
    )
    stitched = audio_dir / "stitched_voice.wav"
    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(stitched),
        ]
    )
    return stitched
