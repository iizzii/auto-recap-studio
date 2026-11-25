"""BGM/SFX synthesis via FFmpeg lavfi, sidechain ducking, and two-pass loudnorm."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pipeline import ffutil
from pipeline.models import Beat

log = logging.getLogger(__name__)

_BGM_MANIFEST = Path(__file__).resolve().parent.parent / "assets" / "bgm" / "bgm_manifest.json"


def _synth_bgm_lavfi(dur_sec: float, out_wav: Path, mood: str) -> Path:
    """Generate a soft rhythmic cinematic suspense pulse + minor-key chord progression via lavfi."""
    root = 98.0 if mood in ("tense", "cliffhanger", "emotional", "hook") else 110.0
    # Subtle 1.6 Hz pulsing bass + warm minor-key triad progression + stereo-ready chorus
    expr = (
        f"(0.045*(0.55+0.45*sin(2*PI*1.6*t))*sin(2*PI*{root}*t)"
        f"+0.025*sin(2*PI*{root * 1.1892}*t)"
        f"+0.020*sin(2*PI*{root * 1.4983}*t)"
        f"+0.015*sin(2*PI*{root * 2.0}*t*(1+0.002*sin(2*PI*0.25*t))))"
    )
    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"aevalsrc={expr}:s=48000:d={dur_sec:.3f}",
            "-af",
            "lowpass=f=900,aecho=0.8:0.7:40:0.25,afade=t=in:st=0:d=0.8",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(out_wav),
        ]
    )
    return out_wav


def _pick_or_synth_bgm(dur_sec: float, work_dir: Path, mood: str) -> Path:
    """Use a local CC0 track from bgm_manifest.json if present, else lavfi synth."""
    if _BGM_MANIFEST.exists():
        data = json.loads(_BGM_MANIFEST.read_text(encoding="utf-8"))
        for track in data.get("tracks", []):
            t_path = Path(track.get("path", ""))
            if t_path.exists():
                looped = work_dir / "bgm_looped.wav"
                ffutil.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-stream_loop",
                        "-1",
                        "-i",
                        str(t_path),
                        "-t",
                        f"{dur_sec:.3f}",
                        "-ar",
                        "48000",
                        "-ac",
                        "1",
                        str(looped),
                    ]
                )
                return looped

    return _synth_bgm_lavfi(dur_sec, work_dir / "bgm_synth.wav", mood)


def mix_audio(
    stitched_voice: Path,
    beats: list[Beat],
    work_dir: Path,
    config: dict[str, Any],
) -> Path:
    """Mix voiceover, ducked BGM, and master to target LUFS AAC."""
    audio_cfg = config.get("audio", {})
    bgm_db = float(audio_cfg.get("bgm_db_below_voice", -22))
    fade_out = float(audio_cfg.get("bgm_fade_out_sec", 1.5))
    target_lufs = float(audio_cfg.get("target_lufs", -14))
    true_peak = float(audio_cfg.get("true_peak_db", -1.0))
    sample_rate = int(audio_cfg.get("sample_rate", 48000))
    aac_bitrate = str(audio_cfg.get("aac_bitrate", "192k"))

    voice_dur = ffutil.duration(stitched_voice)
    dominant_mood = beats[0].mood if beats else "tense"
    bgm_wav = _pick_or_synth_bgm(voice_dur, work_dir, dominant_mood)

    fade_start = max(0.0, voice_dur - fade_out)
    mixed_wav = work_dir / "mixed_pre_norm.wav"
    filter_complex = (
        f"[1:a]volume={bgm_db}dB,afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}[bgm];"
        "[bgm][0:a]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=250[ducked];"
        "[0:a][ducked]amix=inputs=2:duration=first:normalize=0[out]"
    )
    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(stitched_voice),
            "-i",
            str(bgm_wav),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(mixed_wav),
        ]
    )

    master_m4a = work_dir / "master_audio.m4a"
    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mixed_wav),
            "-af",
            f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            aac_bitrate,
            str(master_m4a),
        ]
    )
    return master_m4a
