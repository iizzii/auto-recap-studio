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
    """Generate a warmer, subtle rhythmic minor-key pulse via lavfi (used only when no CC0 tracks found)."""
    root = 98.0 if mood in ("tense", "cliffhanger", "emotional", "hook") else 110.0
    # Warm minor-key pulse: fundamental + minor-third + perfect-fifth + octave overtone,
    # pulsing at 1.6 Hz with gentle chorus (slow LFO detune) and no hollow buzz.
    expr = (
        f"(0.038*(0.55+0.45*sin(2*PI*1.6*t))*sin(2*PI*{root}*t)"
        f"+0.022*sin(2*PI*{root * 1.1892}*t*(1+0.0015*sin(2*PI*0.18*t)))"
        f"+0.018*sin(2*PI*{root * 1.4983}*t)"
        f"+0.012*sin(2*PI*{root * 2.0}*t*(1+0.001*sin(2*PI*0.31*t))))"
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
            "lowpass=f=800,highpass=f=55,aecho=0.6:0.65:38:0.18,afade=t=in:st=0:d=1.2,afade=t=out:st={fade_st:.3f}:d=1.5".format(
                fade_st=max(0.0, dur_sec - 1.5)
            ),
            "-ar",
            "48000",
            "-ac",
            "1",
            str(out_wav),
        ]
    )
    return out_wav


def _synth_whoosh_sfx(dur_sec: float, out_wav: Path) -> Path:
    """Generate a crisp 0.35 s swept-bandpass noise whoosh at comfortable listening level."""
    actual_dur = min(dur_sec, 0.35)
    # Bandpass-filtered noise: sweeps from 800 Hz to 3.5 kHz with clean fade-in/out
    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=d={actual_dur:.3f}:color=white:amplitude=0.9:seed=42:s=48000",
            "-af",
            (
                f"bandpass=f=1800:width_type=o:w=2.5,"
                f"afade=t=in:st=0:d={actual_dur * 0.25:.3f},"
                f"afade=t=out:st={actual_dur * 0.65:.3f}:d={actual_dur * 0.35:.3f},"
                f"volume=0.28"
            ),
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
    """Mix voiceover, ducked BGM (responsive sidechain), optional whoosh SFX, and master to target LUFS AAC."""
    audio_cfg = config.get("audio", {})
    bgm_db = float(audio_cfg.get("bgm_db_below_voice", -18))
    fade_out = float(audio_cfg.get("bgm_fade_out_sec", 1.5))
    target_lufs = float(audio_cfg.get("target_lufs", -14))
    true_peak = float(audio_cfg.get("true_peak_db", -1.0))
    sample_rate = int(audio_cfg.get("sample_rate", 48000))
    aac_bitrate = str(audio_cfg.get("aac_bitrate", "192k"))
    sfx_on = bool(audio_cfg.get("sfx_on_transitions", True))
    sfx_whoosh_db = float(audio_cfg.get("sfx_whoosh_db", -14))
    sfx_whoosh_dur = float(audio_cfg.get("sfx_whoosh_dur_sec", 0.35))

    voice_dur = ffutil.duration(stitched_voice)
    dominant_mood = beats[0].mood if beats else "tense"
    bgm_wav = _pick_or_synth_bgm(voice_dur, work_dir, dominant_mood)

    # Identify act-transition boundaries (mood changes between adjacent beats)
    transition_times: list[float] = []
    if sfx_on and len(beats) > 1:
        # Approximate beat start times from their source windows (not 100% exact but sufficient for SFX)
        for i in range(1, len(beats)):
            prev_b = beats[i - 1]
            curr_b = beats[i]
            if prev_b.mood != curr_b.mood:
                # Rough wall-clock transition point from source window midpoints
                t_approx = (prev_b.source_window_end_sec + curr_b.source_window_start_sec) / 2.0
                transition_times.append(t_approx)

    fade_start = max(0.0, voice_dur - fade_out)
    mixed_wav = work_dir / "mixed_pre_norm.wav"

    # Responsive sidechain: fast attack (5 ms) + 250 ms release for natural music swells in pauses
    filter_complex = (
        f"[1:a]volume={bgm_db}dB,afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}[bgm];"
        "[bgm][0:a]sidechaincompress=threshold=0.015:ratio=8:attack=5:release=250[ducked];"
        "[0:a][ducked]amix=inputs=2:duration=first:normalize=0[out]"
    )

    if sfx_on and transition_times:
        # Build whoosh SFX file and overlay at transition points
        whoosh_path = work_dir / "whoosh_sfx.wav"
        _synth_whoosh_sfx(sfx_whoosh_dur, whoosh_path)

        # For each transition, we adelay the whoosh and amix all together
        # Build filter: generate N delayed whoosh copies then amix with main out
        whoosh_inputs_cmd: list[str] = []
        for t in transition_times:
            whoosh_inputs_cmd += ["-i", str(whoosh_path)]

        n_whoosh = len(transition_times)
        # Build the filter parts
        whoosh_mix_parts: list[str] = []
        for i, t in enumerate(transition_times):
            delay_ms = int(max(0.0, t) * 1000)
            idx = 2 + i  # input index (0=voice, 1=bgm, 2..=whoosh copies)
            whoosh_mix_parts.append(
                f"[{idx}:a]adelay={delay_ms}|{delay_ms},volume={sfx_whoosh_db}dB[w{i}]"
            )
        whoosh_labels = "".join(f"[w{i}]" for i in range(n_whoosh))
        filter_complex = (
            f"[1:a]volume={bgm_db}dB,afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}[bgm];"
            "[bgm][0:a]sidechaincompress=threshold=0.015:ratio=8:attack=5:release=250[ducked];"
            + ";".join(whoosh_mix_parts)
            + f";[0:a][ducked]{whoosh_labels}amix=inputs={2 + n_whoosh}:duration=first:normalize=0[out]"
        )
        ffutil.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(stitched_voice),
                "-i", str(bgm_wav),
                *whoosh_inputs_cmd,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-ar", str(sample_rate),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(mixed_wav),
            ]
        )
    else:
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
