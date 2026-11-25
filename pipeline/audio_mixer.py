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


def infer_bgm_profile(beats: list[Beat], style: str = "commentary") -> str:
    """Classify the video's dominant mood profile for adaptive background music."""
    if not beats:
        return "suspense_action"
    counts: dict[str, int] = {}
    for b in beats:
        counts[b.mood] = counts.get(b.mood, 0) + 1

    funny_score = counts.get("funny", 0) * 2
    emotional_score = counts.get("emotional", 0) + counts.get("calm", 0) + counts.get("reveal", 0)
    action_score = (
        counts.get("tense", 0)
        + counts.get("action", 0)
        + counts.get("cliffhanger", 0)
        + counts.get("hook", 0)
    )

    if funny_score >= max(emotional_score, action_score) and funny_score > 0:
        return "cartoon_playful"
    if emotional_score > action_score:
        return "emotional_story"
    return "suspense_action"


def _synth_bgm_lavfi(dur_sec: float, out_wav: Path, profile: str) -> Path:
    """Generate a light, mood-adaptive musical bed normalized to -16 LUFS via lavfi."""
    if profile == "cartoon_playful":
        # Light, bouncy C-major marimba/pizzicato progression (C4-E4-G4-A4 + alternating C3/G3)
        expr = (
            "(0.38*(0.55+0.45*sin(2*PI*2.4*t))*((0.5+0.5*sin(2*PI*0.25*t))*sin(2*PI*130.81*t)"
            "+(0.5-0.5*sin(2*PI*0.25*t))*sin(2*PI*196.00*t))"
            "+0.24*(0.5+0.5*sin(2*PI*1.2*t))*sin(2*PI*261.63*t)"
            "+0.20*sin(2*PI*329.63*t)"
            "+0.16*sin(2*PI*392.00*t)*(0.6+0.4*sin(2*PI*0.5*t)))"
        )
        lp_freq = 1150
    elif profile == "emotional_story":
        # Warm cinematic Am <-> F/C chord swell (8-second cycle, soft harmonics)
        expr = (
            "(0.36*(0.75+0.25*sin(2*PI*0.75*t))*sin(2*PI*110.00*t)"
            "+0.26*((0.5+0.5*sin(2*PI*0.125*t))*sin(2*PI*220.00*t)"
            "+(0.5-0.5*sin(2*PI*0.125*t))*sin(2*PI*174.61*t))"
            "+0.22*sin(2*PI*261.63*t)"
            "+0.16*sin(2*PI*329.63*t*(1+0.0012*sin(2*PI*0.2*t))))"
        )
        lp_freq = 780
    else:
        # Suspense / action minor pulse (Gm progression, subtle 1.8 Hz rhythm + pad)
        expr = (
            "(0.42*(0.62+0.38*sin(2*PI*1.8*t))*sin(2*PI*98.00*t)"
            "+0.25*sin(2*PI*116.54*t*(1+0.0015*sin(2*PI*0.18*t)))"
            "+0.20*sin(2*PI*146.83*t)"
            "+0.14*(0.5+0.5*sin(2*PI*0.125*t))*sin(2*PI*196.00*t))"
        )
        lp_freq = 850

    fade_st = max(0.0, dur_sec - 1.5)
    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"aevalsrc={expr}:s=48000:d={dur_sec:.3f}",
            "-af",
            (
                f"lowpass=f={lp_freq},highpass=f=55,"
                "aecho=0.7:0.7:42:0.22,"
                "loudnorm=I=-16:TP=-2:LRA=7,"
                f"afade=t=in:st=0:d=1.0,afade=t=out:st={fade_st:.3f}:d=1.5"
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


def _pick_or_synth_bgm(dur_sec: float, work_dir: Path, profile: str) -> Path:
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

    return _synth_bgm_lavfi(dur_sec, work_dir / "bgm_synth.wav", profile)


def mix_audio(
    stitched_voice: Path,
    beats: list[Beat],
    work_dir: Path,
    config: dict[str, Any],
) -> Path:
    """Mix voiceover, ducked BGM (responsive sidechain), optional whoosh SFX, and master to target LUFS AAC."""
    audio_cfg = config.get("audio", {})
    bgm_db = float(audio_cfg.get("bgm_db_below_voice", -16))
    fade_out = float(audio_cfg.get("bgm_fade_out_sec", 1.5))
    target_lufs = float(audio_cfg.get("target_lufs", -14))
    true_peak = float(audio_cfg.get("true_peak_db", -1.0))
    sample_rate = int(audio_cfg.get("sample_rate", 48000))
    aac_bitrate = str(audio_cfg.get("aac_bitrate", "192k"))
    sfx_on = bool(audio_cfg.get("sfx_on_transitions", True))
    sfx_whoosh_db = float(audio_cfg.get("sfx_whoosh_db", -14))
    sfx_whoosh_dur = float(audio_cfg.get("sfx_whoosh_dur_sec", 0.35))

    voice_dur = ffutil.duration(stitched_voice)
    profile = infer_bgm_profile(beats)
    log.info("Audio mixer selected BGM profile '%s' for %d beats.", profile, len(beats))
    bgm_wav = _pick_or_synth_bgm(voice_dur, work_dir, profile)

    # Identify act-transition boundaries on the OUTPUT voiceover timeline
    transition_times: list[float] = []
    if sfx_on and len(beats) > 1:
        cur_out_t = 0.0
        for i, b in enumerate(beats):
            b_dur = float(b.audio_duration_sec) if b.audio_duration_sec else (voice_dur / len(beats))
            cur_out_t += b_dur
            if i + 1 < len(beats) and b.mood != beats[i + 1].mood and cur_out_t < max(0.0, voice_dur - 1.0):
                transition_times.append(cur_out_t)

    fade_start = max(0.0, voice_dur - fade_out)
    mixed_wav = work_dir / "mixed_pre_norm.wav"

    # Light audible BGM with smooth sidechain ducking (ratio=4, fast attack, 320ms release)
    filter_complex = (
        f"[1:a]volume={bgm_db}dB,afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}[bgm];"
        "[bgm][0:a]sidechaincompress=threshold=0.02:ratio=4:attack=15:release=320[ducked];"
        "[0:a][ducked]amix=inputs=2:duration=first:normalize=0[out]"
    )

    if sfx_on and transition_times:
        whoosh_path = work_dir / "whoosh_sfx.wav"
        _synth_whoosh_sfx(sfx_whoosh_dur, whoosh_path)

        whoosh_inputs_cmd: list[str] = []
        for _t in transition_times:
            whoosh_inputs_cmd += ["-i", str(whoosh_path)]

        n_whoosh = len(transition_times)
        whoosh_mix_parts: list[str] = []
        for i, t in enumerate(transition_times):
            delay_ms = int(max(0.0, t - sfx_whoosh_dur * 0.5) * 1000)
            idx = 2 + i
            whoosh_mix_parts.append(
                f"[{idx}:a]adelay={delay_ms}|{delay_ms},volume={sfx_whoosh_db}dB[w{i}]"
            )
        whoosh_labels = "".join(f"[w{i}]" for i in range(n_whoosh))
        filter_complex = (
            f"[1:a]volume={bgm_db}dB,afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}[bgm];"
            "[bgm][0:a]sidechaincompress=threshold=0.02:ratio=4:attack=15:release=320[ducked];"
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
