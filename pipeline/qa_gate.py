"""Final QA gate verifying video/audio specs, sync, loudness, and black/silence gaps."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from pipeline import ffutil
from pipeline.models import Shot

log = logging.getLogger(__name__)

_EXIT_CODE_QA_FAIL = 7


def _measure_lufs(media_path: Path) -> float:
    """Measure integrated LUFS using FFmpeg ebur128 filter."""
    res = ffutil.run(
        [
            "ffmpeg",
            "-i",
            str(media_path),
            "-af",
            "ebur128=framelung=0",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    matches = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", res.stderr)
    return float(matches[-1]) if matches else -99.0


def _max_silence_ms(
    media_path: Path,
    total_dur_sec: float = 0.0,
    ignore_trailing_sec: float = 0.0,
) -> float:
    """Return max detected silence duration in milliseconds, ignoring trailing outro pad."""
    res = ffutil.run(
        [
            "ffmpeg",
            "-i",
            str(media_path),
            "-af",
            "silencedetect=noise=-50dB:d=0.15",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    cutoff = total_dur_sec - ignore_trailing_sec if total_dur_sec > ignore_trailing_sec else 1e9
    valid_durs: list[float] = []
    for line in res.stderr.splitlines():
        m = re.search(r"silence_end:\s*(\d+(?:\.\d+)?)\s*\|\s*silence_duration:\s*(\d+(?:\.\d+)?)", line)
        if m:
            s_end = float(m.group(1))
            s_dur = float(m.group(2))
            if s_end < cutoff:
                valid_durs.append(s_dur)
    return max(valid_durs) * 1000.0 if valid_durs else 0.0


def _max_black_sec(media_path: Path) -> float:
    """Return max detected black segment duration in seconds."""
    res = ffutil.run(
        [
            "ffmpeg",
            "-i",
            str(media_path),
            "-vf",
            "blackdetect=d=0.1:pix_th=0.10",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    durs = [float(x) for x in re.findall(r"black_duration:(\d+(?:\.\d+)?)", res.stderr)]
    return max(durs) if durs else 0.0


def run_qa_gate(
    final_mp4: Path,
    shots: list[Shot],
    intermediates: list[Path],
    caption_match_ratio: float,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run all QA checks, write ``qa_report.json``, and exit 7 on hard failures."""
    rt = config.get("runtime", {})
    qa_cfg = config.get("qa", {})
    shots_cfg = config.get("shots", {})
    audio_cfg = config.get("audio", {})

    exp_w = int(rt.get("width", 1080))
    exp_h = int(rt.get("height", 1920))
    exp_fps = int(rt.get("fps", 30))
    min_dur = float(rt.get("min_duration_sec", 120))
    max_dur = float(rt.get("max_duration_sec", 175))
    sync_tol = float(qa_cfg.get("av_sync_tolerance_sec", 0.15))
    max_sil_ms = float(qa_cfg.get("max_silence_gap_ms", 400))
    hard_max_shot = float(shots_cfg.get("hard_max_shot_sec", 4.5))
    target_lufs = float(audio_cfg.get("target_lufs", -14))
    source_audio_mode = str(audio_cfg.get("source_audio_mode", "off"))

    info = ffutil.probe(final_mp4)
    v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)

    failures: list[str] = []
    warnings: list[str] = []

    if v_stream is None:
        failures.append("No video stream found in final_short.mp4.")
    else:
        w = int(v_stream.get("width", 0))
        h = int(v_stream.get("height", 0))
        if (w, h) != (exp_w, exp_h):
            failures.append(f"Resolution {w}x{h} != {exp_w}x{exp_h}.")
        if v_stream.get("codec_name") != "h264":
            failures.append(f"Video codec {v_stream.get('codec_name')} != h264.")
        if v_stream.get("pix_fmt") != "yuv420p":
            failures.append(f"Pixel format {v_stream.get('pix_fmt')} != yuv420p.")

        r_fps = v_stream.get("r_frame_rate", "0/1")
        num, den = (int(x) for x in str(r_fps).split("/"))
        actual_fps = num / max(1, den)
        if abs(actual_fps - exp_fps) > 0.1:
            failures.append(f"FPS {actual_fps:.2f} != {exp_fps}.")

    if a_stream is None:
        failures.append("No audio stream found in final_short.mp4.")

    v_dur = float(v_stream.get("duration") or info.get("format", {}).get("duration", 0)) if v_stream else 0.0
    a_dur = float(a_stream.get("duration") or v_dur) if a_stream else 0.0
    if not (min_dur <= v_dur <= max_dur):
        failures.append(f"Duration {v_dur:.2f}s outside [{min_dur}, {max_dur}]s.")
    if abs(v_dur - a_dur) > sync_tol:
        failures.append(f"A/V sync delta {abs(v_dur - a_dur):.3f}s > {sync_tol}s.")

    lufs = _measure_lufs(final_mp4)
    if abs(lufs - target_lufs) > 2.0:
        warnings.append(f"Integrated loudness {lufs:.2f} LUFS differs from target {target_lufs}.")

    ending_pad_frames = int(config.get("tts", {}).get("ending_pad_frames", 15))
    ignore_tail_sec = (ending_pad_frames / exp_fps) + 0.2
    silence_ms = _max_silence_ms(final_mp4, total_dur_sec=v_dur, ignore_trailing_sec=ignore_tail_sec)
    if silence_ms > max_sil_ms:
        warnings.append(f"Max silence gap {silence_ms:.0f}ms > {max_sil_ms}ms.")

    black_sec = _max_black_sec(final_mp4)
    if black_sec > 2.0:
        failures.append(f"Black frame run {black_sec:.2f}s > 2.0s.")

    for shot in shots:
        shot_sec = shot.frames / exp_fps
        if shot_sec > hard_max_shot + 0.05:
            failures.append(f"Beat {shot.beat_id} shot {shot.shot_index} is {shot_sec:.2f}s > {hard_max_shot}s.")

    if source_audio_mode == "off":
        for inter in intermediates:
            if inter.exists() and ffutil.has_audio(inter):
                failures.append(f"Intermediate {inter.name} has audio stream when source_audio_mode=off.")
                break

    if caption_match_ratio < 0.95:
        warnings.append(f"Caption-to-script word match {caption_match_ratio:.2%} < 95%.")

    # 1. Post-render combing check
    if config.get("layout", {}).get("auto_deinterlace", True) and v_dur > 2.0:
        from pipeline import deinterlace
        sample_ts = [v_dur * f for f in (0.15, 0.35, 0.50, 0.70, 0.85)]
        if deinterlace.detect_combing(final_mp4, sample_ts, threshold=1.30):
            failures.append("Residual interlacing/combing detected in final_short.mp4.")

    # 3. No frozen/overlong shots & no overlapping source ranges
    hard_max = float(config.get("shots", {}).get("hard_max_shot_sec", 3.5))
    fps = float(config.get("runtime", {}).get("fps", 30))
    for s in shots:
        shot_dur = s.frames / fps
        if shot_dur > hard_max + 0.3:
            failures.append(f"Shot B{s.beat_id}_S{s.shot_index} duration {shot_dur:.2f}s > {hard_max + 0.3:.2f}s.")
    if config.get("shots", {}).get("never_reuse_source_range", True):
        sorted_shots = sorted(shots, key=lambda x: x.src_start_sec)
        for a, b in zip(sorted_shots, sorted_shots[1:]):
            if a.src_end_sec > b.src_start_sec + 0.05:
                failures.append(
                    f"Overlapping source ranges: B{a.beat_id} ({a.src_start_sec}-{a.src_end_sec}s) "
                    f"and B{b.beat_id} ({b.src_start_sec}-{b.src_end_sec}s)."
                )
                break
    # 2. Vision sync score check
    scores = [float(s.match_score) for s in shots if s.match_score is not None]
    avg_match_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    b1_scores = [float(s.match_score) for s in shots if s.beat_id == 1 and s.match_score is not None]
    hook_match_score = round(max(b1_scores), 2) if b1_scores else 0.0
    min_avg = float(qa_cfg.get("min_avg_match_score", 6.0))
    min_hook = float(qa_cfg.get("min_hook_match_score", 7.0))
    if scores and avg_match_score < min_avg:
        warnings.append(f"avg_match_score {avg_match_score:.2f} < {min_avg:.2f}.")
    if b1_scores and hook_match_score < min_hook:
        warnings.append(f"hook_match_score {hook_match_score:.2f} < {min_hook:.2f}.")

    report = {
        "passed": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "video_duration_sec": round(v_dur, 3),
            "audio_duration_sec": round(a_dur, 3),
            "integrated_lufs": round(lufs, 2),
            "max_silence_ms": round(silence_ms, 1),
            "max_black_sec": round(black_sec, 3),
            "caption_match_ratio": round(caption_match_ratio, 4),
            "avg_match_score": avg_match_score,
            "hook_match_score": hook_match_score,
            "total_shots": len(shots),
        },
    }
    (out_dir / "qa_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if failures or (qa_cfg.get("fail_on_warnings", False) and warnings):
        log.error("QA Gate FAILED: %s", failures or warnings)
        sys.exit(_EXIT_CODE_QA_FAIL)
    return report
