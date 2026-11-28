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
            "ebur128",
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


def _black_stats(media_path: Path, total_dur: float) -> tuple[float, float]:
    """Return (max_black_sec, trailing_black_sec_in_final_1s) in one blackdetect pass."""
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
    max_dur = 0.0
    trailing = 0.0
    pat = re.compile(
        r"black_start:(\d+(?:\.\d+)?)\s+black_end:(\d+(?:\.\d+)?)\s+black_duration:(\d+(?:\.\d+)?)"
    )
    for m in pat.finditer(res.stderr):
        b_end = float(m.group(2))
        b_dur = float(m.group(3))
        if b_dur > max_dur:
            max_dur = b_dur
        if total_dur > 1.0 and b_end >= total_dur - 0.25:
            trailing = max(trailing, b_dur)
    return max_dur, trailing


def _find_trailing_punct_caption_tokens(ass_path: Path) -> list[str]:
    """Return any visible ASS Dialogue caption tokens ending with '.' or ','."""
    if not ass_path.exists():
        return []
    bad: list[str] = []
    for line in ass_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        visible = re.sub(r"\{[^}]*\}", "", parts[9]).strip()
        for tok in visible.split():
            if tok.endswith(".") or tok.endswith(","):
                bad.append(tok)
    return bad


def run_chapter_qa_gate(
    final_mp4: Path,
    script: Any,
    timings: list[dict[str, Any]],
    expansion_meta: dict[str, Any],
    out_dir: Path,
    config: dict[str, Any],
    *,
    strict: bool = True,
    mock_llm: bool = False,
    part_number: int = 1,
    total_parts: int = 1,
    ass_path: Path | None = None,
    bleeped_words: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run hard QA checks for chapter_commentary mode, write qa_report.json, exit 7 on failure."""
    from pipeline import chapter_commentator

    rt = config.get("runtime", {})
    qa_cfg = config.get("qa", {})
    cc_cfg = config.get("chapter_commentary", {})
    audio_cfg = config.get("audio", {})

    exp_w = int(rt.get("width", 1080))
    exp_h = int(rt.get("height", 1920))
    exp_fps = int(rt.get("fps", 30))
    sync_tol = float(qa_cfg.get("av_sync_tolerance_sec", 0.15))
    target_lufs = float(audio_cfg.get("target_lufs", -14.0))
    min_share = float(cc_cfg.get("min_commentary_share", 0.35))
    max_share = float(cc_cfg.get("max_commentary_share", 0.65))
    min_emotions = int(cc_cfg.get("min_distinct_emotions", 6))
    min_dur = 3.0 if mock_llm else max(45.0, float(cc_cfg.get("min_chapter_sec", 65.0)) - 15.0)
    max_dur = float(cc_cfg.get("max_chapter_sec", 120.0)) + 20.0

    info = ffutil.probe(final_mp4)
    v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)

    failures: list[str] = []
    warnings: list[str] = []

    if v_stream is None:
        failures.append("No video stream found in final chapter Short.")
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
        failures.append("No audio stream found in final chapter Short.")

    v_dur = float(v_stream.get("duration") or info.get("format", {}).get("duration", 0)) if v_stream else 0.0
    a_dur = float(a_stream.get("duration") or v_dur) if a_stream else 0.0
    if not (min_dur <= v_dur <= max_dur):
        failures.append(f"Duration {v_dur:.2f}s outside [{min_dur:.1f}, {max_dur:.1f}]s.")
    if abs(v_dur - a_dur) > sync_tol:
        failures.append(f"A/V sync delta {abs(v_dur - a_dur):.3f}s > {sync_tol}s.")

    lufs = _measure_lufs(final_mp4)
    if lufs <= -50.0 or abs(lufs - target_lufs) > 3.5:
        failures.append(f"Integrated loudness {lufs:.2f} LUFS out of acceptable range near {target_lufs} LUFS.")
    elif abs(lufs - target_lufs) > 2.0:
        warnings.append(f"Integrated loudness {lufs:.2f} LUFS differs slightly from target {target_lufs}.")

    max_black, trailing_black = _black_stats(final_mp4, v_dur)
    if max_black > 1.5:
        failures.append(f"Black frame run {max_black:.2f}s > 1.5s.")
    if trailing_black >= 0.4:
        failures.append(f"Trailing black screen detected at end of clip ({trailing_black:.2f}s).")

    # Commentary anti-placeholder, anti-cliche, anti-repetition & diversity gate
    script_errs = chapter_commentator.validate_chapter_commentary(
        script,
        chapter_duration_sec=None,
        min_emotions=min_emotions,
        require_freeze_frames=False,
    )
    failures.extend(script_errs)

    active_sec = float(expansion_meta.get("active_commentary_sec", 0.0))
    share = active_sec / max(0.1, v_dur) if v_dur > 0 else float(expansion_meta.get("commentary_share", 0.0))
    freeze_count = int(expansion_meta.get("freeze_count", 0))
    freeze_timestamps = list(expansion_meta.get("freeze_timestamps", []))
    distinct_emotions = sorted({str(t.get("emotion", "neutral")) for t in timings})

    non_freeze_overlaps = [
        float(t.get("dialogue_overlap_sec", 0.0))
        for t in timings
        if not bool(t.get("freeze_frame", False))
    ]
    max_dlg_overlap = max(
        [float(expansion_meta.get("max_dialogue_overlap_sec", 0.0)), *non_freeze_overlaps],
        default=0.0,
    )

    last_cue_end = max((float(t.get("end_sec", 0.0)) for t in timings), default=0.0)
    cta_end_padding = (v_dur - last_cue_end) if v_dur > 0 else float(expansion_meta.get("cta_end_padding_sec", 0.0))
    cta_end_offset = max(0.0, cta_end_padding)

    bad_cap_tokens = _find_trailing_punct_caption_tokens(ass_path) if ass_path is not None else []
    bleeped_list = list(
        bleeped_words if bleeped_words is not None else expansion_meta.get("bleeped_words", [])
    )

    if not mock_llm:
        if share < min_share or share > max_share:
            failures.append(
                f"Active commentary share {share:.1%} outside [{min_share:.0%}, {max_share:.0%}]."
            )
        if freeze_count < 2:
            failures.append(
                f"Expected >=2 freeze-frame roast moments rendered in live chapter commentary, got {freeze_count}."
            )
        if max_dlg_overlap > 0.45:
            failures.append(
                f"Non-freeze commentary cue overlaps BUSY_DIALOGUE by {max_dlg_overlap:.2f}s (>0.45s limit)."
            )
        if v_dur > 0 and last_cue_end > v_dur - 0.20:
            failures.append(
                f"Final commentary cue ends at {last_cue_end:.2f}s (> final_video_duration - 0.20s = {v_dur - 0.20:.2f}s); ending CTA/audio is chopped off!"
            )
        if part_number < total_parts and cta_end_offset > 2.5:
            failures.append(
                f"Part {part_number}/{total_parts} ending CTA finished {cta_end_offset:.2f}s before video end (>2.5s limit)."
            )
        if bad_cap_tokens:
            failures.append(
                f"Caption tokens contain forbidden trailing '.' or ',': {bad_cap_tokens[:6]}"
            )

    report = {
        "passed": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "video_duration_sec": round(v_dur, 3),
            "audio_duration_sec": round(a_dur, 3),
            "av_sync_delta_sec": round(abs(v_dur - a_dur), 4),
            "integrated_lufs": round(lufs, 2),
            "max_black_sec": round(max_black, 3),
            "trailing_black_sec": round(trailing_black, 3),
            "cues_count": len(timings),
            "freeze_count": freeze_count,
            "freeze_timestamps": freeze_timestamps,
            "active_commentary_sec": round(active_sec, 2),
            "commentary_share": round(share, 4),
            "max_dialogue_overlap_sec": round(max_dlg_overlap, 3),
            "cta_end_offset_sec": round(cta_end_offset, 3),
            "cta_end_padding_sec": round(cta_end_padding, 3),
            "bleeped_words": bleeped_list,
            "voice_used": str(expansion_meta.get("voice_used", getattr(script, "voice_name", ""))),
            "distinct_emotions_count": len(distinct_emotions),
            "distinct_emotions": distinct_emotions,
            "top_headline": getattr(script, "top_headline", ""),
            "genre": getattr(script, "genre", ""),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qa_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if failures and strict:
        log.error("Chapter QA Gate FAILED: %s", failures)
        sys.exit(_EXIT_CODE_QA_FAIL)
    return report

