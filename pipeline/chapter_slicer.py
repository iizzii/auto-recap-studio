"""3-Act Director's Cut scene condenser, speech-safe snapper, storyboard extractor, and chapter assembler."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import cv2
from PIL import Image, ImageDraw
from pydantic import BaseModel
from scenedetect import ContentDetector, SceneManager, open_video

from pipeline import chapter_commentator, ffutil, srt_parser, transcriber, video_composer
from pipeline.models import ActChapterPlan, ChapterSlice, SceneSegment, ThreeActSeriesPlan

log = logging.getLogger(__name__)

_DIRECTOR_PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "three_act_director.txt"


# Plain-type wire schemas for google-genai==1.14.0 (no Field(gt=...) / exclusiveMinimum)
class _GeminiSceneSegment(BaseModel):
    start_sec: float
    end_sec: float
    why_kept: str


class _GeminiActChapterPlan(BaseModel):
    part_number: int
    total_parts: int
    top_headline: str
    act_goal: str
    segments: list[_GeminiSceneSegment]
    cliffhanger_summary: str


class _GeminiThreeActSeriesPlan(BaseModel):
    base_slug: str
    work_title: str
    genre: Literal["comedy", "horror_comedy", "action_scifi", "emotional_drama"]
    total_parts: int
    parts: list[_GeminiActChapterPlan]


def _detect_scene_cuts(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    cache_path: Path | None = None,
) -> list[float]:
    """Return sorted scene-cut timestamps (seconds) inside [start_sec, end_sec], caching if requested."""
    if cache_path is not None and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                return [float(t) for t in cached if start_sec < float(t) < end_sec]
        except Exception:
            pass

    cuts: list[float] = []
    try:
        video = open_video(str(video_path))
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=27.0))
        sm.detect_scenes(video, show_progress=False)
        for s_time, _ in sm.get_scene_list():
            t = round(float(s_time.get_seconds()), 3)
            if t > 0.0:
                cuts.append(t)
    except Exception as exc:
        log.warning("Scene detection failed (%s); continuing without scene cuts.", exc)

    cuts = sorted(set(cuts))
    if cache_path is not None and cuts:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cuts, indent=2) + "\n", encoding="utf-8")
    return [t for t in cuts if start_sec < t < end_sec]


def extract_film_dialogue(
    source_path: Path,
    work_dir: Path,
    config: dict[str, Any],
    *,
    srt_source: str | None = None,
    mock_whisper: bool = False,
) -> list[dict[str, Any]]:
    """Extract & cache full-film timestamped dialogue (SRT/embedded first, else faster-whisper base.en)."""
    cache_path = work_dir / "dialogue.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and isinstance(cached.get("segments"), list):
                return cached["segments"]
        except Exception:
            pass

    work_dir.mkdir(parents=True, exist_ok=True)
    segments_out: list[dict[str, Any]] = []

    # 1. Try explicit SRT/VTT source or embedded text subtitle stream
    srt_path: Path | None = None
    if srt_source:
        local = Path(srt_source)
        if local.exists():
            srt_path = local
        else:
            import requests

            resp = requests.get(srt_source, timeout=60)
            resp.raise_for_status()
            srt_path = work_dir / "source_dialogue.srt"
            srt_path.write_bytes(resp.content)
    else:
        lang = str(config.get("source", {}).get("subtitle_language", "en"))
        srt_path = transcriber._extract_embedded_subtitles(source_path, work_dir, lang)

    if srt_path is not None and srt_path.exists():
        for seg in srt_parser.parse_srt(srt_path):
            segments_out.append(
                {
                    "start": round(float(seg["start"]), 3),
                    "end": round(float(seg["end"]), 3),
                    "text": str(seg["text"]).strip(),
                    "words": [],
                }
            )
    elif not mock_whisper and ffutil.has_audio(source_path):
        # 2. Run faster-whisper base.en once on full film audio with VAD + word timestamps
        from faster_whisper import WhisperModel

        wav_16k = work_dir / "full_film_16k.wav"
        ffutil.run(
            [
                "ffmpeg", "-y", "-i", str(source_path),
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                str(wav_16k),
            ]
        )
        threads = int(config.get("runtime", {}).get("threads", 2))
        model_name = str(config.get("source", {}).get("whisper_model_fallback", "base.en"))
        log.info("Running faster-whisper (%s, VAD+word_timestamps) on %s", model_name, source_path.name)
        model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)
        raw_segs, _ = model.transcribe(str(wav_16k), vad_filter=True, word_timestamps=True)
        for seg in raw_segs:
            txt = seg.text.strip()
            if not txt:
                continue
            words_list = []
            for w in getattr(seg, "words", []) or []:
                wtxt = w.word.strip()
                if wtxt:
                    words_list.append(
                        {
                            "word": wtxt,
                            "start": round(float(w.start), 3),
                            "end": round(float(w.end), 3),
                        }
                    )
            segments_out.append(
                {
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "text": txt,
                    "words": words_list,
                }
            )
        wav_16k.unlink(missing_ok=True)

    cache_path.write_text(
        json.dumps({"segments": segments_out}, indent=2) + "\n",
        encoding="utf-8",
    )
    return segments_out


def plan_chapters(
    video_path: Path,
    allowed_start: float,
    allowed_end: float,
    total_parts: int,
    config: dict[str, Any],
    ci_synthetic: bool = False,
) -> list[ChapterSlice]:
    """Legacy sequential scene-snapped slicer retained for unit-test compatibility."""
    cc_cfg = config.get("chapter_commentary", {})
    target_sec = float(cc_cfg.get("target_chapter_sec", 98.0))
    usable_dur = max(5.0, allowed_end - allowed_start)

    if total_parts <= 0:
        total_parts = 1 if usable_dur <= 135.0 else max(1, round(usable_dur / target_sec))

    if total_parts == 1:
        return [
            ChapterSlice(
                part_number=1,
                total_parts=1,
                start_sec=round(allowed_start, 2),
                end_sec=round(allowed_end, 2),
                duration_sec=round(usable_dur, 2),
            )
        ]

    cuts = [] if ci_synthetic else _detect_scene_cuts(video_path, allowed_start, allowed_end)
    boundaries: list[float] = [allowed_start]

    for k in range(1, total_parts):
        t_ideal = allowed_start + k * (usable_dur / total_parts)
        nearby = [c for c in cuts if abs(c - t_ideal) <= 12.0 and c > boundaries[-1] + 10.0]
        snapped = min(nearby, key=lambda c: abs(c - t_ideal)) if nearby else t_ideal
        boundaries.append(round(snapped, 2))
    boundaries.append(round(allowed_end, 2))

    slices: list[ChapterSlice] = []
    for k in range(total_parts):
        s_t = boundaries[k]
        e_t = boundaries[k + 1]
        slices.append(
            ChapterSlice(
                part_number=k + 1,
                total_parts=total_parts,
                start_sec=s_t,
                end_sec=e_t,
                duration_sec=round(max(1.0, e_t - s_t), 2),
            )
        )
    return slices


def extract_labeled_contact_sheet(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    interval_sec: float,
    out_path: Path,
) -> Path:
    """Extract 1 frame every interval_sec with MM:SS pills into a single 4-column JPEG grid."""
    pages = extract_storyboard_pages(
        video_path,
        start_sec,
        end_sec,
        interval_sec,
        out_path.parent,
        prefix=out_path.stem,
        max_pages=1,
        max_tiles_per_page=120,
        relative_labels=True,
        single_out_path=out_path,
    )
    return pages[0]


def extract_storyboard_pages(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    interval_sec: float,
    out_dir: Path,
    *,
    prefix: str = "storyboard",
    max_pages: int = 3,
    max_tiles_per_page: int = 48,
    relative_labels: bool = False,
    single_out_path: Path | None = None,
) -> list[Path]:
    """Extract timestamped 320x180 storyboard grids split cleanly into <= max_pages JPEGs."""
    cols = 4
    tile_w, tile_h = 320, 180
    span = max(1.0, end_sec - start_sec)
    max_total = max(1, max_pages * max_tiles_per_page)
    step = max(1.0, float(interval_sec))
    if span / step > max_total:
        step = span / max_total

    timestamps: list[float] = []
    t = start_sec
    while t < end_sec - 0.05 and len(timestamps) < max_total:
        timestamps.append(round(t, 2))
        t += step
    if not timestamps:
        timestamps = [round(start_sec, 2)]

    out_dir.mkdir(parents=True, exist_ok=True)
    pages_out: list[Path] = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        for page_idx, offset in enumerate(range(0, len(timestamps), max_tiles_per_page), start=1):
            batch = timestamps[offset : offset + max_tiles_per_page]
            rows = max(1, (len(batch) + cols - 1) // cols)
            canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (12, 12, 12))
            draw = ImageDraw.Draw(canvas)

            for idx, ts in enumerate(batch):
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ret, frame = cap.read()
                r, c = divmod(idx, cols)
                x0, y0 = c * tile_w, r * tile_h
                if ret and frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tile = Image.fromarray(rgb).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                    canvas.paste(tile, (x0, y0))
                display_s = max(0, round(ts - start_sec)) if relative_labels else max(0, round(ts))
                lbl = f"{display_s // 60:02d}:{display_s % 60:02d}"
                draw.rectangle([x0 + 4, y0 + 4, x0 + 66, y0 + 24], fill=(0, 0, 0))
                draw.text((x0 + 9, y0 + 6), lbl, fill=(255, 224, 60))

            out_path = single_out_path if (single_out_path is not None and page_idx == 1) else (
                out_dir / f"{prefix}_page{page_idx}.jpg"
            )
            canvas.save(out_path, format="JPEG", quality=86)
            pages_out.append(out_path)
    finally:
        cap.release()

    return pages_out


def extract_chapter_contact_sheets(
    chapter_video: Path,
    duration_sec: float,
    interval_sec: float,
    out_dir: Path,
    *,
    prefix: str = "chapter_sheet",
) -> list[Path]:
    """Extract crisp 2.0s labeled contact sheets of an assembled chapter (up to 2 pages if >30 tiles)."""
    return extract_storyboard_pages(
        chapter_video,
        0.0,
        duration_sec,
        interval_sec,
        out_dir,
        prefix=prefix,
        max_pages=2,
        max_tiles_per_page=30,
        relative_labels=True,
    )


def _speech_intervals(dialogue_segments: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Return sorted spoken intervals (word-level if available, else segment-level)."""
    intervals: list[tuple[float, float]] = []
    for seg in dialogue_segments:
        words = seg.get("words") or []
        if words:
            for w in words:
                ws, we = float(w["start"]), float(w["end"])
                if we > ws:
                    intervals.append((ws, we))
        else:
            ss, se = float(seg["start"]), float(seg["end"])
            if se > ss:
                intervals.append((ss, se))
    intervals.sort()
    return intervals


def _nudge_out_of_speech(
    ts: float,
    speech: list[tuple[float, float]],
    *,
    prefer_earlier: bool,
    max_nudge: float,
    lo: float,
    hi: float,
) -> float:
    """If ts lands inside a spoken word/line, nudge ±max_nudge to nearest silence boundary."""
    for ws, we in speech:
        if ws + 0.04 < ts < we - 0.04:
            cand_early = max(lo, ws - 0.06)
            cand_late = min(hi, we + 0.06)
            d_early = abs(ts - cand_early)
            d_late = abs(cand_late - ts)
            if prefer_earlier:
                if d_early <= max_nudge:
                    return round(cand_early, 3)
                if d_late <= max_nudge:
                    return round(cand_late, 3)
            else:
                if d_late <= max_nudge:
                    return round(cand_late, 3)
                if d_early <= max_nudge:
                    return round(cand_early, 3)
            break
    return round(ts, 3)


def snap_scene_segments(
    segments: list[SceneSegment],
    scene_cuts: list[float],
    dialogue_segments: list[dict[str, Any]],
    allowed_start: float,
    allowed_end: float,
    *,
    cut_snap_sec: float = 2.5,
    speech_nudge_sec: float = 0.6,
    min_seg_sec: float = 6.0,
) -> list[SceneSegment]:
    """Snap segment boundaries to nearest scene cut (±2.5s) and nudge out of spoken words (±0.6s)."""
    speech = _speech_intervals(dialogue_segments)
    sorted_segs = sorted(segments, key=lambda s: s.start_sec)
    snapped: list[SceneSegment] = []
    prev_end = allowed_start

    for idx, seg in enumerate(sorted_segs):
        s_t = max(prev_end, min(allowed_end - min_seg_sec, seg.start_sec))
        next_start = sorted_segs[idx + 1].start_sec if idx + 1 < len(sorted_segs) else allowed_end
        e_t = min(allowed_end, max(s_t + min_seg_sec, seg.end_sec))

        # 1. Snap to nearest scene cut within ±cut_snap_sec
        if scene_cuts:
            near_s = [c for c in scene_cuts if abs(c - s_t) <= cut_snap_sec and prev_end <= c <= e_t - min_seg_sec]
            if near_s:
                s_t = min(near_s, key=lambda c: abs(c - s_t))
            near_e = [
                c
                for c in scene_cuts
                if abs(c - e_t) <= cut_snap_sec and s_t + min_seg_sec <= c <= min(allowed_end, next_start)
            ]
            if near_e:
                e_t = min(near_e, key=lambda c: abs(c - e_t))

        # 2. Speech-safe nudge so cuts never slice mid-word
        s_t = _nudge_out_of_speech(
            s_t, speech, prefer_earlier=True, max_nudge=speech_nudge_sec, lo=prev_end, hi=e_t - min_seg_sec
        )
        e_t = _nudge_out_of_speech(
            e_t,
            speech,
            prefer_earlier=False,
            max_nudge=speech_nudge_sec,
            lo=s_t + min_seg_sec,
            hi=min(allowed_end, next_start),
        )

        if e_t <= s_t + 1.0:
            e_t = min(allowed_end, s_t + min_seg_sec)
        if e_t <= s_t:
            continue

        snapped.append(
            SceneSegment(
                start_sec=round(s_t, 2),
                end_sec=round(e_t, 2),
                why_kept=seg.why_kept,
            )
        )
        prev_end = round(e_t, 2)

    return snapped


def _sanitize_headline(raw: str, part_number: int) -> str:
    clean = " ".join(str(raw or "").strip().upper().split())
    low = clean.lower()
    for cliche in chapter_commentator.BANNED_COMMENTARY_CLICHES:
        if cliche in low:
            clean = ""
            break
    words = clean.split()
    if len(words) < 3:
        defaults = {
            1: "THEY WOKE UP THE FOREST",
            2: "THE TRAP JUST SNAPPED SHUT",
            3: "ONLY ONE SURVIVES THIS NIGHT",
        }
        return defaults.get(part_number, f"ACT {part_number} CHANGES EVERYTHING")
    if len(words) > 6:
        clean = " ".join(words[:6])
    return clean


def sanitize_three_act_plan(
    plan: ThreeActSeriesPlan,
    allowed_start: float,
    allowed_end: float,
    total_parts: int,
    scene_cuts: list[float],
    dialogue_segments: list[dict[str, Any]],
) -> ThreeActSeriesPlan:
    """Validate, enforce strict cross-part chronology, and snap all scene boundaries."""
    parts = sorted(plan.parts, key=lambda p: p.part_number)[:total_parts]
    span = max(30.0, allowed_end - allowed_start)
    cursor = allowed_start
    clean_parts: list[ActChapterPlan] = []

    for idx in range(total_parts):
        p_num = idx + 1
        part_floor = cursor
        # Reserve reasonable room for remaining parts
        remaining_after = total_parts - p_num
        part_ceiling = allowed_end - remaining_after * min(45.0, span / (total_parts * 2.0))

        orig = parts[idx] if idx < len(parts) else None
        raw_segs = orig.segments if orig and orig.segments else []

        # Filter/repair segments strictly inside [part_floor, allowed_end]
        valid_segs: list[SceneSegment] = []
        seg_cursor = part_floor
        for seg in sorted(raw_segs, key=lambda s: s.start_sec):
            s_t = max(seg_cursor, seg.start_sec)
            e_t = min(allowed_end, seg.end_sec)
            if e_t - s_t >= 6.0 and s_t < part_ceiling:
                valid_segs.append(SceneSegment(start_sec=round(s_t, 2), end_sec=round(e_t, 2), why_kept=seg.why_kept))
                seg_cursor = e_t

        if not valid_segs:
            # Deterministic chronological fallback for a malformed single act
            act_span = (allowed_end - allowed_start) / total_parts
            act_s = max(part_floor, allowed_start + idx * act_span)
            act_e = min(allowed_end, allowed_start + (idx + 1) * act_span)
            target_total = min(84.0, max(20.0, act_e - act_s))
            n_blocks = 3 if target_total >= 36.0 else 1
            block_dur = target_total / n_blocks
            stride = (act_e - act_s) / n_blocks
            for b in range(n_blocks):
                bs = round(act_s + b * stride, 2)
                be = round(min(act_e, bs + block_dur), 2)
                if be > bs + 2.0:
                    valid_segs.append(SceneSegment(start_sec=bs, end_sec=be, why_kept=f"Act {p_num} key beat {b + 1}"))

        snapped = snap_scene_segments(
            valid_segs,
            scene_cuts,
            dialogue_segments,
            part_floor,
            allowed_end,
            min_seg_sec=min(6.0, max(2.0, (allowed_end - part_floor) / 4.0)),
        )
        if not snapped:
            snapped = valid_segs

        cursor = snapped[-1].end_sec
        clean_parts.append(
            ActChapterPlan(
                part_number=p_num,
                total_parts=total_parts,
                top_headline=_sanitize_headline(orig.top_headline if orig else "", p_num),
                act_goal=(orig.act_goal if orig and orig.act_goal else f"Act {p_num} story progression"),
                segments=snapped,
                cliffhanger_summary=(orig.cliffhanger_summary if orig else ""),
            )
        )

    return ThreeActSeriesPlan(
        base_slug=plan.base_slug,
        work_title=plan.work_title,
        genre=plan.genre,
        total_parts=total_parts,
        parts=clean_parts,
    )


def _format_dialogue_for_director(
    dialogue_segments: list[dict[str, Any]],
    allowed_start: float,
    allowed_end: float,
) -> str:
    lines: list[str] = []
    for seg in dialogue_segments:
        s_t = float(seg["start"])
        e_t = float(seg["end"])
        if e_t < allowed_start or s_t > allowed_end:
            continue
        sm, ss = divmod(int(s_t), 60)
        em, es = divmod(int(e_t), 60)
        lines.append(f"[{sm:02d}:{ss:02d}-{em:02d}:{es:02d}] ({s_t:.1f}s-{e_t:.1f}s): {seg['text']}")
    return "\n".join(lines) if lines else "(Minimal spoken dialogue; rely primarily on the timestamped visual storyboard pages.)"


def _mock_three_act_plan(
    allowed_start: float,
    allowed_end: float,
    total_parts: int,
    base_slug: str,
    work_title: str,
) -> ThreeActSeriesPlan:
    span = max(15.0, allowed_end - allowed_start)
    act_span = span / max(1, total_parts)
    headlines = [
        "THESE TEENS WOKE UP THE FOREST",
        "THE CUTE MUSHROOMS JUST SNAPPED",
        "ONLY ONE SURVIVED THE AMBUSH",
    ]
    parts: list[ActChapterPlan] = []
    for k in range(total_parts):
        a_s = allowed_start + k * act_span
        a_e = min(allowed_end, allowed_start + (k + 1) * act_span)
        n_blocks = 3 if (a_e - a_s) >= 45.0 else 1
        target_dur = min(82.0, a_e - a_s)
        seg_dur = target_dur / n_blocks
        stride = (a_e - a_s) / n_blocks
        segs = [
            SceneSegment(
                start_sec=round(a_s + b * stride, 2),
                end_sec=round(min(a_e, a_s + b * stride + seg_dur), 2),
                why_kept=f"Act {k + 1} scene block {b + 1}",
            )
            for b in range(n_blocks)
        ]
        parts.append(
            ActChapterPlan(
                part_number=k + 1,
                total_parts=total_parts,
                top_headline=headlines[k % len(headlines)],
                act_goal=f"Act {k + 1} narrative arc for {work_title}",
                segments=segs,
                cliffhanger_summary=f"Act {k + 1} ends right on the turning point." if k + 1 < total_parts else "Final twist payoff.",
            )
        )
    return ThreeActSeriesPlan(
        base_slug=base_slug,
        work_title=work_title,
        genre="horror_comedy",
        total_parts=total_parts,
        parts=parts,
    )


def plan_three_act_series(
    source_path: Path,
    allowed_start: float,
    allowed_end: float,
    total_parts: int,
    base_slug: str,
    work_title: str,
    project_dir: Path,
    work_dir: Path,
    config: dict[str, Any],
    *,
    dialogue_segments: list[dict[str, Any]] | None = None,
    mock_llm: bool = False,
    force: bool = False,
) -> ThreeActSeriesPlan:
    """Create or load the 3-Act Director's Cut plan with scene-cut and speech-safe snapping."""
    plan_path = project_dir / "three_act_plan.json"
    if plan_path.exists() and not force:
        try:
            loaded = ThreeActSeriesPlan(**json.loads(plan_path.read_text(encoding="utf-8")))
            if loaded.total_parts == total_parts and len(loaded.parts) == total_parts:
                log.info("Loaded existing ThreeActSeriesPlan from %s", plan_path)
                return loaded
        except Exception as exc:
            log.warning("Existing %s invalid (%s); regenerating.", plan_path, exc)

    dialogue = dialogue_segments if dialogue_segments is not None else []
    cuts = [] if mock_llm else _detect_scene_cuts(
        source_path, allowed_start, allowed_end, cache_path=work_dir / "scene_cuts.json"
    )

    if mock_llm:
        raw_plan = _mock_three_act_plan(allowed_start, allowed_end, total_parts, base_slug, work_title)
        plan = sanitize_three_act_plan(raw_plan, allowed_start, allowed_end, total_parts, cuts, dialogue)
    else:
        interval = float(config.get("chapter_commentary", {}).get("storyboard_interval_sec", 5.0))
        pages = extract_storyboard_pages(
            source_path,
            allowed_start,
            allowed_end,
            interval,
            work_dir / "storyboard",
            prefix="full_film",
            max_pages=3,
            max_tiles_per_page=48,
            relative_labels=False,
        )
        prompt_tmpl = _DIRECTOR_PROMPT.read_text(encoding="utf-8") if _DIRECTOR_PROMPT.exists() else ""
        prompt_tmpl = prompt_tmpl.replace("{TOTAL_PARTS}", str(total_parts))
        transcript_text = _format_dialogue_for_director(dialogue, allowed_start, allowed_end)
        meta = (
            f"WORK_TITLE: {work_title}\n"
            f"BASE_SLUG: {base_slug}\n"
            f"TOTAL_PARTS: {total_parts}\n"
            f"ALLOWED_START_SEC: {allowed_start:.1f}\n"
            f"ALLOWED_END_SEC: {allowed_end:.1f}\n"
            f"USABLE_FILM_DURATION_SEC: {allowed_end - allowed_start:.1f}\n\n"
            f"FULL_FILM_DIALOGUE_TRANSCRIPT:\n{transcript_text[:14000]}"
        )
        data = chapter_commentator.call_gemini_structured(
            f"{prompt_tmpl}\n\n{meta}",
            pages,
            _GeminiThreeActSeriesPlan,
            config,
            label=f"three_act_director_{base_slug}",
        )
        data["base_slug"] = base_slug
        data["work_title"] = work_title
        data["total_parts"] = total_parts
        raw_plan = ThreeActSeriesPlan(**data)
        plan = sanitize_three_act_plan(raw_plan, allowed_start, allowed_end, total_parts, cuts, dialogue)

    project_dir.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return plan


def remap_dialogue_to_chapter(
    dialogue_segments: list[dict[str, Any]],
    segment_mappings: list[dict[str, Any]],
    assembled_duration: float,
    *,
    min_gap_sec: float = 0.85,
) -> tuple[list[dict[str, Any]], list[dict[str, float]], str]:
    """Remap film dialogue onto assembled chapter timeline and compute OPEN_GAP windows."""
    mapped: list[dict[str, Any]] = []
    for m in segment_mappings:
        src_s = float(m["src_start_sec"])
        src_e = float(m["src_end_sec"])
        ch_s = float(m["chapter_start_sec"])
        for d in dialogue_segments:
            ds, de = float(d["start"]), float(d["end"])
            if de <= src_s or ds >= src_e:
                continue
            ov_s = max(ds, src_s)
            ov_e = min(de, src_e)
            if ov_e - ov_s < 0.08:
                continue
            mapped.append(
                {
                    "start_sec": round(ch_s + (ov_s - src_s), 3),
                    "end_sec": round(ch_s + (ov_e - src_s), 3),
                    "text": str(d["text"]).strip(),
                }
            )
    mapped.sort(key=lambda x: x["start_sec"])

    # Merge busy speech intervals (with 0.12s guard padding) to find clean OPEN_GAPs
    busy: list[tuple[float, float]] = []
    for item in mapped:
        bs = max(0.0, item["start_sec"] - 0.12)
        be = min(assembled_duration, item["end_sec"] + 0.12)
        if busy and bs <= busy[-1][1] + 0.15:
            busy[-1] = (busy[-1][0], max(busy[-1][1], be))
        else:
            busy.append((bs, be))

    open_gaps: list[dict[str, float]] = []
    cur = 0.0
    for bs, be in busy:
        if bs - cur >= min_gap_sec:
            open_gaps.append(
                {"start_sec": round(cur, 2), "end_sec": round(bs, 2), "duration_sec": round(bs - cur, 2)}
            )
        cur = max(cur, be)
    if assembled_duration - cur >= min_gap_sec:
        open_gaps.append(
            {
                "start_sec": round(cur, 2),
                "end_sec": round(assembled_duration, 2),
                "duration_sec": round(assembled_duration - cur, 2),
            }
        )

    # Build interleaved timeline string for Gemini
    events: list[tuple[float, str]] = []
    for item in mapped:
        events.append(
            (
                item["start_sec"],
                f"[{item['start_sec']:05.1f}s-{item['end_sec']:05.1f}s] BUSY_DIALOGUE: \"{item['text']}\"",
            )
        )
    for gap in open_gaps:
        events.append(
            (
                gap["start_sec"],
                f"[{gap['start_sec']:05.1f}s-{gap['end_sec']:05.1f}s] OPEN_GAP ({gap['duration_sec']:.1f}s — ideal for normal commentary)",
            )
        )
    events.sort(key=lambda x: x[0])
    timeline_str = "\n".join(line for _, line in events)
    return mapped, open_gaps, timeline_str


def assemble_condensed_chapter(
    source_path: Path,
    act_plan: ActChapterPlan,
    dialogue_segments: list[dict[str, Any]],
    project_dir: Path,
    work_dir: Path,
    config: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Cut and stitch ActChapterPlan.segments into chapter_raw.mp4 with 0.12s zero-drift audio crossfades."""
    fps = int(config.get("runtime", {}).get("fps", 30))
    xfade_sec = float(config.get("chapter_commentary", {}).get("audio_crossfade_sec", 0.12))
    seg_dir = work_dir / f"part_{act_plan.part_number}_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    info = ffutil.probe(source_path)
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    enc_flags = video_composer.select_h264_encoder(config, intermediate=True)

    vid_paths: list[Path] = []
    aud_paths: list[Path] = []
    mappings: list[dict[str, Any]] = []
    ch_cursor = 0.0
    n_segs = len(act_plan.segments)

    for idx, seg in enumerate(act_plan.segments):
        seg_dur = max(1.0, seg.end_sec - seg.start_sec)
        n_frames = max(fps, round(seg_dur * fps))
        exact_dur = n_frames / float(fps)

        v_out = seg_dir / f"seg_{idx:02d}_v.mp4"
        ffutil.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{seg.start_sec:.3f}",
                "-i", str(source_path),
                "-t", f"{exact_dur + 0.25:.3f}",
                "-vf", f"fps={fps},tpad=stop_mode=clone:stop_duration=1",
                "-frames:v", str(n_frames),
                "-an",
                *enc_flags,
                "-pix_fmt", "yuv420p", "-r", str(fps),
                str(v_out),
            ]
        )
        vid_paths.append(v_out)

        # For zero-drift 0.12s acrossfade between segments i and i+1, extend audio slice i by +xfade_sec
        extra_tail = xfade_sec if (idx < n_segs - 1 and xfade_sec > 0.0) else 0.0
        aud_dur = exact_dur + extra_tail
        a_out = seg_dir / f"seg_{idx:02d}_a.wav"
        if has_audio:
            ffutil.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{seg.start_sec:.3f}",
                    "-i", str(source_path),
                    "-t", f"{aud_dur + 0.2:.3f}",
                    "-vn",
                    "-af", f"apad=whole_dur={aud_dur:.6f},atrim=0:{aud_dur:.6f},asetpts=PTS-STARTPTS",
                    "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                    str(a_out),
                ]
            )
        else:
            ffutil.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={aud_dur:.6f}",
                    "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                    str(a_out),
                ]
            )
        aud_paths.append(a_out)

        mappings.append(
            {
                "segment_index": idx,
                "src_start_sec": round(seg.start_sec, 3),
                "src_end_sec": round(seg.start_sec + exact_dur, 3),
                "chapter_start_sec": round(ch_cursor, 3),
                "chapter_end_sec": round(ch_cursor + exact_dur, 3),
                "frames": n_frames,
                "why_kept": seg.why_kept,
            }
        )
        ch_cursor += exact_dur

    assembled_dur = round(ch_cursor, 3)

    # Concat video stream cleanly
    v_concat_txt = seg_dir / "v_concat.txt"
    v_concat_txt.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in vid_paths) + "\n",
        encoding="utf-8",
    )
    concat_v = seg_dir / "chapter_v.mp4"
    ffutil.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(v_concat_txt), "-c", "copy", str(concat_v),
        ]
    )

    # Chain 0.12s triangular audio crossfades across all jumped scene boundaries (zero drift!)
    concat_a = seg_dir / "chapter_a.wav"
    if len(aud_paths) == 1:
        concat_a = aud_paths[0]
    else:
        a_cmd: list[str] = ["ffmpeg", "-y"]
        for p in aud_paths:
            a_cmd.extend(["-i", str(p)])
        fc_parts: list[str] = []
        prev_lbl = "[0:a]"
        for i in range(1, len(aud_paths)):
            out_lbl = f"[xf{i}]"
            fc_parts.append(f"{prev_lbl}[{i}:a]acrossfade=d={xfade_sec:.3f}:c1=tri:c2=tri{out_lbl}")
            prev_lbl = out_lbl
        fc_parts.append(f"{prev_lbl}atrim=0:{assembled_dur:.6f},asetpts=PTS-STARTPTS[aout]")
        a_cmd.extend(
            [
                "-filter_complex", ";".join(fc_parts),
                "-map", "[aout]",
                "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                str(concat_a),
            ]
        )
        ffutil.run(a_cmd)

    chapter_raw = work_dir / f"part_{act_plan.part_number}_chapter_raw.mp4"
    ffutil.run(
        [
            "ffmpeg", "-y",
            "-i", str(concat_v),
            "-i", str(concat_a),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            str(chapter_raw),
        ]
    )

    mapped_dlg, open_gaps, timeline_str = remap_dialogue_to_chapter(
        dialogue_segments, mappings, assembled_dur
    )
    timeline_map = {
        "part_number": act_plan.part_number,
        "total_parts": act_plan.total_parts,
        "top_headline": act_plan.top_headline,
        "assembled_duration_sec": assembled_dur,
        "segments": mappings,
        "dialogue": mapped_dlg,
        "open_gaps": open_gaps,
        "dialogue_timeline_text": timeline_str,
    }
    project_dir.mkdir(parents=True, exist_ok=True)
    map_path = project_dir / f"part_{act_plan.part_number}_timeline_map.json"
    map_path.write_text(json.dumps(timeline_map, indent=2) + "\n", encoding="utf-8")
    return chapter_raw, map_path, timeline_map
