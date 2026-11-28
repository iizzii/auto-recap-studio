"""Native Gemini video+audio commentary writer, dialogue-collision resolver, dynamic prosody TTS & freeze SFX."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import re
import time
import wave
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from pipeline import caption_engine, ffutil
from pipeline.models import ChapterCommentaryScript, ChapterSlice, CommentaryCue

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "chapter_commentary_writer.txt"

VOICE_FALLBACK_CHAIN: tuple[str, ...] = (
    "en-US-ChristopherNeural",
    "en-US-AndrewNeural",
    "en-US-AndrewMultilingualNeural",
)

EMOTION_VOICE_PROSODY: dict[str, tuple[str, str]] = {
    "laughing": ("+9%", "+8Hz"),
    "shocked": ("+10%", "+7Hz"),
    "hyped": ("+10%", "+6Hz"),
    "horrified": ("+7%", "+4Hz"),
    "smug": ("+2%", "-2Hz"),
    "facepalm": ("+2%", "-2Hz"),
    "sad": ("-4%", "-4Hz"),
    "neutral": ("+4%", "+1Hz"),
}

_GENRE_VOICES: dict[str, tuple[str, str]] = {
    "comedy": ("en-US-ChristopherNeural", "+7%"),
    "horror_comedy": ("en-US-ChristopherNeural", "+7%"),
    "action_scifi": ("en-US-ChristopherNeural", "+8%"),
    "emotional_drama": ("en-US-ChristopherNeural", "+3%"),
}

_RESOLVED_VOICE_CACHE: str | None = None

PLACEHOLDER_REGEX = re.compile(r"watch\s+what\s+happens\s+right\s+here\s+in\s+scene\s+\d+", re.IGNORECASE)

BANNED_COMMENTARY_CLICHES: tuple[str, ...] = (
    "big mistake",
    "bad idea",
    "pure madness",
    "unbelievable",
    "worst idea ever",
    "recipe for disaster",
    "talk about a",
    "watch what happens right here",
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "that", "this", "with", "from", "just", "they", "them",
        "have", "what", "when", "where", "there", "their", "about", "into",
        "right", "here", "going", "like", "really", "actually", "literally",
    }
)

_CTA_HINTS: tuple[str, ...] = ("part 2", "part 3", "part two", "part three", "next part", "profile", "live right now")


# Plain-type Gemini wire schemas (NO Field(gt=...) / Field(ge=...) because
# google-genai==1.14.0 rejects exclusiveMinimum in response_schema).
class _GeminiCommentaryCue(BaseModel):
    cue_id: int
    start_sec: float
    max_duration_sec: float
    text: str
    emphasis_word: str
    emotion: Literal[
        "neutral", "laughing", "shocked", "horrified",
        "smug", "hyped", "sad", "facepalm",
    ]
    pause_emotion: Literal[
        "neutral", "laughing", "shocked", "horrified",
        "smug", "hyped", "sad", "facepalm",
    ]
    freeze_frame: bool
    freeze_duration_sec: float


class _GeminiChapterCommentaryScript(BaseModel):
    genre: Literal["comedy", "horror_comedy", "action_scifi", "emotional_drama"]
    voice_name: str
    voice_rate: str
    chapter_title: str
    top_headline: str
    cliffhanger_summary: str
    cues: list[_GeminiCommentaryCue]


def _extract_status_code(exc: Exception) -> str:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code is not None:
        return str(code)
    m = re.search(r"\b(400|401|403|404|429|500|502|503|504)\b", str(exc))
    return m.group(1) if m else "UNKNOWN"


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _upload_and_wait_active(client: Any, video_path: Path, timeout_sec: float = 120.0) -> Any:
    """Upload MP4 via Gemini Files API and poll every 2s until state == ACTIVE."""
    log.info("Uploading %s (%.2f MB) to Gemini Files API...", video_path.name, video_path.stat().st_size / (1024 * 1024))
    video_file = client.files.upload(file=str(video_path))
    deadline = time.monotonic() + timeout_sec
    while True:
        state_obj = getattr(video_file, "state", None)
        state_name = str(getattr(state_obj, "name", state_obj or "")).upper()
        if state_name == "ACTIVE":
            log.info("Gemini Files API ready: %s (state=ACTIVE)", video_file.name)
            return video_file
        if state_name == "FAILED":
            with contextlib.suppress(Exception):
                client.files.delete(name=video_file.name)
            raise RuntimeError(f"Gemini Files API processing FAILED for {video_path.name}")
        if time.monotonic() > deadline:
            with contextlib.suppress(Exception):
                client.files.delete(name=video_file.name)
            raise TimeoutError(
                f"Gemini Files API timed out after {timeout_sec:.0f}s for {video_path.name} (state={state_name})"
            )
        time.sleep(2.0)
        video_file = client.files.get(name=video_file.name)


def call_gemini_structured(
    prompt: str,
    images: list[Path],
    wire_schema: type[BaseModel],
    config: dict[str, Any],
    *,
    video_path: Path | None = None,
    label: str = "gemini_call",
) -> dict[str, Any]:
    """Call Gemini with native MP4 video+audio (via Files API) or images + constrained JSON wire schema.

    NEVER falls back to mock data. Raises a loud RuntimeError containing the exact
    HTTP status code, model name, and error snippet if all retries fail.
    """
    api_keys: list[tuple[str, str]] = []
    for env_k in ("GEMINI_API_KEY", "GEMINI_API_KEY1"):
        val = os.environ.get(env_k, "").strip()
        if val and all(val != existing for _, existing in api_keys):
            api_keys.append((env_k, val))
    if not api_keys:
        raise RuntimeError(
            f"{label}: GEMINI_API_KEY is missing while mock_llm=False. "
            "Silent mock fallback is permanently disabled."
        )

    from google import genai
    from google.genai import types

    llm_cfg = config.get("llm", {})
    primary = str(llm_cfg.get("model", "gemini-3.5-flash")).strip() or "gemini-3.5-flash"
    extra_models = list(
        llm_cfg.get("fallback_models", ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.0-flash"])
    )
    models_to_try: list[str] = []
    for m_name in [primary, *extra_models]:
        if m_name and m_name not in models_to_try:
            models_to_try.append(m_name)

    temperature = float(llm_cfg.get("temperature", 0.7))
    clients = [
        (lbl, genai.Client(api_key=k_val, http_options=types.HttpOptions(timeout=180_000)))
        for lbl, k_val in api_keys
    ]

    image_parts: list[Any] = []
    if video_path is None:
        for img_path in images:
            if img_path.exists():
                mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
                image_parts.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type=mime))

    backoff_delays = (4.0, 10.0, 22.0)
    errors_seen: list[str] = []

    for key_lbl, client in clients:
        uploaded_file: Any = None
        try:
            if video_path is not None and video_path.exists():
                for up_attempt in range(3):
                    try:
                        uploaded_file = _upload_and_wait_active(client, video_path)
                        break
                    except Exception as up_exc:
                        errors_seen.append(f"key={key_lbl} upload_attempt={up_attempt + 1}/3 detail={str(up_exc)[:220]}")
                        if up_attempt < 2:
                            time.sleep(3.0 * (up_attempt + 1))
                if uploaded_file is None:
                    continue
                contents: list[Any] = [uploaded_file, prompt]
            else:
                contents = [prompt, *image_parts]

            for model_name in models_to_try:
                for attempt in range(4):  # 1 initial + 3 retries
                    try:
                        media_desc = f"video={video_path.name}" if uploaded_file is not None else f"images={len(image_parts)}"
                        log.info(
                            "%s: calling Gemini model=%s (%s, attempt %d/4, %s)",
                            label,
                            model_name,
                            key_lbl,
                            attempt + 1,
                            media_desc,
                        )
                        resp = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                temperature=temperature,
                                response_mime_type="application/json",
                                response_schema=wire_schema,
                            ),
                        )
                        raw_text = (resp.text or "").strip()
                        if not raw_text:
                            raise RuntimeError(f"Empty response text from model={model_name}")
                        return json.loads(_strip_json_fences(raw_text))
                    except Exception as exc:
                        status = _extract_status_code(exc)
                        snippet = str(exc).replace("\n", " ")[:280]
                        err_line = f"key={key_lbl} model={model_name} status={status} attempt={attempt + 1}/4 detail={snippet}"
                        errors_seen.append(err_line)

                        if status == "404" or "NOT_FOUND" in str(exc):
                            log.warning("%s: %s -> moving to next model", label, err_line)
                            break

                        low_exc = str(exc).lower()
                        if status == "429" and ("limit: 0" in low_exc or "perday" in low_exc or attempt >= 1):
                            log.warning("%s: quota/429 on %s/%s -> rotating to next model/key", label, key_lbl, model_name)
                            break

                        is_retryable = status in {"429", "500", "502", "503", "504"} or any(
                            tok in str(exc).upper()
                            for tok in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "TIMED OUT", "JSON")
                        )
                        if is_retryable and attempt < 3:
                            delay = backoff_delays[attempt] + random.uniform(0.0, 1.5)
                            log.warning("%s: retryable error (%s); sleeping %.1fs", label, err_line, delay)
                            time.sleep(delay)
                            continue

                        log.error("%s: non-retryable or exhausted on %s", label, err_line)
                        break
        finally:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                    log.info("Deleted temporary Gemini uploaded file %s", uploaded_file.name)
                except Exception as del_exc:
                    log.warning("Failed to delete Gemini uploaded file %s: %s", uploaded_file.name, del_exc)

    summary = " | ".join(errors_seen[-4:])
    raise RuntimeError(
        f"{label} FAILED after retries across models {models_to_try}. "
        f"Errors: {summary}"
    )


def _tokenize_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w}


def normalize_busy_intervals(
    busy_intervals: list[tuple[float, float]] | list[list[float]] | None,
) -> list[tuple[float, float]]:
    if not busy_intervals:
        return []
    pairs = sorted((float(iv[0]), float(iv[1])) for iv in busy_intervals if len(iv) >= 2 and float(iv[1]) > float(iv[0]))
    merged: list[tuple[float, float]] = []
    for s, e in pairs:
        if merged and s <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def compute_busy_overlap(
    start_sec: float,
    end_sec: float,
    busy_intervals: list[tuple[float, float]],
) -> float:
    """Return total seconds that [start_sec, end_sec] overlaps any BUSY_DIALOGUE span."""
    if end_sec <= start_sec or not busy_intervals:
        return 0.0
    total = 0.0
    for bs, be in busy_intervals:
        if be <= start_sec:
            continue
        if bs >= end_sec:
            break
        total += max(0.0, min(end_sec, be) - max(start_sec, bs))
    return round(total, 4)


def _is_cta_text(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in _CTA_HINTS)


def resolve_dialogue_collisions_and_cta(
    script: ChapterCommentaryScript,
    chapter_duration: float,
    busy_intervals: list[tuple[float, float]] | list[list[float]] | None,
    open_gaps: list[dict[str, float]] | None,
    *,
    part_number: int,
    total_parts: int,
) -> ChapterCommentaryScript:
    """Enforce zero dialogue interruption (>0.35s) and exact end-of-clip cliffhanger freeze CTA on Part < N."""
    busy = normalize_busy_intervals(busy_intervals)
    gaps = sorted(open_gaps or [], key=lambda g: float(g.get("start_sec", 0.0)))
    cues = [c.model_copy(deep=True) for c in sorted(script.cues, key=lambda x: x.start_sec)]
    if not cues:
        return script

    # 1. Exact End-of-Clip Cliffhanger Freeze CTA for non-final parts (fixes 7s-early CTA permanently!)
    cta_cue: CommentaryCue | None = None
    if part_number < total_parts and chapter_duration >= 15.0:
        cta_idx = len(cues) - 1
        for idx in range(len(cues) - 1, -1, -1):
            if _is_cta_text(cues[idx].text):
                cta_idx = idx
                break
        cta_cue = cues.pop(cta_idx)
        cta_cue.start_sec = round(max(0.0, chapter_duration - 0.15), 2)
        cta_cue.freeze_frame = True
        est_cta_dur = max(1.6, len(cta_cue.text.split()) / 3.85)
        cta_cue.freeze_duration_sec = round(min(2.8, max(1.8, est_cta_dur + 0.15)), 2)

        # Ensure preceding cues finish cleanly ahead of the final cliffhanger line
        for c in cues:
            if c.start_sec > chapter_duration - 2.2:
                c.start_sec = round(max(0.5, chapter_duration - 2.8), 2)

    # 2. Dialogue-collision resolver on all non-freeze cues
    if busy:
        used_starts: list[float] = []
        for idx, cue in enumerate(cues):
            est_dur = max(1.1, len(cue.text.split()) / 3.85)
            if cue.freeze_frame:
                used_starts.append(cue.start_sec)
                continue

            overlap = compute_busy_overlap(cue.start_sec, cue.start_sec + est_dur, busy)
            if overlap <= 0.35:
                used_starts.append(cue.start_sec)
                continue

            # Opening hook (Cue 1 near 0.0s) colliding with 0.0s dialogue -> promote to crisp 00:00 opening freeze!
            if idx == 0 and cue.start_sec <= 0.6:
                cue.start_sec = 0.05
                cue.freeze_frame = True
                cue.freeze_duration_sec = round(min(2.8, max(1.5, est_dur + 0.15)), 2)
                used_starts.append(cue.start_sec)
                continue

            # Try nudging into the nearest OPEN_GAP within ±4.5s (strongly preferring closest)
            best_cand: float | None = None
            best_dist = 999.0
            for g in gaps:
                gs = float(g.get("start_sec", 0.0))
                ge = float(g.get("end_sec", 0.0))
                if ge - gs < 0.8:
                    continue
                for cand in (gs + 0.05, max(gs + 0.05, ge - est_dur - 0.05), min(ge - 0.6, max(gs + 0.05, cue.start_sec))):
                    if cand < 0.0 or cand > chapter_duration - 1.5:
                        continue
                    dist = abs(cand - cue.start_sec)
                    if dist > 4.5:
                        continue
                    if any(abs(cand - u) < 1.8 for u in used_starts):
                        continue
                    cand_ov = compute_busy_overlap(cand, cand + est_dur, busy)
                    if cand_ov <= 0.30 and dist < best_dist:
                        best_dist = dist
                        best_cand = cand

            if best_cand is not None:
                cue.start_sec = round(best_cand, 2)
            else:
                # Promote to crisp freeze-frame roast right AFTER the overlapping spoken line finishes!
                ov_end = cue.start_sec
                for bs, be in busy:
                    if be > cue.start_sec and bs < cue.start_sec + est_dur:
                        ov_end = max(ov_end, be)
                target_t = min(chapter_duration - 1.5, ov_end + 0.06)
                cue.start_sec = round(max(0.2, target_t), 2)
                cue.freeze_frame = True
                cue.freeze_duration_sec = round(min(2.8, max(1.5, est_dur + 0.15)), 2)
            used_starts.append(cue.start_sec)

    if cta_cue is not None:
        cues.append(cta_cue)

    cues.sort(key=lambda c: c.start_sec)
    for new_id, c in enumerate(cues, start=1):
        c.cue_id = new_id

    updated = script.model_copy(deep=True)
    updated.cues = cues
    return updated


def validate_chapter_commentary(
    script: ChapterCommentaryScript,
    *,
    chapter_duration_sec: float | None = None,
    min_emotions: int = 5,
    require_freeze_frames: bool = False,
) -> list[str]:
    """Validate a ChapterCommentaryScript against placeholder, repetition, cliché, and diversity rules."""
    errors: list[str] = []
    cues = script.cues
    if not cues:
        return ["Script contains zero commentary cues."]

    if script.top_headline:
        low_head = script.top_headline.lower()
        for cliche in BANNED_COMMENTARY_CLICHES:
            if cliche in low_head:
                errors.append(f"top_headline uses banned cliche '{cliche}': {script.top_headline!r}")

    tokenized: list[tuple[int, set[str]]] = []
    for cue in cues:
        txt = cue.text.strip()
        if not txt:
            errors.append(f"Cue {cue.cue_id} has empty text.")
            continue
        if PLACEHOLDER_REGEX.search(txt):
            errors.append(f"Cue {cue.cue_id} contains forbidden placeholder text: {txt!r}")
        low = txt.lower()
        for cliche in BANNED_COMMENTARY_CLICHES:
            if cliche in low:
                errors.append(f"Cue {cue.cue_id} uses banned cliche '{cliche}': {txt!r}")
        tokenized.append((cue.cue_id, _tokenize_words(txt)))

    # Pairwise >70% token overlap rejection
    for i in range(len(tokenized)):
        id_a, tok_a = tokenized[i]
        if len(tok_a) < 3:
            continue
        for j in range(i + 1, len(tokenized)):
            id_b, tok_b = tokenized[j]
            if len(tok_b) < 3:
                continue
            inter = len(tok_a & tok_b)
            union = len(tok_a | tok_b)
            max_len = max(len(tok_a), len(tok_b))
            if (union > 0 and (inter / union) > 0.70) or (max_len > 0 and (inter / max_len) > 0.70):
                errors.append(
                    f"Cues {id_a} and {id_b} share >70% token overlap ({inter}/{max_len})."
                )

    # Emotion diversity
    speaking_emotions = {c.emotion for c in cues}
    req_em = min(min_emotions, len(cues))
    if len(speaking_emotions) < req_em:
        errors.append(
            f"Only {len(speaking_emotions)} distinct emotions used ({sorted(speaking_emotions)}); "
            f"need at least {req_em}."
        )

    # First cue hook check
    first_start = min(c.start_sec for c in cues)
    if first_start > 1.0:
        errors.append(f"Cue 1 starts at {first_start:.2f}s; must start near 0.0s for instant hook.")

    # Freeze-frame & estimated presence checks on full-length chapters
    if chapter_duration_sec is not None and chapter_duration_sec >= 40.0:
        freeze_cues = [c for c in cues if c.freeze_frame]
        if require_freeze_frames and len(freeze_cues) < 2:
            errors.append(
                f"Expected 2-3 freeze_frame=true roast cues, got {len(freeze_cues)}."
            )
        if len(cues) < 12:
            errors.append(
                f"Too few cues ({len(cues)}) for a {chapter_duration_sec:.0f}s chapter; "
                "need 13-17 cues totaling ~195-230 words so Kiko hits 44-55% active speaking time."
            )

        total_words = sum(len(c.text.split()) for c in cues)
        est_talk = sum(max(0.95, len(c.text.split()) / 4.30) for c in cues)
        est_freeze = sum(min(3.0, max(1.4, c.freeze_duration_sec)) for c in freeze_cues)
        est_total = max(1.0, chapter_duration_sec + est_freeze)
        est_share = est_talk / est_total
        if est_share < 0.42:
            needed_words = round(0.50 * est_total * 4.30)
            errors.append(
                f"Estimated commentary share {est_share:.0%} ({total_words} words) is too low (<42%); "
                f"expand to ~{needed_words} total words across 13-17 punchy cues (12-16 words each)."
            )
        elif est_share > 0.62:
            errors.append(
                f"Estimated commentary share {est_share:.0%} ({total_words} words) is too high (>62%); "
                "tighten cue lengths slightly so the movie breathes."
            )

    return errors


def _pick_emphasis_word(text: str, given: str) -> str:
    tokens = [w.strip(".,!?;:\"'—–-") for w in text.split() if w.strip(".,!?;:\"'—–-")]
    if not tokens:
        return ""
    given_norm = re.sub(r"[^a-z0-9]", "", given.lower())
    if given_norm:
        for tok in tokens:
            if re.sub(r"[^a-z0-9]", "", tok.lower()) == given_norm:
                return tok.upper()
    candidates = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) >= 4]
    chosen = max(candidates or tokens, key=len)
    return chosen.upper()


def _mock_commentary_script(
    chapter_slice: ChapterSlice,
    work_title: str,
    *,
    genre_hint: str = "comedy",
    top_headline: str = "",
) -> ChapterCommentaryScript:
    """Deterministic valid non-placeholder script used ONLY when mock_llm=True in unit tests."""
    dur = chapter_slice.duration_sec
    templates = [
        ("Hold up, nobody brings a heavy cricket bat on a quiet camp hike!", "BAT", "smug", "shocked", False, 0.0),
        ("Look at those tiny glowing spores creeping behind his boots already.", "SPORES", "shocked", "neutral", False, 0.0),
        ("Freeze right there—that smirk screams zero survival instincts!", "SURVIVAL", "laughing", "smug", True, 2.0),
        ("Ellie literally tried warning the crew while they mocked every leaf.", "WARNING", "facepalm", "horrified", False, 0.0),
        ("Oh no, the cute forest critters just flipped into attack formation!", "ATTACK", "horrified", "shocked", True, 2.0),
        ("Wait till you see what erupts next—Part 2 is live right now!", "ERUPTS", "hyped", "shocked", True, 2.2),
    ]
    n_cues = min(len(templates), max(5, int(dur // 2.5)))
    chosen = templates[:n_cues]
    step = max(1.8, (dur - 1.5) / max(1, len(chosen)))
    cues: list[CommentaryCue] = []
    for i, (txt, emp, em, pem, is_frz, fdur) in enumerate(chosen):
        if i == 0:
            st = 0.0
        elif i == len(chosen) - 1 and chapter_slice.part_number < chapter_slice.total_parts and dur >= 15.0:
            st = round(dur - 0.15, 2)
        else:
            st = round(min(dur - 0.8, i * step), 2)
        allow_frz = is_frz and dur >= 10.0 and (st >= 1.5 or st == 0.05)
        cues.append(
            CommentaryCue(
                cue_id=i + 1,
                start_sec=st,
                max_duration_sec=round(max(1.5, step - 0.3), 2),
                text=txt,
                emphasis_word=emp,
                emotion=em,  # type: ignore[arg-type]
                pause_emotion=pem,  # type: ignore[arg-type]
                freeze_frame=allow_frz,
                freeze_duration_sec=fdur if allow_frz else 0.0,
            )
        )
    genre = genre_hint if genre_hint in _GENRE_VOICES else "comedy"
    v_name, v_rate = _GENRE_VOICES[genre]
    return ChapterCommentaryScript(
        genre=genre,  # type: ignore[arg-type]
        voice_name=v_name,
        voice_rate=v_rate,
        chapter_title=f"{work_title} (Part {chapter_slice.part_number}/{chapter_slice.total_parts})",
        top_headline=top_headline or chapter_slice.top_headline or "THE FOREST JUST WOKE UP",
        cliffhanger_summary=(
            "The forest creatures turn menacing right as night falls."
            if chapter_slice.part_number < chapter_slice.total_parts
            else ""
        ),
        cues=cues,
    )


def generate_chapter_commentary(
    contact_sheet_path: Path | list[Path] | None,
    chapter_slice: ChapterSlice,
    work_title: str,
    config: dict[str, Any],
    *,
    video_proxy_path: Path | None = None,
    subtitles_text: str = "",
    dialogue_timeline: str = "",
    busy_intervals: list[tuple[float, float]] | list[list[float]] | None = None,
    open_gaps: list[dict[str, float]] | None = None,
    previous_cliffhanger: str = "",
    next_cliffhanger_goal: str = "",
    genre_hint: str = "",
    top_headline: str = "",
    mock_llm: bool = False,
) -> ChapterCommentaryScript:
    """Generate a validated ChapterCommentaryScript by natively watching+hearing the MP4 proxy.

    Fail-fast: when mock_llm=False, NEVER silently falls back to mock/placeholder lines.
    """
    dur = chapter_slice.duration_sec
    if mock_llm:
        script = _mock_commentary_script(
            chapter_slice, work_title, genre_hint=genre_hint or "comedy", top_headline=top_headline
        )
        errs = validate_chapter_commentary(script, chapter_duration_sec=None, min_emotions=5)
        if errs:
            raise RuntimeError(f"Mock commentary script failed validation: {errs}")
        return script

    sheets: list[Path] = []
    if video_proxy_path is None and contact_sheet_path is not None:
        sheets = [contact_sheet_path] if isinstance(contact_sheet_path, Path) else list(contact_sheet_path)

    sys_prompt = _PROMPT_PATH.read_text(encoding="utf-8") if _PROMPT_PATH.exists() else ""

    gaps_str = ""
    if open_gaps:
        gap_lines = [
            f"  - {g['start_sec']:.1f}s to {g['end_sec']:.1f}s ({g['duration_sec']:.1f}s OPEN_GAP)"
            for g in open_gaps
            if g.get("duration_sec", 0.0) >= 0.85
        ]
        gaps_str = "\n".join(gap_lines[:45])

    timeline_block = dialogue_timeline.strip() or subtitles_text.strip() or "(Mostly visual / non-verbal audio)"
    headline_str = top_headline or chapter_slice.top_headline or ""
    target_words = max(140, round(0.50 * (dur + 8.0) * 4.30))
    cta_rule = (
        f"MANDATORY ENDING CTA TIMESTAMP: {dur - 0.15:.2f}s (freeze_frame=true, freeze_duration_sec=2.2) — AFTER final dialogue!"
        if chapter_slice.part_number < chapter_slice.total_parts
        else "FINAL PART: close on the final twist payoff with a witty reaction + quick comment-bait question."
    )
    meta = (
        f"WORK_TITLE: {work_title}\n"
        f"GENRE_HINT: {genre_hint or 'auto-detect'}\n"
        f"PART_NUMBER: {chapter_slice.part_number}/{chapter_slice.total_parts}\n"
        f"TOP_HEADLINE: {headline_str}\n"
        f"CHAPTER_DURATION_SEC: {dur:.2f}s\n"
        f"{cta_rule}\n"
        f"TARGET_TOTAL_SPOKEN_WORDS: ~{target_words} words across 13-17 cues (to hit 44-55% active time)\n"
        f"PREVIOUS_CLIFFHANGER: {previous_cliffhanger or 'None (Part 1 opening)'}\n"
        f"NEXT_CLIFFHANGER_GOAL: {next_cliffhanger_goal or 'Deliver a sharp final twist/payoff'}\n"
        f"OPEN_GAPS (best windows for freeze_frame=false cues):\n{gaps_str or '  (see timeline below)'}\n\n"
        f"DIALOGUE_TIMELINE:\n{timeline_block[:7000]}"
    )

    min_emotions = int(config.get("chapter_commentary", {}).get("min_distinct_emotions", 5))
    last_errors: list[str] = []
    prev_json = ""

    for attempt in range(1, 5):
        feedback = ""
        if last_errors:
            feedback = (
                "\n\nYOUR PREVIOUS JSON FAILED QUALITY VALIDATION. FIX EVERY ERROR BELOW:\n- "
                + "\n- ".join(last_errors)
            )
            if prev_json:
                feedback += f"\n\nPREVIOUS JSON TO FIX:\n{prev_json[:5000]}"

        data = call_gemini_structured(
            f"{sys_prompt}\n\n{meta}{feedback}",
            sheets,
            _GeminiChapterCommentaryScript,
            config,
            video_path=video_proxy_path,
            label=f"chapter_commentary_part{chapter_slice.part_number}_attempt{attempt}",
        )
        prev_json = json.dumps(data, indent=2)

        raw_cues = sorted(data.get("cues", []), key=lambda x: float(x.get("start_sec", 0.0)))
        norm_cues: list[dict[str, Any]] = []
        for idx, c in enumerate(raw_cues, start=1):
            st = max(0.0, min(dur - 0.10, float(c.get("start_sec", 0.0))))
            if idx == 1:
                st = 0.0
            max_d = max(1.0, float(c.get("max_duration_sec", 3.5)))
            is_frz = bool(c.get("freeze_frame", False))
            fdur = float(c.get("freeze_duration_sec", 0.0)) if is_frz else 0.0
            if is_frz:
                fdur = round(min(2.8, max(1.4, fdur or 2.2)), 2)
            txt = str(c.get("text", "")).strip()
            norm_cues.append(
                {
                    "cue_id": idx,
                    "start_sec": round(st, 2),
                    "max_duration_sec": round(max_d, 2),
                    "text": txt,
                    "emphasis_word": _pick_emphasis_word(txt, str(c.get("emphasis_word", ""))),
                    "emotion": c.get("emotion", "neutral"),
                    "pause_emotion": c.get("pause_emotion", "neutral"),
                    "freeze_frame": is_frz,
                    "freeze_duration_sec": fdur,
                }
            )

        data["cues"] = norm_cues
        if headline_str:
            data["top_headline"] = headline_str
        elif data.get("top_headline"):
            data["top_headline"] = str(data["top_headline"]).strip().upper()

        script = ChapterCommentaryScript(**data)
        if genre_hint in _GENRE_VOICES:
            script.genre = genre_hint  # type: ignore[assignment]
        v_name, v_rate = _GENRE_VOICES.get(script.genre, ("en-US-ChristopherNeural", "+7%"))
        script.voice_name = v_name
        script.voice_rate = v_rate

        # Deterministic post-generation dialogue collision resolver + exact ending CTA placement
        script = resolve_dialogue_collisions_and_cta(
            script,
            chapter_duration=dur,
            busy_intervals=busy_intervals,
            open_gaps=open_gaps,
            part_number=chapter_slice.part_number,
            total_parts=chapter_slice.total_parts,
        )

        errs = validate_chapter_commentary(
            script,
            chapter_duration_sec=dur,
            min_emotions=min_emotions,
            require_freeze_frames=(dur >= 40.0),
        )
        if not errs:
            return script
        last_errors = errs
        log.warning(
            "Chapter commentary attempt %d/4 failed validation: %s",
            attempt,
            errs,
        )

    raise RuntimeError(
        f"Chapter commentary failed quality validation after 4 attempts: {last_errors}"
    )


async def _list_edge_voices() -> set[str]:
    import edge_tts

    voices = await edge_tts.list_voices()
    return {str(v.get("ShortName", "")) for v in voices if v.get("ShortName")}


def resolve_expressive_voice(
    preferred: str | None = None,
    *,
    mock_tts: bool = False,
) -> str:
    """Resolve best available expressive edge-tts voice from fallback chain (cached per run)."""
    global _RESOLVED_VOICE_CACHE
    if mock_tts:
        return preferred or VOICE_FALLBACK_CHAIN[0]
    if _RESOLVED_VOICE_CACHE is not None:
        return _RESOLVED_VOICE_CACHE

    chain: list[str] = []
    for v in (preferred, *VOICE_FALLBACK_CHAIN):
        if v and v not in chain:
            chain.append(v)

    try:
        available = asyncio.run(_list_edge_voices())
        for candidate in chain:
            if candidate in available:
                log.info("Selected expressive TTS voice: %s", candidate)
                _RESOLVED_VOICE_CACHE = candidate
                return candidate
    except Exception as exc:
        log.warning("edge-tts list_voices failed (%s); defaulting to %s", exc, chain[0])

    _RESOLVED_VOICE_CACHE = chain[0]
    return _RESOLVED_VOICE_CACHE


def get_emotion_prosody(emotion: str) -> tuple[str, str]:
    """Return dynamic (rate, pitch) for edge-tts based on cue emotion."""
    return EMOTION_VOICE_PROSODY.get(emotion, EMOTION_VOICE_PROSODY["neutral"])


async def _edge_tts_to_mp3(text: str, voice: str, rate: str, pitch: str, out_mp3: Path) -> None:
    import edge_tts

    comm = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    await comm.save(str(out_mp3))


def _synth_cue_wav(
    text: str,
    voice: str,
    rate: str,
    out_wav: Path,
    *,
    pitch: str = "+0Hz",
    mock_tts: bool = False,
    sr: int = 48000,
) -> float:
    """Synthesize a single cue to 48kHz mono WAV (dynamic rate+pitch, silence trim, -16 LUFS)."""
    if mock_tts:
        dur = max(0.9, min(3.0, len(text.split()) * 0.26))
        n = round(dur * sr)
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        tone = 0.25 * np.sin(2.0 * np.pi * 280.0 * t)
        pcm = np.clip(tone * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return dur

    tmp_mp3 = out_wav.with_suffix(".mp3")
    last_exc: Exception | None = None
    # Try requested voice first, falling back along VOICE_FALLBACK_CHAIN on persistent errors
    voice_candidates = [voice, *[v for v in VOICE_FALLBACK_CHAIN if v != voice]]
    for attempt in range(4):
        cur_voice = voice_candidates[min(attempt // 2, len(voice_candidates) - 1)]
        eff_pitch = "+0Hz" if "Multilingual" in cur_voice else pitch
        try:
            asyncio.run(_edge_tts_to_mp3(text, cur_voice, rate, eff_pitch, tmp_mp3))
            if tmp_mp3.exists() and tmp_mp3.stat().st_size > 200:
                break
            raise RuntimeError("edge-tts produced empty MP3 output")
        except Exception as exc:
            last_exc = exc
            if attempt < 3:
                delay = 1.5 * (2**attempt)
                log.warning("edge-tts attempt %d/4 (%s) failed (%s); retrying in %.1fs", attempt + 1, cur_voice, exc, delay)
                time.sleep(delay)
    else:
        raise RuntimeError(f"edge-tts failed after 4 retries for cue '{text[:40]}': {last_exc}")

    ffutil.run(
        [
            "ffmpeg", "-y", "-i", str(tmp_mp3),
            "-af",
            (
                "silenceremove=start_periods=1:start_silence=0.02:start_threshold=-45dB,"
                "areverse,silenceremove=start_periods=1:start_silence=0.02:start_threshold=-45dB,areverse,"
                "loudnorm=I=-16:TP=-1.5:LRA=9"
            ),
            "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le",
            str(out_wav),
        ]
    )
    return ffutil.duration(out_wav)


def _synth_freeze_sfx_track(
    freeze_events: list[dict[str, Any]],
    total_duration: float,
    out_wav: Path,
    sr: int = 48000,
) -> Path:
    """Synthesize crisp 100ms vinyl-stop + camera-pop freeze SFX at -13 dBFS (zero clipping)."""
    total_samples = max(sr, round(total_duration * sr))
    track = np.zeros(total_samples, dtype=np.float32)

    # 100ms punchy vinyl-stop pitch drop (780Hz -> 95Hz) + subtle crisp 12ms shutter transient
    pop_dur = 0.10
    n_pop = round(pop_dur * sr)
    t = np.linspace(0.0, pop_dur, n_pop, endpoint=False, dtype=np.float32)
    freq = 685.0 * np.exp(-26.0 * t) + 95.0
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    # Soft 4ms attack and 12ms release cosine fade -> zero harsh clicks
    fade_in = np.minimum(t / 0.004, 1.0)
    fade_out = np.clip((pop_dur - t) / 0.014, 0.0, 1.0)
    sweep = np.sin(phase) * np.exp(-14.0 * t) * fade_in * fade_out

    # Crisp camera-pop click harmonic (1.8kHz damped burst over first 14ms)
    click = 0.42 * np.sin(2.0 * np.pi * 1850.0 * t) * np.exp(-190.0 * t) * fade_in
    raw_sfx = (sweep + click).astype(np.float32)
    peak = float(np.max(np.abs(raw_sfx))) or 1.0
    target_peak = 10.0 ** (-13.0 / 20.0)  # Exact -13 dBFS peak (~0.2239)
    pop = (raw_sfx / peak) * target_peak

    for ev in freeze_events:
        s_idx = round(float(ev["expanded_start_sec"]) * sr)
        if 0 <= s_idx < total_samples:
            e_idx = min(total_samples, s_idx + n_pop)
            track[s_idx:e_idx] += pop[: e_idx - s_idx]

    track = np.clip(track, -0.95, 0.95)
    stereo = np.stack([track, track], axis=1)
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return out_wav


def _speed_up_wav(
    wav_path: Path,
    speed: float,
    out_wav: Path,
    *,
    sr: int = 48000,
    mock_tts: bool = False,
) -> tuple[Path, float]:
    """Speed up a mono WAV via NumPy (when mock_tts=True) or FFmpeg atempo (live mode)."""
    if mock_tts:
        with wave.open(str(wav_path), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
        new_len = max(1, round(len(pcm) / max(0.5, speed)))
        idx = np.linspace(0, max(0, len(pcm) - 1), new_len)
        resampled = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.int16)
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(resampled.tobytes())
        return out_wav, new_len / float(sr)

    ffutil.run(
        [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-af", f"atempo={speed:.3f}",
            "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le",
            str(out_wav),
        ]
    )
    return out_wav, ffutil.duration(out_wav)


def synthesize_timed_commentary(
    script: ChapterCommentaryScript,
    chapter_duration: float,
    out_dir: Path,
    config: dict[str, Any],
    mock_tts: bool = False,
    *,
    busy_intervals: list[tuple[float, float]] | list[list[float]] | None = None,
    seam_y: int = 946,
) -> tuple[Path, Path, Path]:
    """Synthesize dynamic-prosody cues onto freeze-expanded timeline -> (full_wav, timings_json, ass_path).

    Applies warm broadcast vocal EQ/compression, crisp -13 dBFS freeze SFX, runtime dialogue-overlap
    protection, and writes ``03_timeline_expansion.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cc_cfg = config.get("chapter_commentary", {})
    fps = int(config.get("runtime", {}).get("fps", 30))
    min_gap = float(cc_cfg.get("min_gap_between_lines_sec", 0.30))
    max_speed = float(cc_cfg.get("max_line_speedup", 1.15))
    sr = 48000

    busy = normalize_busy_intervals(busy_intervals)
    voice_pref = str(config.get("tts", {}).get("voice") or script.voice_name or VOICE_FALLBACK_CHAIN[0])
    chosen_voice = resolve_expressive_voice(voice_pref, mock_tts=mock_tts)
    cues = sorted(script.cues, key=lambda c: c.start_sec)

    # 1. Synthesize each cue WAV with dynamic emotion-driven rate & pitch
    synth_cues: list[tuple[CommentaryCue, Path, float]] = []
    for cue in cues:
        rate_str, pitch_str = get_emotion_prosody(cue.emotion)
        raw_wav = out_dir / f"cue_{cue.cue_id:03d}.wav"
        dur = _synth_cue_wav(
            cue.text,
            chosen_voice,
            rate_str,
            raw_wav,
            pitch=pitch_str,
            mock_tts=mock_tts,
            sr=sr,
        )
        synth_cues.append((cue, raw_wav, dur))

    # 2. Compute freeze-frame timeline expansion & collision-free exact placement
    cumulative_shift = 0.0
    cursor_exp = 0.0
    last_freeze_raw = -999.0
    placed_entries: list[tuple[dict[str, Any], Path]] = []
    freeze_events: list[dict[str, Any]] = []
    max_dialogue_overlap = 0.0

    for idx, (cue, wav_path, dur) in enumerate(synth_cues):
        is_last = idx == len(synth_cues) - 1
        max_raw_pos = chapter_duration - 0.05 if (is_last and cue.freeze_frame) else (chapter_duration - 0.5)
        t_raw = max(0.0, min(max(0.0, max_raw_pos), float(cue.start_sec)))

        # Runtime safety check: if a non-freeze cue's actual WAV overlaps > 0.35s of BUSY_DIALOGUE,
        # tighten with mild speedup or promote cleanly to a freeze roast!
        want_freeze = (
            cue.freeze_frame
            and chapter_duration >= 10.0
            and ((t_raw - last_freeze_raw) >= 3.0 or is_last)
        )
        if not want_freeze and busy:
            t_exp_tent = max(cursor_exp, t_raw + cumulative_shift)
            raw_eff_start = max(0.0, t_exp_tent - cumulative_shift)
            ov = compute_busy_overlap(raw_eff_start, raw_eff_start + dur, busy)
            if ov > 0.35:
                # Check if slight speedup <= max_speed keeps overlap <= 0.35s
                fixed_by_speed = False
                for test_spd in (1.08, max_speed):
                    if compute_busy_overlap(raw_eff_start, raw_eff_start + (dur / test_spd), busy) <= 0.35:
                        sped_wav = out_dir / f"cue_{cue.cue_id:03d}_fit.wav"
                        wav_path, dur = _speed_up_wav(wav_path, test_spd, sped_wav, sr=sr, mock_tts=mock_tts)
                        fixed_by_speed = True
                        break
                if not fixed_by_speed and chapter_duration >= 10.0:
                    want_freeze = True
                    t_raw = round(min(chapter_duration - 0.15, raw_eff_start), 3)

        if want_freeze:
            t_exp = t_raw + cumulative_shift
            if t_exp < cursor_exp:
                push = cursor_exp - t_exp
                if t_raw + push <= (chapter_duration - 0.05 if is_last else chapter_duration - 0.3):
                    t_raw = round(t_raw + push, 3)
                    t_exp = t_raw + cumulative_shift
                elif not is_last:
                    want_freeze = False

        if want_freeze:
            # Cap freeze hold to [1.2, 3.2]s snapped to exact frame count so full roast finishes cleanly
            raw_hold = min(3.2, max(1.2, dur + 0.18))
            freeze_frames = max(int(round(1.2 * fps)), min(int(round(3.2 * fps)), round(raw_hold * fps)))
            freeze_dur = round(freeze_frames / fps, 6)

            usable_window = max(0.8, freeze_dur - 0.12)
            if dur > usable_window:
                needed_speed = min(1.20, dur / usable_window)
                if needed_speed > 1.01:
                    sped_wav = out_dir / f"cue_{cue.cue_id:03d}_sped.wav"
                    wav_path, dur = _speed_up_wav(wav_path, needed_speed, sped_wav, sr=sr, mock_tts=mock_tts)

            t_exp = round(t_raw + cumulative_shift, 3)
            voice_start = round(t_exp + 0.06, 3)
            voice_end = round(min(t_exp + freeze_dur, voice_start + dur), 3)

            freeze_events.append(
                {
                    "cue_id": cue.cue_id,
                    "raw_time_sec": round(t_raw, 3),
                    "expanded_start_sec": t_exp,
                    "expanded_end_sec": round(t_exp + freeze_dur, 3),
                    "freeze_dur_sec": round(freeze_dur, 3),
                    "freeze_frames": freeze_frames,
                }
            )
            cumulative_shift = round(cumulative_shift + freeze_dur, 6)
            last_freeze_raw = t_raw
            cursor_exp = t_exp + freeze_dur + min_gap

            placed_entries.append(
                (
                    {
                        "cue_id": cue.cue_id,
                        "raw_start_sec": round(t_raw, 3),
                        "start_sec": voice_start,
                        "end_sec": voice_end,
                        "text": cue.text,
                        "emphasis_word": cue.emphasis_word,
                        "emotion": cue.emotion,
                        "pause_emotion": cue.pause_emotion,
                        "freeze_frame": True,
                        "freeze_duration_sec": round(freeze_dur, 3),
                        "dialogue_overlap_sec": 0.0,
                    },
                    wav_path,
                )
            )
        else:
            t_exp = max(cursor_exp, t_raw + cumulative_shift)
            if idx + 1 < len(synth_cues):
                next_raw = float(synth_cues[idx + 1][0].start_sec)
                next_exp = next_raw + cumulative_shift
            else:
                next_exp = chapter_duration + cumulative_shift
            avail = max(0.5, (next_exp - t_exp) - min_gap)

            if dur > avail:
                needed_speed = min(max_speed, dur / avail)
                if needed_speed > 1.01:
                    sped_wav = out_dir / f"cue_{cue.cue_id:03d}_sped.wav"
                    wav_path, dur = _speed_up_wav(wav_path, needed_speed, sped_wav, sr=sr, mock_tts=mock_tts)

            voice_start = round(t_exp, 3)
            voice_end = round(voice_start + dur, 3)
            cursor_exp = voice_end + min_gap

            raw_eff_s = max(0.0, voice_start - cumulative_shift)
            ov = compute_busy_overlap(raw_eff_s, raw_eff_s + dur, busy)
            max_dialogue_overlap = max(max_dialogue_overlap, ov)

            placed_entries.append(
                (
                    {
                        "cue_id": cue.cue_id,
                        "raw_start_sec": round(t_raw, 3),
                        "start_sec": voice_start,
                        "end_sec": voice_end,
                        "text": cue.text,
                        "emphasis_word": cue.emphasis_word,
                        "emotion": cue.emotion,
                        "pause_emotion": cue.pause_emotion,
                        "freeze_frame": False,
                        "freeze_duration_sec": 0.0,
                        "dialogue_overlap_sec": ov,
                    },
                    wav_path,
                )
            )

    expanded_duration = round(chapter_duration + cumulative_shift, 3)
    total_samples = max(sr, round(expanded_duration * sr))
    timeline_audio = np.zeros(total_samples, dtype=np.float32)

    placed_timings: list[dict[str, Any]] = []
    active_commentary_sec = 0.0

    for entry, wav_path in placed_entries:
        if entry["start_sec"] >= expanded_duration - 0.10:
            continue
        with wave.open(str(wav_path), "rb") as wf:
            pcm = (
                np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
                / 32768.0
            )
        s_idx = min(total_samples - 1, max(0, round(entry["start_sec"] * sr)))
        e_idx = min(total_samples, s_idx + len(pcm))
        if e_idx > s_idx:
            timeline_audio[s_idx:e_idx] += pcm[: e_idx - s_idx]
        actual_end = round(min(expanded_duration, entry["start_sec"] + (e_idx - s_idx) / sr), 3)
        entry["end_sec"] = actual_end
        active_commentary_sec += max(0.0, actual_end - entry["start_sec"])
        placed_timings.append(entry)

    raw_full_wav = out_dir / "03_commentary_raw.wav"
    full_wav = out_dir / "03_commentary_full.wav"
    out_pcm = np.clip(timeline_audio * 32767.0, -32768, 32767).astype(np.int16)
    target_wav = full_wav if mock_tts else raw_full_wav
    with wave.open(str(target_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out_pcm.tobytes())

    # Warm broadcast vocal EQ + gentle compression on Kiko's master commentary track
    if not mock_tts:
        ffutil.run(
            [
                "ffmpeg", "-y", "-i", str(raw_full_wav),
                "-af", "highpass=f=85,equalizer=f=3000:t=q:w=1.2:g=2.5,acompressor=threshold=-18dB:ratio=3:attack=10:release=120",
                "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le",
                str(full_wav),
            ]
        )

    sfx_wav_path = ""
    if freeze_events:
        sfx_wav = _synth_freeze_sfx_track(
            freeze_events, expanded_duration, out_dir / "03_freeze_sfx.wav", sr=sr
        )
        sfx_wav_path = str(sfx_wav)

    timings_json = out_dir / "03_cue_timings.json"
    timings_json.write_text(json.dumps(placed_timings, indent=2) + "\n", encoding="utf-8")

    last_cue_end = max((float(t["end_sec"]) for t in placed_timings), default=0.0)
    cta_end_offset = round(max(0.0, expanded_duration - last_cue_end), 3)

    expansion_meta = {
        "raw_duration_sec": round(chapter_duration, 3),
        "expanded_duration_sec": expanded_duration,
        "voice_used": chosen_voice,
        "freeze_events": freeze_events,
        "freeze_timestamps": [round(float(e["expanded_start_sec"]), 2) for e in freeze_events],
        "freeze_count": len(freeze_events),
        "active_commentary_sec": round(active_commentary_sec, 3),
        "commentary_share": round(active_commentary_sec / max(0.1, expanded_duration), 4),
        "max_dialogue_overlap_sec": round(max_dialogue_overlap, 3),
        "cta_end_offset_sec": cta_end_offset,
        "distinct_emotions": sorted({t["emotion"] for t in placed_timings}),
        "sfx_wav": sfx_wav_path,
    }
    (out_dir / "03_timeline_expansion.json").write_text(
        json.dumps(expansion_meta, indent=2) + "\n", encoding="utf-8"
    )

    ass_path = out_dir / "05_captions.ass"
    if placed_timings:
        caption_engine.build_chapter_pop_captions(
            placed_timings, full_wav, ass_path, config, mock_tts=mock_tts, seam_y=seam_y
        )
    else:
        ass_path.write_text("", encoding="utf-8")

    return full_wav, timings_json, ass_path
