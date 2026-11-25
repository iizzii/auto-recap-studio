"""Two-pass LLM script generator with validation loop and mock mode."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from pipeline import ffutil, srt_parser
from pipeline.budget import TimeBudget
from pipeline.models import Beat, ProjectManifest, RightsRecord
from pipeline.rights_guard import build_attribution_block
from pipeline.script_validator import validate_manifest

log = logging.getLogger(__name__)

_EXIT_CODE_SCRIPT_FAIL = 5
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}

_MOOD_SYNONYMS: dict[str, str] = {
    "tension": "tense",
    "suspense": "tense",
    "suspenseful": "tense",
    "sad": "emotional",
    "dramatic": "emotional",
    "heartfelt": "emotional",
    "humorous": "funny",
    "comedy": "funny",
    "comedic": "funny",
    "exciting": "action",
    "energetic": "action",
    "mystery": "reveal",
    "mysterious": "reveal",
    "peaceful": "calm",
    "quiet": "calm",
    "opening": "hook",
    "ending": "cliffhanger",
}


class EventItem(BaseModel):
    """Structured event item returned by Pass 1 plot extraction."""

    start: str
    end: str
    summary: str
    characters: list[str]
    visual_notes: str
    craft_notes: str = ""
    intensity: int
    hook_candidate: bool = False


class EventLedgerResponse(BaseModel):
    """Pass 1 Gemini constrained-decoding schema."""

    events: list[EventItem]


class BeatDraft(BaseModel):
    """Structured beat item returned by Pass 2 script writer."""

    beat_id: int
    narration: str
    source_window_start_sec: float
    source_window_end_sec: float
    visual_cue: str
    mood: Literal[
        "hook", "funny", "tense", "emotional", "action", "cliffhanger", "reveal", "calm"
    ]
    kind: Literal["story", "commentary"]


class ScriptDraftResponse(BaseModel):
    """Pass 2 Gemini constrained-decoding schema."""

    title: str
    banner_text: str
    description: str
    hashtags: list[str]
    beats: list[BeatDraft]


def _normalize_manifest_data(
    data: dict[str, Any],
    allowed_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Normalize common LLM mood/kind synonym drift and enforce valid monotonic windows."""
    min_a, max_a = allowed_range if allowed_range else (0.0, 1e9)
    prev_start = min_a
    prev_end = min_a
    for idx, beat in enumerate(data.get("beats", [])):
        if isinstance(beat, dict):
            beat["beat_id"] = idx + 1
            raw_mood = str(beat.get("mood", "action")).strip().lower()
            beat["mood"] = _MOOD_SYNONYMS.get(raw_mood, raw_mood)
            raw_kind = str(beat.get("kind", "story")).strip().lower()
            if raw_kind not in ("story", "commentary"):
                beat["kind"] = "story"
            else:
                beat["kind"] = raw_kind
            s_val = float(beat.get("source_window_start_sec", prev_end))
            e_val = float(beat.get("source_window_end_sec", s_val + 5.0))
            if idx > 0:
                s_val = max(s_val, prev_start, prev_end - 1.0)
            s_val = max(min_a, min(s_val, max_a - 2.0))
            e_val = min(max_a, max(e_val, s_val + 2.0))
            beat["source_window_start_sec"] = round(s_val, 2)
            beat["source_window_end_sec"] = round(e_val, 2)
            prev_start = s_val
            prev_end = e_val
    return data


def _strip_json_fences(raw: str) -> str:
    """Strip markdown code fences if the LLM wrapped its JSON."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _extract_pass1_contact_sheets(
    source_path: Path,
    allowed_range: tuple[float, float],
    work_dir: Path,
    config: dict[str, Any],
) -> list[Path]:
    """Sample climax-weighted frames within *allowed_range* and pack into 1-3 labeled contact sheets.

    Strictly clamps to [allowed_start, max(allowed_start+1.0, allowed_end-4.0)] so we
    never sample end-credits or tail frames outside what clip_selector is allowed to pick.
    65% of frames cover the first 75% of allowed_range; 35% densely sample the final 25%
    (where climax actions and twists unfold rapidly).
    """
    from PIL import Image, ImageDraw

    src_cfg = config.get("source", {})
    max_frames = int(src_cfg.get("pass1_max_frames", 48))
    cols = int(src_cfg.get("pass1_grid_cols", 4))
    rows = int(src_cfg.get("pass1_grid_rows", 4))
    per_sheet = max(1, cols * rows)

    # Strictly clamp the sampling window so we never see tail/credits
    start_s, end_s = allowed_range
    safe_end = max(start_s + 1.0, end_s - 4.0)
    dur = max(2.0, safe_end - start_s)
    n_frames = min(max_frames, max(8, int(dur // 8)))

    # Climax-weighted split: 65% over first 75% of range, 35% over final 25%
    n_early = max(1, int(round(n_frames * 0.65)))
    n_climax = max(1, n_frames - n_early)
    boundary = start_s + dur * 0.75

    early_step = max(0.1, (boundary - start_s) / max(1, n_early))
    climax_step = max(0.1, (safe_end - boundary) / max(1, n_climax))

    timestamps: list[float] = []
    for i in range(n_early):
        timestamps.append(start_s + (i + 0.5) * early_step)
    for i in range(n_climax):
        timestamps.append(boundary + (i + 0.5) * climax_step)

    sheets_dir = work_dir / "pass1_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 320, 180
    tiles: list[Image.Image] = []

    for ts in timestamps:
        tile_path = sheets_dir / f"tile_{i:03d}.jpg"
        try:
            ffutil.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{ts:.2f}",
                    "-i",
                    str(source_path),
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale={tile_w}:{tile_h}",
                    "-q:v",
                    "5",
                    str(tile_path),
                ]
            )
            img = Image.open(tile_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (tile_w, tile_h), (20, 20, 20))
        finally:
            tile_path.unlink(missing_ok=True)

        draw = ImageDraw.Draw(img)
        mm, ss = divmod(int(ts), 60)
        label = f"[{mm:02d}:{ss:02d}]"
        draw.rectangle([4, 4, 72, 22], fill=(0, 0, 0))
        draw.text((8, 7), label, fill=(255, 255, 255))
        tiles.append(img)

    sheet_paths: list[Path] = []
    for s_idx in range(0, len(tiles), per_sheet):
        batch = tiles[s_idx : s_idx + per_sheet]
        canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (12, 12, 12))
        for idx, tile in enumerate(batch):
            r, c = divmod(idx, cols)
            canvas.paste(tile, (c * tile_w, r * tile_h))
        if canvas.width > 1280 or canvas.height > 720:
            canvas.thumbnail((1280, 720))
        out_sheet = sheets_dir / f"sheet_{len(sheet_paths) + 1:02d}.jpg"
        canvas.save(out_sheet, format="JPEG", quality=78, optimize=True)
        sheet_paths.append(out_sheet)
    return sheet_paths


def _call_llm(
    prompt: str,
    config: dict[str, Any],
    response_schema: Any = None,
    budget: TimeBudget | None = None,
    images: list[Path] | None = None,
) -> str:
    """Call Gemini (google-genai) or an OpenAI-compatible endpoint."""
    llm_cfg = config.get("llm", {})
    provider = str(llm_cfg.get("provider", "gemini"))
    temperature = float(llm_cfg.get("temperature", 0.7))

    if provider == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))
        primary_model = str(llm_cfg.get("model", "gemini-3.8-flash"))
        models_to_try = [primary_model] + [
            m for m in ("gemini-3.5-flash", "gemini-3.5-flash-lite") if m != primary_model
        ]
        backoff_schedule = (5.0, 12.0, 25.0, 45.0)
        last_exc: Exception | None = None

        for model_name in models_to_try:
            now = time.monotonic()
            cooldown_until = _MODEL_COOLDOWN_UNTIL.get(model_name, 0.0)
            if cooldown_until > now:
                log.warning(
                    "Skipping %s (429 cooldown for %.1fs more).",
                    model_name,
                    cooldown_until - now,
                )
                continue

            for attempt in range(4):
                if budget is not None:
                    budget.check_or_exit("script_generator")
                try:
                    gen_kwargs: dict[str, Any] = {
                        "temperature": temperature,
                        "response_mime_type": "application/json",
                    }
                    if response_schema is not None:
                        gen_kwargs["response_schema"] = response_schema
                    contents: Any = prompt
                    if images:
                        contents = [prompt] + [
                            types.Part.from_bytes(data=p.read_bytes(), mime_type="image/jpeg")
                            for p in images
                            if p.exists()
                        ]
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(**gen_kwargs),
                    )
                    return resp.text or ""
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc)
                    if "404" in msg or "NOT_FOUND" in msg:
                        log.warning("Gemini model %s not found; trying next fallback.", model_name)
                        break
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        _MODEL_COOLDOWN_UNTIL[model_name] = time.monotonic() + 65.0
                        log.warning(
                            "Gemini model %s rate-limited (429); cooling down 65s and trying fallback.",
                            model_name,
                        )
                        break
                    transient = ("500", "502", "503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED", "timed out")
                    if any(code in msg for code in transient):
                        if attempt < 3:
                            sleep_s = backoff_schedule[attempt] + random.uniform(0.0, 2.0)
                            if budget is not None and budget.remaining_sec() < sleep_s + 120.0:
                                log.warning("Low time budget; skipping remaining retries for %s.", model_name)
                                break
                            log.warning(
                                "Gemini model %s transient error (attempt %d/4): %s; retrying in %.1fs",
                                model_name,
                                attempt + 1,
                                msg[:100],
                                sleep_s,
                            )
                            time.sleep(sleep_s)
                            continue
                        break
                    raise
        raise RuntimeError(f"Gemini API failed across {models_to_try}: {last_exc}")

    if provider == "openai_compatible":
        from openai import OpenAI

        oai_cfg = llm_cfg.get("openai_compatible", {})
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = str(oai_cfg.get("base_url", "")) or None
        model_name = str(oai_cfg.get("model", "")) or "gpt-4o-mini"
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _mock_ledger(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a deterministic event ledger from parsed subtitle segments."""
    events: list[dict[str, Any]] = []
    step = max(1, len(segments) // 5)
    for i in range(0, len(segments), step):
        seg = segments[i]
        end_seg = segments[min(len(segments) - 1, i + step - 1)]
        events.append(
            {
                "start": srt_parser.format_timestamp(seg["start"]).strip("[]"),
                "end": srt_parser.format_timestamp(end_seg["end"]).strip("[]"),
                "summary": f"A character reacts as {seg['text'][:60]}",
                "characters": ["Hero"],
                "visual_notes": "Moving geometric shapes across color bars.",
                "craft_notes": "Fast pacing builds tension across the cut.",
                "intensity": 4 if i == 0 else 3,
                "hook_candidate": i == 0,
            }
        )
    return {"events": events}


def _mock_manifest(
    slug: str,
    style: str,
    split_part: int,
    rights: RightsRecord,
    allowed_range: tuple[float, float],
    config: dict[str, Any],
) -> ProjectManifest:
    """Create a deterministic valid ProjectManifest for CI / --mock-llm."""
    script_cfg = config.get("script", {})
    min_beats = int(script_cfg.get("min_beats", 4))
    min_words = int(script_cfg.get("min_words", 40))
    beat_min = int(script_cfg.get("beat_min_words", 6))

    n_beats = max(min_beats, 5)
    start_a, end_a = allowed_range
    span = max(5.0, end_a - start_a)
    win = span / n_beats

    target_per_beat = max(beat_min + 2, (min_words // n_beats) + 2)
    sample_words = [
        "Something", "strange", "erupts", "across", "the", "frame", "as",
        "bright", "shapes", "race", "forward", "through", "the", "scene",
        "and", "shift", "the", "stakes", "for", "every", "single", "moment",
        "with", "sharp", "visual", "energy", "and", "clean", "framing", "now.",
    ]

    beats: list[Beat] = []
    for idx in range(n_beats):
        b_start = round(start_a + idx * win, 2)
        b_end = round(min(end_a, b_start + max(3.0, win - 0.5)), 2)
        if b_end <= b_start:
            b_end = round(b_start + 1.0, 2)
        words = sample_words[:target_per_beat]
        if idx == 0:
            words[0] = "Suddenly"
        kind = "commentary" if (style == "commentary" and idx % 3 == 1) else "story"
        mood = "hook" if idx == 0 else ("cliffhanger" if idx == n_beats - 1 else "action")
        beats.append(
            Beat(
                beat_id=idx + 1,
                narration=" ".join(words),
                source_window_start_sec=b_start,
                source_window_end_sec=b_end,
                visual_cue="bright moving shapes on color bars",
                mood=mood,
                kind=kind,
            )
        )

    attr = build_attribution_block(rights, style, split_part)
    return ProjectManifest(
        project_slug=slug,
        style=style,  # type: ignore[arg-type]
        part_number=split_part,
        title=f"{rights.work_title} hides a wild secret in plain sight #shorts"[:75],
        banner_text="STRANGE SIGNAL UNCOVERED",
        description=(
            f"A rapid look at {rights.work_title}. "
            "Every cut reveals unexpected motion and visual rhythm. "
            "Watch how the sequence escalates from the very first frame."
        ),
        hashtags=["#shorts", "#recap", "#film"],
        source_title=rights.work_title,
        attribution_block=attr,
        beats=beats,
    )


def generate_or_load_script(
    slug: str,
    project_dir: Path,
    work_dir: Path,
    segments: list[dict[str, Any]],
    rights: RightsRecord,
    allowed_range: tuple[float, float],
    config: dict[str, Any],
    *,
    mode: str,
    style: str,
    split_part: int,
    mock_llm: bool = False,
    force: bool = False,
    budget: TimeBudget | None = None,
    source_path: Path | None = None,
) -> tuple[ProjectManifest, list[str]]:
    """Load an existing committed script manifest or generate one via two-pass LLM."""
    manifest_path = project_dir / "02_script_manifest.json"
    if manifest_path.exists() and (mode == "render_only" or not force):
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ProjectManifest(**raw)
        errors, warnings = validate_manifest(manifest, config, allowed_range)
        if errors:
            log.error("Existing 02_script_manifest.json failed validation: %s", errors)
            sys.exit(_EXIT_CODE_SCRIPT_FAIL)
        return manifest, warnings

    if mode == "render_only":
        log.error("render_only mode requires existing %s", manifest_path)
        sys.exit(_EXIT_CODE_SCRIPT_FAIL)

    if mock_llm:
        ledger = _mock_ledger(segments)
        manifest = _mock_manifest(slug, style, split_part, rights, allowed_range, config)
        errors, warnings = validate_manifest(manifest, config, allowed_range, ledger)
        if errors:
            log.error("Mock manifest validation failed: %s", errors)
            sys.exit(_EXIT_CODE_SCRIPT_FAIL)
        project_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return manifest, warnings

    # Pass 1: Map-reduce event ledger
    pass1_tmpl = (_PROMPTS_DIR / "pass1_plot_extract.txt").read_text(encoding="utf-8")
    llm_cfg = config.get("llm", {})
    chunk_tokens = int(llm_cfg.get("chunk_tokens", 2500))
    overlap_tokens = int(llm_cfg.get("chunk_overlap_tokens", 200))
    text_chunks = srt_parser.format_chunks_for_llm(segments, chunk_tokens, overlap_tokens)

    pass1_sheets: list[Path] = []
    if (
        source_path is not None
        and config.get("source", {}).get("pass1_visual_overview", True)
    ):
        pass1_sheets = _extract_pass1_contact_sheets(
            source_path, allowed_range, work_dir, config
        )

    all_events: list[dict[str, Any]] = []
    for idx, chunk in enumerate(text_chunks):
        prompt = f"{pass1_tmpl}\n\nTRANSCRIPT CHUNK {idx + 1}/{len(text_chunks)}:\n{chunk}"
        raw_out = _call_llm(
            prompt,
            config,
            response_schema=EventLedgerResponse,
            budget=budget,
            images=pass1_sheets if idx == 0 else None,
        )
        parsed = json.loads(_strip_json_fences(raw_out))
        all_events.extend(parsed.get("events", []))

    ledger = {"events": all_events}
    (work_dir / "01_event_ledger.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8"
    )

    # Pass 2: Recap writer + validation loop
    pass2_tmpl = (_PROMPTS_DIR / "pass2_recap_writer.txt").read_text(encoding="utf-8")
    script_cfg = config.get("script", {})
    max_retries = int(llm_cfg.get("max_retries", 3))
    attr = build_attribution_block(rights, style, split_part)

    payload_context = json.dumps(
        {
            "LEDGER": ledger,
            "STYLE": style,
            "SPLIT_PART": split_part,
            "SOURCE_TITLE": rights.work_title,
            "ALLOWED_RANGE": {"start_sec": allowed_range[0], "end_sec": allowed_range[1]},
            "TARGET_WORDS": {
                "min": int(script_cfg.get("min_words", 220)),
                "max": int(script_cfg.get("max_words", 430)),
                "recommended": 320,
            },
            "BEAT_LIMITS": {
                "min_beats": int(script_cfg.get("min_beats", 12)),
                "max_beats": int(script_cfg.get("max_beats", 24)),
                "recommended_beats": 16,
                "beat_min_words": int(script_cfg.get("beat_min_words", 10)),
                "beat_max_words": int(script_cfg.get("beat_max_words", 28)),
            },
        },
        indent=2,
    )

    last_errors: list[str] = []
    best_raw = ""
    prev_valid_json = ""
    for attempt in range(1, max_retries + 2):
        feedback = ""
        if last_errors:
            feedback = (
                "\n\nPREVIOUS ATTEMPT FAILED VALIDATION. FIX ALL ERRORS:\n- "
                + "\n- ".join(last_errors)
            )
            if prev_valid_json:
                feedback += f"\n\nPREVIOUS DRAFT TO REVISE (keep valid timestamps, expand short beats to 16-22 words each):\n{prev_valid_json}"
        prompt = f"{pass2_tmpl}\n\nINPUTS:\n{payload_context}{feedback}"
        best_raw = _call_llm(
            prompt, config, response_schema=ScriptDraftResponse, budget=budget
        )
        try:
            data = json.loads(_strip_json_fences(best_raw))
            data = _normalize_manifest_data(data, allowed_range)
            data["schema_version"] = 1
            data["project_slug"] = slug
            data["style"] = style
            data["part_number"] = split_part
            data["source_title"] = rights.work_title
            data["attribution_block"] = attr
            manifest = ProjectManifest(**data)
            errors, warnings = validate_manifest(manifest, config, allowed_range, ledger)
            if not errors:
                project_dir.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
                return manifest, warnings
            last_errors = errors
            if not any("overlaps" in e or "chronological" in e for e in errors):
                prev_valid_json = best_raw
            # Enrich feedback with exact counts/windows so smaller models fix violations in 1 retry
            n_b = len(manifest.beats)
            min_comm = max(1, int(-(-n_b * float(script_cfg.get("min_commentary_share", 0.25)) // 1)))
            if any("Commentary share" in e for e in errors):
                last_errors.append(
                    f"Action: Change at least {min_comm} of your {n_b} beats to kind='commentary'."
                )
            if any("Total word count" in e for e in errors):
                total_w = sum(len(b.narration.split()) for b in manifest.beats)
                last_errors.append(
                    f"Action: Current total words={total_w}; adjust beat lengths to hit [{script_cfg.get('min_words')}, {script_cfg.get('max_words')}]."
                )
            if any("overlaps previous beat" in e or "chronological order" in e for e in errors):
                last_errors.append(
                    "Action: Make each beat's source_window_start_sec >= previous beat's source_window_end_sec - 1.0."
                )
        except Exception as exc:
            last_errors = [f"JSON/Schema error on attempt {attempt}: {exc}"]
        log.warning("Script attempt %d failed: %s", attempt, last_errors)

    debug_dir = work_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir / "failed_script_attempt.txt"
    debug_file.write_text(best_raw, encoding="utf-8")
    log.error(
        "Script validation failed after %d retries. Errors: %s. Best attempt: %s",
        max_retries,
        last_errors,
        debug_file,
    )
    sys.exit(_EXIT_CODE_SCRIPT_FAIL)
