"""Two-pass LLM script generator with validation loop and mock mode."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from pipeline import srt_parser
from pipeline.models import Beat, ProjectManifest, RightsRecord
from pipeline.rights_guard import build_attribution_block
from pipeline.script_validator import validate_manifest

log = logging.getLogger(__name__)

_EXIT_CODE_SCRIPT_FAIL = 5
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _strip_json_fences(raw: str) -> str:
    """Strip markdown code fences if the LLM wrapped its JSON."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _call_llm(prompt: str, config: dict[str, Any]) -> str:
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
        client = genai.Client(api_key=api_key)
        model_name = str(llm_cfg.get("model", "gemini-3.8-flash"))
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        response_mime_type="application/json",
                    ),
                )
                return resp.text or ""
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                if any(code in msg for code in ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED")) and attempt < 3:
                    sleep_s = 5.0 * (2**attempt)
                    log.warning("Gemini API transient error (attempt %d/4): %s; retrying in %.0fs", attempt + 1, msg[:120], sleep_s)
                    time.sleep(sleep_s)
                    continue
                raise
        raise RuntimeError(f"Gemini API failed: {last_exc}")

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

    all_events: list[dict[str, Any]] = []
    for idx, chunk in enumerate(text_chunks):
        prompt = f"{pass1_tmpl}\n\nTRANSCRIPT CHUNK {idx + 1}/{len(text_chunks)}:\n{chunk}"
        raw_out = _call_llm(prompt, config)
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
                "min": int(script_cfg.get("min_words", 360)),
                "max": int(script_cfg.get("max_words", 430)),
            },
            "BEAT_LIMITS": {
                "min_beats": int(script_cfg.get("min_beats", 18)),
                "max_beats": int(script_cfg.get("max_beats", 26)),
                "beat_min_words": int(script_cfg.get("beat_min_words", 12)),
                "beat_max_words": int(script_cfg.get("beat_max_words", 28)),
            },
        },
        indent=2,
    )

    last_errors: list[str] = []
    best_raw = ""
    for attempt in range(1, max_retries + 2):
        feedback = ""
        if last_errors:
            feedback = (
                "\n\nPREVIOUS ATTEMPT FAILED VALIDATION. FIX ALL ERRORS:\n- "
                + "\n- ".join(last_errors)
            )
        prompt = f"{pass2_tmpl}\n\nINPUTS:\n{payload_context}{feedback}"
        best_raw = _call_llm(prompt, config)
        try:
            data = json.loads(_strip_json_fences(best_raw))
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
