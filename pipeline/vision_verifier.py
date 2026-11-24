"""Multimodal vision verifier for candidate shot contact sheets."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from pipeline.models import Beat

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _build_contact_sheet(
    beat: Beat,
    candidates: list[dict[str, Any]],
    max_px: int = 1280,
) -> Image.Image:
    """Build a labeled 3-frame-per-candidate contact sheet (A, B, C...) for *beat*."""
    cell_w, cell_h = 320, 180
    rows = max(1, len(candidates))
    canvas = Image.new("RGB", (cell_w * 3 + 60, rows * cell_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)

    for r_idx, cand in enumerate(candidates):
        label = chr(ord("A") + r_idx)
        y0 = r_idx * cell_h
        draw.text((16, y0 + cell_h // 2 - 10), label, fill=(255, 230, 0))
        frames: list[Image.Image] = cand.get("frames", [])
        for c_idx in range(3):
            if c_idx < len(frames):
                thumb = frames[c_idx].resize((cell_w - 4, cell_h - 4))
                canvas.paste(thumb, (60 + c_idx * cell_w + 2, y0 + 2))

    if canvas.width > max_px:
        new_h = int(canvas.height * (max_px / canvas.width))
        canvas = canvas.resize((max_px, new_h))
    return canvas


def _strip_json_fences(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", raw.strip(), re.DOTALL)
    return match.group(1).strip() if match else raw.strip()


def verify_beats(
    beats: list[Beat],
    beat_candidates: dict[int, list[dict[str, Any]]],
    config: dict[str, Any],
    *,
    mock_vision: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    """Score candidate shots per beat using Gemini Vision or deterministic mock."""
    if mock_vision:
        out: dict[int, list[dict[str, Any]]] = {}
        for beat in beats:
            cands = beat_candidates.get(beat.beat_id, [])
            out[beat.beat_id] = [
                {
                    "id": chr(ord("A") + i),
                    "score": max(5, 9 - i),
                    "matches_cue": True,
                    "problems": [],
                    "note": "Mock vision match",
                }
                for i in range(len(cands))
            ]
        return out

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        log.warning("GEMINI_API_KEY missing; falling back to motion-only selection.")
        return {}

    from google import genai
    from google.genai import types

    vision_cfg = config.get("vision", {})
    llm_cfg = config.get("llm", {})
    max_px = int(vision_cfg.get("contact_sheet_max_px", 1280))
    batch_size = max(1, int(vision_cfg.get("batch_beats_per_call", 4)))
    sleep_sec = float(vision_cfg.get("rate_limit_sleep_sec", 4))
    model_name = str(llm_cfg.get("model", "gemini-2.5-flash"))
    sys_prompt = (_PROMPTS_DIR / "vision_shot_verifier.txt").read_text(encoding="utf-8")

    client = genai.Client(api_key=api_key)
    results: dict[int, list[dict[str, Any]]] = {}

    for start in range(0, len(beats), batch_size):
        batch = beats[start : start + batch_size]
        parts: list[Any] = [sys_prompt]
        for b in batch:
            cands = beat_candidates.get(b.beat_id, [])
            if not cands:
                continue
            sheet = _build_contact_sheet(b, cands, max_px=max_px)
            buf = io.BytesIO()
            sheet.save(buf, format="JPEG", quality=80)
            parts.append(
                f"\nBEAT {b.beat_id}: narration=\"{b.narration}\" visual_cue=\"{b.visual_cue}\""
            )
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))

        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=parts,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                data = json.loads(_strip_json_fences(resp.text or "{}"))
                for b_res in data.get("beats", []):
                    bid = int(b_res["beat_id"])
                    results[bid] = b_res.get("candidates", [])
                break
            except Exception as exc:
                log.warning("Vision batch attempt %d failed: %s", attempt + 1, exc)
                time.sleep(sleep_sec * (2**attempt))

        if start + batch_size < len(beats):
            time.sleep(sleep_sec)

    return results
