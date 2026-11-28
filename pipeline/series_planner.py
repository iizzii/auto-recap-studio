"""Series Architect: splits a full Event Ledger into N connected chronological episodes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pipeline.models import CharacterEntry, SeriesBlueprint, SeriesPartPlan

log = logging.getLogger(__name__)

_ARCHITECT_PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "series_architect.txt"


def sanitize_series_blueprint(
    blueprint: SeriesBlueprint,
    allowed_range: tuple[float, float],
    total_parts: int,
) -> SeriesBlueprint:
    """Ensure strict chronological, non-overlapping windows with >= 15% width per part."""
    a_start, a_end = allowed_range
    total_span = max(30.0, a_end - a_start)
    min_width = max(15.0, total_span * 0.15)

    parts = sorted(blueprint.parts, key=lambda p: p.part_number)[:total_parts]
    needs_repair = len(parts) != total_parts

    if not needs_repair:
        parts[0].window_start_sec = max(a_start, parts[0].window_start_sec)
        parts[-1].window_end_sec = min(a_end, parts[-1].window_end_sec)
        for i, p in enumerate(parts):
            if p.window_end_sec - p.window_start_sec < min_width:
                needs_repair = True
                break
            if i + 1 < len(parts) and p.window_end_sec > parts[i + 1].window_start_sec + 2.0:
                needs_repair = True
                break

    if needs_repair:
        slice_len = total_span / total_parts
        repaired: list[SeriesPartPlan] = []
        for k in range(total_parts):
            ws = round(a_start + k * slice_len, 2)
            we = round(min(a_end, a_start + (k + 1) * slice_len), 2)
            orig = parts[k] if k < len(parts) else None
            repaired.append(
                SeriesPartPlan(
                    part_number=k + 1,
                    total_parts=total_parts,
                    window_start_sec=ws,
                    window_end_sec=we,
                    arc_goal=orig.arc_goal if orig else f"Part {k + 1} story arc",
                    previous_cliffhanger="" if k == 0 else (orig.previous_cliffhanger if orig else "Continuing from the cliffhanger."),
                    ending_hook=orig.ending_hook if orig else ("Shocking cliffhanger." if k < total_parts - 1 else "Final climax and payoff."),
                    forbidden_future_spoiler=orig.forbidden_future_spoiler if orig else "",
                )
            )
        blueprint.parts = repaired
    else:
        blueprint.parts = parts
    return blueprint


def build_series_blueprint(
    ledger: dict[str, Any],
    total_parts: int,
    allowed_range: tuple[float, float],
    base_slug: str,
    work_title: str,
    config: dict[str, Any],
    mock_llm: bool = False,
) -> SeriesBlueprint:
    """Build and sanitize a SeriesBlueprint from the full Pass 1 Event Ledger."""
    a_start, a_end = allowed_range
    if mock_llm:
        slice_len = max(5.0, (a_end - a_start) / max(1, total_parts))
        bp = SeriesBlueprint(
            base_slug=base_slug,
            work_title=work_title,
            total_parts=total_parts,
            characters=[CharacterEntry(name="Hero", short_role="main protagonist")],
            parts=[
                SeriesPartPlan(
                    part_number=k + 1,
                    total_parts=total_parts,
                    window_start_sec=round(a_start + k * slice_len, 2),
                    window_end_sec=round(min(a_end, a_start + (k + 1) * slice_len), 2),
                    arc_goal=f"Advance plot in part {k + 1}",
                    previous_cliffhanger="" if k == 0 else f"Hero escaped the trap in part {k}.",
                    ending_hook="Sudden danger strikes." if k < total_parts - 1 else "Final victory.",
                    forbidden_future_spoiler="" if k == total_parts - 1 else "Final ending.",
                )
                for k in range(total_parts)
            ],
        )
        return sanitize_series_blueprint(bp, allowed_range, total_parts)

    from pydantic import BaseModel as _BaseModel

    class _CharWire(_BaseModel):
        name: str
        short_role: str

    class _PartWire(_BaseModel):
        part_number: int
        total_parts: int
        window_start_sec: float
        window_end_sec: float
        arc_goal: str
        previous_cliffhanger: str
        ending_hook: str
        forbidden_future_spoiler: str

    class _BlueprintWire(_BaseModel):
        base_slug: str
        work_title: str
        total_parts: int
        characters: list[_CharWire]
        parts: list[_PartWire]

    from pipeline.chapter_commentator import call_gemini_structured

    sys_prompt = _ARCHITECT_PROMPT.read_text(encoding="utf-8") if _ARCHITECT_PROMPT.exists() else ""
    payload = json.dumps(
        {
            "base_slug": base_slug,
            "work_title": work_title,
            "total_parts": total_parts,
            "allowed_start": a_start,
            "allowed_end": a_end,
            "events": ledger.get("events", []),
        },
        indent=2,
    )
    data = call_gemini_structured(
        f"{sys_prompt}\n\n{payload}",
        [],
        _BlueprintWire,
        config,
        label=f"series_architect_{base_slug}",
    )
    bp = SeriesBlueprint(**data)
    return sanitize_series_blueprint(bp, allowed_range, total_parts)
