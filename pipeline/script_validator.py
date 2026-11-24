"""Script validation (hard constraints + soft proper-noun grounding)."""

from __future__ import annotations

import re
from typing import Any

from pipeline.models import ProjectManifest

_BANNED_BEAT1_OPENERS = (
    "in this movie",
    "in this film",
    "today we",
    "welcome back",
    "hey guys",
    "hello everyone",
    "meet ",
)

_PROPER_NOUN_RE = re.compile(r"(?<!^)(?<!\. )(?<!\? )(?<!\! )\b([A-Z][a-z]{2,})\b")


def validate_manifest(
    manifest: ProjectManifest,
    config: dict[str, Any],
    allowed_range: tuple[float, float],
    ledger: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Validate *manifest* against script config rules.

    Returns:
        ``(hard_errors, soft_warnings)``
    """
    script_cfg = config.get("script", {})
    min_words = int(script_cfg.get("min_words", 360))
    max_words = int(script_cfg.get("max_words", 430))
    min_beats = int(script_cfg.get("min_beats", 18))
    max_beats = int(script_cfg.get("max_beats", 26))
    beat_min_words = int(script_cfg.get("beat_min_words", 12))
    beat_max_words = int(script_cfg.get("beat_max_words", 28))
    min_commentary_share = float(script_cfg.get("min_commentary_share", 0.25))

    errors: list[str] = []
    warnings: list[str] = []

    beats = manifest.beats
    if not (min_beats <= len(beats) <= max_beats):
        errors.append(
            f"Beat count {len(beats)} out of range [{min_beats}, {max_beats}]."
        )

    total_words = 0
    prev_start = -1.0
    prev_end = -1.0
    commentary_count = 0
    min_allowed, max_allowed = allowed_range

    for idx, beat in enumerate(beats, start=1):
        if beat.beat_id != idx:
            errors.append(f"Beat {idx} has non-contiguous beat_id={beat.beat_id}.")

        words = beat.narration.split()
        w_count = len(words)
        total_words += w_count
        if not (beat_min_words <= w_count <= beat_max_words):
            errors.append(
                f"Beat {beat.beat_id} word count {w_count} out of range "
                f"[{beat_min_words}, {beat_max_words}]."
            )

        if idx == 1:
            low = beat.narration.strip().lower()
            for opener in _BANNED_BEAT1_OPENERS:
                if low.startswith(opener):
                    errors.append(f"Beat 1 uses banned opener '{opener.strip()}'.")

        if beat.source_window_start_sec < min_allowed - 0.01:
            errors.append(
                f"Beat {beat.beat_id} start {beat.source_window_start_sec:.2f}s "
                f"< allowed_range start {min_allowed:.2f}s."
            )
        if beat.source_window_end_sec > max_allowed + 0.01:
            errors.append(
                f"Beat {beat.beat_id} end {beat.source_window_end_sec:.2f}s "
                f"> allowed_range end {max_allowed:.2f}s."
            )

        if beat.source_window_start_sec < prev_start:
            errors.append(
                f"Beat {beat.beat_id} start {beat.source_window_start_sec:.2f}s "
                f"breaks chronological order (prev {prev_start:.2f}s)."
            )
        if prev_end >= 0 and (prev_end - beat.source_window_start_sec) > 2.0:
            errors.append(
                f"Beat {beat.beat_id} overlaps previous beat by > 2.0s."
            )

        prev_start = beat.source_window_start_sec
        prev_end = beat.source_window_end_sec

        if beat.kind == "commentary":
            commentary_count += 1

    if not (min_words <= total_words <= max_words):
        errors.append(
            f"Total word count {total_words} out of range [{min_words}, {max_words}]."
        )

    if manifest.style == "commentary" and beats:
        share = commentary_count / len(beats)
        if share < min_commentary_share:
            errors.append(
                f"Commentary share {share:.2f} < minimum {min_commentary_share:.2f}."
            )
    elif manifest.style == "recap" and commentary_count > 0:
        errors.append("Style 'recap' must not contain 'commentary' beats.")

    if len(manifest.title) > 75:
        errors.append(f"Title length {len(manifest.title)} > 75 chars.")

    banner_words = manifest.banner_text.split()
    if not (3 <= len(banner_words) <= 6):
        errors.append(f"Banner word count {len(banner_words)} not in [3, 6].")

    # Soft grounding check against ledger proper nouns
    if ledger:
        ledger_text = " ".join(
            [
                str(ev.get("summary", ""))
                + " "
                + " ".join(ev.get("characters", []))
                + " "
                + str(ev.get("visual_notes", ""))
                + " "
                + str(ev.get("craft_notes", ""))
                for ev in ledger.get("events", [])
            ]
        ).lower()
        for beat in beats:
            for match in _PROPER_NOUN_RE.finditer(beat.narration):
                noun = match.group(1)
                if noun.lower() not in ledger_text:
                    warnings.append(
                        f"Beat {beat.beat_id}: proper noun '{noun}' not found in ledger."
                    )

    return errors, warnings
