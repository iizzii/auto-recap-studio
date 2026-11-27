"""Script validation (hard constraints + soft proper-noun grounding)."""

from __future__ import annotations

import re
from typing import Any

from pipeline.models import ProjectManifest

BANNED_BEAT1_PHRASES: tuple[str, ...] = (
    "in this movie",
    "in this film",
    "today we",
    "welcome back",
    "hey guys",
    "hello everyone",
    "meet ",
    "enjoys a quiet",
    "peaceful morning",
    "starts off peacefully",
    "begins on a quiet",
)

BANNED_META_PHRASES: tuple[str, ...] = (
    "establishes",
    "backstory",
    "pacing shift",
    "ramps up",
    "viewer tension",
    "emotional stakes",
    "emotional beat",
    "character motivation",
    "narrative arc",
    "the protagonist",
    "unresolved",
    "cinematic",
    "visceral",
    "masterfully",
    "incredibly intense situation",
    "dialogue reveal",
    "color saturation",
    "lighting design",
    "casts long shadows",
    "menacing atmosphere",
    "mechanical design",
    "traditional human proportions",
    "cartoon proportions",
    "exaggerated proportions",
    "animation framing",
    "animated lighting",
    "throughout the sequence",
    "chase sequences",
    "physical comedy",
    "comedic weight",
    "personal risk",
    "immediate peril",
    "malicious escalation",
    "forest ecosystem",
    "direct assault on",
    "highlights an",
    "highlights every",
    "emphasizes",
    "amplifies",
    "maximizes",
    "underscores",
    "contrast sharply",
    "here's the detail most people miss",
    "what she doesn't realize yet",
    "what he doesn't realize yet",
    "that single choice changes everything",
    "the tragic twist is",
    "final, humiliating punishment",
    "final humiliating punishment",
    "restores peace",
    "big mistake",
    "bad idea",
    "worst idea ever",
    "worst timing possible",
    "pure madness",
    "unbelievable",
    "talk about a",
    "recipe for disaster",
    "wild beyond belief",
    "pure unadulterated",
    "total recipe",
    "asking for serious trouble",
    "absolute middle of nowhere",
    "ever imaginable",
)

BANNED_AUDIO_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "sound effects",
    "sound effect",
    "music swells",
    "soundtrack",
    "deafening roar",
    "loud music",
)

_VAGUE_BEAT1_PLACEHOLDERS = (
    "someone finally",
    "a mysterious stranger",
)

# Visual-pointer phrases — max 1 combined occurrence allowed across the whole script
_VISUAL_POINTER_PHRASES = (
    "look closely",
    "look at",
    "notice how",
    "pay attention",
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
    min_commentary_share = float(script_cfg.get("min_commentary_share", 0.12))
    max_commentary_share = float(script_cfg.get("max_commentary_share", 0.25))
    min_story_between = int(script_cfg.get("min_story_beats_between_commentary", 2))

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
            for opener in BANNED_BEAT1_PHRASES:
                if opener in low:
                    errors.append(
                        "Beat 1 starts with a slow/generic setup instead of an immediate conflict hook."
                    )
                    break
            if ledger and any(ev.get("characters") or ev.get("visual_notes") for ev in ledger.get("events", [])):
                for vague in _VAGUE_BEAT1_PLACEHOLDERS:
                    if vague in low:
                        errors.append(f"Beat 1 uses vague placeholder '{vague}'; name the character/visual.")
            # No character name as the very first word of Beat 1
            first_word = beat.narration.strip().split()[0].rstrip(",.!?") if beat.narration.strip() else ""
            if ledger:
                ledger_names: set[str] = set()
                for ev in ledger.get("events", []):
                    for ch in ev.get("characters", []):
                        ledger_names.add(ch.strip().lower())
                if first_word.lower() in ledger_names and first_word.lower() not in {"a", "the", "this", "that"}:
                    errors.append(
                        f"Beat 1 starts with a character name '{first_word}'. "
                        "Open with a universal hook ('This girl...', 'A lone traveler...', 'After finding...') instead."
                    )

        low_nar = beat.narration.lower()
        for cliche in BANNED_META_PHRASES:
            if cliche in low_nar:
                errors.append(f"Beat {beat.beat_id} uses banned film-essay cliche '{cliche}'.")
        if config.get("audio", {}).get("source_audio_mode", "off") == "off":
            for aph in BANNED_AUDIO_HALLUCINATION_PHRASES:
                if aph in low_nar:
                    errors.append(
                        f"Narration references source audio ('{aph}'), which is stripped in this pipeline."
                    )

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

    if manifest.part_number == 0 and beats:
        span = max(1.0, max_allowed - min_allowed)
        if beats[-1].source_window_end_sec < min_allowed + 0.78 * span:
            errors.append(
                "Final beat stops too early in the film; include the actual ending payoff."
            )

    if not (min_words <= total_words <= max_words):
        errors.append(
            f"Total word count {total_words} out of range [{min_words}, {max_words}]."
        )

    # Max 1 visual-pointer phrase across the entire script
    all_narration_lower = " ".join(b.narration.lower() for b in beats)
    pointer_count = sum(all_narration_lower.count(phrase) for phrase in _VISUAL_POINTER_PHRASES)
    if pointer_count > 1:
        errors.append(
            f"Script uses visual-pointer phrases ('look closely', 'look at', 'notice how', "
            f"'pay attention') {pointer_count} times total; maximum allowed is 1. "
            "Rewrite extra commentary beats using natural storytelling pivots "
            "('The tragic twist is...', 'What she doesn't realize...', 'That single choice...')."
        )

    if manifest.style == "commentary" and beats:
        share = commentary_count / len(beats)
        if share < min_commentary_share:
            errors.append(
                f"Commentary share {share:.2f} < minimum {min_commentary_share:.2f}."
            )
        if share > max_commentary_share:
            errors.append(
                f"Commentary share {share:.2f} > maximum {max_commentary_share:.2f}."
            )
        last_comm_idx = -999
        for idx, b in enumerate(beats):
            if b.kind == "commentary":
                if len(b.narration.split()) > 12:
                    errors.append(
                        f"Beat {b.beat_id} commentary exceeds 12 words ({len(b.narration.split())})."
                    )
                if idx - last_comm_idx - 1 < min_story_between:
                    errors.append(
                        f"Beat {b.beat_id} commentary too close to previous commentary (need >= {min_story_between} story beats between)."
                    )
                last_comm_idx = idx
    elif manifest.style == "recap" and commentary_count > 0:
        errors.append("Style 'recap' must not contain 'commentary' beats.")

    # Grammar & adjacent 7+ char word repetition guard
    _a_vowel_re = re.compile(r"\ba\s+([aeiou]\w+)", re.IGNORECASE)
    _a_exceptions = {"one", "unique", "united", "uniform", "usual", "useful", "once"}
    _char_names: set[str] = set()
    if ledger:
        for ev in ledger.get("events", []):
            for ch in ev.get("characters", []):
                _char_names.update(w.lower() for w in ch.split())
    prev_long_words: set[str] = set()
    for b in beats:
        for m in _a_vowel_re.finditer(b.narration):
            nxt = m.group(1).lower()
            if nxt not in _a_exceptions:
                errors.append(f"Beat {b.beat_id} grammar error: 'a {m.group(1)}' should use 'an'.")
        cur_long = {
            w.lower().strip(".,!?;:\"'")
            for w in b.narration.split()
            if len(w.strip(".,!?;:\"'")) >= 7 and w.lower().strip(".,!?;:\"'") not in _char_names
        }
        overlap = prev_long_words & cur_long
        if overlap:
            errors.append(f"Beat {b.beat_id} repeats 7+ char word from previous beat: {sorted(overlap)[0]}.")
        prev_long_words = cur_long

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
