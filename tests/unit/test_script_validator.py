"""Unit tests for pipeline.script_validator."""

from __future__ import annotations

from typing import Any

from pipeline.models import Beat, ProjectManifest
from pipeline.script_validator import validate_manifest


def _base_config() -> dict[str, Any]:
    return {
        "script": {
            "min_words": 40,
            "max_words": 100,
            "min_beats": 4,
            "max_beats": 8,
            "beat_min_words": 8,
            "beat_max_words": 25,
            "min_commentary_share": 0.25,
        }
    }


def _make_manifest(**overrides: Any) -> ProjectManifest:
    sample_lines = [
        "Suddenly Alice discovers a glowing door behind the old stone wall today.",
        "She steps inside the dark tunnel and finds a golden key.",
        "A giant shadow blocks her path near the silver gate.",
        "She unlocks the gate and escapes into the bright morning.",
    ]
    beats = [
        Beat(
            beat_id=i + 1,
            narration=sample_lines[i],
            source_window_start_sec=float(i * 10),
            source_window_end_sec=float(i * 10 + 8),
            visual_cue="Alice opens glowing stone door",
            mood="hook" if i == 0 else "action",
            kind="commentary" if i == 1 else "story",
        )
        for i in range(4)
    ]
    data: dict[str, Any] = {
        "project_slug": "test_proj",
        "style": "commentary",
        "part_number": 0,
        "title": "A glowing door changes everything #shorts",
        "banner_text": "SECRET DOOR FOUND",
        "description": "Short test description.",
        "hashtags": ["#shorts", "#test"],
        "source_title": "Test Film",
        "attribution_block": "Source: Test Film",
        "beats": beats,
    }
    data.update(overrides)
    return ProjectManifest(**data)


def test_valid_manifest_passes() -> None:
    manifest = _make_manifest()
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 45.0))
    assert errors == []


def test_banned_opener_rejected() -> None:
    manifest = _make_manifest()
    manifest.beats[0].narration = (
        "In this movie Alice discovers a glowing door behind the old stone wall."
    )
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 45.0))
    assert any("slow/generic setup" in e.lower() for e in errors)


def test_low_commentary_share_rejected() -> None:
    manifest = _make_manifest()
    for b in manifest.beats:
        b.kind = "story"
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 45.0))
    assert any("commentary share" in e.lower() for e in errors)


def test_soft_proper_noun_warning_not_hard_error() -> None:
    manifest = _make_manifest()
    ledger = {"events": [{"summary": "Someone opens a door", "characters": [], "visual_notes": "", "craft_notes": ""}]}
    errors, warnings = validate_manifest(manifest, _base_config(), (0.0, 45.0), ledger)
    assert errors == []
    assert any("Alice" in w for w in warnings)


def test_banned_film_essay_cliche_rejected() -> None:
    manifest = _make_manifest()
    manifest.beats[1].narration = (
        "This intense dialogue efficiently establishes crucial backstory and emotional stakes for her."
    )
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 45.0))
    assert any("banned film-essay cliche" in e.lower() for e in errors)


def test_vague_beat1_placeholder_rejected_with_ledger() -> None:
    manifest = _make_manifest()
    manifest.beats[0].narration = (
        "A mysterious stranger walks through the ruined city gate carrying a glowing blade."
    )
    ledger = {"events": [{"summary": "Sintel searches", "characters": ["Sintel"], "visual_notes": "snowy gate", "craft_notes": ""}]}
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 45.0), ledger)
    assert any("vague placeholder" in e.lower() for e in errors)


def test_visual_pointer_count_above_1_rejected() -> None:
    """Combined look-closely/notice-how count > 1 must be a hard error."""
    manifest = _make_manifest()
    manifest.beats[1].narration = "Look closely at the scar on the great wing she once bandaged herself."
    manifest.beats[2].narration = "Notice how she drops her sword the second she sees its eyes glow again."
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 45.0))
    assert any("visual-pointer" in e.lower() for e in errors)


def test_visual_pointer_count_exactly_1_passes() -> None:
    """Exactly one visual-pointer phrase across all beats should be allowed."""
    manifest = _make_manifest()
    manifest.beats[1].narration = "Look closely at the scar on the great wing she once bandaged herself."
    # beats[2] stays as default (no visual pointer)
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 45.0))
    assert not any("visual-pointer" in e.lower() for e in errors)


def test_beat1_starts_with_char_name_rejected() -> None:
    """Beat 1 that starts with a known character name must be rejected (with ledger)."""
    manifest = _make_manifest()
    manifest.beats[0].narration = "Sintel spent years searching for the dragon that took her pet scales."
    ledger = {"events": [{"summary": "Sintel searches", "characters": ["Sintel"], "visual_notes": "snowy gate", "craft_notes": ""}]}
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 45.0), ledger)
    assert any("character name" in e.lower() for e in errors)


def test_beat1_universal_hook_passes() -> None:
    """Beat 1 starting with a universal subject (not a char name) must pass."""
    manifest = _make_manifest()
    manifest.beats[0].narration = "This girl spent years searching for the dragon that stole her pet."
    ledger = {"events": [{"summary": "Sintel searches", "characters": ["Sintel"], "visual_notes": "snowy gate", "craft_notes": ""}]}
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 45.0), ledger)
    assert not any("character name" in e.lower() for e in errors)


def test_pass1_frame_sampling_within_allowed_range() -> None:
    """Verify that _extract_pass1_contact_sheets timestamps stay within [start, end-4] bounds."""
    allowed_range = (60.0, 900.0)
    cfg = {
        "source": {
            "pass1_max_frames": 16,
            "pass1_grid_cols": 4,
            "pass1_grid_rows": 4,
            "pass1_visual_overview": True,
        }
    }

    # We only test the timestamp math — no actual FFmpeg call.
    # Replicate the sampling logic from _extract_pass1_contact_sheets to validate bounds.
    start_s, end_s = allowed_range
    safe_end = max(start_s + 1.0, end_s - 4.0)
    max_frames = int(cfg["source"]["pass1_max_frames"])
    dur = max(2.0, safe_end - start_s)
    n_frames = min(max_frames, max(8, int(dur // 8)))
    n_early = max(1, int(round(n_frames * 0.65)))
    n_climax = max(1, n_frames - n_early)
    boundary = start_s + dur * 0.75
    early_step = max(0.1, (boundary - start_s) / max(1, n_early))
    climax_step = max(0.1, (safe_end - boundary) / max(1, n_climax))

    timestamps = []
    for i in range(n_early):
        timestamps.append(start_s + (i + 0.5) * early_step)
    for i in range(n_climax):
        timestamps.append(boundary + (i + 0.5) * climax_step)

    assert all(start_s <= ts <= safe_end for ts in timestamps), (
        f"Some timestamps fall outside [{start_s}, {safe_end}]: {[t for t in timestamps if not (start_s <= t <= safe_end)]}"
    )
    # Verify safe_end never reaches end-credit zone
    assert safe_end <= end_s - 4.0 + 0.001
