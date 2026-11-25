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
    beats = [
        Beat(
            beat_id=i + 1,
            narration="Suddenly Alice discovers a glowing door behind the old stone wall today.",
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
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 50.0))
    assert errors == []


def test_banned_opener_rejected() -> None:
    manifest = _make_manifest()
    manifest.beats[0].narration = (
        "In this movie Alice discovers a glowing door behind the old stone wall."
    )
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 50.0))
    assert any("banned opener" in e.lower() for e in errors)


def test_low_commentary_share_rejected() -> None:
    manifest = _make_manifest()
    for b in manifest.beats:
        b.kind = "story"
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 50.0))
    assert any("commentary share" in e.lower() for e in errors)


def test_soft_proper_noun_warning_not_hard_error() -> None:
    manifest = _make_manifest()
    ledger = {"events": [{"summary": "Someone opens a door", "characters": [], "visual_notes": "", "craft_notes": ""}]}
    errors, warnings = validate_manifest(manifest, _base_config(), (0.0, 50.0), ledger)
    assert errors == []
    assert any("Alice" in w for w in warnings)


def test_banned_film_essay_cliche_rejected() -> None:
    manifest = _make_manifest()
    manifest.beats[1].narration = (
        "This intense dialogue efficiently establishes crucial backstory and emotional stakes for her."
    )
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 50.0))
    assert any("banned film-essay cliche" in e.lower() for e in errors)


def test_vague_beat1_placeholder_rejected_with_ledger() -> None:
    manifest = _make_manifest()
    manifest.beats[0].narration = (
        "A mysterious stranger walks through the ruined city gate carrying a glowing blade."
    )
    ledger = {"events": [{"summary": "Sintel searches", "characters": ["Sintel"], "visual_notes": "snowy gate", "craft_notes": ""}]}
    errors, _ = validate_manifest(manifest, _base_config(), (0.0, 50.0), ledger)
    assert any("vague placeholder" in e.lower() for e in errors)
