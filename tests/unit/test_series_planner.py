"""Unit tests for connected multi-part series planner and series-aware validator."""

from __future__ import annotations

from pipeline.models import Beat, CharacterEntry, ProjectManifest, SeriesBlueprint, SeriesPartPlan
from pipeline.script_validator import validate_manifest
from pipeline.series_planner import build_series_blueprint, sanitize_series_blueprint


def test_sanitize_series_blueprint_repairs_overlapping_windows() -> None:
    bp = SeriesBlueprint(
        base_slug="sprite",
        work_title="Sprite Fright",
        total_parts=3,
        characters=[CharacterEntry(name="Ellie", short_role="camper")],
        parts=[
            SeriesPartPlan(
                part_number=1, total_parts=3, window_start_sec=20.0, window_end_sec=250.0,
                arc_goal="Intro", previous_cliffhanger="", ending_hook="Turn",
            ),
            SeriesPartPlan(
                part_number=2, total_parts=3, window_start_sec=100.0, window_end_sec=200.0,  # overlaps!
                arc_goal="Mid", previous_cliffhanger="Turn", ending_hook="Crisis",
            ),
            SeriesPartPlan(
                part_number=3, total_parts=3, window_start_sec=200.0, window_end_sec=320.0,
                arc_goal="End", previous_cliffhanger="Crisis", ending_hook="Finale",
            ),
        ],
    )
    repaired = sanitize_series_blueprint(bp, (20.0, 320.0), 3)
    assert len(repaired.parts) == 3
    assert repaired.parts[0].window_end_sec <= repaired.parts[1].window_start_sec
    assert repaired.parts[1].window_end_sec <= repaired.parts[2].window_start_sec
    assert repaired.parts[0].arc_goal == "Intro"


def test_build_series_blueprint_mock() -> None:
    bp = build_series_blueprint(
        {"events": []}, total_parts=3, allowed_range=(10.0, 310.0),
        base_slug="test_s", work_title="Test", config={}, mock_llm=True,
    )
    assert len(bp.parts) == 3
    assert bp.parts[0].window_start_sec == 10.0
    assert bp.parts[2].window_end_sec == 310.0


def test_series_aware_validator_part2_rules() -> None:
    cfg = {
        "script": {
            "min_words": 20, "max_words": 200, "min_beats": 4, "max_beats": 8,
            "beat_min_words": 4, "beat_max_words": 20,
            "min_commentary_share": 0.10, "max_commentary_share": 0.25,
            "min_story_beats_between_commentary": 2,
        }
    }
    ledger = {"events": [{"characters": ["Ellie"], "summary": "Ellie runs."}]}
    part2 = SeriesPartPlan(
        part_number=2, total_parts=3, window_start_sec=100.0, window_end_sec=200.0,
        arc_goal="Escape", previous_cliffhanger="Sprites attacked.", ending_hook="Cornered.",
    )
    beats = [
        Beat(beat_id=1, narration="Ellie sprints away from the glowing mushroom cave.", source_window_start_sec=100.0, source_window_end_sec=120.0, visual_cue="Ellie runs", mood="action", kind="story"),
        Beat(beat_id=2, narration="She hides behind a hollow log.", source_window_start_sec=121.0, source_window_end_sec=145.0, visual_cue="Ellie hides", mood="tense", kind="commentary"),
        Beat(beat_id=3, narration="A loud snap gives away her spot.", source_window_start_sec=146.0, source_window_end_sec=170.0, visual_cue="twig snaps", mood="tense", kind="story"),
        Beat(beat_id=4, narration="The tiny sprites surround her in the dark.", source_window_start_sec=171.0, source_window_end_sec=195.0, visual_cue="sprites surround", mood="cliffhanger", kind="story"),
    ]
    manifest = ProjectManifest(
        project_slug="test_s", style="commentary", part_number=2,
        title="Part 2/3: Ellie is cornered #shorts", banner_text="ELLIE IS CORNERED",
        description="Test.", hashtags=["#shorts"], source_title="Test", attribution_block="Test", beats=beats,
    )
    # Part 2 starting with canonical name "Ellie" should PASS
    errors, _ = validate_manifest(manifest, cfg, (100.0, 200.0), ledger=ledger, series_part_plan=part2)
    assert errors == []

    # Part 2 resetting with "a young camper" should FAIL
    manifest.beats[0].narration = "A young camper sprints away from the glowing cave."
    errors2, _ = validate_manifest(manifest, cfg, (100.0, 200.0), ledger=ledger, series_part_plan=part2)
    assert any("resets characters as strangers" in e for e in errors2)
