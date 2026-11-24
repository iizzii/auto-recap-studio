"""Unit tests for pipeline.models Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline.models import (
    ApprovalRecord,
    Beat,
    ProjectManifest,
    QueueFile,
    QueueItem,
    RightsRecord,
    RunReport,
    Shot,
)


# ---------------------------------------------------------------------------
# Beat
# ---------------------------------------------------------------------------

class TestBeat:
    """Validate Beat construction and constraints."""

    def test_valid_beat(self) -> None:
        beat = Beat(
            beat_id=1,
            narration="A quick recap of the opening scene.",
            source_window_start_sec=0.0,
            source_window_end_sec=10.5,
            visual_cue="wide establishing shot",
            mood="tense",
            kind="story",
        )
        assert beat.beat_id == 1
        assert beat.narration == "A quick recap of the opening scene."
        assert beat.source_window_start_sec == 0.0
        assert beat.source_window_end_sec == 10.5
        assert beat.visual_cue == "wide establishing shot"
        assert beat.mood == "tense"
        assert beat.kind == "story"
        assert beat.audio_duration_sec is None
        assert beat.audio_frames is None

    def test_end_before_start_fails(self) -> None:
        with pytest.raises(ValidationError):
            Beat(
                beat_id=1,
                narration="text",
                source_window_start_sec=20.0,
                source_window_end_sec=10.0,
                visual_cue="cue",
                mood="tense",
                kind="story",
            )

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Beat(beat_id=1)  # type: ignore[call-arg]

    def test_beat_id_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Beat(
                beat_id=0,
                narration="text",
                source_window_start_sec=0.0,
                source_window_end_sec=5.0,
                visual_cue="cue",
                mood="tense",
                kind="story",
            )

    def test_optional_audio_fields(self) -> None:
        beat = Beat(
            beat_id=2,
            narration="narration",
            source_window_start_sec=0.0,
            source_window_end_sec=5.0,
            visual_cue="cue",
            mood="calm",
            kind="commentary",
            audio_duration_sec=4.5,
            audio_frames=108000,
        )
        assert beat.audio_duration_sec == 4.5
        assert beat.audio_frames == 108000


# ---------------------------------------------------------------------------
# Shot
# ---------------------------------------------------------------------------

class TestShot:
    """Validate Shot construction."""

    def test_valid_shot(self) -> None:
        shot = Shot(
            beat_id=1,
            shot_index=0,
            src_start_sec=0.0,
            src_end_sec=3.0,
            speed_factor=1.0,
            frames=72,
        )
        assert shot.beat_id == 1
        assert shot.shot_index == 0
        assert shot.speed_factor == 1.0
        assert shot.frames == 72
        assert shot.match_score is None
        assert shot.flag is None

    def test_shot_with_flag(self) -> None:
        shot = Shot(
            beat_id=2,
            shot_index=1,
            src_start_sec=5.0,
            src_end_sec=8.0,
            speed_factor=0.5,
            frames=144,
            match_score=0.85,
            flag="low_match",
        )
        assert shot.flag == "low_match"
        assert shot.match_score == 0.85

    def test_invalid_flag_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Shot(
                beat_id=1,
                shot_index=0,
                src_start_sec=0.0,
                src_end_sec=1.0,
                speed_factor=1.0,
                frames=24,
                flag="invalid_flag",
            )

    def test_missing_required_shot_fields(self) -> None:
        with pytest.raises(ValidationError):
            Shot(beat_id=1, shot_index=0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ProjectManifest
# ---------------------------------------------------------------------------

class TestProjectManifest:
    """Validate ProjectManifest construction and banner_text rules."""

    @staticmethod
    def _make_manifest(**overrides: object) -> ProjectManifest:
        defaults: dict[str, object] = {
            "schema_version": 1,
            "project_slug": "my-recap",
            "style": "recap",
            "part_number": 1,
            "title": "Episode 1 Recap",
            "banner_text": "AMAZING RECAP VIDEO",
            "description": "A short description.",
            "hashtags": ["#recap"],
            "source_title": "Original Movie",
            "attribution_block": "CC-BY Some Creator",
            "beats": [
                Beat(
                    beat_id=1,
                    narration="Narration text.",
                    source_window_start_sec=0.0,
                    source_window_end_sec=10.0,
                    visual_cue="cue",
                    mood="tense",
                    kind="story",
                ),
            ],
            "shots": [
                Shot(
                    beat_id=1,
                    shot_index=0,
                    src_start_sec=0.0,
                    src_end_sec=5.0,
                    speed_factor=1.0,
                    frames=120,
                ),
            ],
        }
        defaults.update(overrides)
        return ProjectManifest(**defaults)  # type: ignore[arg-type]

    def test_valid_manifest(self) -> None:
        m = self._make_manifest()
        assert m.project_slug == "my-recap"
        assert m.schema_version == 1
        assert len(m.beats) == 1
        assert len(m.shots) == 1

    def test_title_too_long(self) -> None:
        with pytest.raises(ValidationError):
            self._make_manifest(title="A" * 76)

    def test_banner_text_too_few_words(self) -> None:
        with pytest.raises(ValidationError):
            self._make_manifest(banner_text="AB")

    def test_banner_text_too_many_words(self) -> None:
        with pytest.raises(ValidationError):
            self._make_manifest(banner_text="ONE TWO THREE FOUR FIVE SIX SEVEN")

    def test_banner_text_must_be_uppercase(self) -> None:
        with pytest.raises(ValidationError):
            self._make_manifest(banner_text="not uppercase text here")

    def test_style_literal(self) -> None:
        with pytest.raises(ValidationError):
            self._make_manifest(style="trailer")


# ---------------------------------------------------------------------------
# RightsRecord
# ---------------------------------------------------------------------------

class TestRightsRecord:
    """Validate RightsRecord for all license types."""

    @staticmethod
    def _base(**overrides: object) -> dict[str, object]:
        defaults: dict[str, object] = {
            "work_title": "Test Film",
            "creator": "Creator Name",
            "year": 2024,
            "license": "cc_by",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source_page_url": "https://example.com/film",
            "commercial_use_allowed": True,
            "derivatives_allowed": True,
            "attribution_text": "Credit: Creator Name",
            "pd_verified_by_human": False,
            "soundtrack_rights_note": "Original score – same license.",
            "verified_by": "unit-test",
            "verified_on": "2024-01-01",
        }
        defaults.update(overrides)
        return defaults

    def test_cc_by_valid(self) -> None:
        r = RightsRecord(**self._base())  # type: ignore[arg-type]
        assert r.license == "cc_by"
        assert r.commercial_use_allowed is True

    def test_public_domain(self) -> None:
        r = RightsRecord(
            **self._base(
                license="public_domain",
                license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                pd_basis="expired copyright – published 1920",
                pd_verified_by_human=True,
            )
        )  # type: ignore[arg-type]
        assert r.license == "public_domain"
        assert r.pd_basis is not None

    def test_cc_by_sa(self) -> None:
        r = RightsRecord(
            **self._base(
                license="cc_by_sa",
                license_url="https://creativecommons.org/licenses/by-sa/4.0/",
            )
        )  # type: ignore[arg-type]
        assert r.license == "cc_by_sa"

    def test_cc_by_nc_valid_model(self) -> None:
        r = RightsRecord(
            **self._base(
                license="cc_by_nc",
                license_url="https://creativecommons.org/licenses/by-nc/4.0/",
                commercial_use_allowed=False,
            )
        )  # type: ignore[arg-type]
        assert r.license == "cc_by_nc"
        assert r.commercial_use_allowed is False

    def test_owned_license(self) -> None:
        r = RightsRecord(
            **self._base(
                license="owned",
                license_url="",
                owner_statement="I own the rights to this work.",
            )
        )  # type: ignore[arg-type]
        assert r.license == "owned"
        assert r.owner_statement == "I own the rights to this work."

    def test_licensed_license(self) -> None:
        r = RightsRecord(
            **self._base(
                license="licensed",
                license_url="",
                license_document_url="https://example.com/license-doc.pdf",
            )
        )  # type: ignore[arg-type]
        assert r.license == "licensed"

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            RightsRecord(work_title="X")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ApprovalRecord
# ---------------------------------------------------------------------------

class TestApprovalRecord:
    """Validate ApprovalRecord defaults."""

    def test_defaults(self) -> None:
        a = ApprovalRecord(approved=False)
        assert a.approved is False
        assert a.approved_by is None
        assert a.approved_on is None

    def test_approved_with_details(self) -> None:
        a = ApprovalRecord(
            approved=True,
            approved_by="reviewer",
            approved_on="2024-06-15",
        )
        assert a.approved is True
        assert a.approved_by == "reviewer"
        assert a.approved_on == "2024-06-15"

    def test_approved_false_allows_none_fields(self) -> None:
        a = ApprovalRecord(approved=False, approved_by=None, approved_on=None)
        assert a.approved_by is None


# ---------------------------------------------------------------------------
# QueueItem / QueueFile
# ---------------------------------------------------------------------------

class TestQueueModels:
    """Validate QueueItem and QueueFile."""

    def test_queue_item_construction(self) -> None:
        qi = QueueItem(
            project_slug="recap-01",
            part_number=1,
            status="pending",
        )
        assert qi.project_slug == "recap-01"
        assert qi.part_number == 1
        assert qi.status == "pending"

    def test_queue_file_with_items(self) -> None:
        qi = QueueItem(project_slug="recap-01", part_number=1, status="pending")
        qf = QueueFile(items=[qi])
        assert len(qf.items) == 1
        assert qf.items[0].project_slug == "recap-01"

    def test_queue_file_empty(self) -> None:
        qf = QueueFile(items=[])
        assert qf.items == []


# ---------------------------------------------------------------------------
# RunReport
# ---------------------------------------------------------------------------

class TestRunReport:
    """Validate RunReport model."""

    def test_run_report_construction(self) -> None:
        rr = RunReport(
            project_slug="recap-01",
            part_number=1,
            status="success",
            duration_sec=120.5,
            output_path="/out/recap.mp4",
        )
        assert rr.project_slug == "recap-01"
        assert rr.status == "success"
        assert rr.duration_sec == 120.5

    def test_run_report_failed(self) -> None:
        rr = RunReport(
            project_slug="recap-01",
            part_number=1,
            status="failed",
            duration_sec=5.0,
            error="ffmpeg crashed",
        )
        assert rr.status == "failed"
        assert rr.error == "ffmpeg crashed"

    def test_run_report_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            RunReport(project_slug="x")  # type: ignore[call-arg]
