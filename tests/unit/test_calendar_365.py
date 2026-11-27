"""Unit tests for 365-day ranked release calendar and daily publisher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import main as main_mod
from pipeline.models import CalendarSlot, HistoryRecord, PlatformStatus, PublishedHistory, ReleaseCalendar365
from scripts.build_365_calendar import build_calendar, compute_total_parts


def test_compute_total_parts() -> None:
    assert compute_total_parts(240) == 1
    assert compute_total_parts(600) == 2
    assert compute_total_parts(1500) == 3
    assert compute_total_parts(3000) == 6
    assert compute_total_parts(4800) == 12


def test_build_calendar_dedup_and_golden_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assets" / "catalogs").mkdir(parents=True)
    golden = {
        "schema_version": 1,
        "tier1": [{"title": "Agent 327", "parts": 1, "license": "cc_by", "provider": "blender_peertube"}],
        "tier2": [{"title": "Sita Sings the Blues", "parts": 12, "license": "cc0", "provider": "archive_pd"}],
        "tier3": [],
    }
    (tmp_path / "assets" / "catalogs" / "golden_launch_80.json").write_text(json.dumps(golden), encoding="utf-8")

    # Mark Agent 327 as already published
    h = PublishedHistory(records=[
        HistoryRecord(
            slug="agent_327", work_title="Agent 327", normalized_title="agent 327",
            platforms={"youtube": PlatformStatus(published=True, url="https://youtu.be/1")},
        )
    ])
    (tmp_path / "published_history.json").write_text(h.model_dump_json(), encoding="utf-8")

    cal = build_calendar(target_days=365, mock_llm=True, mock_http=True)
    assert cal.total_slots == 12
    assert cal.slots[0].work_title == "Sita Sings the Blues"
    assert cal.slots[0].part_number == 1
    assert cal.slots[11].part_number == 12


def test_run_next_day_skips_pd_and_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cal = ReleaseCalendar365(
        generated_at="2026-11-27T00:00:00Z",
        total_slots=2,
        slots=[
            CalendarSlot(
                day_number=1, slot_id="day_001_pd", base_slug="pd_film", work_title="PD Film",
                part_number=1, total_parts=1, source_provider="archive_pd",
                license="public_domain", license_url="http://cc.org", source_page_url="http://ex.com",
                video_url="http://ex.com/v.mp4", attribution_text="PD", duration_sec=300,
                editorial_score=80.0, viral_hook_angle="Hook", pd_verified_by_human=False,
            ),
            CalendarSlot(
                day_number=2, slot_id="day_002_cc", base_slug="cc_film", work_title="CC Film",
                part_number=1, total_parts=1, source_provider="blender_peertube",
                license="cc_by", license_url="http://cc.org", source_page_url="http://ex.com",
                video_url="http://ex.com/v2.mp4", attribution_text="CC", duration_sec=300,
                editorial_score=90.0, viral_hook_angle="Hook 2", pd_verified_by_human=True,
            ),
        ],
    )
    (tmp_path / "release_calendar_365.json").write_text(cal.model_dump_json(), encoding="utf-8")
    (tmp_path / "published_history.json").write_text(PublishedHistory().model_dump_json(), encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(main_mod.app, ["run-next-day", "--dry-run"])
    assert res.exit_code == 0
    assert "Selected Day 2: CC Film" in res.stdout
    updated = ReleaseCalendar365(**json.loads((tmp_path / "release_calendar_365.json").read_text()))
    assert updated.slots[0].status == "awaiting_pd_verification"
