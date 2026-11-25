"""Unit tests for Milestone B: published history ledger & top-5 story scout QA module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import history as hist_mod
from pipeline import story_scout
from pipeline.models import PublishedHistory


def test_history_duplicate_detection_and_mark_published(tmp_path: Path) -> None:
    h_path = tmp_path / "published_history.json"
    h = PublishedHistory()
    hist_mod.upsert_rendered(
        h,
        slug="sintel_01",
        work_title="Sintel (2010)",
        source_url="https://example.com/sintel.mp4",
        source_id="Sintel",
        source_sha256="abc123",
    )
    hist_mod.save(h, h_path)

    loaded = hist_mod.load(h_path)
    assert hist_mod.is_duplicate(loaded, slug="sintel_01") is not None
    assert hist_mod.is_duplicate(loaded, source_url="https://example.com/sintel.mp4") is not None
    assert hist_mod.is_duplicate(loaded, source_sha256="abc123") is not None
    assert hist_mod.is_duplicate(loaded, work_title="Sintel") is not None
    assert hist_mod.is_published_anywhere(loaded.records[0]) is False

    # Mark published on YouTube -> published_partial
    hist_mod.mark_published(loaded, slug="sintel_01", platform="youtube", url="https://youtu.be/xyz")
    assert loaded.records[0].status == "published_partial"
    assert hist_mod.is_published_anywhere(loaded.records[0]) is True

    # Mark published on Instagram -> published_all
    hist_mod.mark_published(loaded, slug="sintel_01", platform="instagram", url="https://instagram.com/reel/123")
    assert loaded.records[0].status == "published_all"
    hist_mod.save(loaded, h_path)

    # Duplicate check exits with code 10
    with pytest.raises(SystemExit) as exc:
        hist_mod.check_no_duplicate_or_exit(
            slug="sintel_01",
            work_title="Sintel",
            path=h_path,
        )
    assert exc.value.code == 10


def test_story_scout_validation_and_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assets" / "catalogs").mkdir(parents=True)
    (tmp_path / "projects").mkdir()

    catalog = {
        "schema_version": 1,
        "items": [
            {
                "title": "Spring",
                "year": 2019,
                "license": "cc_by",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "source_id": "Spring_2019",
                "video_url": "https://example.com/spring.mp4",
                "srt_url": "",
                "runtime_min": 7.7,
                "resolution_height": 1080,
                "downloads": 50000,
                "subjects": ["fantasy"],
                "description": "A shepherd girl faces ancient spirits.",
            },
            {
                "title": "NC Rejected Film",
                "year": 2020,
                "license": "cc_by_nc",
                "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
                "source_id": "NC_2020",
                "video_url": "https://example.com/nc.mp4",
                "srt_url": "",
                "runtime_min": 6.0,
                "resolution_height": 1080,
                "downloads": 10000,
                "subjects": ["test"],
                "description": "Should be rejected.",
            },
        ],
    }
    (tmp_path / "assets" / "catalogs" / "open_movies.json").write_text(
        json.dumps(catalog), encoding="utf-8"
    )

    rep = story_scout.run_scout(
        candidate_count=5,
        license_scope="cc_only",
        mock_llm=True,
        mock_http=True,
    )
    assert rep["winner_slug"] == "spring_19"
    assert len(rep["candidates"]) == 1
    assert (tmp_path / "projects" / "spring_19" / "rights.json").exists()
    assert (tmp_path / "queue.json").exists()
