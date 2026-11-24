"""Unit tests for queue selection, weekly upload cap, and approval gate."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from pipeline.models import ApprovalRecord, QueueFile, QueueItem
from pipeline.publisher import check_public_approval, count_weekly_public_uploads


def test_weekly_cap_counts_recent_only(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    p1 = projects_root / "proj_one"
    p1.mkdir(parents=True)
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    old_iso = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)).isoformat()

    (p1 / "run_report.json").write_text(
        json.dumps({"publish_results": {"youtube": {"privacy": "public", "uploaded_at": now_iso}}}),
        encoding="utf-8",
    )
    p2 = projects_root / "proj_old"
    p2.mkdir(parents=True)
    (p2 / "run_report.json").write_text(
        json.dumps({"publish_results": {"youtube": {"privacy": "public", "uploaded_at": old_iso}}}),
        encoding="utf-8",
    )

    queue_path = tmp_path / "queue.json"
    queue_path.write_text(QueueFile().model_dump_json(), encoding="utf-8")
    assert count_weekly_public_uploads(projects_root, queue_path) == 1


def test_approval_gate_blocks_unapproved(tmp_path: Path) -> None:
    proj_dir = tmp_path / "projects" / "demo"
    proj_dir.mkdir(parents=True)
    (proj_dir / "approval.json").write_text(
        ApprovalRecord(approved=False).model_dump_json(), encoding="utf-8"
    )
    cfg: dict[str, Any] = {"publishing": {"max_public_uploads_per_week": 3}}
    allowed, reason = check_public_approval(
        proj_dir, cfg, projects_root=tmp_path / "projects", queue_path=tmp_path / "queue.json"
    )
    assert not allowed
    assert "approved=false" in reason


def test_approval_gate_allows_when_approved_and_under_cap(tmp_path: Path) -> None:
    proj_dir = tmp_path / "projects" / "demo"
    proj_dir.mkdir(parents=True)
    (proj_dir / "approval.json").write_text(
        ApprovalRecord(approved=True, approved_by="operator", approved_on="2026-11-24").model_dump_json(),
        encoding="utf-8",
    )
    cfg: dict[str, Any] = {"publishing": {"max_public_uploads_per_week": 3}}
    allowed, _ = check_public_approval(
        proj_dir, cfg, projects_root=tmp_path / "projects", queue_path=tmp_path / "queue.json"
    )
    assert allowed


def test_queue_pending_selection() -> None:
    q = QueueFile(
        items=[
            QueueItem(slug="done_one", video_source_url="https://example.com/1.mp4", status="done"),
            QueueItem(slug="next_one", video_source_url="https://example.com/2.mp4", status="pending"),
        ]
    )
    nxt = next((i for i in q.items if i.status == "pending"), None)
    assert nxt is not None
    assert nxt.slug == "next_one"
