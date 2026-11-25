"""Unit tests for mood-adaptive BGM, Telegram GitHub fallback, and story discovery."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from pipeline.audio_mixer import infer_bgm_profile
from pipeline.models import Beat
from pipeline.publisher import _publish_telegram
from scripts.discover_stories import _pick_best_video_and_srt, classify_license_url, make_slug


def _beat(bid: int, mood: str) -> Beat:
    return Beat(
        beat_id=bid,
        narration="Suddenly something wild happened across the whole scene right now.",
        source_window_start_sec=float(bid * 5),
        source_window_end_sec=float(bid * 5 + 4),
        visual_cue="character moves across frame",
        mood=mood,  # type: ignore[arg-type]
        kind="story",
    )


def test_infer_bgm_profile_cartoon_emotional_suspense() -> None:
    assert infer_bgm_profile([_beat(1, "funny"), _beat(2, "funny"), _beat(3, "action")]) == "cartoon_playful"
    assert infer_bgm_profile([_beat(1, "emotional"), _beat(2, "calm"), _beat(3, "reveal")]) == "emotional_story"
    assert infer_bgm_profile([_beat(1, "tense"), _beat(2, "action"), _beat(3, "cliffhanger")]) == "suspense_action"


def test_classify_license_url_allows_only_cleared_licenses() -> None:
    assert classify_license_url("https://creativecommons.org/licenses/by/3.0/") == "cc_by"
    assert classify_license_url("https://creativecommons.org/publicdomain/zero/1.0/") == "cc0"
    assert classify_license_url("http://creativecommons.org/licenses/by-sa/4.0/") == "cc_by_sa"
    assert classify_license_url("http://creativecommons.org/publicdomain/mark/1.0/") == "public_domain"
    assert classify_license_url("https://creativecommons.org/licenses/by-nc-nd/3.0/") is None
    assert classify_license_url(None) is None


def test_make_slug_sanitizes_and_deduplicates() -> None:
    existing = {"superman_1941"}
    assert make_slug("Superman-1941!", existing) == "superman_1941_2"


def test_pick_best_video_and_srt_prefers_non_512kb_mp4() -> None:
    files = [
        {"name": "clip_512kb.mp4", "length": "420", "width": "320"},
        {"name": "clip_hd.mp4", "length": "420", "width": "640"},
        {"name": "clip_en.srt"},
    ]
    vid, srt, dur = _pick_best_video_and_srt(files, "item1")
    assert vid == "https://archive.org/download/item1/clip_hd.mp4"
    assert srt == "https://archive.org/download/item1/clip_en.srt"
    assert dur == 420.0


def test_publish_telegram_includes_github_link_on_oversized_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "iizzii/auto-recap-studio")
    monkeypatch.setenv("GITHUB_RUN_ID", "99999")

    work = tmp_path / "sintel_01"
    out = work / "output"
    out.mkdir(parents=True)
    final_mp4 = out / "final_short.mp4"
    final_mp4.write_bytes(b"0" * (52 * 1024 * 1024))
    thumb = out / "thumbnail.jpg"
    thumb.write_bytes(b"")

    meta = {"title": "Test Title", "description": "Test desc."}
    cfg = {"publishing": {"telegram": {"max_video_mb": 50}}}

    posted_payloads: list[dict] = []

    def fake_post(url: str, **kwargs):
        posted_payloads.append({"url": url, **kwargs})
        resp = MagicMock()
        if url.endswith("/sendVideo"):
            raise requests.HTTPError("413 Entity Too Large")
        resp.raise_for_status.return_value = None
        return resp

    with (
        patch("pipeline.publisher.ffutil.duration", return_value=75.0),
        patch("pipeline.publisher.ffutil.run") as mock_ff,
        patch("pipeline.publisher.requests.post", side_effect=fake_post),
    ):
        mock_ff.side_effect = lambda cmd: (work / "telegram_upload.mp4").write_bytes(b"1" * 1024)
        res = _publish_telegram(final_mp4, thumb, meta, cfg)

    assert res["status"] == "notice_sent"
    assert res["github_url"] == "https://github.com/iizzii/auto-recap-studio/actions/runs/99999"
    assert any("99999" in str(p.get("json", {}).get("text", "")) for p in posted_payloads)
