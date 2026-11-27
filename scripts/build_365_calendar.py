"""Automated 365-day ranked release calendar harvester & ranker."""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from pipeline import history as hist_mod
from pipeline import story_scout
from pipeline.models import CalendarSlot, ReleaseCalendar365

log = logging.getLogger(__name__)

_GOLDEN_PATH = Path("assets/catalogs/golden_launch_80.json")
_CALENDAR_JSON = Path("release_calendar_365.json")
_CALENDAR_MD = Path("RELEASE_CALENDAR_365.md")
_HISTORY_PATH = Path("published_history.json")

_PIRACY_TERMS = {
    "disney", "pixar", "dreamworks", "illumination", "warner",
    "marvel", "dc comics", "looney tunes", "tom and jerry", "minion",
    "superman", "popeye", "betty boop", "bugs bunny", "mickey",
}


def compute_total_parts(duration_sec: float) -> int:
    """Compute recommended series part count from source duration."""
    if duration_sec < 570:
        return 1
    if duration_sec < 1020:
        return 2
    if duration_sec < 2100:
        return 3
    if duration_sec < 3900:
        return 6
    return 12


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _is_trademark_or_piracy(title: str, desc: str = "") -> bool:
    txt = (title + " " + desc).lower()
    return any(t in txt for t in _PIRACY_TERMS)


def _fetch_youtube_cc_candidates(max_results: int = 30) -> list[dict[str, Any]]:
    """Harvest high-like CC-BY animated short films via yt-dlp (PERSONAL_MAC only)."""
    out: list[dict[str, Any]] = []
    try:
        res = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                "--flat-playlist=False",
                f"ytsearch{max_results}:creative commons animated short film",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in res.stdout.splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            lic = str(d.get("license") or "")
            dur = float(d.get("duration") or 0)
            likes = int(d.get("like_count") or 0)
            title = str(d.get("title") or "")
            desc = str(d.get("description") or "")
            if "Creative Commons Attribution" not in lic:
                continue
            if not (180 <= dur <= 2700) or likes < 500:
                continue
            if _is_trademark_or_piracy(title, desc):
                continue
            out.append({
                "title": title,
                "year": None,
                "source_provider": "youtube_cc",
                "runner_target": "personal_mac",
                "license": "cc_by",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "source_page_url": str(d.get("webpage_url") or ""),
                "video_url": str(d.get("webpage_url") or ""),
                "srt_url": None,
                "attribution_text": f'"{title}" by {d.get("uploader", "YouTube Creator")}, CC BY',
                "duration_sec": dur,
                "resolution_height": int(d.get("height") or 1080),
                "likes": likes,
                "views": int(d.get("view_count") or 0),
                "source_id": f"yt_{d.get('id', '')}",
                "description": desc[:400],
            })
    except Exception as exc:
        log.warning("youtube_cc harvest failed: %s", exc)
    return out


def _score_candidate(c: dict[str, Any], max_likes: int, mock_llm: bool = False) -> float:
    """Compute 100-point composite score (35 engagement + 25 tech/rights + 40 story)."""
    likes = max(1, int(c.get("likes") or 0))
    eng = min(35.0, 35.0 * (math.log10(likes + 1) / max(1.0, math.log10(max_likes + 1))))

    lic = c.get("license", "public_domain")
    lic_pts = 12.0 if lic in ("cc0", "cc_by") else (10.0 if lic == "cc_by_sa" else 6.0)
    h = int(c.get("resolution_height") or 720)
    res_pts = 8.0 if h >= 1080 else (5.0 if h >= 720 else 2.0)
    sub_pts = 5.0 if c.get("srt_url") else 0.0
    tech = lic_pts + res_pts + sub_pts

    story = 34.0 if mock_llm else float(c.get("story_score", 32.0))
    return round(min(100.0, eng + tech + story), 2)


def build_calendar(
    target_days: int = 365,
    include_youtube_cc: bool = False,
    mock_llm: bool = False,
    mock_http: bool = False,
) -> ReleaseCalendar365:
    """Harvest, deduplicate, rank, and expand stories into a 365-day ReleaseCalendar365."""
    history = hist_mod.load(_HISTORY_PATH)
    published_norms = {
        r.normalized_title for r in history.records if hist_mod.is_published_anywhere(r)
    }

    raw: list[dict[str, Any]] = []

    # 1. Golden Launch 80
    if _GOLDEN_PATH.exists():
        g = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
        for tier_idx, key in enumerate(("tier1", "tier2", "tier3"), start=1):
            for item in g.get(key, []):
                raw.append({
                    "title": item["title"],
                    "year": item.get("year"),
                    "source_provider": item.get("provider", "blender_peertube"),
                    "runner_target": "github_actions",
                    "license": item.get("license", "cc_by"),
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "source_page_url": item.get("source_page_url", "https://video.blender.org"),
                    "video_url": item.get("video_url", f"https://archive.org/details/{item.get('archive_id', story_scout.make_slug(item['title']))}"),
                    "srt_url": None,
                    "attribution_text": f'"{item["title"]}" ({item.get("license", "cc_by").upper()})',
                    "duration_sec": float(item.get("parts", 1) * 420),
                    "resolution_height": 1080 if tier_idx == 1 else 720,
                    "likes": 5000 if tier_idx == 1 else (2000 if tier_idx == 2 else 800),
                    "views": 50000,
                    "source_id": item.get("archive_id") or story_scout.make_slug(item["title"]),
                    "golden_tier": tier_idx,
                    "forced_parts": item.get("parts"),
                    "pd_verified_by_human": item.get("license", "cc_by") != "public_domain",
                })

    # 2. Live HD providers (when not mock_http)
    if not mock_http:
        for c in story_scout._fetch_blender_peertube():
            c["source_provider"] = "blender_peertube"
            c["runner_target"] = "github_actions"
            c["license"] = c.get("license_key", "cc_by")
            c["duration_sec"] = float(c.get("runtime_min", 5)) * 60.0
            c["likes"] = int(c.get("downloads", 1000))
            raw.append(c)
        for c in story_scout._fetch_wikimedia_commons():
            c["source_provider"] = "wikimedia_commons"
            c["runner_target"] = "github_actions"
            c["license"] = c.get("license_key", "cc_by")
            c["duration_sec"] = float(c.get("runtime_min", 5)) * 60.0
            c["likes"] = int(c.get("downloads", 500))
            raw.append(c)
        for q in ("animated short film", "3d animation short", "cartoon short", "sci-fi animation"):
            for c in story_scout._fetch_sepiasearch(query=q):
                c["source_provider"] = "sepiasearch"
                c["runner_target"] = "github_actions"
                c["license"] = c.get("license_key", "cc_by")
                c["duration_sec"] = float(c.get("runtime_min", 5)) * 60.0
                c["likes"] = int(c.get("downloads", 300))
                raw.append(c)
        if include_youtube_cc:
            raw.extend(_fetch_youtube_cc_candidates())

    # 3. Dedup & safety filter
    seen_norms: set[str] = set()
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for c in raw:
        norm = _norm_title(c["title"])
        sid = str(c.get("source_id") or norm)
        if not norm or norm in seen_norms or sid in seen_ids or norm in published_norms:
            continue
        if _is_trademark_or_piracy(c["title"], str(c.get("description", ""))):
            continue
        lic = c.get("license", "unknown")
        if lic not in ("cc0", "cc_by", "cc_by_sa", "public_domain"):
            continue
        seen_norms.add(norm)
        seen_ids.add(sid)
        deduped.append(c)

    max_likes = max([int(c.get("likes") or 1) for c in deduped] or [1])
    for c in deduped:
        c["editorial_score"] = _score_candidate(c, max_likes, mock_llm=mock_llm)

    # Sort: Golden Tier 1 first, then Tier 2, then descending editorial_score
    deduped.sort(key=lambda x: (x.get("golden_tier", 99), -x["editorial_score"]))

    # 4. Expand into up to target_days slots
    slots: list[CalendarSlot] = []
    day = 1
    for c in deduped:
        if day > target_days:
            break
        dur = float(c.get("duration_sec") or 300.0)
        parts = int(c.get("forced_parts") or compute_total_parts(dur))
        if day + parts - 1 > target_days:
            continue
        base_slug = story_scout.make_slug(c["title"], c.get("year"))
        for p in range(1, parts + 1):
            slot_id = f"day_{day:03d}_{base_slug}_p{p}"
            slots.append(
                CalendarSlot(
                    day_number=day,
                    slot_id=slot_id,
                    base_slug=base_slug,
                    work_title=c["title"],
                    year=c.get("year"),
                    part_number=p,
                    total_parts=parts,
                    source_provider=c.get("source_provider", "blender_peertube"),
                    runner_target=c.get("runner_target", "github_actions"),
                    license=c["license"],
                    license_url=c.get("license_url") or "https://creativecommons.org/licenses/by/4.0/",
                    source_page_url=c.get("source_page_url") or c.get("video_url", ""),
                    video_url=c.get("video_url", ""),
                    srt_url=c.get("srt_url"),
                    attribution_text=c.get("attribution_text") or c["title"],
                    duration_sec=dur,
                    likes=int(c.get("likes") or 0),
                    views=int(c.get("views") or 0),
                    editorial_score=c["editorial_score"],
                    viral_hook_angle=f"Watch {c['title']} ({p}/{parts})",
                    pd_verified_by_human=bool(c.get("pd_verified_by_human", c["license"] != "public_domain")),
                    status="scheduled",
                )
            )
            day += 1

    cal = ReleaseCalendar365(
        schema_version=1,
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        total_slots=len(slots),
        slots=slots,
    )
    _CALENDAR_JSON.write_text(cal.model_dump_json(indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# 365-Day Ranked Release Calendar",
        "",
        "| Day | Title | Part | License | Likes | Score | Runner | Hook |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in slots:
        md_lines.append(
            f"| {s.day_number} | {s.work_title} | {s.part_number}/{s.total_parts} | "
            f"{s.license} | {s.likes} | {s.editorial_score} | {s.runner_target} | {s.viral_hook_angle} |"
        )
    _CALENDAR_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return cal
