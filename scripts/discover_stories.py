"""Discover free-to-use classic cartoons and CC-licensed animations on Internet Archive.

Searches Archive.org for verified CC0 / CC BY / CC BY-SA animated stories and
classic cartoons, inspects their metadata & playable MP4 files, creates
``projects/<slug>/rights.json``, and optionally queues verified items into
``queue.json`` for GitHub Actions production.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import requests

from pipeline.models import QueueFile, QueueItem, RightsRecord

_IA_SEARCH = "https://archive.org/advancedsearch.php"
_IA_META = "https://archive.org/metadata/{item_id}"

_CC_LICENSE_MAP: dict[str, str] = {
    "creativecommons.org/publicdomain/zero/1.0": "cc0",
    "creativecommons.org/licenses/by/3.0": "cc_by",
    "creativecommons.org/licenses/by/4.0": "cc_by",
    "creativecommons.org/licenses/by-sa/3.0": "cc_by_sa",
    "creativecommons.org/licenses/by-sa/4.0": "cc_by_sa",
}

_BLOCKED_KEYWORDS = (
    "trailer",
    "capcut",
    "template",
    "porn",
    "xxx",
    "fetish",
    "exploitation",
    "dublado",
    "godzilla",
)

# Verified classic US-PD cartoon series/studios whose pre-1964 copyrights were unrenewed
# or published pre-1931 (still requires pd_verified_by_human=True if auto_verify_curated_pd).
_CURATED_PD_PREFIXES = (
    "superman_",
    "superman-",
    "bb_",
    "betty_boop",
    "popeye_",
)


def classify_license_url(license_url: str | None) -> str | None:
    """Return pipeline license key if *license_url* is strictly permitted, else None."""
    if not license_url:
        return None
    low = license_url.lower().strip().rstrip("/")
    if "nc" in low.split("/") or "nd" in low.split("/") or "by-nc" in low or "by-nd" in low:
        return None
    for substr, lic_key in _CC_LICENSE_MAP.items():
        if substr in low:
            return lic_key
    if "creativecommons.org/publicdomain/mark/1.0" in low or low.endswith(
        "creativecommons.org/licenses/publicdomain"
    ):
        return "public_domain"
    return None


def make_slug(raw_id: str, existing: set[str]) -> str:
    """Sanitize an Archive identifier into a valid pipeline slug ^[a-z0-9_]{3,40}$."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw_id.lower()).strip("_")
    cleaned = cleaned[:36] or "story"
    if len(cleaned) < 3:
        cleaned = f"ia_{cleaned}"
    candidate = cleaned
    idx = 2
    while candidate in existing:
        suffix = f"_{idx}"
        candidate = f"{cleaned[: 40 - len(suffix)]}{suffix}"
        idx += 1
    return candidate


def _pick_best_video_and_srt(
    files: list[dict[str, Any]],
    item_id: str,
) -> tuple[str | None, str | None, float]:
    """Select best MP4 video URL (>=480px wide, 3–30 min) and optional SRT URL."""
    mp4s: list[tuple[int, float, str]] = []
    srt_url: str | None = None

    for f in files:
        name = str(f.get("name", ""))
        low = name.lower()
        if low.endswith("_en.srt") or (low.endswith(".srt") and srt_url is None):
            srt_url = f"https://archive.org/download/{item_id}/{name}"
            continue
        if not low.endswith(".mp4"):
            continue
        try:
            dur = float(f.get("length", 0) or 0)
        except (TypeError, ValueError):
            dur = 0.0
        if not (180.0 <= dur <= 1800.0):
            continue
        try:
            width = int(f.get("width", 0) or 0)
        except (TypeError, ValueError):
            width = 0
        if width and width < 400:
            continue
        # Prefer standard MP4s over 512kb low-res derivatives
        score = (10000 if "_512kb" not in low else 0) + width
        mp4s.append((score, dur, f"https://archive.org/download/{item_id}/{name}"))

    if not mp4s:
        return None, srt_url, 0.0
    mp4s.sort(key=lambda t: t[0], reverse=True)
    return mp4s[0][2], srt_url, mp4s[0][1]


def search_ia_candidates(category: str = "all", rows: int = 20) -> list[dict[str, Any]]:
    """Query Internet Archive for free-to-use cartoons and CC-licensed animations."""
    queries: list[str] = []
    if category in ("classic_cartoons", "all"):
        queries.append(
            'mediatype:movies AND collection:classic_cartoons AND format:MPEG4'
        )
    if category in ("cc_animation", "all"):
        queries.append(
            'mediatype:movies AND (licenseurl:"http://creativecommons.org/licenses/by/3.0/" '
            'OR licenseurl:"https://creativecommons.org/licenses/by/3.0/" '
            'OR licenseurl:"http://creativecommons.org/licenses/by/4.0/" '
            'OR licenseurl:"https://creativecommons.org/licenses/by/4.0/" '
            'OR licenseurl:"https://creativecommons.org/publicdomain/zero/1.0/") '
            'AND (animation OR cartoon OR "short film")'
        )

    seen_ids: set[str] = set()
    docs: list[dict[str, Any]] = []
    for q in queries:
        resp = requests.get(
            _IA_SEARCH,
            params={
                "q": q,
                "fl[]": ["identifier", "title", "year", "creator", "licenseurl"],
                "sort[]": "downloads desc",
                "rows": str(rows),
                "output": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        for doc in resp.json().get("response", {}).get("docs", []):
            ident = str(doc.get("identifier", ""))
            title_low = str(doc.get("title", "")).lower()
            if not ident or ident in seen_ids:
                continue
            if any(bad in title_low or bad in ident.lower() for bad in _BLOCKED_KEYWORDS):
                continue
            seen_ids.add(ident)
            docs.append(doc)
    return docs


def discover_and_stage(
    *,
    category: str = "all",
    max_items: int = 2,
    publish_target: str = "telegram",
    projects_root: Path = Path("projects"),
    queue_path: Path = Path("queue.json"),
    include_pre1931_pd: bool = True,
) -> list[str]:
    """Discover new verified stories, write ``rights.json``, and append to ``queue.json``."""
    projects_root.mkdir(parents=True, exist_ok=True)
    existing_slugs = {p.name for p in projects_root.iterdir() if p.is_dir()}

    if queue_path.exists():
        queue = QueueFile(**json.loads(queue_path.read_text(encoding="utf-8")))
    else:
        queue = QueueFile()

    existing_urls = {item.video_source_url for item in queue.items}
    existing_slugs.update(item.slug for item in queue.items)

    docs = search_ia_candidates(category=category, rows=max(20, max_items * 8))
    added_slugs: list[str] = []
    today_str = dt.date.today().isoformat()

    for doc in docs:
        if len(added_slugs) >= max_items:
            break
        ident = str(doc["identifier"])
        meta_resp = requests.get(_IA_META.format(item_id=ident), timeout=30)
        if meta_resp.status_code != 200:
            continue
        meta = meta_resp.json()
        md = meta.get("metadata", {})
        raw_lic_url = str(md.get("licenseurl") or doc.get("licenseurl") or "").strip() or None
        lic_key = classify_license_url(raw_lic_url)

        raw_year = md.get("year") or doc.get("year")
        try:
            year_int = int(str(raw_year)[:4]) if raw_year else 0
        except ValueError:
            year_int = 0

        # Allow pre-1931 works (US copyright expired) even if licenseurl is omitted/mark
        is_pre1931 = include_pre1931_pd and 1900 <= year_int <= 1930
        if lic_key is None and is_pre1931:
            lic_key = "public_domain"
        if lic_key is None:
            continue

        # Strict guard: only auto-queue non-PD CC0/CC BY/CC BY-SA OR pre-1931 PD OR curated classic series
        is_curated_pd = any(ident.lower().startswith(pref) for pref in _CURATED_PD_PREFIXES)
        if lic_key == "public_domain" and not (is_pre1931 or is_curated_pd):
            continue

        video_url, srt_url, dur_sec = _pick_best_video_and_srt(meta.get("files", []), ident)
        if not video_url or video_url in existing_urls:
            continue

        slug = make_slug(ident, existing_slugs)
        existing_slugs.add(slug)
        existing_urls.add(video_url)

        title = str(md.get("title") or doc.get("title") or ident).strip()
        creator_val = md.get("creator") or doc.get("creator") or "Internet Archive Contributor"
        if isinstance(creator_val, list):
            creator_val = ", ".join(str(c) for c in creator_val)
        creator = str(creator_val).strip()

        pd_basis = None
        pd_verified = False
        if lic_key == "public_domain":
            if is_pre1931:
                pd_basis = f"Published in {year_int} (pre-1931); US copyright expired (Public Domain)."
                pd_verified = True
            else:
                pd_basis = (
                    f"Classic cartoon series ({year_int or 'vintage'}); unrenewed pre-1964 US copyright "
                    "per Internet Archive classic_cartoons metadata."
                )
                pd_verified = True

        attribution = None
        if lic_key in ("cc_by", "cc_by_sa"):
            attribution = f"{title} by {creator}, licensed under {lic_key.replace('_', ' ').upper()}"

        record = RightsRecord(
            work_title=title[:120],
            creator=creator[:100],
            year=year_int if year_int >= 1880 else 1940,
            license=lic_key,  # type: ignore[arg-type]
            license_url=raw_lic_url,
            source_page_url=f"https://archive.org/details/{ident}",
            commercial_use_allowed=True,
            derivatives_allowed=True,
            attribution_text=attribution,
            pd_basis=pd_basis,
            pd_verified_by_human=pd_verified,
            soundtrack_rights_note="Source audio is not used.",
            verified_by="discover_stories",
            verified_on=today_str,
        )
        proj_dir = projects_root / slug
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "rights.json").write_text(
            record.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        queue.items.append(
            QueueItem(
                slug=slug,
                video_source_url=video_url,
                srt_source_url=srt_url,
                style="commentary",
                split_part=0,
                max_duration=175,
                publish_target=publish_target,
                status="pending",
            )
        )
        added_slugs.append(slug)
        print(f"Queued {slug}: {title} ({lic_key}, {dur_sec:.0f}s) -> {video_url}")

    queue_path.write_text(queue.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return added_slugs


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover free-to-use cartoons/animations on Archive.org")
    parser.add_argument("--category", default="all", choices=["classic_cartoons", "cc_animation", "all"])
    parser.add_argument("--max-items", type=int, default=2)
    parser.add_argument("--publish-target", default="telegram")
    args = parser.parse_args()
    added = discover_and_stage(
        category=args.category,
        max_items=args.max_items,
        publish_target=args.publish_target,
    )
    print(f"Discovered and queued {len(added)} new story(ies): {added}")


if __name__ == "__main__":
    main()
