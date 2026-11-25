"""Published story history ledger — prevents re-publishing already-released videos."""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from pathlib import Path

from pipeline.models import HistoryRecord, PublishedHistory

log = logging.getLogger(__name__)

_HISTORY_PATH = Path("published_history.json")
_EXIT_CODE_DUPLICATE = 10


def _normalize(title: str) -> str:
    """Lowercase, strip punctuation and trailing years for fuzzy matching."""
    t = title.lower()
    t = re.sub(r"[\W_]+", " ", t)
    t = re.sub(r"\b(19|20)\d{2}\b", "", t)
    return t.strip()


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load(path: Path = _HISTORY_PATH) -> PublishedHistory:
    if path.exists():
        try:
            return PublishedHistory(**json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            log.warning("history.load: parse error %s, starting fresh", exc)
    return PublishedHistory()


def save(history: PublishedHistory, path: Path = _HISTORY_PATH) -> None:
    path.write_text(
        json.dumps(history.model_dump(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def is_duplicate(
    history: PublishedHistory,
    *,
    slug: str = "",
    source_url: str = "",
    source_sha256: str = "",
    work_title: str = "",
) -> HistoryRecord | None:
    """Return the matching HistoryRecord if already tracked, else None.

    Matches on any of: slug, source_url, source_sha256, or normalized_title.
    """
    norm = _normalize(work_title) if work_title else ""
    for rec in history.records:
        if slug and rec.slug == slug:
            return rec
        if source_url and rec.source_url and rec.source_url == source_url:
            return rec
        if source_sha256 and rec.source_sha256 and rec.source_sha256 == source_sha256:
            return rec
        if norm and rec.normalized_title and rec.normalized_title == norm:
            return rec
    return None


def is_published_anywhere(rec: HistoryRecord) -> bool:
    return any(p.published for p in rec.platforms.values())


def upsert_rendered(
    history: PublishedHistory,
    *,
    slug: str,
    work_title: str,
    source_url: str = "",
    source_id: str = "",
    source_sha256: str = "",
) -> HistoryRecord:
    """Upsert a record as 'rendered', preserving existing platform publish flags."""
    now = _now_iso()
    norm = _normalize(work_title)
    existing = is_duplicate(history, slug=slug)
    if existing is not None:
        existing.updated_at = now
        if existing.status == "rendered":
            # keep status, just update metadata
            existing.work_title = work_title
            existing.source_url = source_url or existing.source_url
            existing.source_sha256 = source_sha256 or existing.source_sha256
        return existing
    rec = HistoryRecord(
        slug=slug,
        work_title=work_title,
        normalized_title=norm,
        source_id=source_id,
        source_url=source_url,
        source_sha256=source_sha256,
        status="rendered",
        created_at=now,
        updated_at=now,
    )
    history.records.append(rec)
    return rec


def mark_published(
    history: PublishedHistory,
    *,
    slug: str,
    platform: str,
    url: str = "",
    privacy: str = "",
) -> HistoryRecord:
    """Mark a slug as published on 'platform'. Creates a stub record if missing."""
    now = _now_iso()
    rec = is_duplicate(history, slug=slug)
    if rec is None:
        rec = HistoryRecord(
            slug=slug,
            work_title=slug,
            normalized_title=_normalize(slug),
            created_at=now,
            updated_at=now,
        )
        history.records.append(rec)
    ps = rec.platforms.setdefault(platform, type(list(rec.platforms.values())[0])())  # PlatformStatus
    ps.published = True
    ps.url = url or ps.url
    ps.published_at = now
    if privacy:
        ps.privacy = privacy
    # Update aggregate status
    pub_platforms = [p for p, v in rec.platforms.items() if v.published]
    if len(pub_platforms) >= 2:
        rec.status = "published_all"
    elif len(pub_platforms) == 1:
        rec.status = "published_partial"
    rec.updated_at = now
    return rec


def check_no_duplicate_or_exit(
    slug: str,
    work_title: str,
    source_url: str = "",
    source_sha256: str = "",
    *,
    allow_reproduce: bool = False,
    path: Path = _HISTORY_PATH,
) -> None:
    """Exit with code 10 if this story was already published (and --allow-reproduce not set)."""
    if allow_reproduce:
        return
    history = load(path)
    rec = is_duplicate(
        history,
        slug=slug,
        source_url=source_url,
        source_sha256=source_sha256,
        work_title=work_title,
    )
    if rec is not None and is_published_anywhere(rec):
        pub_urls = [
            f"{p}: {v.url}" for p, v in rec.platforms.items() if v.published and v.url
        ]
        msg = (
            f"Story '{rec.slug}' ('{rec.work_title}') was already published "
            f"[{', '.join(pub_urls) or 'no URL recorded'}]. "
            "Use --allow-reproduce to override."
        )
        log.error(msg)
        import sys
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(_EXIT_CODE_DUPLICATE)
