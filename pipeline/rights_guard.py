"""Rights Guard – validates source-material licensing before pipeline runs.

Loads ``projects/<slug>/rights.json``, checks the licence against an
allow-list, captures live evidence from the source page, and builds the
attribution block embedded into the final video description.

Exit code **3** signals a rights failure so CI can distinguish it from
generic crashes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from pipeline.models import RightsRecord

__all__ = [
    "validate_rights",
    "check_rights",
    "capture_evidence",
    "build_attribution_block",
]

log = logging.getLogger(__name__)

# ── licence allow-list ──────────────────────────────────────────────────
_ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        "public_domain",
        "cc0",
        "cc_by",
        "cc_by_sa",
        "owned",
        "licensed",
    }
)

_REJECTED_PREFIXES: tuple[str, ...] = (
    "cc_by_nc",
    "cc_by_nd",
)

_REJECTED_EXACT: frozenset[str] = frozenset(
    {
        "all_rights_reserved",
        "unknown",
    }
)

_EXIT_CODE_RIGHTS = 3


# ── helpers ─────────────────────────────────────────────────────────────


def _fail(msg: str) -> None:
    """Log an error and exit with code 3."""
    log.error("Rights Guard FAIL: %s", msg)
    sys.exit(_EXIT_CODE_RIGHTS)


def _is_http(url: str) -> bool:
    return url.lower().startswith(("http://", "https://"))


def _is_archive_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname is not None and parsed.hostname.endswith("archive.org")


def _archive_item_id(url: str) -> str | None:
    """Extract the Internet Archive item identifier from a URL.

    Handles URLs like ``https://archive.org/details/<id>`` and
    ``https://archive.org/download/<id>/...``.
    """
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in {"details", "download", "embed", "stream"}:
        return parts[1]
    return None


# ── core validation ─────────────────────────────────────────────────────


def validate_rights(
    slug: str,
    project_dir: Path,
    *,
    strict: bool = True,
) -> RightsRecord:
    """Load and validate ``rights.json`` for *slug*.

    Args:
        slug: Project slug (used only for log messages).
        project_dir: Path to ``projects/<slug>/`` directory.
        strict: When *True* (default), exit on any validation failure.

    Returns:
        A validated :class:`RightsRecord`.
    """
    rights_path = project_dir / "rights.json"

    # 1. File must exist
    if not rights_path.exists():
        _fail(f"[{slug}] rights.json not found at {rights_path}")

    # 2. Parse JSON → Pydantic model
    try:
        raw = json.loads(rights_path.read_text(encoding="utf-8"))
        record = RightsRecord(**raw)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        _fail(f"[{slug}] rights.json is invalid: {exc}")
        raise  # unreachable – keeps type-checkers happy

    # 3. Reject disallowed licences
    lic = record.license
    if lic in _REJECTED_EXACT or any(lic.startswith(p) for p in _REJECTED_PREFIXES):
        _fail(f"[{slug}] Licence '{lic}' is not permitted for this pipeline.")

    if lic not in _ALLOWED_LICENSES:
        _fail(f"[{slug}] Licence '{lic}' is not on the allow-list.")

    # 4. Global flags
    if not record.commercial_use_allowed:
        _fail(f"[{slug}] commercial_use_allowed must be True.")
    if not record.derivatives_allowed:
        _fail(f"[{slug}] derivatives_allowed must be True.")

    # 5. Per-licence extra checks
    if lic == "public_domain":
        if not record.pd_basis:
            _fail(f"[{slug}] public_domain requires a non-empty pd_basis.")
        if not record.pd_verified_by_human:
            _fail(f"[{slug}] public_domain requires pd_verified_by_human == True.")

    elif lic in {"cc_by", "cc_by_sa"}:
        if not record.attribution_text:
            _fail(f"[{slug}] {lic} requires non-empty attribution_text.")
        if not record.license_url:
            _fail(f"[{slug}] {lic} requires a non-empty license_url.")
        if lic == "cc_by_sa":
            log.warning(
                "[%s] ShareAlike (cc_by_sa): derivative must be released "
                "under the same or a compatible licence.",
                slug,
            )

    elif lic == "licensed":
        if not record.license_document_url:
            _fail(f"[{slug}] 'licensed' requires a license_document_url.")
        if not record.commercial_use_allowed:
            _fail(f"[{slug}] 'licensed' requires commercial_use_allowed == True.")
        if not record.derivatives_allowed:
            _fail(f"[{slug}] 'licensed' requires derivatives_allowed == True.")

    elif lic == "owned":
        if not record.owner_statement:
            _fail(f"[{slug}] 'owned' requires a non-empty owner_statement.")

    # cc0 has no additional constraints beyond the global flags.

    log.info("[%s] Rights validated: licence=%s", slug, lic)
    return record


# ── evidence capture ────────────────────────────────────────────────────


def capture_evidence(
    record: RightsRecord,
    project_dir: Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Fetch the source page and persist a hashed evidence snapshot.

    For Internet Archive URLs the IA metadata API is also queried and the
    ``licenseurl`` field is compared against the declared licence URL.

    Returns the evidence dict (also written to ``rights_evidence.json``).
    """
    evidence: dict[str, Any] = {
        "source_page_url": record.source_page_url,
        "captured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }

    url = record.source_page_url

    # Skip synthetic / non-HTTP URLs
    if not _is_http(url):
        evidence["skipped"] = True
        evidence["reason"] = "non-HTTP source_page_url"
        _write_evidence(evidence, project_dir)
        return evidence

    # Internet Archive metadata cross-check
    if _is_archive_url(url):
        item_id = _archive_item_id(url)
        if item_id:
            evidence["ia_item_id"] = item_id
            try:
                meta_resp = requests.get(
                    f"https://archive.org/metadata/{item_id}", timeout=30
                )
                if meta_resp.status_code == 200:
                    meta = meta_resp.json()
                    ia_license_url: str = (
                        meta.get("metadata", {}).get("licenseurl", "") or ""
                    )
                    evidence["ia_licenseurl"] = ia_license_url
                    declared = record.license_url or ""
                    if ia_license_url and declared:
                        match = (
                            ia_license_url.rstrip("/").lower()
                            == declared.rstrip("/").lower()
                        )
                        evidence["ia_license_match"] = match
                        if not match:
                            msg = (
                                f"IA metadata licenseurl ({ia_license_url}) does not match "
                                f"declared license_url ({declared})."
                            )
                            if strict:
                                _write_evidence(evidence, project_dir)
                                _fail(msg)
                            log.warning(msg)
                else:
                    evidence["ia_meta_status"] = meta_resp.status_code
            except requests.RequestException as exc:
                evidence["ia_meta_error"] = str(exc)
                log.warning("IA metadata fetch failed for %s: %s", item_id, exc)

    # Fetch the source page
    try:
        resp = requests.get(url, timeout=30)
        evidence["http_status"] = resp.status_code
        body_text = resp.text if isinstance(resp.text, str) else ""
        evidence["body_sha256"] = hashlib.sha256(
            body_text.encode("utf-8")
        ).hexdigest()
        evidence["body_length"] = len(body_text)
    except requests.RequestException as exc:
        evidence["fetch_error"] = str(exc)
        log.warning("Evidence fetch failed for %s: %s", url, exc)

    _write_evidence(evidence, project_dir)
    return evidence


def check_rights(rights_path: Path, *, strict: bool = True) -> bool:
    """Validate a ``rights.json`` file path and optionally capture IA evidence."""
    record = validate_rights(rights_path.parent.name, rights_path.parent, strict=strict)
    if _is_http(record.source_page_url):
        capture_evidence(record, rights_path.parent, strict=strict)
    return True


def _write_evidence(evidence: dict[str, Any], project_dir: Path) -> None:
    out_path = project_dir / "rights_evidence.json"
    out_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Evidence written to %s", out_path)


# ── attribution block builder ───────────────────────────────────────────


def build_attribution_block(
    record: RightsRecord | dict[str, Any],
    style: str = "commentary",
    part_number: int = 0,
) -> str:
    """Build the attribution string for the video description."""
    if isinstance(record, dict):
        record = RightsRecord(**record)
    license_label = record.license.replace("_", " ").upper()
    license_ref = f" ({record.license_url})" if record.license_url else ""

    lines: list[str] = [
        (
            f'Source: "{record.work_title}" by {record.creator} '
            f"({record.year}), {license_label}{license_ref}. "
            f"Original: {record.source_page_url}."
        ),
        (
            "Narration, selection, editing, and commentary created for "
            "this channel. Narration generated with text-to-speech."
        ),
    ]

    if record.license == "cc_by_sa":
        lines.append(
            "This derivative work is released under the same ShareAlike "
            "licence as the original."
        )

    return " ".join(lines)
