"""Story Discovery, Validation & Top-5 Scoring QA Module.

Finds legally cleared free animations/cartoons, validates them, scores on a
100-point rubric (50 deterministic + 50 LLM), and scaffolds the #1 winner.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

from pipeline import history as hist_mod
from pipeline.models import PublishedHistory, QueueFile, QueueItem

log = logging.getLogger(__name__)

_CATALOG_PATH = Path("assets/catalogs/open_movies.json")
_QUEUE_PATH = Path("queue.json")
_HISTORY_PATH = Path("published_history.json")
_SCOUT_REPORTS_DIR = Path("scout_reports")

_TRADEMARK_TERMS = {
    "superman", "popeye", "betty boop", "bugs bunny", "mickey",
    "disney", "warner", "looney tunes", "batman", "spider-man", "spiderman",
}

_REJECTED_LICENSE_TOKENS = {"nc", "nd", "all_rights_reserved", "unknown"}

_IA_API = "https://archive.org/advancedsearch.php"
_IA_META = "https://archive.org/metadata/{id}"


# ── License helpers ──────────────────────────────────────────────────────────

def classify_license_url(license_url: str) -> str:
    """Classify a license URL into a canonical bucket or 'unknown'."""
    u = license_url.lower()
    if "creativecommons.org/publicdomain/zero" in u or "cc0" in u:
        return "cc0"
    if "creativecommons.org/licenses/by/" in u and "nc" not in u and "nd" not in u:
        return "cc_by"
    if "creativecommons.org/licenses/by-sa" in u:
        return "cc_by_sa"
    if "publicdomain" in u or "public_domain" in u:
        return "public_domain"
    return "unknown"


def _is_license_ok(license_key: str, license_scope: str) -> bool:
    if any(tok in license_key for tok in _REJECTED_LICENSE_TOKENS):
        return False
    if license_scope == "cc_only":
        return license_key in {"cc0", "cc_by"}
    if license_scope == "cc_and_pd":
        return license_key in {"cc0", "cc_by", "cc_by_sa", "public_domain"}
    return False


def _has_trademark(title: str, description: str = "") -> bool:
    text = (title + " " + description).lower()
    return any(tm in text for tm in _TRADEMARK_TERMS)


# ── Slug helpers ─────────────────────────────────────────────────────────────

def make_slug(title: str, year: int | None = None) -> str:
    """Generate a valid pipeline slug from a title + optional year."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower())
    slug = slug.strip("_")[:30]
    if year:
        slug = f"{slug}_{str(year)[-2:]}"
    slug = slug[:40]
    if len(slug) < 3:
        slug = (slug + "___")[:3]
    return slug


# ── Internet Archive search ───────────────────────────────────────────────────

def _ia_search(query: str, rows: int = 20) -> list[dict[str, Any]]:
    """Search Archive.org for open-license animation/movie items."""
    params = {
        "q": f"({query}) AND mediatype:movies AND (subject:animation OR subject:cartoon OR collection:animationandcartoons OR collection:classic_cartoons) AND (licenseurl:creativecommons OR licenseurl:publicdomain OR licenseurl:creativecommons.org/publicdomain)",
        "fl[]": "identifier,title,description,year,licenseurl,subject,downloads",
        "sort[]": "downloads desc",
        "rows": rows,
        "page": 1,
        "output": "json",
    }
    try:
        resp = requests.get(_IA_API, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json().get("response", {}).get("docs", [])
    except Exception as exc:
        log.warning("IA search error: %s", exc)
        return []


def _ia_metadata(item_id: str) -> dict[str, Any]:
    try:
        resp = requests.get(_IA_META.format(id=item_id), timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("IA metadata error for %s: %s", item_id, exc)
        return {}


def _pick_best_video_and_srt(files: list[dict[str, Any]], item_id: str) -> tuple[str, str]:
    """Pick the best MP4/MKV and optional SRT from an IA file list."""
    video_url = ""
    srt_url = ""
    base = f"https://archive.org/download/{item_id}/"

    # Prefer 720p+ MP4, then any MP4, then MKV
    mp4s = [f for f in files if str(f.get("name", "")).lower().endswith(".mp4")]
    mkv = [f for f in files if str(f.get("name", "")).lower().endswith(".mkv")]
    srts = [f for f in files if str(f.get("name", "")).lower().endswith((".srt", ".vtt"))]

    video_pool = mp4s or mkv
    # Pick largest (best quality proxy)
    if video_pool:
        best = max(video_pool, key=lambda f: int(f.get("size", 0)))
        video_url = base + best["name"]

    en_srts = [f for f in srts if "en" in str(f.get("name", "")).lower()]
    if en_srts:
        srt_url = base + en_srts[0]["name"]
    elif srts:
        srt_url = base + srts[0]["name"]

    return video_url, srt_url


# ── HTTP HEAD check ───────────────────────────────────────────────────────────

def _head_check(url: str, min_mb: float = 10.0, max_gb: float = 3.5) -> tuple[bool, int]:
    """Return (ok, size_bytes). ok=False if unreachable or out of size range."""
    try:
        r = requests.head(url, timeout=15, allow_redirects=True)
        if r.status_code not in range(200, 400):
            return False, 0
        size = int(r.headers.get("content-length", 0))
        min_b = int(min_mb * 1024 * 1024)
        max_b = int(max_gb * 1024 * 1024 * 1024)
        if size > 0 and not (min_b <= size <= max_b):
            return False, size
        return True, size
    except Exception:
        return False, 0


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_technical(
    license_key: str,
    has_trademark: bool,
    resolution_height: int,
    has_srt: bool,
    runtime_min: float,
    downloads: int,
) -> int:
    """Deterministic tech+rights score (0–50)."""
    # License cleanness (0-15)
    if license_key in {"cc0", "cc_by"} and not has_trademark:
        lic_score = 15
    elif license_key == "public_domain" and not has_trademark:
        lic_score = 10
    elif license_key in {"cc0", "cc_by"} and has_trademark:
        lic_score = 8
    elif license_key == "public_domain" and has_trademark:
        lic_score = 4
    else:
        lic_score = 0

    # Visual quality (0-12)
    if resolution_height >= 1080:
        qual_score = 12
    elif resolution_height >= 720:
        qual_score = 9
    elif resolution_height >= 480:
        qual_score = 5
    else:
        qual_score = 2

    # Subtitle (0-10)
    sub_score = 10 if has_srt else 4

    # Runtime sweet spot (0-8): ideal 5-18 min
    if 5 <= runtime_min <= 18:
        rt_score = 8
    elif 3 <= runtime_min < 5 or 18 < runtime_min <= 30:
        rt_score = 5
    else:
        rt_score = 2

    # Popularity (0-5): normalize downloads
    pop_score = min(5, int(downloads / 2000))

    return lic_score + qual_score + sub_score + rt_score + pop_score


def _mock_llm_story_scores(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic mock LLM scores for CI (zero network)."""
    results = []
    for i, c in enumerate(candidates):
        # Assign deterministic scores based on index (highest to first)
        hook = max(5, 15 - i * 2)
        visual = max(5, 15 - i * 2)
        twist = max(2, 10 - i)
        commentary = max(2, 10 - i)
        results.append({
            "hook_strength": hook,
            "visual_storytelling": visual,
            "twist_or_payoff": twist,
            "commentary_potential": commentary,
            "story_score": hook + visual + twist + commentary,
            "pitch_angle": f"Mock pitch for {c.get('title', 'unknown')}",
            "verdict": "mock",
        })
    return results


def _llm_story_scores(candidates: list[dict[str, Any]], gemini_api_key: str) -> list[dict[str, Any]]:
    """Call Gemini to score candidates' story/hook potential (batched)."""
    try:
        import google.generativeai as genai  # type: ignore[import]
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt_path = Path("prompts/story_scout_scorer.txt")
        if prompt_path.exists():
            system_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            system_prompt = (
                "You are a short-form video content strategist. Score each candidate 0-15 on "
                "hook_strength, visual_storytelling, 0-10 on twist_or_payoff and commentary_potential. "
                "Return ONLY valid JSON: {\"scores\": [{\"hook_strength\":N,\"visual_storytelling\":N,"
                "\"twist_or_payoff\":N,\"commentary_potential\":N,\"pitch_angle\":\"...\",\"verdict\":\"...\"}]}"
            )
        payload = json.dumps(
            [{"title": c.get("title"), "year": c.get("year"), "runtime_min": c.get("runtime_min"),
              "description": str(c.get("description", ""))[:300], "subjects": c.get("subjects", [])}
             for c in candidates],
            indent=2,
        )
        resp = model.generate_content(
            f"{system_prompt}\n\nCANDIDATES:\n{payload}",
            generation_config={"temperature": 0.3},
        )
        raw = resp.text.strip().lstrip("```json").rstrip("```").strip()
        data = json.loads(raw)
        scores = data.get("scores", [])
        result = []
        for s in scores:
            s["story_score"] = (
                s.get("hook_strength", 0) + s.get("visual_storytelling", 0)
                + s.get("twist_or_payoff", 0) + s.get("commentary_potential", 0)
            )
            result.append(s)
        return result
    except Exception as exc:
        log.warning("LLM story scoring failed: %s, using mock", exc)
        return _mock_llm_story_scores(candidates)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_candidate(
    item: dict[str, Any],
    history: PublishedHistory,
    license_scope: str,
    min_runtime_min: float = 3.0,
    max_runtime_min: float = 35.0,
    mock_http: bool = False,
) -> tuple[bool, str]:
    """Return (ok, rejection_reason). ok=True means candidate passes all hard filters."""
    slug = make_slug(item.get("title", ""), item.get("year"))
    # Already tracked
    existing = hist_mod.is_duplicate(
        history,
        slug=slug,
        source_url=item.get("video_url", ""),
        work_title=item.get("title", ""),
    )
    if existing is not None:
        return False, f"already tracked: {existing.slug}"

    # Already in projects/ dir
    if (Path("projects") / slug).exists():
        return False, f"projects/{slug} already exists"

    # License check
    license_key = item.get("license_key", "unknown")
    if not _is_license_ok(license_key, license_scope):
        return False, f"license '{license_key}' rejected for scope '{license_scope}'"

    # Runtime
    runtime_min = float(item.get("runtime_min", 0))
    if not (min_runtime_min <= runtime_min <= max_runtime_min):
        return False, f"runtime {runtime_min:.1f}min outside [{min_runtime_min}, {max_runtime_min}]"

    # HTTP HEAD check (skipped in mock mode)
    video_url = item.get("video_url", "")
    if not video_url:
        return False, "no video_url"
    if not mock_http:
        ok, _size = _head_check(video_url)
        if not ok:
            return False, f"HEAD check failed or size out of range for {video_url}"

    return True, ""


# ── Main scout function ───────────────────────────────────────────────────────

def run_scout(
    query: str = "",
    candidate_count: int = 5,
    license_scope: str = "cc_only",
    mock_llm: bool = False,
    mock_http: bool = False,
    gemini_api_key: str = "",
    auto_produce: bool = False,
) -> dict[str, Any]:
    """Run the full story discovery + scoring pipeline. Returns the scout report dict."""
    history = hist_mod.load(_HISTORY_PATH)
    queue_path = _QUEUE_PATH

    candidates_raw: list[dict[str, Any]] = []

    # Source 1: curated catalog
    if _CATALOG_PATH.exists():
        catalog = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
        for item in catalog.get("items", []):
            license_url = item.get("license_url", "")
            license_key = classify_license_url(license_url) if license_url else item.get("license", "unknown")
            candidates_raw.append({
                "title": item.get("title", ""),
                "year": item.get("year"),
                "description": item.get("description", ""),
                "subjects": item.get("subjects", []),
                "license_key": license_key,
                "license_url": license_url,
                "video_url": item.get("video_url", ""),
                "srt_url": item.get("srt_url", ""),
                "runtime_min": float(item.get("runtime_min", 0)),
                "resolution_height": int(item.get("resolution_height", 0)),
                "downloads": int(item.get("downloads", 0)),
                "source": "catalog",
                "source_id": item.get("source_id", ""),
                "has_trademark": _has_trademark(item.get("title", ""), item.get("description", "")),
            })

    # Source 2: Internet Archive (live only)
    if not mock_http:
        ia_query = query or "animation cartoon open license"
        ia_results = _ia_search(ia_query, rows=30)
        for doc in ia_results:
            item_id = doc.get("identifier", "")
            if not item_id:
                continue
            meta = _ia_metadata(item_id)
            files = meta.get("files", [])
            video_url, srt_url = _pick_best_video_and_srt(files, item_id)
            if not video_url:
                continue
            license_url = str(doc.get("licenseurl", ""))
            license_key = classify_license_url(license_url)
            # Estimate runtime from metadata
            runtime_min = 0.0
            for f in files:
                if str(f.get("name", "")).lower().endswith(".mp4"):
                    try:
                        runtime_min = float(f.get("length", 0)) / 60.0
                        break
                    except (TypeError, ValueError):
                        pass
            subjects = doc.get("subject", [])
            if isinstance(subjects, str):
                subjects = [subjects]
            title = str(doc.get("title", item_id))
            year = doc.get("year")
            candidates_raw.append({
                "title": title,
                "year": year,
                "description": str(doc.get("description", ""))[:500],
                "subjects": subjects,
                "license_key": license_key,
                "license_url": license_url,
                "video_url": video_url,
                "srt_url": srt_url,
                "runtime_min": runtime_min,
                "resolution_height": 480,  # unknown without probing
                "downloads": int(doc.get("downloads", 0)),
                "source": "internet_archive",
                "source_id": item_id,
                "has_trademark": _has_trademark(title, str(doc.get("description", ""))),
            })

    # Validate and collect up to candidate_count
    validated: list[dict[str, Any]] = []
    for item in candidates_raw:
        if len(validated) >= candidate_count:
            break
        ok, reason = validate_candidate(item, history, license_scope, mock_http=mock_http)
        if ok:
            item["trademark_caution"] = item.get("has_trademark", False)
            validated.append(item)
        else:
            log.info("Rejected '%s': %s", item.get("title"), reason)

    if not validated:
        return {"error": "No valid candidates found", "candidates": [], "winner": None}

    # Technical scores
    for item in validated:
        item["tech_score"] = score_technical(
            license_key=item["license_key"],
            has_trademark=item["trademark_caution"],
            resolution_height=item["resolution_height"],
            has_srt=bool(item.get("srt_url")),
            runtime_min=item["runtime_min"],
            downloads=item["downloads"],
        )

    # LLM story scores
    if mock_llm:
        llm_scores = _mock_llm_story_scores(validated)
    else:
        llm_scores = _llm_story_scores(validated, gemini_api_key)

    for i, item in enumerate(validated):
        ls = llm_scores[i] if i < len(llm_scores) else {"story_score": 0}
        item["story_score"] = int(ls.get("story_score", 0))
        item["total_score"] = item["tech_score"] + item["story_score"]
        item["pitch_angle"] = ls.get("pitch_angle", "")
        item["verdict"] = ls.get("verdict", "")
        item["llm_detail"] = {k: v for k, v in ls.items() if k not in {"story_score", "pitch_angle", "verdict"}}

    # Rank
    validated.sort(key=lambda x: x["total_score"], reverse=True)
    for rank, item in enumerate(validated, start=1):
        item["rank"] = rank

    winner = validated[0]

    # Scaffold winner
    winner_slug = make_slug(winner["title"], winner.get("year"))
    winner_project_dir = Path("projects") / winner_slug
    winner_project_dir.mkdir(parents=True, exist_ok=True)

    # rights.json
    is_pd = winner["license_key"] == "public_domain"
    rights = {
        "work_title": winner["title"],
        "creator": winner.get("creator", "Unknown"),
        "year": winner.get("year") or 0,
        "license": winner["license_key"] if winner["license_key"] in {"cc0", "cc_by", "cc_by_sa", "public_domain", "owned", "licensed"} else "public_domain",
        "license_url": winner.get("license_url"),
        "source_page_url": winner.get("video_url", ""),
        "commercial_use_allowed": not winner["trademark_caution"],
        "derivatives_allowed": True,
        "attribution_text": f"{winner['title']}{' (' + str(winner.get('year')) + ')' if winner.get('year') else ''}. License: {winner['license_key']}.",
        "pd_basis": "Public domain via curated catalog." if is_pd else None,
        "pd_verified_by_human": False,
        "trademark_caution": winner["trademark_caution"],
        "verified_by": "story_scout",
        "verified_on": dt.date.today().isoformat(),
    }
    (winner_project_dir / "rights.json").write_text(json.dumps(rights, indent=2) + "\n", encoding="utf-8")

    # rights_evidence.json
    evidence = {"source": winner.get("source"), "source_id": winner.get("source_id"), "license_url": winner.get("license_url"), "video_url": winner.get("video_url"), "srt_url": winner.get("srt_url"), "scout_verified": True}
    (winner_project_dir / "rights_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    # Append to queue.json
    q_status = "awaiting_pd_verification" if is_pd and not auto_produce else "pending"
    if queue_path.exists():
        try:
            q_data = QueueFile(**json.loads(queue_path.read_text(encoding="utf-8")))
        except Exception:
            q_data = QueueFile()
    else:
        q_data = QueueFile()

    already_queued = any(qi.slug == winner_slug for qi in q_data.items)
    if not already_queued:
        q_data.items.append(
            QueueItem(
                slug=winner_slug,
                video_source_url=winner["video_url"],
                srt_source_url=winner.get("srt_url") or None,
                style="commentary",
                split_part=0,
                max_duration=175,
                publish_target="artifact_only",
                status="pending" if (not is_pd or auto_produce) else "pending",
            )
        )
        queue_path.write_text(q_data.model_dump_json(indent=2) + "\n", encoding="utf-8")

    # Save scout report
    _SCOUT_REPORTS_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = _SCOUT_REPORTS_DIR / f"scout_{ts}.json"
    report = {
        "run_at": ts,
        "query": query,
        "license_scope": license_scope,
        "candidate_count": len(validated),
        "winner_slug": winner_slug,
        "candidates": [
            {k: v for k, v in c.items() if k != "llm_detail"}
            for c in validated
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    log.info("Scout report saved: %s", report_path)

    return {
        "winner_slug": winner_slug,
        "winner": winner,
        "candidates": validated,
        "report_path": str(report_path),
        "queue_status": q_status,
        "is_pd": is_pd,
    }
