"""Zero-local delivery publisher (Artifact, Telegram, R2, YouTube) with approval gate."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import boto3
import cv2
import requests

from pipeline import ffutil
from pipeline.models import ApprovalRecord, ProjectManifest, QueueFile, RightsRecord, Shot

log = logging.getLogger(__name__)

_EXIT_CODE_ALL_PUBLISH_FAILED = 9


def extract_thumbnail(source_path: Path, shots: list[Shot], out_jpg: Path) -> Path:
    """Save the highest-scoring Beat 1 frame as ``thumbnail.jpg``."""
    b1_shots = [s for s in shots if s.beat_id == 1]
    best = max(
        b1_shots or shots,
        key=lambda s: s.match_score if s.match_score is not None else 0.0,
    )
    mid_ms = ((best.src_start_sec + best.src_end_sec) * 0.5) * 1000.0
    cap = cv2.VideoCapture(str(source_path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_ms)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(out_jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return out_jpg
    finally:
        cap.release()
    out_jpg.write_bytes(b"")
    return out_jpg


def count_weekly_public_uploads(projects_root: Path, queue_path: Path) -> int:
    """Count public YouTube uploads completed in the last 7 days."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)
    count = 0

    if projects_root.exists():
        for report_file in projects_root.glob("*/run_report.json"):
            try:
                data = json.loads(report_file.read_text(encoding="utf-8"))
                yt = data.get("publish_results", {}).get("youtube", {})
                if yt.get("privacy") == "public" and yt.get("uploaded_at"):
                    ts = dt.datetime.fromisoformat(yt["uploaded_at"])
                    if ts >= cutoff:
                        count += 1
            except Exception:
                continue

    if queue_path.exists():
        try:
            q = QueueFile(**json.loads(queue_path.read_text(encoding="utf-8")))
            for item in q.items:
                if item.status == "done" and item.publish_target == "youtube_public" and item.completed_on:
                    ts = dt.datetime.fromisoformat(item.completed_on)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=dt.timezone.utc)
                    if ts >= cutoff:
                        count += 1
        except Exception:
            pass
    return count


def check_public_approval(
    project_dir: Path,
    config: dict[str, Any],
    *,
    projects_root: Path = Path("projects"),
    queue_path: Path = Path("queue.json"),
) -> tuple[bool, str]:
    """Return ``(allowed, reason)`` for public YouTube publishing."""
    approval_path = project_dir / "approval.json"
    if not approval_path.exists():
        return False, "approval.json missing"
    approval = ApprovalRecord(**json.loads(approval_path.read_text(encoding="utf-8")))
    if not approval.approved:
        return False, "approval.json has approved=false"

    max_per_week = int(config.get("publishing", {}).get("max_public_uploads_per_week", 3))
    recent = count_weekly_public_uploads(projects_root, queue_path)
    if recent >= max_per_week:
        return False, f"weekly public upload cap reached ({recent}/{max_per_week})"
    return True, "approved"


def build_metadata(
    manifest: ProjectManifest,
    rights: RightsRecord,
    out_dir: Path,
) -> dict[str, Any]:
    """Build ``metadata.json`` with mandatory attribution and synthetic media disclosure."""
    full_desc = (
        f"{manifest.description.strip()}\n\n"
        f"{manifest.attribution_block.strip()}\n"
        "Narration generated with text-to-speech."
    )
    meta = {
        "title": manifest.title,
        "description": full_desc,
        "hashtags": manifest.hashtags,
        "source_title": rights.work_title,
        "license": rights.license,
        "attribution_block": manifest.attribution_block,
        "contains_synthetic_media": True,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _publish_telegram(
    final_mp4: Path,
    thumb_jpg: Path,
    meta: dict[str, Any],
    config: dict[str, Any],
    r2_url: str | None = None,
) -> dict[str, Any]:
    """Send thumbnail + caption and video (if <= max_video_mb) to Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return {"status": "skipped", "reason": "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing"}

    max_mb = float(config.get("publishing", {}).get("telegram", {}).get("max_video_mb", 50))
    base = f"https://api.telegram.org/bot{token}"
    caption = f"{meta['title']}\n\n{meta['description'][:700]}"
    if r2_url:
        caption += f"\n\nDownload: {r2_url}"

    if thumb_jpg.exists() and thumb_jpg.stat().st_size > 0:
        with thumb_jpg.open("rb") as fh:
            r = requests.post(
                f"{base}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption[:1024]},
                files={"photo": fh},
                timeout=60,
            )
            r.raise_for_status()

    size_mb = final_mp4.stat().st_size / (1024 * 1024)
    upload_mp4 = final_mp4
    if size_mb > max_mb:
        # Re-encode a Telegram-sized copy targeting ~90% of max_mb so sendVideo succeeds
        dur = max(1.0, ffutil.duration(final_mp4))
        target_total_kbps = int(((max_mb * 0.88) * 8 * 1024) / dur)
        audio_kbps = 128
        video_kbps = max(400, target_total_kbps - audio_kbps)
        tg_copy = final_mp4.parent.parent / "telegram_upload.mp4"
        ffutil.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(final_mp4),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-b:v",
                f"{video_kbps}k",
                "-maxrate",
                f"{int(video_kbps * 1.15)}k",
                "-bufsize",
                f"{video_kbps * 2}k",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                f"{audio_kbps}k",
                "-movflags",
                "+faststart",
                str(tg_copy),
            ]
        )
        if tg_copy.exists() and tg_copy.stat().st_size / (1024 * 1024) <= max_mb:
            upload_mp4 = tg_copy
            size_mb = upload_mp4.stat().st_size / (1024 * 1024)

    if size_mb <= max_mb:
        with upload_mp4.open("rb") as fh:
            r = requests.post(
                f"{base}/sendVideo",
                data={"chat_id": chat_id, "caption": meta["title"][:1024]},
                files={"video": fh},
                timeout=180,
            )
            r.raise_for_status()
        return {"status": "sent", "video_mb": round(size_mb, 2)}

    requests.post(
        f"{base}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": f"Video is {size_mb:.1f} MB (> {max_mb} MB limit). Check GitHub Artifact or R2 link.",
        },
        timeout=30,
    ).raise_for_status()
    return {"status": "notice_sent", "video_mb": round(size_mb, 2)}


def _publish_r2(
    final_mp4: Path,
    slug: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Upload ``final_short.mp4`` to Cloudflare R2 / S3 and return presigned URL."""
    endpoint = os.environ.get("R2_ENDPOINT_URL", "").strip()
    key_id = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    bucket = os.environ.get("R2_OUTPUT_BUCKET", "").strip()
    if not all([endpoint, key_id, secret, bucket]):
        return {"status": "skipped", "reason": "R2 credentials missing"}

    days = int(config.get("publishing", {}).get("r2", {}).get("presign_days", 7))
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
    )
    object_key = f"recaps/{slug}/{final_mp4.name}"
    client.upload_file(str(final_mp4), bucket, object_key)
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=days * 86400,
    )
    return {"status": "uploaded", "url": url, "expires_days": days}


def _publish_youtube(
    final_mp4: Path,
    thumb_jpg: Path,
    meta: dict[str, Any],
    privacy: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Upload video to YouTube Data API v3 using refresh-token OAuth."""
    cid = os.environ.get("YT_CLIENT_ID", "").strip()
    csec = os.environ.get("YT_CLIENT_SECRET", "").strip()
    rtok = os.environ.get("YT_REFRESH_TOKEN", "").strip()
    if not all([cid, csec, rtok]):
        return {"status": "skipped", "reason": "YouTube OAuth secrets missing"}

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(
        token=None,
        refresh_token=rtok,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=cid,
        client_secret=csec,
    )
    yt = build("youtube", "v3", credentials=creds)
    yt_cfg = config.get("publishing", {}).get("youtube", {})
    body = {
        "snippet": {
            "title": meta["title"][:100],
            "description": meta["description"][:5000],
            "tags": [t.lstrip("#") for t in meta["hashtags"]],
            "categoryId": str(yt_cfg.get("category_id", "1")),
        },
        "status": {
            "privacyStatus": privacy,
            "containsSyntheticMedia": True,
        },
    }
    media = MediaFileUpload(str(final_mp4), mimetype="video/mp4", resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = req.execute()
    vid_id = resp.get("id")
    actual_privacy = resp.get("status", {}).get("privacyStatus", privacy)
    if privacy == "public" and actual_privacy != "public":
        log.warning(
            "YouTube forced upload %s to '%s' (unverified API project restriction).",
            vid_id,
            actual_privacy,
        )

    if vid_id and thumb_jpg.exists() and thumb_jpg.stat().st_size > 0:
        try:
            yt.thumbnails().set(
                videoId=vid_id,
                media_body=MediaFileUpload(str(thumb_jpg), mimetype="image/jpeg"),
            ).execute()
        except Exception as exc:
            log.warning("Thumbnail upload failed: %s", exc)

    return {
        "status": "uploaded",
        "video_id": vid_id,
        "privacy": actual_privacy,
        "requested_privacy": privacy,
        "uploaded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def publish_outputs(
    slug: str,
    publish_target: str,
    final_mp4: Path,
    source_path: Path,
    manifest: ProjectManifest,
    rights: RightsRecord,
    shots: list[Shot],
    project_dir: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate thumbnail/metadata and deliver to requested targets."""
    thumb_jpg = extract_thumbnail(source_path, shots, out_dir / "thumbnail.jpg")
    meta = build_metadata(manifest, rights, out_dir)

    if publish_target == "artifact_only":
        return {"artifact": {"status": "ready", "path": str(final_mp4)}}

    targets: list[str]
    if publish_target == "all_private":
        targets = ["r2_link", "telegram", "youtube_private"]
    else:
        targets = [publish_target]

    results: dict[str, Any] = {}
    errors: list[str] = []
    r2_url: str | None = None

    if "r2_link" in targets:
        try:
            res = _publish_r2(final_mp4, slug, config)
            results["r2_link"] = res
            r2_url = res.get("url")
        except Exception as exc:
            errors.append(f"r2_link: {exc}")
            results["r2_link"] = {"status": "failed", "error": str(exc)}

    if "telegram" in targets:
        try:
            results["telegram"] = _publish_telegram(final_mp4, thumb_jpg, meta, config, r2_url)
        except Exception as exc:
            errors.append(f"telegram: {exc}")
            results["telegram"] = {"status": "failed", "error": str(exc)}

    if "youtube_private" in targets or "youtube_public" in targets:
        privacy = "private"
        if "youtube_public" in targets:
            allowed, reason = check_public_approval(project_dir, config)
            if allowed:
                privacy = "public"
            else:
                log.warning("Downgrading youtube_public to private: %s", reason)
                results["approval_downgrade_reason"] = reason
        try:
            results["youtube"] = _publish_youtube(final_mp4, thumb_jpg, meta, privacy, config)
        except Exception as exc:
            errors.append(f"youtube: {exc}")
            results["youtube"] = {"status": "failed", "error": str(exc)}

    if len(errors) == len(targets) and targets:
        log.error("All requested publish targets failed: %s", errors)
        sys.exit(_EXIT_CODE_ALL_PUBLISH_FAILED)

    return results
