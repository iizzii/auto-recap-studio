"""Source video fetcher supporting HTTP/HTTPS, Internet Archive, Google Drive, R2/S3, and local /tmp."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import gdown
import requests

from pipeline import ffutil

log = logging.getLogger(__name__)

_CHUNK_BYTES = 1024 * 1024  # 1 MiB


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def _check_disk_space(work_dir: Path, min_gb: float) -> None:
    """Ensure *work_dir* (or its nearest existing parent) has at least *min_gb* free."""
    check_path = work_dir
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent
    usage = shutil.disk_usage(check_path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_gb:
        log.error(
            "Insufficient free disk space in %s: %.2f GB free < %.2f GB required.",
            check_path,
            free_gb,
            min_gb,
        )
        sys.exit(2)


def _download_http(url: str, dest: Path, max_gb: float) -> Path:
    """Stream-download an HTTP(S) file with Range resume and *max_gb* cap."""
    max_bytes = int(max_gb * (1024**3))
    existing_size = dest.stat().st_size if dest.exists() else 0
    headers: dict[str, str] = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code == 416:
            return dest
        resp.raise_for_status()

        content_length = resp.headers.get("Content-Length")
        if content_length is not None:
            total = int(content_length) + (existing_size if resp.status_code == 206 else 0)
            if total > max_bytes:
                raise ValueError(
                    f"Source size ({total / (1024**3):.2f} GB) exceeds max_source_gb ({max_gb} GB)."
                )

        mode = "ab" if resp.status_code == 206 else "wb"
        written = existing_size if resp.status_code == 206 else 0
        with dest.open(mode) as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK_BYTES):
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(
                        f"Downloaded bytes exceeded max_source_gb ({max_gb} GB)."
                    )
                fh.write(chunk)
    return dest


def _resolve_internet_archive_url(url: str) -> str:
    """Resolve an archive.org/details/<id> page URL to a direct video download URL."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "details":
        item_id = parts[1]
        meta_url = f"https://archive.org/metadata/{item_id}"
        resp = requests.get(meta_url, timeout=30)
        resp.raise_for_status()
        meta: dict[str, Any] = resp.json()
        files = meta.get("files", [])
        for ext in (".mp4", ".mkv", ".mov", ".avi", ".webm"):
            for f in files:
                name = str(f.get("name", ""))
                if name.lower().endswith(ext):
                    return f"https://archive.org/download/{item_id}/{name}"
        raise ValueError(f"No playable video file found in Internet Archive item '{item_id}'.")
    return url


def _download_gdrive(url_or_id: str, dest: Path) -> Path:
    """Download a Google Drive file via service account JSON or gdown."""
    sa_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    match = re.search(r"(?:id=|/d/)([A-Za-z0-9_-]{10,})", url_or_id)
    file_id = match.group(1) if match else url_or_id.strip()

    if sa_json:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        service = build("drive", "v3", credentials=creds)
        request = service.files().get_media(fileId=file_id)
        with dest.open("wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return dest

    url = f"https://drive.google.com/uc?id={file_id}"
    out = gdown.download(url, str(dest), quiet=False, fuzzy=True)
    if not out or not dest.exists():
        raise RuntimeError(f"gdown failed to download Google Drive file '{url_or_id}'.")
    return dest


def _download_r2_s3(uri: str, dest: Path) -> Path:
    """Download an r2://bucket/key or s3://bucket/key object via boto3."""
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    endpoint_url = os.environ.get("R2_ENDPOINT_URL") or None
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=(
            os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
        ),
    )
    client.download_file(bucket, key, str(dest))
    return dest


def _copy_local(path: str, dest: Path) -> Path:
    """Copy a local /tmp file (for CI smoke tests) to *dest*."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Local source file not found: {src}")
    if src.resolve() == dest.resolve():
        return dest
    shutil.copy2(src, dest)
    return dest


def fetch_source(url: str, work_dir: Path, config: dict[str, Any]) -> tuple[Path, str]:
    """Fetch source video into *work_dir*, probe video stream, and return (path, sha256)."""
    runtime_cfg = config.get("runtime", {})
    source_cfg = config.get("source", {})
    min_gb = float(runtime_cfg.get("min_free_disk_gb", 4.0))
    max_gb = float(source_cfg.get("max_source_gb", 6.0))

    work_dir.mkdir(parents=True, exist_ok=True)
    _check_disk_space(work_dir, min_gb)

    parsed = urlparse(url)
    ext = Path(parsed.path).suffix or ".mp4"
    dest = work_dir / f"source{ext}"

    if parsed.scheme in ("r2", "s3"):
        _download_r2_s3(url, dest)
    elif parsed.scheme in ("http", "https"):
        host = parsed.netloc.lower()
        if "drive.google.com" in host:
            _download_gdrive(url, dest)
        elif "archive.org" in host:
            resolved = _resolve_internet_archive_url(url)
            _download_http(resolved, dest, max_gb)
        else:
            _download_http(url, dest, max_gb)
    elif parsed.scheme == "" and Path(url).exists():
        _copy_local(url, dest)
    elif re.fullmatch(r"[A-Za-z0-9_-]{20,}", url.strip()):
        _download_gdrive(url, dest)
    else:
        raise ValueError(f"Unsupported video source URL or missing local path: {url}")

    info = ffutil.probe(dest)
    has_video = any(s.get("codec_type") == "video" for s in info.get("streams", []))
    if not has_video:
        raise ValueError(f"Downloaded file {dest} does not contain a valid video stream.")

    sha256 = compute_sha256(dest)
    return dest, sha256
