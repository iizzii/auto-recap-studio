#!/usr/bin/env python3
"""Generate synthetic test media for CI end-to-end runs.

Creates a 45-second 1920×1080 30fps colour-bar video with a 1 kHz sine tone
and a matching 24-line SRT subtitle file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

DURATION_SEC: int = 45
WIDTH: int = 1920
HEIGHT: int = 1080
FPS: int = 30
SRT_LINES: int = 24
AUDIO_FREQ_HZ: int = 1000


def _run_ffmpeg(args: Sequence[str]) -> None:
    """Run an FFmpeg command, preferring ffutil if importable."""
    try:
        from pipeline.ffutil import run as ffutil_run  # type: ignore[import-untyped]

        ffutil_run(list(args))
    except ImportError:
        subprocess.run(["ffmpeg", *args], check=True)


def generate_video(out_dir: Path) -> Path:
    """Create a test video with colour bars, moving shapes, and a sine tone."""
    video_path = out_dir / "source.mp4"
    _run_ffmpeg(
        [
            "-y",
            "-f", "lavfi",
            "-i", (
                f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}:duration={DURATION_SEC}"
            ),
            "-f", "lavfi",
            "-i", f"sine=frequency={AUDIO_FREQ_HZ}:duration={DURATION_SEC}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "64k",
            "-shortest",
            str(video_path),
        ]
    )
    return video_path


def generate_srt(out_dir: Path) -> Path:
    """Create a 24-line SRT file with timestamps spread across the video."""
    srt_path = out_dir / "source.srt"
    interval = DURATION_SEC / SRT_LINES
    lines: list[str] = []
    for i in range(SRT_LINES):
        idx = i + 1
        start = i * interval
        end = start + interval - 0.1
        lines.append(str(idx))
        lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
        lines.append(f"Synthetic subtitle line {idx}.")
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def _fmt_ts(seconds: float) -> str:
    """Format seconds as an SRT timestamp ``HH:MM:SS,mmm``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_rights(out_dir: Path) -> Path:
    """Create a minimal rights.json declaring synthetic/owned content."""
    rights_dir = out_dir / "projects" / "ci_synth"
    rights_dir.mkdir(parents=True, exist_ok=True)
    rights_path = rights_dir / "rights.json"
    rights: dict[str, object] = {
        "work_title": "CI Synthetic Test Fixture",
        "creator": "automated-ci",
        "year": 2025,
        "license": "owned",
        "license_url": "",
        "source_page_url": "synthetic://ci",
        "commercial_use_allowed": True,
        "derivatives_allowed": True,
        "attribution_text": "Synthetic test content – no attribution required.",
        "pd_basis": None,
        "pd_verified_by_human": False,
        "license_document_url": None,
        "owner_statement": "Generated for CI testing purposes.",
        "soundtrack_rights_note": "Sine tone – no rights issues.",
        "verified_by": "ci",
        "verified_on": "auto",
    }
    rights_path.write_text(json.dumps(rights, indent=2) + "\n", encoding="utf-8")
    return rights_path


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic media fixtures for CI.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write source.mp4, source.srt, and rights.json into.",
    )
    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[synth] Generating {DURATION_SEC}s test video → {out_dir}", file=sys.stderr)
    video = generate_video(out_dir)
    print(f"[synth] Video: {video}", file=sys.stderr)

    srt = generate_srt(out_dir)
    print(f"[synth] SRT:   {srt}", file=sys.stderr)

    rights = generate_rights(out_dir)
    print(f"[synth] Rights: {rights}", file=sys.stderr)

    print("[synth] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
