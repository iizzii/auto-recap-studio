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
    """Create a 6-scene, 44-second adventure SRT with >=1.0s gaps between scenes."""
    srt_path = out_dir / "source.srt"
    scenes: list[list[tuple[float, float, str]]] = [
        [
            (0.5, 3.2, "MILO: Look under the rusted bridge, Robo!"),
            (3.4, 6.2, "A glowing blue signal cube is pulsing in the mud."),
        ],
        [
            (7.5, 10.4, "ROBO: Warning, Milo. Its core clock is counting down fast."),
            (10.6, 13.4, "If it reaches zero before sunset, the canyon relay explodes."),
        ],
        [
            (14.7, 17.6, "Searchlight drones sweep across the ridge behind them."),
            (17.8, 20.6, "Milo snatches the cube and sprints into the drainage tunnel."),
        ],
        [
            (22.0, 25.0, "ROBO: Holding the flooded vault door!"),
            (25.2, 28.0, "Sparks rain down as Robo welds the jammed blast hatch shut."),
        ],
        [
            (29.4, 32.4, "Milo slots the pulsing cube into the ancient star-map console."),
            (32.6, 35.4, "Spinning holographic rings ignite across the dark chamber."),
        ],
        [
            (36.8, 40.0, "MILO: The ceiling is opening! That was never a bomb."),
            (40.2, 43.8, "Above them waits a hidden launch bay and an untouched starship."),
        ],
    ]
    lines: list[str] = []
    idx = 1
    for scene in scenes:
        for start, end, text in scene:
            lines.append(str(idx))
            lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
            lines.append(text)
            lines.append("")
            idx += 1
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
