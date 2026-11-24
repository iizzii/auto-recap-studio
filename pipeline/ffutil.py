"""FFmpeg / ffprobe utility helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

_STDERR_TAIL_LINES = 40


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe subprocess fails."""

    def __init__(
        self,
        cmd_or_stderr: list[str] | str,
        stderr: str | None = None,
    ) -> None:
        if isinstance(cmd_or_stderr, str) and stderr is None:
            self.cmd: list[str] = []
            raw_stderr = cmd_or_stderr
        else:
            self.cmd = cmd_or_stderr if isinstance(cmd_or_stderr, list) else [str(cmd_or_stderr)]
            raw_stderr = stderr or ""
        lines = raw_stderr.splitlines()
        self.last_lines: list[str] = lines[-_STDERR_TAIL_LINES:] if lines else []
        self.stderr_tail = "\n".join(self.last_lines)
        prefix = f"FFmpeg failed: {' '.join(self.cmd[:4])}...\n" if self.cmd else "FFmpeg failed:\n"
        super().__init__(f"{prefix}{self.stderr_tail}")


def run(
    cmd: list[str],
    *,
    timeout: int = 600,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg/ffprobe command.

    Args:
        cmd: Full argument list (no shell).
        timeout: Seconds before the process is killed.
        check: If *True*, raise :class:`FFmpegError` on non-zero exit.

    Returns:
        The completed process result.
    """
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise FFmpegError(cmd, result.stderr)
    return result


def probe(path: str | Path) -> dict[str, Any]:
    """Return the full ffprobe JSON for *path*."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = run(cmd)
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def duration(path: str | Path) -> float:
    """Return the duration of a media file in seconds."""
    info = probe(path)
    fmt = info.get("format", {})
    dur = fmt.get("duration")
    if dur is not None:
        return float(dur)
    # Fall back to the first stream with a duration.
    for stream in info.get("streams", []):
        dur = stream.get("duration")
        if dur is not None:
            return float(dur)
    raise FFmpegError(
        ["ffprobe", str(path)],
        "No duration found in format or streams",
    )


def has_audio(path: str | Path) -> bool:
    """Return *True* if the file contains at least one audio stream."""
    info = probe(path)
    return any(
        s.get("codec_type") == "audio" for s in info.get("streams", [])
    )


def frame_count(path: str | Path) -> int:
    """Return the video frame count using ffprobe *nb_frames* or packet counting."""
    info = probe(path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        nb = stream.get("nb_frames")
        if nb is not None:
            return int(nb)

    # Fallback: count packets with ffprobe.
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-print_format", "json",
        str(path),
    ]
    result = run(cmd)
    data: dict[str, Any] = json.loads(result.stdout)
    for stream in data.get("streams", []):
        nb = stream.get("nb_read_packets")
        if nb is not None:
            return int(nb)
    raise FFmpegError(
        ["ffprobe", str(path)],
        "Could not determine frame count",
    )
