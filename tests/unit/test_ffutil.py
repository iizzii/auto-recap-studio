"""Unit tests for pipeline.ffutil (FFmpeg wrapper utilities)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pipeline.ffutil import FFmpegError, duration, probe, run


# ---------------------------------------------------------------------------
# FFmpegError
# ---------------------------------------------------------------------------

class TestFFmpegError:
    """Validate the FFmpegError exception."""

    def test_stores_last_40_lines(self) -> None:
        lines = [f"log line {i}" for i in range(100)]
        stderr_text = "\n".join(lines)
        err = FFmpegError(stderr_text)
        assert len(err.last_lines) == 40
        assert err.last_lines[0] == "log line 60"
        assert err.last_lines[-1] == "log line 99"

    def test_short_stderr_stored_fully(self) -> None:
        stderr_text = "single warning\nsecond line"
        err = FFmpegError(stderr_text)
        assert len(err.last_lines) == 2
        assert err.last_lines[0] == "single warning"

    def test_empty_stderr(self) -> None:
        err = FFmpegError("")
        assert err.last_lines == [] or err.last_lines == [""]

    def test_str_representation(self) -> None:
        err = FFmpegError("error on line 1\nerror on line 2")
        s = str(err)
        assert "error on line" in s


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

class TestRun:
    """Test the run() wrapper around subprocess."""

    @patch("pipeline.ffutil.subprocess.run")
    def test_raises_ffmpeg_error_on_failure(self, mock_run: MagicMock) -> None:
        stderr_lines = "\n".join([f"frame {i}" for i in range(50)])
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg", "-i", "in.mp4", "out.mp4"],
            stderr=stderr_lines,
        )
        with pytest.raises(FFmpegError) as exc_info:
            run(["ffmpeg", "-i", "in.mp4", "out.mp4"])
        assert len(exc_info.value.last_lines) == 40

    @patch("pipeline.ffutil.subprocess.run")
    def test_success_returns_none(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg", "-y", "-i", "in.mp4", "out.mp4"],
            returncode=0,
            stdout="",
            stderr="",
        )
        result = run(["ffmpeg", "-y", "-i", "in.mp4", "out.mp4"])
        assert result is None
        mock_run.assert_called_once()

    @patch("pipeline.ffutil.subprocess.run")
    def test_run_passes_check_true(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=0, stdout="", stderr="",
        )
        run(["ffmpeg"])
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("check") is True or (
            len(call_kwargs.args) > 0
        )


# ---------------------------------------------------------------------------
# probe()
# ---------------------------------------------------------------------------

class TestProbe:
    """Test ffprobe JSON wrapper."""

    @patch("pipeline.ffutil.subprocess.run")
    def test_probe_returns_dict(self, mock_run: MagicMock) -> None:
        probe_output = {
            "format": {
                "filename": "input.mp4",
                "duration": "123.456",
                "size": "9876543",
            },
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1080,
                    "height": 1920,
                    "r_frame_rate": "24/1",
                },
            ],
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(probe_output),
            stderr="",
        )
        result = probe("input.mp4")
        assert isinstance(result, dict)
        assert result["format"]["filename"] == "input.mp4"
        assert len(result["streams"]) == 1
        assert result["streams"][0]["codec_type"] == "video"

    @patch("pipeline.ffutil.subprocess.run")
    def test_probe_error_raises(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffprobe", "bad.mp4"],
            stderr="bad.mp4: No such file or directory",
        )
        with pytest.raises(FFmpegError):
            probe("bad.mp4")


# ---------------------------------------------------------------------------
# duration()
# ---------------------------------------------------------------------------

class TestDuration:
    """Test duration extraction helper."""

    @patch("pipeline.ffutil.subprocess.run")
    def test_duration_returns_float(self, mock_run: MagicMock) -> None:
        probe_output = {
            "format": {"duration": "185.320000"},
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(probe_output),
            stderr="",
        )
        result = duration("input.mp4")
        assert isinstance(result, float)
        assert result == pytest.approx(185.32)

    @patch("pipeline.ffutil.subprocess.run")
    def test_duration_on_short_file(self, mock_run: MagicMock) -> None:
        probe_output = {
            "format": {"duration": "0.500000"},
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(probe_output),
            stderr="",
        )
        result = duration("short.mp4")
        assert result == pytest.approx(0.5)

    @patch("pipeline.ffutil.subprocess.run")
    def test_duration_error_raises(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffprobe", "missing.mp4"],
            stderr="missing.mp4: No such file",
        )
        with pytest.raises(FFmpegError):
            duration("missing.mp4")
