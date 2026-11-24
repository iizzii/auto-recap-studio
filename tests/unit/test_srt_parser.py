"""Unit tests for pipeline.srt_parser."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from pipeline.srt_parser import (
    chunk_segments,
    compute_speech_density,
    merge_segments,
    parse_srt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_SRT = dedent("""\
    1
    00:00:01,000 --> 00:00:03,500
    Hello, world!

    2
    00:00:04,000 --> 00:00:06,000
    This is a test.

    3
    00:00:10,000 --> 00:00:14,000
    Final subtitle line.
""")

HTML_SRT = dedent("""\
    1
    00:00:01,000 --> 00:00:03,000
    <i>Italic text</i> and <b>bold</b>.
""")

SOUND_SRT = dedent("""\
    1
    00:00:00,500 --> 00:00:02,000
    [music] Hello there.

    2
    00:00:03,000 --> 00:00:04,500
    [applause]
""")

SPEAKER_SRT = dedent("""\
    1
    00:00:01,000 --> 00:00:03,000
    NARRATOR: Once upon a time.

    2
    00:00:04,000 --> 00:00:06,000
    ALICE: Hi there!
""")

MERGE_SRT = dedent("""\
    1
    00:00:01,000 --> 00:00:02,000
    Part one

    2
    00:00:02,300 --> 00:00:03,500
    part two

    3
    00:00:10,000 --> 00:00:12,000
    Far away line.
""")


def _write_srt(tmp_path: Path, content: str) -> Path:
    fp = tmp_path / "test.srt"
    fp.write_text(content, encoding="utf-8")
    return fp


# ---------------------------------------------------------------------------
# parse_srt
# ---------------------------------------------------------------------------

class TestParseSrt:
    """Core SRT parsing."""

    def test_valid_srt_parsing(self, tmp_path: Path) -> None:
        fp = _write_srt(tmp_path, VALID_SRT)
        segments = parse_srt(fp)
        assert len(segments) == 3
        assert segments[0]["text"] == "Hello, world!"
        assert segments[0]["start"] == pytest.approx(1.0)
        assert segments[0]["end"] == pytest.approx(3.5)
        assert segments[2]["text"] == "Final subtitle line."

    def test_html_tag_stripping(self, tmp_path: Path) -> None:
        fp = _write_srt(tmp_path, HTML_SRT)
        segments = parse_srt(fp)
        assert len(segments) == 1
        text = segments[0]["text"]
        assert "<i>" not in text
        assert "<b>" not in text
        assert "Italic text" in text
        assert "bold" in text

    def test_sound_removal(self, tmp_path: Path) -> None:
        fp = _write_srt(tmp_path, SOUND_SRT)
        segments = parse_srt(fp)
        # [music] should be removed; "Hello there." kept
        texts = [s["text"] for s in segments if s["text"].strip()]
        combined = " ".join(texts)
        assert "[music]" not in combined
        assert "[applause]" not in combined
        assert "Hello there" in combined

    def test_speaker_prefix_removal(self, tmp_path: Path) -> None:
        fp = _write_srt(tmp_path, SPEAKER_SRT)
        segments = parse_srt(fp)
        texts = [s["text"] for s in segments]
        for t in texts:
            assert "NARRATOR:" not in t
            assert "ALICE:" not in t
        assert any("Once upon a time" in t for t in texts)
        assert any("Hi there" in t for t in texts)


# ---------------------------------------------------------------------------
# merge_segments
# ---------------------------------------------------------------------------

class TestMergeSegments:
    """Merge segments whose gap is <= 500 ms."""

    def test_merge_within_500ms(self, tmp_path: Path) -> None:
        fp = _write_srt(tmp_path, MERGE_SRT)
        segments = parse_srt(fp)
        merged = merge_segments(segments, max_gap_ms=500)
        # First two are 300 ms apart → merged.  Third is far away → separate.
        assert len(merged) == 2
        assert "Part one" in merged[0]["text"]
        assert "part two" in merged[0]["text"]
        assert merged[0]["start"] == pytest.approx(1.0)
        assert merged[0]["end"] == pytest.approx(3.5)

    def test_no_merge_when_gap_too_large(self, tmp_path: Path) -> None:
        fp = _write_srt(tmp_path, VALID_SRT)
        segments = parse_srt(fp)
        merged = merge_segments(segments, max_gap_ms=100)
        # Gaps > 100 ms → no merges
        assert len(merged) == 3


# ---------------------------------------------------------------------------
# chunk_segments
# ---------------------------------------------------------------------------

class TestChunkSegments:
    """Chunk segments with overlap."""

    def test_chunking_with_overlap(self, tmp_path: Path) -> None:
        fp = _write_srt(tmp_path, VALID_SRT)
        segments = parse_srt(fp)
        chunks = chunk_segments(segments, chunk_size=2, overlap=1)
        # 3 segments, chunks of 2 with overlap 1 → 2 chunks
        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        # overlap: last of chunk 0 == first of chunk 1
        assert chunks[0][-1]["text"] == chunks[1][0]["text"]

    def test_chunking_no_overlap(self, tmp_path: Path) -> None:
        fp = _write_srt(tmp_path, VALID_SRT)
        segments = parse_srt(fp)
        chunks = chunk_segments(segments, chunk_size=2, overlap=0)
        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 1


# ---------------------------------------------------------------------------
# compute_speech_density
# ---------------------------------------------------------------------------

class TestSpeechDensity:
    """Speech density curve computation."""

    def test_density_curve_length(self, tmp_path: Path) -> None:
        fp = _write_srt(tmp_path, VALID_SRT)
        segments = parse_srt(fp)
        density = compute_speech_density(segments, bucket_sec=1.0)
        # Segments span 1–14 s → at least 14 buckets
        assert len(density) >= 14

    def test_density_nonzero_in_speech(self, tmp_path: Path) -> None:
        fp = _write_srt(tmp_path, VALID_SRT)
        segments = parse_srt(fp)
        density = compute_speech_density(segments, bucket_sec=1.0)
        # Bucket covering 1–2 s should have speech
        assert density[1] > 0.0

    def test_density_zero_in_silence(self, tmp_path: Path) -> None:
        fp = _write_srt(tmp_path, VALID_SRT)
        segments = parse_srt(fp)
        density = compute_speech_density(segments, bucket_sec=1.0)
        # Bucket covering 7–8 s is a gap between segment 2 (ends 6 s) and 3 (starts 10 s)
        assert density[7] == pytest.approx(0.0)
