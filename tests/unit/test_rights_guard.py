"""Unit tests for pipeline.rights_guard – zero real network calls."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pipeline.rights_guard import (
    build_attribution_block,
    check_rights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rights_dict(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid CC-BY rights dict, with optional overrides."""
    base: dict[str, Any] = {
        "work_title": "Test Film",
        "creator": "Jane Doe",
        "year": 2024,
        "license": "cc_by",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_page_url": "https://archive.org/details/test-film",
        "commercial_use_allowed": True,
        "derivatives_allowed": True,
        "attribution_text": "Credit: Jane Doe (CC BY 4.0)",
        "pd_basis": None,
        "pd_verified_by_human": False,
        "license_document_url": None,
        "owner_statement": None,
        "soundtrack_rights_note": "Original score – same license.",
        "verified_by": "unit-test",
        "verified_on": "2024-01-01",
    }
    base.update(overrides)
    return base


def _write_rights(tmp_path: Path, data: dict[str, Any] | str) -> Path:
    """Write a rights.json into *tmp_path* and return the file path."""
    fp = tmp_path / "rights.json"
    fp.write_text(json.dumps(data) if isinstance(data, dict) else data, encoding="utf-8")
    return fp


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestCheckRightsPass:
    """Cases that should pass validation."""

    def test_valid_cc_by_passes(self, tmp_path: Path) -> None:
        fp = _write_rights(tmp_path, _rights_dict())
        result = check_rights(fp)
        assert result is True

    def test_valid_public_domain(self, tmp_path: Path) -> None:
        fp = _write_rights(
            tmp_path,
            _rights_dict(
                license="public_domain",
                license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                pd_basis="expired copyright – published 1920",
                pd_verified_by_human=True,
            ),
        )
        result = check_rights(fp)
        assert result is True


# ---------------------------------------------------------------------------
# Rejection / exit-code-3 tests
# ---------------------------------------------------------------------------

class TestCheckRightsReject:
    """Cases that must be rejected (exit code 3 or raised exception)."""

    def test_cc_by_nc_rejected(self, tmp_path: Path) -> None:
        fp = _write_rights(
            tmp_path,
            _rights_dict(
                license="cc_by_nc",
                license_url="https://creativecommons.org/licenses/by-nc/4.0/",
                commercial_use_allowed=False,
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            check_rights(fp)
        assert exc_info.value.code == 3

    def test_missing_rights_json(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"
        with pytest.raises(SystemExit) as exc_info:
            check_rights(missing)
        assert exc_info.value.code == 3

    def test_invalid_json(self, tmp_path: Path) -> None:
        fp = _write_rights(tmp_path, "{not valid json!!!")
        with pytest.raises(SystemExit) as exc_info:
            check_rights(fp)
        assert exc_info.value.code == 3

    def test_public_domain_without_pd_basis(self, tmp_path: Path) -> None:
        fp = _write_rights(
            tmp_path,
            _rights_dict(
                license="public_domain",
                license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                pd_basis=None,
                pd_verified_by_human=True,
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            check_rights(fp)
        assert exc_info.value.code == 3

    def test_owned_without_owner_statement(self, tmp_path: Path) -> None:
        fp = _write_rights(
            tmp_path,
            _rights_dict(
                license="owned",
                license_url="",
                owner_statement=None,
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            check_rights(fp)
        assert exc_info.value.code == 3

    def test_licensed_without_license_document_url(self, tmp_path: Path) -> None:
        fp = _write_rights(
            tmp_path,
            _rights_dict(
                license="licensed",
                license_url="",
                license_document_url=None,
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            check_rights(fp)
        assert exc_info.value.code == 3


# ---------------------------------------------------------------------------
# Attribution block
# ---------------------------------------------------------------------------

class TestBuildAttributionBlock:
    """Verify the textual attribution output."""

    def test_output_contains_creator(self) -> None:
        block = build_attribution_block(_rights_dict())
        assert "Jane Doe" in block

    def test_output_contains_license(self) -> None:
        block = build_attribution_block(_rights_dict())
        assert "CC" in block.upper() or "cc_by" in block.lower() or "creativecommons" in block.lower()

    def test_output_contains_title(self) -> None:
        block = build_attribution_block(_rights_dict(work_title="My Great Film"))
        assert "My Great Film" in block


# ---------------------------------------------------------------------------
# ShareAlike warning
# ---------------------------------------------------------------------------

class TestShareAlikeWarning:
    """cc_by_sa should emit a warning but still pass."""

    def test_cc_by_sa_logs_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        fp = _write_rights(
            tmp_path,
            _rights_dict(
                license="cc_by_sa",
                license_url="https://creativecommons.org/licenses/by-sa/4.0/",
            ),
        )
        with caplog.at_level(logging.WARNING):
            result = check_rights(fp)
        assert result is True
        assert any("sharealike" in rec.message.lower() or "share-alike" in rec.message.lower() or "sa" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Internet Archive metadata mismatch (strict mode, mocked HTTP)
# ---------------------------------------------------------------------------

class TestInternetArchiveCheck:
    """Mock requests to test IA metadata verification."""

    @patch("pipeline.rights_guard.requests.get")
    def test_ia_metadata_mismatch_strict(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """When IA metadata disagrees with rights.json in strict mode, exit 3."""
        ia_response = MagicMock()
        ia_response.status_code = 200
        ia_response.json.return_value = {
            "metadata": {
                "licenseurl": "https://creativecommons.org/licenses/by-nc/4.0/",
                "title": "Test Film",
            },
        }
        mock_get.return_value = ia_response

        fp = _write_rights(
            tmp_path,
            _rights_dict(
                source_page_url="https://archive.org/details/test-film",
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            check_rights(fp, strict=True)
        assert exc_info.value.code == 3
        mock_get.assert_called_once()

    @patch("pipeline.rights_guard.requests.get")
    def test_synthetic_ci_skips_fetch(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """source_page_url='synthetic://ci' must skip the HTTP fetch entirely."""
        fp = _write_rights(
            tmp_path,
            _rights_dict(source_page_url="synthetic://ci"),
        )
        result = check_rights(fp)
        assert result is True
        mock_get.assert_not_called()
