#!/usr/bin/env python3
"""Print a ready-to-paste rights.json template to stdout.

Each field includes an inline comment (via a ``"_comment_*"`` key) explaining
its purpose.  Strip the comment keys before committing.
"""

from __future__ import annotations

import json
import sys

TEMPLATE: dict[str, object] = {
    "_comment_work_title": "Title of the source work (film, show, video, etc.).",
    "work_title": "FILL_IN",
    "_comment_creator": "Original creator or rights-holder name.",
    "creator": "FILL_IN",
    "_comment_year": "Year the work was first published or released.",
    "year": 0,
    "_comment_license": "One of: owned | cc0 | cc-by | cc-by-sa | cc-by-nc | cc-by-nc-sa | cc-by-nd | cc-by-nc-nd | public_domain | fair_use.",
    "license": "FILL_IN",
    "_comment_license_url": "URL to the full license text (e.g. Creative Commons deed).",
    "license_url": "",
    "_comment_source_page_url": "URL where the source material was obtained.",
    "source_page_url": "FILL_IN",
    "_comment_commercial_use_allowed": "true if the license permits commercial use.",
    "commercial_use_allowed": False,
    "_comment_derivatives_allowed": "true if the license permits derivative works.",
    "derivatives_allowed": False,
    "_comment_attribution_text": "Required attribution line to include in the video description.",
    "attribution_text": "FILL_IN",
    "_comment_pd_basis": "If public domain: legal basis (e.g. 'US copyright expired pre-1929'). null otherwise.",
    "pd_basis": None,
    "_comment_pd_verified_by_human": "true only if a human verified the public-domain status.",
    "pd_verified_by_human": False,
    "_comment_license_document_url": "URL to the specific license document or deed, if separate from license_url.",
    "license_document_url": None,
    "_comment_owner_statement": "Direct statement from the rights-holder granting permission, if applicable.",
    "owner_statement": None,
    "_comment_soundtrack_rights_note": "Notes on soundtrack/music rights – required even if 'no music used'.",
    "soundtrack_rights_note": "FILL_IN",
    "_comment_verified_by": "Name or handle of the person who verified the rights information.",
    "verified_by": "FILL_IN",
    "_comment_verified_on": "ISO-8601 date when verification was performed (e.g. 2025-01-15).",
    "verified_on": "FILL_IN",
}


def main() -> None:
    """Dump the template as pretty-printed JSON."""
    json.dump(TEMPLATE, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
