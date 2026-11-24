"""Time-budget tracker for pipeline stages."""

from __future__ import annotations

import logging
import sys
import time

log = logging.getLogger(__name__)

_EXIT_CODE_BUDGET = 8


class TimeBudget:
    """Track remaining wall-clock time for the pipeline run.

    Args:
        minutes: Total budget in minutes.
    """

    def __init__(self, minutes: float) -> None:
        self._budget_sec = minutes * 60.0
        self._start = time.monotonic()

    def remaining_sec(self) -> float:
        """Seconds remaining in the budget."""
        elapsed = time.monotonic() - self._start
        return max(0.0, self._budget_sec - elapsed)

    def has_margin(self, min_minutes: float = 8) -> bool:
        """Return *True* if at least *min_minutes* remain."""
        return self.remaining_sec() >= min_minutes * 60.0

    def check_or_exit(self, stage_name: str) -> None:
        """Exit the process with code 8 if no margin remains.

        Args:
            stage_name: Label of the stage about to run (used in log).
        """
        if not self.has_margin():
            remaining = self.remaining_sec()
            log.warning(
                "Budget exhausted before stage '%s' (%.1fs left). "
                "Exiting with code %d.",
                stage_name,
                remaining,
                _EXIT_CODE_BUDGET,
            )
            sys.exit(_EXIT_CODE_BUDGET)
