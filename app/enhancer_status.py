"""Process-wide enhancer status, written by the enhancer loop and read by
the webapp. Single mutable singleton — both threads update fields directly.

Python's GIL makes single-field assignments effectively atomic, so we don't
bother with a lock for reads; writes happen in only one thread (the enhancer).
"""
from __future__ import annotations

import threading
import time
from typing import Any


class EnhancerStatus:
    def __init__(self) -> None:
        self.state: str = "starting"  # starting | fetching | enhancing | sleeping | idle | error
        self.cycle_number: int = 0
        self.last_started_at: float | None = None
        self.last_finished_at: float | None = None
        self.sleep_until: float | None = None
        self.current_group: str | None = None
        self.progress_done: int = 0
        self.progress_total: int = 0
        self.last_summary: dict[str, int] = {}
        self.last_error: str | None = None

        # Manual-trigger plumbing: the webapp sets this, the enhancer loop
        # observes it at the top of its sleep poll and runs early.
        self._run_now = threading.Event()

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "cycleNumber": self.cycle_number,
            "lastStartedAt": self.last_started_at,
            "lastFinishedAt": self.last_finished_at,
            "sleepUntil": self.sleep_until,
            "currentGroup": self.current_group,
            "progressDone": self.progress_done,
            "progressTotal": self.progress_total,
            "lastSummary": self.last_summary,
            "lastError": self.last_error,
            "now": int(time.time() * 1000),
        }

    def request_run_now(self) -> None:
        self._run_now.set()

    def consume_run_now(self) -> bool:
        if self._run_now.is_set():
            self._run_now.clear()
            return True
        return False


# Module-level singleton shared between the enhancer loop and the webapp.
STATUS = EnhancerStatus()
