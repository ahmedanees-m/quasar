"""Progress reporting for long checks.

Lines go to stderr, so a check's report on stdout can be redirected on its own, and every
write is flushed. The remaining-time estimate is a running mean over completed items.
"""

from __future__ import annotations

import sys
import time
from typing import TextIO


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


class Progress:
    """A counter that prints one line per completed item, with elapsed time and an estimate.

    Parameters
    ----------
    total
        Number of items expected. Used for the fraction and the estimate; exceeding it is
        reported.
    label
        Short name printed on every line.
    stream
        Defaults to stderr.
    """

    def __init__(self, total: int, label: str, stream: TextIO | None = None) -> None:
        self.total = int(total)
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.done = 0
        self.started = time.monotonic()

    def step(self, note: str = "") -> None:
        self.done += 1
        elapsed = time.monotonic() - self.started
        per_item = elapsed / self.done
        remaining = max(self.total - self.done, 0) * per_item
        fraction = f"{self.done}/{self.total}"
        if self.done > self.total:
            fraction += " (over the declared total)"
        line = (
            f"[{self.label}] {fraction}  elapsed {_duration(elapsed)}  "
            f"eta {_duration(remaining)}  {per_item:.1f}s/item"
        )
        if note:
            line += f"  {note}"
        print(line, file=self.stream, flush=True)

    def finish(self) -> None:
        elapsed = time.monotonic() - self.started
        print(
            f"[{self.label}] complete, {self.done} items in {_duration(elapsed)}",
            file=self.stream,
            flush=True,
        )
