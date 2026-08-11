"""Progress reporting for gates that run for hours.

Written after G-6 ran for the better part of a day, died in a LAPACK call, and left nothing
behind. The log held one line saying it had started and one saying it had exited 1, so there
was no way to tell whether it had failed on its first case or its last, how much of the grid
had been covered, or whether the crash was near a particular size. Every conclusion about
where it broke had to come from re-running it.

Output goes to stderr, so a gate's report on stdout stays a clean artefact-shaped thing that
can be redirected on its own, and every write is flushed. An unflushed progress line is worse
than none: it is still sitting in a buffer when the process dies, which is exactly the moment
it was needed.

The estimate is a running mean over completed items rather than a smoothed or windowed one.
These grids are strongly inhomogeneous in cost, usually growing with L, so the estimate is
honest early on about being provisional and tightens as the expensive sizes arrive. It is
labelled an estimate for that reason.
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
        How many items the caller intends to complete. Used only for the fraction and the
        estimate; overrunning it is reported rather than hidden, because a total that turns
        out to be wrong is itself worth seeing in the log.
    label
        Short name for the work, printed on every line so interleaved output stays readable.
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
