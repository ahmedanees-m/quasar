"""Shared fixtures and seed control.

Every stochastic component takes an explicit generator, so a test that draws numbers uses one
of these fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic generator with seed 0."""
    return np.random.default_rng(0)


@pytest.fixture(params=[0, 1, 2, 3, 4])
def seed(request: pytest.FixtureRequest) -> int:
    """Landscape seeds 0 to 4, for seed-robustness checks."""
    return int(request.param)
