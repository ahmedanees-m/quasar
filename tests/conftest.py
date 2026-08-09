"""Shared fixtures and seed control.

Every stochastic component in this project takes an explicit generator. There is no
implicit global randomness, so a test that draws numbers must ask for one of these
fixtures and the seed is visible in the test.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic generator. Seed 0 is the project default for tests."""
    return np.random.default_rng(0)


@pytest.fixture(params=[0, 1, 2, 3, 4])
def seed(request: pytest.FixtureRequest) -> int:
    """The first five of the ten registered landscape seeds, for seed-robustness checks."""
    return int(request.param)
