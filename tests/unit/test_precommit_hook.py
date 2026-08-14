"""The pre-commit configuration must not drift away from what CI runs.

Hooks that check less than CI give false confidence: they pass, you push, CI fails, and the
loop the hooks exist to close stays open. That open loop is the single most repeated defect in
this project. CI's claims step failed for a fortnight while `UNRESOLVED` in its output read
like a status word rather than a failure; `black --check` was red for a day after the wrong
formatter was used; and a laptop-produced record reached a commit because ADR-0006 was policy
with nothing enforcing it.

The specific gap this test was written to catch: `.pre-commit-config.yaml` ran black, ruff and
mypy but neither `check_claims.py` nor `check_results_provenance.py`, which are the two checks
guarding the evidence the manuscript will cite.

The test suite is exempt by name. It takes minutes, and hooks slow enough to be resented get
disabled, which is worse than not having them. It stays in CI and in `make test-all`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / ".pre-commit-config.yaml"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# Slow by nature, left to CI and `make test-all`. Named so the gap stays deliberate.
EXEMPT = {"pytest"}


def ci_tools() -> set[str]:
    """The identifying tool of every `run:` line in the workflow."""
    tools = set()
    for line in re.findall(r"^\s*run:\s*(.+)$", WORKFLOW.read_text(encoding="utf-8"), re.M):
        command = line.strip()
        if command == "|":
            continue
        parts = command.split()
        # `python scripts/check_claims.py` is identified by its script, not by `python`.
        if parts[0] == "python" and len(parts) > 1:
            tools.add(Path(parts[1]).name)
        else:
            tools.add(parts[0])
    return tools


def config_text() -> str:
    return CONFIG.read_text(encoding="utf-8")


def test_the_configuration_parses() -> None:
    data = yaml.safe_load(config_text())
    assert data.get("repos"), "pre-commit config has no repos"


def test_every_check_ci_runs_is_also_a_hook() -> None:
    text = config_text()
    missing = sorted(tool for tool in ci_tools() if tool not in EXEMPT and tool not in text)
    assert not missing, (
        f"CI runs {missing} and pre-commit does not. Hooks that check less than CI pass "
        f"locally and fail on push, which is the loop they exist to close. Add them to "
        f".pre-commit-config.yaml, or to EXEMPT here with a reason."
    )


def test_the_exemptions_are_still_things_ci_runs() -> None:
    """An exemption for a check CI no longer runs is stale and should go."""
    stale = sorted(name for name in EXEMPT if name not in ci_tools())
    assert not stale, f"{stale} are exempted here but CI does not run them any more"


def test_the_evidence_checks_run_on_the_whole_tree() -> None:
    """Both inspect all of `results/`, so limiting them to staged files would miss breakage.

    A claim breaks when an artefact it names disappears, and that artefact is usually not
    itself among the staged files.
    """
    data = yaml.safe_load(config_text())
    local = [h for repo in data["repos"] if repo["repo"] == "local" for h in repo["hooks"]]
    ids = {h["id"] for h in local}
    assert {"check-claims", "check-provenance"} <= ids, f"local hooks are {sorted(ids)}"
    for hook in local:
        assert hook.get("always_run") is True, f"{hook['id']} must always run"
        assert hook.get("pass_filenames") is False, f"{hook['id']} must not take staged files"


def test_make_setup_installs_the_hooks() -> None:
    """Hooks nobody installs are hooks nobody runs; `.git/hooks` is not version controlled."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "pre-commit install" in makefile, "make setup must install the hooks"
