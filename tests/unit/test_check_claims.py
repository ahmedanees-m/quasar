"""The claims-ledger checker, which had been skipping rows in silence.

`CLAIMS.md` is one of the four documents carrying scientific weight, and this script is what
makes it more than a table. It was accepting identifiers of the form C12 only, so rows like
C4b and C5b, added for a second claim belonging to the same gate, matched nothing and were
passed over without a word. Three claims sat in the ledger looking checked and were not.

A checker that quietly ignores what it cannot parse is worse than no checker, because it
produces a green tick that means nothing. These tests pin both halves of the fix: suffixed
identifiers parse, and anything claim-shaped that still fails to parse is reported.
"""

from __future__ import annotations

import pytest

from scripts.check_claims import LEDGER, check, parse_ledger

pytestmark = pytest.mark.fast

HEADER = "| # | Claim | Gate | Artefact | Script | Status |\n|---|---|---|---|---|---|\n"


def test_plain_and_suffixed_identifiers_both_parse() -> None:
    text = HEADER + (
        "| C1 | first | G-R.1 | `results/a.json` | `scripts/x.py` | planned |\n"
        "| C1b | second, same gate | G-R.1 | `results/b.json` | `scripts/x.py` | planned |\n"
        "| C12ab | third | G-R.2 | `results/c.json` | `scripts/y.py` | planned |\n"
    )
    claims, unparsed = parse_ledger(text)
    assert [c.ident for c in claims] == ["C1", "C1b", "C12ab"]
    assert unparsed == []


@pytest.mark.parametrize(
    ("row", "why"),
    [
        ("| C7 | too | few |", "identifier fine, columns short"),
        (
            "| CX | bad id | G | `a.json` | `s.py` | planned |",
            "identifier not of the expected form",
        ),
        ("| C7-alt | dash | G | `a.json` | `s.py` | planned |", "unexpected punctuation in the id"),
    ],
)
def test_claim_shaped_rows_that_do_not_parse_are_reported(row: str, why: str) -> None:
    """The failure mode that motivated all of this: silence.

    Each of these looks like a claim to a reader of the ledger. None of them parses. All of
    them must be surfaced rather than passed over.
    """
    claims, unparsed = parse_ledger(HEADER + row + "\n")
    assert claims == [], why
    assert unparsed, f"a malformed claim row must be surfaced, not skipped: {why}"


def test_unparsed_rows_become_failures_not_warnings() -> None:
    text = HEADER + "| C7 | too | few |\n| C8 | ok | G | `results/a.json` | `s.py` | planned |\n"
    claims, unparsed = parse_ledger(text)
    problems = check(claims, allow_planned=True)
    problems.extend(f"row looks like a claim but did not parse: {row}" for row in unparsed)
    assert any("did not parse" in p for p in problems)
    assert [c.ident for c in claims] == ["C8"]


def test_the_table_separator_is_not_mistaken_for_a_claim() -> None:
    claims, unparsed = parse_ledger(HEADER)
    assert claims == []
    assert unparsed == []


def test_the_real_ledger_parses_completely() -> None:
    """No row in the project's own ledger may go unchecked."""
    claims, unparsed = parse_ledger(LEDGER.read_text(encoding="utf-8"))
    assert not unparsed, f"unparsed claim rows in CLAIMS.md: {unparsed}"
    assert len(claims) > 30
    assert any(
        c.ident.endswith("b") for c in claims
    ), "the ledger uses suffixed identifiers, so the parser must handle them"


def test_a_pass_row_without_its_artefact_fails_even_in_ci_mode() -> None:
    """`--allow-planned` relaxes planned rows only. A claim marked pass must resolve."""
    text = HEADER + "| C9 | claimed | G | `results/does_not_exist.json` | `s.py` | pass |\n"
    claims, _ = parse_ledger(text)
    problems = check(claims, allow_planned=True)
    assert any("missing" in p for p in problems)
