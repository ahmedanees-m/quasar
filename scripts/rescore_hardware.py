"""Reconstruct the hardware result from archived raw counts, offline.

This closes the last dependency the hardware result had on an external service. Given
`raw_counts.json` from the deposit, it rebuilds the reference distributions, applies readout
mitigation, decodes and scores, and compares the outcome against the published record. No
network, no account, no job identifier.

The reconstruction is not a replay of stored numbers. The reference quasispecies is recomputed
from the analytic solution and the variational parameters are reconverged, so agreement means
the published values follow from the deposited measurements and the deposited code together.

    python scripts/rescore_hardware.py <raw_counts.json> [--record results/_local/g_8.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from quasarstack.analytic.crow_kimura import class_quasispecies  # noqa: E402
from quasarstack.backends.execution import mitigate_readout  # noqa: E402
from quasarstack.classical.landscapes import single_peak_classes  # noqa: E402
from quasarstack.io.conventions import (  # noqa: E402
    decode_from_measurement,
    genotype_to_index,
    qiskit_bitstring_to_genotype,
)
from quasarstack.scoring.metrics import score  # noqa: E402

# The submitted plan, in the order the circuits were sent. Sweep points first for each size,
# then that size's readout-calibration circuits, exactly as `qpu_sweep.build_plan` emits them.
SIZES = [2, 3, 4]
MU_RATIOS = {
    2: [0.4, 0.7, 1.0, 1.3, 1.6],
    3: [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6],
    4: [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6],
}
PEAK_HEIGHT = 1.0


def counts_to_distribution(counts: dict[str, int], n_sites: int) -> np.ndarray:
    probabilities = np.zeros(1 << n_sites, dtype=np.float64)
    for bitstring, count in counts.items():
        genotype = qiskit_bitstring_to_genotype(bitstring.replace(" ", ""))
        probabilities[genotype_to_index(genotype)] += count
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("a circuit returned no counts")
    return probabilities / total


def rebuild_plan() -> list[dict]:
    plan = []
    for n_sites in SIZES:
        mu_c = PEAK_HEIGHT / n_sites
        for ratio in MU_RATIOS[n_sites]:
            plan.append({"kind": "sweep", "L": n_sites, "mu_over_mu_c": ratio, "mu": ratio * mu_c})
        for index in range(1 << n_sites):
            plan.append({"kind": "calibration", "L": n_sites, "prepared_index": index})
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts", help="raw_counts.json from the deposit")
    parser.add_argument("--record", default="results/_local/g_8.json")
    arguments = parser.parse_args()

    payload = json.loads(pathlib.Path(arguments.counts).read_text(encoding="utf-8"))
    pubs = payload["pubs"]
    plan = rebuild_plan()
    if len(pubs) != len(plan):
        raise SystemExit(
            f"{len(pubs)} circuits in the archive against {len(plan)} in the rebuilt plan. "
            f"The archive and this script describe different submissions."
        )
    for entry, pub in zip(plan, pubs, strict=True):
        entry["counts"] = pub["counts"]

    assignment: dict[int, np.ndarray] = {}
    for n_sites in SIZES:
        calibration = sorted(
            (e for e in plan if e["kind"] == "calibration" and e["L"] == n_sites),
            key=lambda e: e["prepared_index"],
        )
        matrix = np.zeros((1 << n_sites, 1 << n_sites), dtype=np.float64)
        for entry in calibration:
            matrix[:, entry["prepared_index"]] = counts_to_distribution(entry["counts"], n_sites)
        assignment[n_sites] = matrix

    print(f"{'L':>2} {'mu/mu_c':>8} {'raw':>9} {'mitigated':>10}")
    rebuilt = {}
    for entry in plan:
        if entry["kind"] != "sweep":
            continue
        n_sites = entry["L"]
        raw = counts_to_distribution(entry["counts"], n_sites)
        mitigated = mitigate_readout(raw, assignment[n_sites])
        reference, _, _ = class_quasispecies(single_peak_classes(n_sites, PEAK_HEIGHT), entry["mu"])
        reference = np.asarray(reference, dtype=np.float64)
        scored = score(decode_from_measurement(mitigated), reference)
        rebuilt[(n_sites, round(entry["mu_over_mu_c"], 2))] = scored["cosine"]
        print(
            f"{n_sites:>2} {entry['mu_over_mu_c']:>8.2f} "
            f"{score(decode_from_measurement(raw), reference)['cosine']:>9.5f} "
            f"{scored['cosine']:>10.5f}"
        )

    record_path = pathlib.Path(arguments.record)
    if not record_path.is_file():
        print(f"\nno record at {record_path} to compare against")
        return 0
    published = {
        (c["L"], round(c["mu_over_mu_c"], 2)): c["decoded_mitigated_cosine"]
        for c in json.loads(record_path.read_text(encoding="utf-8"))["cases"]
    }
    worst = max(abs(rebuilt[k] - published[k]) for k in published)
    print(f"\nreconstructed {len(published)} points from counts alone")
    print(f"largest difference against the published record: {worst:.2e}")
    if worst > 1e-12:
        print("the reconstruction does not match the record")
        return 1
    print("the published hardware result follows from the deposited measurements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
