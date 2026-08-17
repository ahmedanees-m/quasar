"""Generate every supplementary table and data file from committed records.

Nothing here is transcribed by hand. Each table is derived from the record it cites, so a
supplementary number and the artefact behind it cannot drift apart, and regenerating after a
rerun is one command rather than an editing pass.

    python scripts/make_supplementary.py [--out DIR]

Output is written as CSV, one file per item, plus a README naming the source record for each.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load(relative: str):
    path = RESULTS / relative
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def lines(relative: str):
    path = RESULTS / relative
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(row) for row in text.splitlines() if row.strip()]


def write(out: pathlib.Path, name: str, header: list[str], rows: list[list]) -> str:
    target = out / name
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return f"{name}: {len(rows)} rows"


def table_s2(out):
    rows = []
    for path in sorted((RESULTS / "wp_r").glob("g_r_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        measured = record["measured"]
        for key, value in measured.items():
            if isinstance(value, (int, float, str, bool)) and key != "seconds":
                rows.append([record["gate"], key, value, record["passed"]])
            elif isinstance(value, dict):
                for sub, subvalue in value.items():
                    if isinstance(subvalue, (int, float, str, bool)):
                        rows.append([record["gate"], f"{key}.{sub}", subvalue, record["passed"]])
    return write(
        out,
        "Supplementary_Table_S2_validation_suite.csv",
        ["gate", "quantity", "value", "gate_passed"],
        rows,
    )


def table_s3(out):
    record = load("wp3/g_3.json")
    rows = []
    for entry in record["measured"]["criterion_3_monotone_ruggedness"]["by_size"]:
        for k, optima, corr in zip(
            entry["K"], entry["mean_local_optima"], entry["mean_correlation_length"], strict=False
        ):
            rows.append([entry["L"], k, optima, corr])
    return write(
        out,
        "Supplementary_Table_S3_landscape_families.csv",
        ["L", "K", "mean_local_optima_strict", "mean_correlation_length"],
        rows,
    )


def table_s4(out):
    """Threshold location per size, with the two finite-size estimators and their disagreement."""
    record = load("wp1/g_1.json")
    measured = record["measured"]
    c2 = measured["criterion_2_threshold_location"]
    rows = []
    for entry in measured["gap_map"]["gap_closing_at_threshold"]:
        rows.append(
            [
                entry["L"],
                entry["mu_star"],
                entry["mu_star_times_L"],
                entry["min_gap"],
                entry["min_gap_off_the_fine_grid"],
                entry["grid_overestimates_by"],
                entry["worst_eigenvector_condition"],
            ]
        )
    write(
        out,
        "Supplementary_Table_S4_threshold_estimators.csv",
        [
            "L",
            "mu_star",
            "mu_star_times_L",
            "min_gap_arbitrary_precision",
            "min_gap_on_fine_grid",
            "grid_overestimates_by",
            "worst_eigenvector_condition",
        ],
        rows,
    )
    summary = [[k, v] for k, v in c2.items() if isinstance(v, (int, float, str, bool))]
    summary.append(["sizes_that_decide", ";".join(map(str, c2.get("sizes_that_decide", [])))])
    return (
        write(
            out,
            "Supplementary_Table_S4b_threshold_criterion_summary.csv",
            ["quantity", "value"],
            summary,
        )
        + f" (plus S4 with {len(rows)} rows)"
    )


def table_s5(out):
    record = load("wp2/wp2_route_cost.json")
    measured = record["measured"]
    rows = []
    for section in ("simulator_currency", "quantum_currency", "accuracy"):
        for key, value in measured.get(section, {}).items():
            rows.append([section, key, value])
    return write(
        out,
        "Supplementary_Table_S5_route_resource_model.csv",
        ["section", "quantity", "value"],
        rows,
    )


def table_s6(out):
    record = load("wp7/g_7_budget_sensitivity.json")
    measured = record["measured"]
    rows = []
    for arm, values in measured["arms"].items():
        for key, value in values.items():
            rows.append(["arm", arm, key, value])
    for entry in measured["by_size"]:
        for key, value in entry.items():
            if key != "L":
                rows.append(["by_size", f"L={entry['L']}", key, value])
    return write(
        out,
        "Supplementary_Table_S6_budget_exclusions.csv",
        ["section", "group", "quantity", "value"],
        rows,
    )


def table_s7(out):
    record = load("wp6/g_6_3.json")
    rows = [
        [
            c["L"],
            c["family"],
            c["mpo_bond_dimension"],
            c["middle_cut_ceiling"],
            c["fraction_of_ceiling"],
            c["saturates_the_ceiling"],
            c.get("pauli_terms"),
            c.get("ordering_ratio"),
        ]
        for c in record["cases"]
    ]
    return write(
        out,
        "Supplementary_Table_S7_operator_bond_dimension.csv",
        [
            "L",
            "family",
            "mpo_bond_dimension",
            "middle_cut_ceiling",
            "fraction_of_ceiling",
            "saturates_ceiling",
            "pauli_terms",
            "ordering_ratio",
        ],
        rows,
    )


def table_s8(out):
    record = load("wp6/g_6.json")
    rows = [
        [
            c["L"],
            c["family"],
            c.get("mu"),
            c.get("mu_over_mu_c"),
            c.get("chi_needed"),
            c.get("cosine"),
            c.get("converged"),
            c.get("budget_limited"),
            c.get("state_ceiling"),
            c.get("fitness_bond_dimension"),
            c.get("max_discarded_weight_in_one_step"),
        ]
        for c in record["cases"]
    ]
    return write(
        out,
        "Supplementary_Table_S8_state_bond_dimension.csv",
        [
            "L",
            "family",
            "mu",
            "mu_over_mu_c",
            "chi_needed",
            "cosine",
            "converged",
            "budget_limited",
            "state_ceiling",
            "fitness_bond_dimension",
            "max_discarded_weight",
        ],
        rows,
    )


def table_s9(out):
    record = load("wp7/g_7.json")
    measured = record["measured"]
    rows = [
        [condition, count]
        for condition, count in measured["conditions_failed_by_group_count"].items()
    ]
    # Both fields may be a count or the list itself depending on the scorer version, so the
    # length is taken when it is a list. Writing the list into a cell produced a single table
    # cell taller than the page, which is how this was noticed.
    for label, key in (("groups scored", "groups_scored"), ("groups excluded", "groups_excluded")):
        value = measured[key]
        rows.append([label, len(value) if isinstance(value, list) else value])
    return write(
        out,
        "Supplementary_Table_S9_failing_conditions.csv",
        ["condition_or_quantity", "group_count"],
        rows,
    )


def table_s10(out):
    record = load("_local/g_8.json")
    rows = [
        [
            c["L"],
            c["mu_over_mu_c"],
            c["mu"],
            c["shots"],
            c["transpiled_depth"],
            c["two_qubit_gates"],
            c["decoded_raw_cosine"],
            c["decoded_mitigated_cosine"],
            c["decoded_raw_total_variation"],
            c["decoded_mitigated_total_variation"],
            c["shot_noise_floor_tv"],
            c["total_variation_above_floor"],
            c["readout_mitigation_applied"],
            c["assignment_diagonal_min"],
        ]
        for c in record["cases"]
    ]
    return write(
        out,
        "Supplementary_Table_S10_hardware_per_point.csv",
        [
            "L",
            "mu_over_mu_c",
            "mu",
            "shots",
            "transpiled_depth",
            "two_qubit_gates",
            "raw_cosine",
            "mitigated_cosine",
            "raw_total_variation",
            "mitigated_total_variation",
            "shot_noise_floor_tv",
            "total_variation_above_floor",
            "mitigation_applied",
            "assignment_diagonal_min",
        ],
        rows,
    )


def table_s11(out):
    text = (ROOT / "PRIOR_ART.md").read_text(encoding="utf-8")
    rows = []
    for match in re.finditer(r"^### ([IVX]+\.\d+[a-z]?) (.+?): `([^`]+)`", text, re.M):
        rows.append([match.group(1), match.group(2).strip(), match.group(3)])
    return write(
        out,
        "Supplementary_Table_S11_prior_art_status.csv",
        ["entry", "citation", "verification_status"],
        rows,
    )


def data_1(out):
    classical = lines("wp7/sweep_registered.jsonl")
    quantum = {
        (
            c["family"],
            c["L"],
            c["mu"],
            c["seed"],
            c.get("K"),
            c.get("roughness"),
            c.get("block_size"),
        ): c
        for c in lines("wp7/sweep_registered_quantum.jsonl")
    }
    rows = []
    for cell in classical:
        key = (
            cell["family"],
            cell["L"],
            cell["mu"],
            cell["seed"],
            cell.get("K"),
            cell.get("roughness"),
            cell.get("block_size"),
        )
        twin = quantum.get(key, {})
        row = [
            cell["family"],
            cell["L"],
            cell.get("K"),
            cell.get("roughness"),
            cell.get("block_size"),
            cell["seed"],
            cell["mu"],
            cell["mu_over_mu_c"],
            cell.get("order_parameter_of_reference"),
        ]
        for method in (
            "baseline_a_wright_fisher",
            "baseline_b_exact_class",
            "baseline_c_tensor_network",
        ):
            entry = cell["methods"].get(method, {})
            row += [
                entry.get("applicable"),
                entry.get("cosine"),
                entry.get("total_variation"),
                entry.get("seconds_used"),
                entry.get("budget_exhausted"),
            ]
        for method in ("route_a_varqite", "route_b_qsvt_filter"):
            entry = twin.get("methods", {}).get(method, {})
            row += [entry.get("applicable"), entry.get("cosine"), entry.get("seconds_used")]
        rows.append(row)
    header = [
        "family",
        "L",
        "K",
        "roughness",
        "block_size",
        "seed",
        "mu",
        "mu_over_mu_c",
        "order_parameter_of_reference",
    ]
    for method in ("wright_fisher", "exact_class", "tensor_network"):
        header += [
            f"{method}_applicable",
            f"{method}_cosine",
            f"{method}_total_variation",
            f"{method}_seconds",
            f"{method}_over_budget",
        ]
    for method in ("route_a_varqite", "route_b_qsvt"):
        header += [f"{method}_applicable", f"{method}_cosine", f"{method}_seconds"]
    return write(out, "Supplementary_Data_1_sweep_per_instance.csv", header, rows)


def data_2(out):
    record = load("_local/g_8.json")
    rows = []
    for c in record["cases"]:
        rows.append(
            [
                c["L"],
                c["mu_over_mu_c"],
                c["mu"],
                c["shots"],
                ";".join(f"{p:.10g}" for p in c["raw_distribution"]),
                ";".join(f"{p:.10g}" for p in c["mitigated_distribution"]),
                c["decoded_raw_cosine"],
                c["decoded_mitigated_cosine"],
                c["shot_noise_floor_tv"],
            ]
        )
    return write(
        out,
        "Supplementary_Data_2_hardware_per_point.csv",
        [
            "L",
            "mu_over_mu_c",
            "mu",
            "shots",
            "raw_distribution",
            "mitigated_distribution",
            "raw_cosine",
            "mitigated_cosine",
            "shot_noise_floor_tv",
        ],
        rows,
    )


BUILDERS = [
    table_s2,
    table_s3,
    table_s4,
    table_s5,
    table_s6,
    table_s7,
    table_s8,
    table_s9,
    table_s10,
    table_s11,
    data_1,
    data_2,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(
            ROOT.parent / "Submission_package" / "Supplementary_Information" / "Tables_and_Data"
        ),
    )
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    built, skipped = 0, 0
    for builder in BUILDERS:
        try:
            print("  " + builder(out))
            built += 1
        except Exception as error:  # noqa: BLE001
            print(f"  {builder.__name__}: SKIPPED, {type(error).__name__}: {error}")
            skipped += 1
    print(f"\n{built} written, {skipped} skipped, into {out}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
