"""Fetch the raw hardware data for a completed job and write it to an archive directory.

The result records hold normalised distributions, not counts. That is enough to reproduce the
published numbers but not enough for anyone to re-derive them differently: a reader who wants
a different mitigation scheme, or a different estimator, needs the measurements themselves.

Job identifiers allow retrieval only while the service, the account and the retention window
all persist, and none of those is guaranteed over the life of an archive. So the counts are
pulled once and deposited. After that the hardware result is reconstructible offline by
`scripts/rescore_hardware.py`, with no network and no account.

    python scripts/archive_hardware.py <job-id> <destination>
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any


def credential() -> str:
    key = pathlib.Path("G:/My Drive/Qubis_HiQ/QUASAR/IBM/apikey.json")
    if not key.is_file():
        raise SystemExit(f"no credential at {key}")
    return str(json.loads(key.read_text(encoding="utf-8"))["apikey"])


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    return str(value)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    job_id, destination = sys.argv[1], pathlib.Path(sys.argv[2])
    destination.mkdir(parents=True, exist_ok=True)

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=credential())
    job = service.job(job_id)
    print(f"job {job_id}: {job.status()}")

    result = job.result()
    counts = []
    for index, pub in enumerate(result):
        data = pub.data
        register = next(iter(data.__dict__)) if hasattr(data, "__dict__") else "meas"
        bits = getattr(data, register)
        counts.append({"index": index, "register": register, "counts": bits.get_counts()})
    (destination / "raw_counts.json").write_text(
        json.dumps({"job_id": job_id, "pubs": counts}, indent=2), encoding="utf-8"
    )
    total = sum(sum(c["counts"].values()) for c in counts)
    print(f"  {len(counts)} circuits, {total} measurements -> raw_counts.json")

    metadata = {"job_id": job_id, "status": str(job.status())}
    for field, getter in (
        ("backend", lambda: job.backend().name),
        ("creation_date", lambda: job.creation_date),
        ("metrics", lambda: job.metrics()),
        ("usage_estimation", lambda: job.usage_estimation),
        ("inputs", lambda: {k: v for k, v in job.inputs.items() if k != "pubs"}),
    ):
        try:
            metadata[field] = jsonable(getter())
        except Exception as error:  # noqa: BLE001
            metadata[field] = f"unavailable: {type(error).__name__}"
    (destination / "job_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("  job_metadata.json")

    try:
        backend = service.backend(metadata["backend"])
        properties: dict[str, Any] = {
            "name": backend.name,
            "num_qubits": backend.num_qubits,
            "basis_gates": sorted(backend.basis_gates),
        }
        target = backend.target
        properties["qubit_properties"] = [
            {
                "qubit": q,
                "t1": target.qubit_properties[q].t1,
                "t2": target.qubit_properties[q].t2,
                "frequency": target.qubit_properties[q].frequency,
            }
            for q in range(backend.num_qubits)
        ]
        errors = {}
        for gate in ("cz", "ecr", "cx", "sx", "x", "measure"):
            if gate in target:
                errors[gate] = {
                    ",".join(map(str, k)): v.error
                    for k, v in target[gate].items()
                    if v is not None and v.error is not None
                }
        properties["gate_errors"] = errors
        (destination / "backend_properties.json").write_text(
            json.dumps(jsonable(properties), indent=2), encoding="utf-8"
        )
        print("  backend_properties.json")
    except Exception as error:  # noqa: BLE001
        print(f"  backend properties unavailable: {type(error).__name__}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
