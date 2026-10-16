"""Frozen literature inputs for strain- and temperature-conditioned NV analyses.

This module intentionally stops short of modifying the EDT/XY8 score model.  It
provides a provenance-preserving boundary between literature extraction and the
later environment-conditioned analysis.  Experimental validation records are
loaded separately from parameter inputs so that they cannot be used silently
to tune the frozen laws.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "nv" / "environment_literature"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "nv_environment_literature_manifest.json"
DEFAULT_FREEZE_RECORD = DEFAULT_DATA_DIR / "nv_environment_literature_freeze.json"


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without changing it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_temperature_laws(
    path: str | Path = DEFAULT_DATA_DIR / "nv_temperature_laws.json",
) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bose_occupation(temperature_K: float | np.ndarray, energy_meV: float) -> np.ndarray:
    """Bose occupation for an excitation energy expressed in meV."""
    temperatures = np.asarray(temperature_K, dtype=float)
    energy = float(energy_meV)
    if np.any(~np.isfinite(temperatures)) or np.any(temperatures < 0.0):
        raise ValueError("temperature_K must contain finite, non-negative values.")
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy_meV must be positive and finite.")

    result = np.zeros_like(temperatures, dtype=float)
    positive = temperatures > 0.0
    k_b = load_temperature_laws()["constants"]["boltzmann_meV_per_K"]
    result[positive] = 1.0 / np.expm1(energy / (k_b * temperatures[positive]))
    return result


def cambria_zfs_MHz(
    temperature_K: float | np.ndarray,
    laws: dict[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate the frozen Cambria two-representative-phonon D(T) law."""
    if laws is None:
        laws = load_temperature_laws()
    parameters = laws["zfs_two_mode_law"]["parameters"]
    temperatures = np.asarray(temperature_K, dtype=float)
    d0 = parameters["D0_MHz"]["value"]
    c1 = parameters["c1_MHz"]["value"]
    c2 = parameters["c2_MHz"]["value"]
    delta1 = parameters["delta1_meV"]["value"]
    delta2 = parameters["delta2_meV"]["value"]
    return d0 + c1 * bose_occupation(temperatures, delta1) + c2 * bose_occupation(
        temperatures, delta2
    )


def cambria_zfs_shift_MHz(
    temperature_K: float | np.ndarray,
    *,
    reference_temperature_K: float = 300.0,
    laws: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return D(T)-D(T_reference), avoiding use of an arbitrary absolute offset."""
    return cambria_zfs_MHz(temperature_K, laws) - cambria_zfs_MHz(
        reference_temperature_K, laws
    )


def numerical_slope_kHz_per_K(
    function,
    temperature_K: float,
    *,
    step_K: float = 0.01,
) -> float:
    """Evaluate a central finite-difference temperature slope."""
    temperature = float(temperature_K)
    step = float(step_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be positive and finite.")
    if not math.isfinite(step) or step <= 0.0 or step >= temperature:
        raise ValueError("step_K must be positive, finite, and below temperature_K.")
    slope_MHz_per_K = (
        float(function(temperature + step)) - float(function(temperature - step))
    ) / (2.0 * step)
    return 1000.0 * slope_MHz_per_K


def interval_slope_kHz_per_K(function, lower_K: float, upper_K: float) -> float:
    """Return the secant slope over a declared temperature interval."""
    lower = float(lower_K)
    upper = float(upper_K)
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 <= lower < upper):
        raise ValueError("Require finite temperatures with 0 <= lower_K < upper_K.")
    return 1000.0 * (float(function(upper)) - float(function(lower))) / (upper - lower)


def jarmola_t1_rate_s_inv(
    temperature_K: float | np.ndarray,
    *,
    sample: str = "S8",
    laws: dict[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate the frozen Jarmola experimental ensemble-NV T1(T) law."""
    if laws is None:
        laws = load_temperature_laws()
    record = laws["jarmola_t1_law"]
    shared = record["shared_parameters"]
    try:
        a1 = record["sample_A1_s_inv"][sample]["value"]
    except KeyError as error:
        choices = ", ".join(sorted(record["sample_A1_s_inv"]))
        raise ValueError(f"Unknown sample {sample!r}; choose one of {choices}.") from error
    temperatures = np.asarray(temperature_K, dtype=float)
    if np.any(~np.isfinite(temperatures)) or np.any(temperatures < 0.0):
        raise ValueError("temperature_K must contain finite, non-negative values.")
    a2 = shared["A2_s_inv"]["value"]
    delta = shared["delta_meV"]["value"]
    a3 = shared["A3_K_minus5_s_inv"]["value"]
    return a1 + a2 * bose_occupation(temperatures, delta) + a3 * temperatures**5


def load_spin_strain_coefficients(
    path: str | Path = DEFAULT_DATA_DIR / "udvarhelyi_2018_spin_strain_coefficients.csv",
) -> dict[str, dict[str, float | str]]:
    """Load the six directly tabulated spin-strain coefficients."""
    records: dict[str, dict[str, float | str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records[row["coefficient"]] = {
                "value_MHz_per_strain": float(row["strain_value_MHz_per_strain"]),
                "uncertainty_MHz_per_strain": float(
                    row["strain_uncertainty_MHz_per_strain"]
                ),
                "stress_coefficient": row["stress_coefficient"],
                "stress_value_MHz_per_GPa": float(row["stress_value_MHz_per_GPa"]),
                "stress_uncertainty_MHz_per_GPa": float(
                    row["stress_uncertainty_MHz_per_GPa"]
                ),
                "hamiltonian_sector": row["hamiltonian_sector"],
            }
    expected = {"h43", "h41", "h25", "h26", "h15", "h16"}
    if set(records) != expected:
        raise ValueError(f"Expected coefficients {sorted(expected)}, received {sorted(records)}.")
    return records


def _spin_one_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_two = math.sqrt(2.0)
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / root_two
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / root_two
    sz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    return sx, sy, sz


def _anticommutator(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left @ right + right @ left


def spin_strain_hamiltonian_MHz(
    strain_tensor: Iterable[Iterable[float]],
    coefficients: dict[str, dict[str, float | str]] | None = None,
) -> np.ndarray:
    """Return H_strain/h in MHz for a symmetric NV-frame strain tensor.

    The implementation follows Eqs. (3)-(5) of Udvarhelyi et al. and uses the
    directly tabulated ``h`` coefficients.  Tensor axes must be expressed in
    the NV reference frame used by that work.
    """
    strain = np.asarray(strain_tensor, dtype=float)
    if strain.shape != (3, 3) or not np.all(np.isfinite(strain)):
        raise ValueError("strain_tensor must be a finite 3-by-3 array.")
    if not np.allclose(strain, strain.T, rtol=0.0, atol=1e-14):
        raise ValueError("strain_tensor must be symmetric.")
    if coefficients is None:
        coefficients = load_spin_strain_coefficients()
    h = {name: float(record["value_MHz_per_strain"]) for name, record in coefficients.items()}
    exx, eyy, ezz = strain[0, 0], strain[1, 1], strain[2, 2]
    exy, ezx, eyz = strain[0, 1], strain[2, 0], strain[1, 2]
    sx, sy, sz = _spin_one_matrices()

    h0 = (h["h41"] * (exx + eyy) + h["h43"] * ezz) * (sz @ sz)
    h1 = 0.5 * (h["h26"] * ezx - 0.5 * h["h25"] * (exx - eyy)) * _anticommutator(sx, sz)
    h1 += 0.5 * (h["h26"] * eyz + h["h25"] * exy) * _anticommutator(sy, sz)
    h2 = 0.5 * (h["h16"] * ezx - 0.5 * h["h15"] * (exx - eyy)) * (sy @ sy - sx @ sx)
    h2 += 0.5 * (h["h16"] * eyz + h["h15"] * exy) * _anticommutator(sx, sy)
    result = h0 + h1 + h2
    if not np.allclose(result, result.conj().T, rtol=0.0, atol=1e-12):
        raise RuntimeError("Constructed spin-strain Hamiltonian is not Hermitian.")
    return result


def _load_numeric_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_mondal_t1_curve(
    path: str | Path = DEFAULT_DATA_DIR / "mondal_2023_fig3_nv_t1_dft.csv",
) -> list[dict[str, float | str]]:
    """Load digitized vector coordinates and their extraction uncertainty."""
    result: list[dict[str, float | str]] = []
    for row in _load_numeric_csv(path):
        result.append(
            {
                "temperature_K": float(row["temperature_K"]),
                "t1_rate_s_inv": float(row["t1_rate_s_inv"]),
                "temperature_digitization_uncertainty_K": float(
                    row["temperature_digitization_uncertainty_K"]
                ),
                "t1_rate_digitization_uncertainty_s_inv": float(
                    row["t1_rate_digitization_uncertainty_s_inv"]
                ),
                "series": row["series"],
                "source_figure": row["source_figure"],
                "extraction_status": row["extraction_status"],
            }
        )
    return result


def load_acosta_validation(
    path: str | Path = DEFAULT_DATA_DIR / "acosta_2011_zfs_slope_validation.csv",
) -> dict[str, float | str]:
    """Load the validation-only corrected experimental D(T) slope."""
    rows = _load_numeric_csv(path)
    if len(rows) != 1:
        raise ValueError("Expected exactly one Acosta validation record.")
    row = rows[0]
    return {
        "temperature_min_K": float(row["temperature_min_K"]),
        "temperature_max_K": float(row["temperature_max_K"]),
        "dD_dT_kHz_per_K": float(row["dD_dT_kHz_per_K"]),
        "uncertainty_1sigma_kHz_per_K": float(row["uncertainty_1sigma_kHz_per_K"]),
        "dataset_role": row["dataset_role"],
        "source_location": row["source_location"],
        "correction_status": row["correction_status"],
    }


def validate_tang_against_acosta(
    laws: dict[str, Any] | None = None,
    validation: dict[str, float | str] | None = None,
) -> dict[str, float | bool | str]:
    """Compare an untuned first-principles slope with held-out experiment."""
    if laws is None:
        laws = load_temperature_laws()
    if validation is None:
        validation = load_acosta_validation()
    predicted = float(laws["first_principles_room_temperature_slope"]["value_kHz_per_K"])
    measured = float(validation["dD_dT_kHz_per_K"])
    sigma = float(validation["uncertainty_1sigma_kHz_per_K"])
    absolute_error = abs(predicted - measured)
    standardized_error = absolute_error / sigma
    return {
        "prediction_source": laws["first_principles_room_temperature_slope"]["source_id"],
        "validation_source": "acosta_2011_zfs_erratum",
        "predicted_kHz_per_K": predicted,
        "measured_kHz_per_K": measured,
        "measurement_uncertainty_1sigma_kHz_per_K": sigma,
        "absolute_error_kHz_per_K": absolute_error,
        "absolute_error_in_measurement_sigma": standardized_error,
        "within_two_measurement_sigma": standardized_error <= 2.0,
        "separation_status": "validation_not_used_for_parameter_estimation",
    }


def bundle_digest(entries: list[dict[str, Any]]) -> str:
    """Return a deterministic aggregate digest from sorted file records."""
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_freeze_record(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    frozen_at: str | None = None,
) -> dict[str, Any]:
    """Compute the immutable file inventory declared by the source manifest."""
    manifest_file = Path(manifest_path)
    manifest = load_manifest(manifest_file)
    entries = []
    for relative_path in manifest["files_to_freeze"]:
        path = manifest_file.parent / relative_path
        entries.append(
            {
                "path": relative_path,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    record = {
        "schema_version": 1,
        "bundle_id": manifest["bundle_id"],
        "freeze_status": manifest["freeze_status"],
        "frozen_at": frozen_at or manifest["freeze_date"],
        "hash_algorithm": "sha256",
        "files": sorted(entries, key=lambda item: item["path"]),
    }
    record["bundle_sha256"] = bundle_digest(record["files"])
    return record


def verify_frozen_bundle(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    freeze_record_path: str | Path = DEFAULT_FREEZE_RECORD,
) -> dict[str, Any]:
    """Verify every frozen file and the aggregate digest."""
    expected = json.loads(Path(freeze_record_path).read_text(encoding="utf-8"))
    actual = build_freeze_record(manifest_path, frozen_at=expected["frozen_at"])
    file_status = []
    expected_by_path = {item["path"]: item for item in expected["files"]}
    actual_by_path = {item["path"]: item for item in actual["files"]}
    for path in sorted(set(expected_by_path) | set(actual_by_path)):
        file_status.append(
            {
                "path": path,
                "matches": expected_by_path.get(path) == actual_by_path.get(path),
            }
        )
    verified = (
        expected.get("bundle_id") == actual.get("bundle_id")
        and expected.get("freeze_status") == actual.get("freeze_status")
        and expected.get("bundle_sha256") == actual.get("bundle_sha256")
        and all(item["matches"] for item in file_status)
    )
    return {
        "verified": verified,
        "bundle_id": actual["bundle_id"],
        "bundle_sha256": actual["bundle_sha256"],
        "expected_bundle_sha256": expected.get("bundle_sha256"),
        "files": file_status,
    }
