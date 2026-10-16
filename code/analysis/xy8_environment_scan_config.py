"""Validation and freezing helpers for the preregistered environment scan."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT
    / "data"
    / "raw"
    / "nv"
    / "environment_literature"
    / "xy8_environment_scan_config.json"
)
DEFAULT_FREEZE_RECORD = DEFAULT_CONFIG.with_name("xy8_environment_scan_freeze.json")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_sorted_unique(name: str, values: list[float | int]) -> None:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional list.")
    if not np.all(np.diff(array) > 0.0):
        raise ValueError(f"{name} must be strictly increasing and unique.")


def _resolved_dependency_path(config_path: Path, declared_path: str) -> Path:
    path = Path(declared_path)
    if path.is_absolute():
        return path
    return ROOT / path


def validate_config(
    config: dict[str, Any],
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    verify_dependencies: bool = True,
) -> dict[str, Any]:
    """Validate scientific conventions and all frozen dependency digests."""
    if config.get("freeze_status") != "predeclared_before_environment_conditioned_edt_analysis":
        raise ValueError("Configuration does not declare the required preregistration status.")
    preregistration = config["preregistration"]
    if preregistration["environment_conditioned_edt_results_inspected_before_freeze"] is not False:
        raise ValueError("The scan must be frozen before inspecting conditioned results.")
    if preregistration["post_freeze_parameter_tuning_allowed"] is not False:
        raise ValueError("Post-freeze parameter tuning must be disabled.")

    temperatures = config["temperature_axes"]
    for name, axis in temperatures.items():
        _require_sorted_unique(f"temperature_axes.{name}.values_K", axis["values_K"])
        if 300.0 not in axis["values_K"]:
            raise ValueError(f"Temperature axis {name} must contain the 300 K reference.")
    operational = temperatures["operational"]["values_K"]
    if min(operational) < 280.0 or max(operational) > 330.0:
        raise ValueError("The operational temperature axis must remain within 280-330 K.")
    extended = temperatures["extended_stress_test"]["values_K"]
    if min(extended) < 100.0 or max(extended) > 400.0:
        raise ValueError("The extended axis would extrapolate the digitized Mondal curve.")

    amplitudes = config["strain_axes"]["amplitudes_microstrain"]
    _require_sorted_unique("strain_axes.amplitudes_microstrain", amplitudes)
    if 0.0 not in amplitudes or min(amplitudes) != -100.0 or max(amplitudes) != 100.0:
        raise ValueError("Version 1 strain amplitudes must span -100 to 100 microstrain and include zero.")
    paths = config["strain_axes"]["paths"]
    expected_paths = {"zero_strain", "axial_zz", "transverse_differential", "shear_xz"}
    if {path["id"] for path in paths} != expected_paths:
        raise ValueError("Version 1 must declare exactly the four preregistered strain paths.")
    for path in paths:
        tensor = np.asarray(path["tensor_per_unit_amplitude"], dtype=float)
        if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
            raise ValueError(f"Strain path {path['id']} must contain a finite 3-by-3 tensor.")
        if not np.allclose(tensor, tensor.T, atol=0.0, rtol=0.0):
            raise ValueError(f"Strain path {path['id']} must define a symmetric tensor.")

    signal = config["signal_and_control"]
    frequency = float(signal["target_frequency_Hz"])
    expected_tau = 1.0 / (4.0 * frequency)
    if not math.isclose(signal["primary_resonant_tau_s"], expected_tau, rel_tol=1e-12):
        raise ValueError("primary_resonant_tau_s is inconsistent with 1/(4*f).")
    if signal["depth_convention"] != "N counts complete XY8 blocks and the total pi-pulse count is 8N":
        raise ValueError("The canonical XY8 depth convention must be stated exactly.")

    depths = config["control_grids"]["depths_N"]
    _require_sorted_unique("control_grids.depths_N", depths)
    if depths[0] != 1 or depths[-1] != 128 or 20 not in depths:
        raise ValueError("The depth sweep must include N=1, N=20, and N=128.")
    for grid_name in ("discovery_timing_grid", "confirmation_timing_grid"):
        grid = config["control_grids"][grid_name]
        if not grid["insert_exact_odd_harmonic_resonances"]:
            raise ValueError(f"{grid_name} must include exact resonances.")
        if not grid["tau_min_s"] < expected_tau < grid["tau_max_s"]:
            raise ValueError(f"{grid_name} does not bracket the primary resonance.")

    policies = {policy["id"] for policy in config["carrier_policies"]}
    if policies != {"fixed_reference_carrier", "ideal_retracked_carrier"}:
        raise ValueError("Both preregistered carrier policies are required.")
    limits = config["topology_analysis"]["relative_limits"]
    _require_sorted_unique("topology_analysis.relative_limits", limits)
    if config["topology_analysis"]["primary_relative_limit"] != 1.1:
        raise ValueError("The primary EDT threshold must remain 10 percent.")
    if config["topology_analysis"]["connectivity_primary"] != 4:
        raise ValueError("Four-neighbour connectivity must be the primary definition.")

    dependencies = []
    if verify_dependencies:
        config_file = Path(config_path)
        frozen = config["frozen_dependencies"]
        for name in ("nominal_hardware_scenario", "canonical_reference_surface"):
            record = frozen[name]
            path = _resolved_dependency_path(config_file, record["path"])
            actual = file_sha256(path)
            if actual != record["sha256"]:
                raise ValueError(f"Frozen dependency digest mismatch for {name}: {path}")
            dependencies.append({"id": name, "path": record["path"], "sha256": actual})

        literature_record = frozen["literature_bundle"]
        literature_path = _resolved_dependency_path(config_file, literature_record["path"])
        literature = json.loads(literature_path.read_text(encoding="utf-8"))
        if literature["bundle_id"] != literature_record["bundle_id"]:
            raise ValueError("Frozen literature bundle id does not match the scan declaration.")
        if literature["bundle_sha256"] != literature_record["bundle_sha256"]:
            raise ValueError("Frozen literature aggregate digest does not match the scan declaration.")
        dependencies.append(
            {
                "id": "literature_bundle_record",
                "path": literature_record["path"],
                "sha256": file_sha256(literature_path),
                "bundle_sha256": literature["bundle_sha256"],
            }
        )

    return {
        "valid": True,
        "configuration_id": config["configuration_id"],
        "operational_temperature_count": len(operational),
        "extended_temperature_count": len(extended),
        "strain_amplitude_count": len(amplitudes),
        "strain_path_count": len(paths),
        "carrier_policy_count": len(policies),
        "depth_count": len(depths),
        "dependencies": dependencies,
    }


def build_freeze_record(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(config_path)
    config = load_config(path)
    validation = validate_config(config, config_path=path, verify_dependencies=True)
    return {
        "schema_version": "1.0",
        "configuration_id": config["configuration_id"],
        "freeze_status": config["freeze_status"],
        "frozen_at": config["freeze_date"],
        "hash_algorithm": "sha256",
        "configuration_path": str(path.relative_to(ROOT)),
        "configuration_sha256": file_sha256(path),
        "frozen_dependencies": validation["dependencies"],
        "analysis_started": False,
    }


def verify_freeze_record(
    config_path: str | Path = DEFAULT_CONFIG,
    freeze_path: str | Path = DEFAULT_FREEZE_RECORD,
) -> dict[str, Any]:
    path = Path(config_path)
    expected = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    actual = build_freeze_record(path)
    checks = {
        "configuration_id": expected.get("configuration_id") == actual["configuration_id"],
        "freeze_status": expected.get("freeze_status") == actual["freeze_status"],
        "configuration_sha256": expected.get("configuration_sha256") == actual["configuration_sha256"],
        "frozen_dependencies": expected.get("frozen_dependencies") == actual["frozen_dependencies"],
        "analysis_not_started_at_freeze": expected.get("analysis_started") is False,
    }
    return {
        "verified": all(checks.values()),
        "configuration_id": actual["configuration_id"],
        "configuration_sha256": actual["configuration_sha256"],
        "checks": checks,
    }
