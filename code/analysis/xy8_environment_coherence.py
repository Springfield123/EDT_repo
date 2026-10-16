"""Anchored temperature-dependent coherence for the frozen NV environment scan.

The nominal ``T2_1`` already represents total coherence loss at 300 K.  This
module therefore does not multiply the existing envelope by another complete
T1 decay.  Instead it freezes a residual dephasing rate for each independent
T1 source and varies only the longitudinal contribution away from 300 K:

    1/T2_base(T) = 1/T2_reference + [R1(T) - R1(T_reference)]/2.

The experimental Jarmola law and digitized first-principles Mondal curve are
kept as separate variants and are never averaged or fitted to EDT results.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

import xy8_environment_scan_config as scan_config
import xy8_nv_environment_literature as literature


JARMOLA_VARIANT = "jarmola_experimental_S8"
MONDAL_VARIANT = "mondal_first_principles_digitized"
SUPPORTED_VARIANTS = (JARMOLA_VARIANT, MONDAL_VARIANT)


@dataclass(frozen=True)
class T1RateEstimate:
    temperature_K: float
    variant: str
    rate_s_inv: float
    uncertainty_1sigma_s_inv: float
    t1_s: float
    t1_uncertainty_1sigma_s: float
    source_temperature_range_K: tuple[float, float]
    uncertainty_scope: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AnchoredCoherence:
    temperature_K: float
    reference_temperature_K: float
    variant: str
    reference_T2_s: float
    t1_rate_s_inv: float
    reference_t1_rate_s_inv: float
    residual_dephasing_rate_s_inv: float
    effective_T2_s: float
    effective_T2_uncertainty_1sigma_s: float
    t1_s: float
    t1_uncertainty_1sigma_s: float
    warnings: tuple[str, ...]
    uncertainty_scope: str


def _validate_temperature(temperature_K: float) -> float:
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be positive and finite.")
    return temperature


def _range_guard(
    temperature_K: float,
    source_range_K: tuple[float, float],
    *,
    variant: str,
    allow_extrapolation: bool,
) -> tuple[str, ...]:
    lower, upper = source_range_K
    if lower <= temperature_K <= upper:
        return ()
    message = (
        f"temperature_outside_{variant}_source_range:"
        f"{temperature_K:g}K_not_in_{lower:g}-{upper:g}K"
    )
    if not allow_extrapolation:
        raise ValueError(message + "; extrapolation is disabled")
    return (message,)


def _jarmola_rate_and_parameter_gradient(
    temperature_K: float,
    *,
    sample: str,
    laws: dict[str, Any],
) -> tuple[float, float, dict[str, float]]:
    record = laws["jarmola_t1_law"]
    shared = record["shared_parameters"]
    try:
        a1_record = record["sample_A1_s_inv"][sample]
    except KeyError as error:
        choices = ", ".join(sorted(record["sample_A1_s_inv"]))
        raise ValueError(f"Unknown Jarmola sample {sample!r}; choose one of {choices}.") from error
    k_b = float(laws["constants"]["boltzmann_meV_per_K"])
    a1 = float(a1_record["value"])
    a2 = float(shared["A2_s_inv"]["value"])
    delta = float(shared["delta_meV"]["value"])
    a3 = float(shared["A3_K_minus5_s_inv"]["value"])
    occupation = float(literature.bose_occupation(temperature_K, delta))
    rate = a1 + a2 * occupation + a3 * temperature_K**5
    derivative_delta = -a2 * occupation * (occupation + 1.0) / (k_b * temperature_K)
    gradient = {
        "A1": 1.0,
        "A2": occupation,
        "delta": derivative_delta,
        "A3": temperature_K**5,
    }
    variances = {
        "A1": float(a1_record["uncertainty_1sigma"]) ** 2,
        "A2": float(shared["A2_s_inv"]["uncertainty_1sigma"]) ** 2,
        "delta": float(shared["delta_meV"]["uncertainty_1sigma"]) ** 2,
        "A3": float(shared["A3_K_minus5_s_inv"]["uncertainty_1sigma"]) ** 2,
    }
    uncertainty = math.sqrt(
        sum(gradient[name] ** 2 * variances[name] for name in gradient)
    )
    return rate, uncertainty, gradient


def jarmola_t1_rate(
    temperature_K: float,
    *,
    sample: str = "S8",
    laws: dict[str, Any] | None = None,
    allow_extrapolation: bool = False,
) -> T1RateEstimate:
    """Evaluate the experimental analytical T1 law with first-order uncertainty."""
    temperature = _validate_temperature(temperature_K)
    if laws is None:
        laws = literature.load_temperature_laws()
    source_range = tuple(float(value) for value in laws["jarmola_t1_law"]["temperature_range_K"])
    messages = _range_guard(
        temperature,
        source_range,
        variant="jarmola",
        allow_extrapolation=allow_extrapolation,
    )
    rate, uncertainty, _ = _jarmola_rate_and_parameter_gradient(
        temperature, sample=sample, laws=laws
    )
    return T1RateEstimate(
        temperature_K=temperature,
        variant=f"jarmola_experimental_{sample}",
        rate_s_inv=rate,
        uncertainty_1sigma_s_inv=uncertainty,
        t1_s=1.0 / rate,
        t1_uncertainty_1sigma_s=uncertainty / rate**2,
        source_temperature_range_K=source_range,
        uncertainty_scope=(
            "First-order propagation of reported A1, A2, delta, and A3 uncertainties, "
            "treated as independent because source covariance is unavailable."
        ),
        warnings=messages,
    )


def _mondal_curve_arrays(curve: list[dict[str, float | str]]):
    temperatures = np.array([float(row["temperature_K"]) for row in curve])
    rates = np.array([float(row["t1_rate_s_inv"]) for row in curve])
    temperature_uncertainties = np.array(
        [float(row["temperature_digitization_uncertainty_K"]) for row in curve]
    )
    rate_uncertainties = np.array(
        [float(row["t1_rate_digitization_uncertainty_s_inv"]) for row in curve]
    )
    return temperatures, rates, temperature_uncertainties, rate_uncertainties


def _log_interpolation_segment(temperatures: np.ndarray, temperature_K: float) -> tuple[int, int, float]:
    if temperature_K <= temperatures[0]:
        lower, upper = 0, 1
    elif temperature_K >= temperatures[-1]:
        lower, upper = len(temperatures) - 2, len(temperatures) - 1
    else:
        upper = int(np.searchsorted(temperatures, temperature_K, side="right"))
        lower = upper - 1
    log_lower = math.log(float(temperatures[lower]))
    log_upper = math.log(float(temperatures[upper]))
    weight = (math.log(temperature_K) - log_lower) / (log_upper - log_lower)
    return lower, upper, weight


def mondal_t1_rate(
    temperature_K: float,
    *,
    curve: list[dict[str, float | str]] | None = None,
    allow_extrapolation: bool = False,
) -> T1RateEstimate:
    """Log-log interpolate the digitized first-principles curve and uncertainty."""
    temperature = _validate_temperature(temperature_K)
    if curve is None:
        curve = literature.load_mondal_t1_curve()
    temperatures, rates, sigma_t, sigma_r = _mondal_curve_arrays(curve)
    source_range = (float(temperatures[0]), float(temperatures[-1]))
    messages = _range_guard(
        temperature,
        source_range,
        variant="mondal",
        allow_extrapolation=allow_extrapolation,
    )
    lower, upper, weight = _log_interpolation_segment(temperatures, temperature)
    log_rate = (1.0 - weight) * math.log(float(rates[lower])) + weight * math.log(
        float(rates[upper])
    )
    rate = math.exp(log_rate)

    # Coordinate-extraction uncertainty is propagated locally. Vertical
    # uncertainty is interpolated in relative units; horizontal uncertainty is
    # converted to rate uncertainty using the local log-log slope.
    vertical_relative = math.sqrt(
        ((1.0 - weight) * sigma_r[lower] / rates[lower]) ** 2
        + (weight * sigma_r[upper] / rates[upper]) ** 2
    )
    local_log_slope = (
        math.log(float(rates[upper])) - math.log(float(rates[lower]))
    ) / (math.log(float(temperatures[upper])) - math.log(float(temperatures[lower])))
    interpolated_sigma_t = math.sqrt(
        ((1.0 - weight) * sigma_t[lower]) ** 2 + (weight * sigma_t[upper]) ** 2
    )
    horizontal_relative = abs(local_log_slope) * interpolated_sigma_t / temperature
    uncertainty = rate * math.sqrt(vertical_relative**2 + horizontal_relative**2)
    return T1RateEstimate(
        temperature_K=temperature,
        variant=MONDAL_VARIANT,
        rate_s_inv=rate,
        uncertainty_1sigma_s_inv=uncertainty,
        t1_s=1.0 / rate,
        t1_uncertainty_1sigma_s=uncertainty / rate**2,
        source_temperature_range_K=source_range,
        uncertainty_scope=(
            "First-order coordinate-extraction uncertainty from the digitized vector curve; "
            "does not include first-principles model discrepancy."
        ),
        warnings=messages,
    )


def t1_rate(
    temperature_K: float,
    *,
    variant: str,
    laws: dict[str, Any] | None = None,
    curve: list[dict[str, float | str]] | None = None,
    allow_extrapolation: bool = False,
) -> T1RateEstimate:
    if variant == JARMOLA_VARIANT:
        return jarmola_t1_rate(
            temperature_K,
            sample="S8",
            laws=laws,
            allow_extrapolation=allow_extrapolation,
        )
    if variant == MONDAL_VARIANT:
        return mondal_t1_rate(
            temperature_K,
            curve=curve,
            allow_extrapolation=allow_extrapolation,
        )
    raise ValueError(f"Unknown T1 variant {variant!r}; choose one of {SUPPORTED_VARIANTS}.")


def _jarmola_rate_difference_uncertainty(
    temperature_K: float,
    reference_temperature_K: float,
    *,
    sample: str,
    laws: dict[str, Any],
) -> float:
    record = laws["jarmola_t1_law"]
    shared = record["shared_parameters"]
    _, _, gradient = _jarmola_rate_and_parameter_gradient(
        temperature_K, sample=sample, laws=laws
    )
    _, _, reference_gradient = _jarmola_rate_and_parameter_gradient(
        reference_temperature_K, sample=sample, laws=laws
    )
    variances = {
        "A1": float(record["sample_A1_s_inv"][sample]["uncertainty_1sigma"]) ** 2,
        "A2": float(shared["A2_s_inv"]["uncertainty_1sigma"]) ** 2,
        "delta": float(shared["delta_meV"]["uncertainty_1sigma"]) ** 2,
        "A3": float(shared["A3_K_minus5_s_inv"]["uncertainty_1sigma"]) ** 2,
    }
    return math.sqrt(
        sum(
            (gradient[name] - reference_gradient[name]) ** 2 * variances[name]
            for name in gradient
        )
    )


def anchored_coherence(
    temperature_K: float,
    *,
    variant: str,
    config: dict[str, Any] | None = None,
    laws: dict[str, Any] | None = None,
    curve: list[dict[str, float | str]] | None = None,
    allow_extrapolation: bool = False,
) -> AnchoredCoherence:
    """Return temperature-conditioned base T2 while preserving its 300 K value."""
    temperature = _validate_temperature(temperature_K)
    if config is None:
        config = scan_config.load_config()
    if laws is None:
        laws = literature.load_temperature_laws()
    reference_temperature = float(config["reference_environment"]["temperature_K"])
    reference_t2 = float(config["relaxation_policy"]["reference_T2_s"])
    current = t1_rate(
        temperature,
        variant=variant,
        laws=laws,
        curve=curve,
        allow_extrapolation=allow_extrapolation,
    )
    reference = t1_rate(
        reference_temperature,
        variant=variant,
        laws=laws,
        curve=curve,
        allow_extrapolation=False,
    )
    residual_dephasing_rate = 1.0 / reference_t2 - 0.5 * reference.rate_s_inv
    if residual_dephasing_rate <= 0.0:
        raise ValueError(
            "The selected T1 source is incompatible with the reference T2: "
            "the inferred residual dephasing rate is not positive."
        )
    effective_rate = residual_dephasing_rate + 0.5 * current.rate_s_inv
    effective_t2 = 1.0 / effective_rate
    if variant == JARMOLA_VARIANT:
        difference_uncertainty = _jarmola_rate_difference_uncertainty(
            temperature, reference_temperature, sample="S8", laws=laws
        )
        uncertainty_scope = (
            "Correlated first-order propagation of the reported Jarmola parameter "
            "uncertainties through R1(T)-R1(300 K); the sample-specific A1 cancels."
        )
    else:
        if math.isclose(temperature, reference_temperature, rel_tol=0.0, abs_tol=1e-12):
            difference_uncertainty = 0.0
        else:
            difference_uncertainty = math.sqrt(
                current.uncertainty_1sigma_s_inv**2
                + reference.uncertainty_1sigma_s_inv**2
            )
        uncertainty_scope = (
            "Conservative independent propagation of digitized rate-coordinate "
            "uncertainties at T and 300 K; shared-curve covariance is unavailable."
        )
    effective_rate_uncertainty = 0.5 * difference_uncertainty
    effective_t2_uncertainty = effective_rate_uncertainty / effective_rate**2
    messages = tuple(dict.fromkeys(current.warnings + reference.warnings))
    return AnchoredCoherence(
        temperature_K=temperature,
        reference_temperature_K=reference_temperature,
        variant=variant,
        reference_T2_s=reference_t2,
        t1_rate_s_inv=current.rate_s_inv,
        reference_t1_rate_s_inv=reference.rate_s_inv,
        residual_dephasing_rate_s_inv=residual_dephasing_rate,
        effective_T2_s=effective_t2,
        effective_T2_uncertainty_1sigma_s=effective_t2_uncertainty,
        t1_s=current.t1_s,
        t1_uncertainty_1sigma_s=current.t1_uncertainty_1sigma_s,
        warnings=messages,
        uncertainty_scope=uncertainty_scope,
    )


def temperature_conditioned_t2_params(
    base_params: dict[str, Any],
    temperature_K: float,
    *,
    variant: str,
    config: dict[str, Any] | None = None,
    laws: dict[str, Any] | None = None,
    curve: list[dict[str, float | str]] | None = None,
) -> tuple[dict[str, Any], AnchoredCoherence]:
    """Replace only T2_1 while preserving depth/timing scaling parameters."""
    required = {"T2_1", "alpha", "beta", "tau0", "p"}
    missing = required - set(base_params)
    if missing:
        raise ValueError("base_params is missing: " + ", ".join(sorted(missing)))
    result = anchored_coherence(
        temperature_K,
        variant=variant,
        config=config,
        laws=laws,
        curve=curve,
    )
    conditioned = dict(base_params)
    conditioned["T2_1"] = result.effective_T2_s
    conditioned["T1_limit_s"] = result.t1_s
    conditioned["environment_coherence_variant"] = variant
    conditioned["environment_temperature_K"] = float(temperature_K)
    conditioned["environment_T2_1_uncertainty_1sigma_s"] = (
        result.effective_T2_uncertainty_1sigma_s
    )
    conditioned["environment_anchoring_rule"] = (
        "1/T2_1(T)=1/T2_1(300K)+[R1(T)-R1(300K)]/2"
    )
    return conditioned, result


def sequence_coherence_time_s(
    blocks: int,
    tau_s: float,
    base_params: dict[str, Any],
    temperature_K: float,
    *,
    variant: str,
    config: dict[str, Any] | None = None,
) -> tuple[float, bool, AnchoredCoherence]:
    """Return depth-scaled coherence with the physical T2 <= 2*T1 guard.

    This mirrors ``nvratemodel_api.t2_xy8_empirical`` without importing the
    score model into the literature-conditioning layer.
    """
    if isinstance(blocks, bool) or int(blocks) != blocks or int(blocks) < 1:
        raise ValueError("blocks must be a positive integer.")
    tau = float(tau_s)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau_s must be positive and finite.")
    conditioned, result = temperature_conditioned_t2_params(
        base_params, temperature_K, variant=variant, config=config
    )
    empirical = (
        float(conditioned["T2_1"])
        * int(blocks) ** float(conditioned["alpha"])
        * (tau / float(conditioned["tau0"])) ** float(conditioned["beta"])
    )
    limit = 2.0 * result.t1_s
    return min(empirical, limit), empirical > limit, result
