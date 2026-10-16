"""Connected components of a sampled XY8 performance landscape.

The EDT near-optimal set at relative limit ``L`` is the discrete sublevel set

    R_L = {(tau, N): eta(tau, N) <= L * min(eta)}.

This module measures its zero-dimensional topology: the number of connected
components (often denoted beta_0).  Connectivity is evaluated on the sampled
``(tau, N)`` grid and therefore does not, by itself, prove connectivity of the
underlying continuous landscape.  Grid-refinement checks belong in the later
topology-validation stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage


FOUR_NEIGHBOUR_STRUCTURE = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)

EIGHT_NEIGHBOUR_STRUCTURE = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class ScoreSurface:
    """A complete rectangular XY8 score grid.

    Rows follow increasing pulse spacing and columns follow increasing XY8
    block count.  ``eta`` must have shape ``(len(taus_s), len(depths))``.
    """

    taus_s: np.ndarray
    depths: np.ndarray
    eta: np.ndarray

    @property
    def eta_min(self) -> float:
        return float(np.min(self.eta))

    @property
    def relative_score(self) -> np.ndarray:
        return self.eta / self.eta_min


@dataclass(frozen=True)
class ConnectedRegionResult:
    """Labelled EDT sublevel set for one relative performance limit."""

    surface: ScoreSurface
    relative_limit: float
    connectivity: int
    mask: np.ndarray
    labels: np.ndarray
    component_count: int

    @property
    def beta_0(self) -> int:
        """Return the number of connected components in the sampled set."""
        return self.component_count

    @property
    def threshold_percent(self) -> float:
        return 100.0 * (self.relative_limit - 1.0)


@dataclass(frozen=True)
class RegionCharacterization:
    """Physical and grid-level description of one connected component."""

    component_id: str
    label: int
    relative_limit: float
    point_count: int
    sampled_tau_count: int
    sampled_depth_count: int
    eta_min: float
    relative_min: float
    best_tau_s: float
    best_depth: int
    tau_min_s: float
    tau_max_s: float
    tau_width_s: float
    depth_min: int
    depth_max: int
    depth_span_blocks: int
    contains_global_optimum: bool
    resonance_orders: tuple[int, ...]
    resonance_taus_s: tuple[float, ...]
    touches_tau_lower_boundary: bool
    touches_tau_upper_boundary: bool
    touches_depth_lower_boundary: bool
    touches_depth_upper_boundary: bool
    contains_depth_16: bool
    contains_depth_20: bool

    @property
    def touches_tau_boundary(self) -> bool:
        return self.touches_tau_lower_boundary or self.touches_tau_upper_boundary

    @property
    def touches_depth_boundary(self) -> bool:
        return self.touches_depth_lower_boundary or self.touches_depth_upper_boundary

    @property
    def boundary_status(self) -> str:
        boundaries = []
        if self.touches_tau_lower_boundary:
            boundaries.append("tau_lower")
        if self.touches_tau_upper_boundary:
            boundaries.append("tau_upper")
        if self.touches_depth_lower_boundary:
            boundaries.append("depth_lower")
        if self.touches_depth_upper_boundary:
            boundaries.append("depth_upper")
        return ";".join(boundaries) if boundaries else "resolved_interior"

    @property
    def sampling_status(self) -> str:
        flags = []
        if self.sampled_tau_count == 1:
            flags.append("timing_single_grid_row")
        if self.sampled_depth_count == 1:
            flags.append("depth_single_grid_column")
        return ";".join(flags) if flags else "multiple_grid_points_on_both_axes"

    @property
    def resonance_identity(self) -> str:
        if not self.resonance_orders:
            return "no_exact_resonance_contained"
        return ";".join(f"m={order}" for order in self.resonance_orders)


def _as_one_dimensional(name: str, values, *, dtype) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def validate_surface(taus_s, depths, eta) -> ScoreSurface:
    """Validate and normalize an already rectangular performance surface."""
    taus = _as_one_dimensional("taus_s", taus_s, dtype=float)
    raw_depths = _as_one_dimensional("depths", depths, dtype=float)
    if np.any(raw_depths != np.rint(raw_depths)):
        raise ValueError("depths must contain integer XY8 block counts.")
    normalized_depths = raw_depths.astype(int)
    scores = np.asarray(eta, dtype=float)

    expected_shape = (taus.size, normalized_depths.size)
    if scores.shape != expected_shape:
        raise ValueError(
            f"eta must have shape {expected_shape}, received {scores.shape}."
        )
    if not np.all(np.isfinite(scores)) or np.any(scores <= 0.0):
        raise ValueError("eta must contain only positive finite scores.")
    if np.any(taus <= 0.0):
        raise ValueError("Pulse spacings must be positive.")
    if np.any(normalized_depths <= 0):
        raise ValueError("XY8 block counts must be positive.")
    if np.unique(taus).size != taus.size:
        raise ValueError("Pulse spacings must be unique.")
    if np.unique(normalized_depths).size != normalized_depths.size:
        raise ValueError("XY8 block counts must be unique.")

    tau_order = np.argsort(taus)
    depth_order = np.argsort(normalized_depths)
    return ScoreSurface(
        taus_s=taus[tau_order].copy(),
        depths=normalized_depths[depth_order].copy(),
        eta=scores[np.ix_(tau_order, depth_order)].copy(),
    )


def surface_from_points(
    points: Iterable[tuple[float, int, float]],
) -> ScoreSurface:
    """Build a complete score surface from ``(tau_s, N, eta)`` records.

    Duplicate settings and incomplete Cartesian grids are rejected.  This is
    deliberately strict because silently filling absent cells could create or
    remove apparent connections between resonant regions.
    """
    records = list(points)
    if not records:
        raise ValueError("At least one (tau_s, N, eta) point is required.")

    parsed: list[tuple[float, int, float]] = []
    for index, record in enumerate(records):
        if len(record) != 3:
            raise ValueError(
                f"Point {index} must contain exactly (tau_s, N, eta)."
            )
        tau, depth, score = record
        depth_float = float(depth)
        if not depth_float.is_integer():
            raise ValueError("N must be an integer XY8 block count.")
        parsed.append((float(tau), int(depth_float), float(score)))

    taus = np.array(sorted({row[0] for row in parsed}), dtype=float)
    depths = np.array(sorted({row[1] for row in parsed}), dtype=int)
    expected_count = int(taus.size * depths.size)
    if len(parsed) != expected_count:
        raise ValueError(
            "Points must form a complete rectangular (tau, N) grid with no "
            f"duplicates; expected {expected_count} rows and received {len(parsed)}."
        )

    tau_index = {value: index for index, value in enumerate(taus)}
    depth_index = {value: index for index, value in enumerate(depths)}
    scores = np.full((taus.size, depths.size), np.nan, dtype=float)
    for tau, depth, score in parsed:
        row = tau_index[tau]
        column = depth_index[depth]
        if np.isfinite(scores[row, column]):
            raise ValueError(f"Duplicate point for tau={tau}, N={depth}.")
        scores[row, column] = score
    if np.isnan(scores).any():
        raise ValueError("Points do not cover the complete rectangular grid.")
    return validate_surface(taus, depths, scores)


def connected_regions(
    surface: ScoreSurface,
    *,
    relative_limit: float = 1.10,
    connectivity: int = 4,
) -> ConnectedRegionResult:
    """Label connected regions within a relative score limit.

    ``relative_limit=1.10`` selects settings no more than 10% above the global
    minimum.  Four-neighbour connectivity permits movement along exactly one
    sampled coordinate at a time.  Eight-neighbour connectivity additionally
    permits diagonal contact and is provided for sensitivity analysis.
    """
    checked = validate_surface(surface.taus_s, surface.depths, surface.eta)
    limit = float(relative_limit)
    if not np.isfinite(limit) or limit < 1.0:
        raise ValueError("relative_limit must be finite and at least 1.0.")
    if connectivity == 4:
        structure = FOUR_NEIGHBOUR_STRUCTURE
    elif connectivity == 8:
        structure = EIGHT_NEIGHBOUR_STRUCTURE
    else:
        raise ValueError("connectivity must be either 4 or 8.")

    mask = checked.relative_score <= limit
    labels, component_count = ndimage.label(mask, structure=structure)
    return ConnectedRegionResult(
        surface=checked,
        relative_limit=limit,
        connectivity=connectivity,
        mask=mask,
        labels=labels.astype(np.int32, copy=False),
        component_count=int(component_count),
    )


def count_connected_regions(
    surface: ScoreSurface,
    *,
    relative_limit: float = 1.10,
    connectivity: int = 4,
) -> int:
    """Convenience wrapper returning only beta_0."""
    return connected_regions(
        surface,
        relative_limit=relative_limit,
        connectivity=connectivity,
    ).component_count


def resonant_taus(
    target_frequency_hz: float,
    tau_min_s: float,
    tau_max_s: float,
) -> tuple[tuple[int, float], ...]:
    """Return exact XY8 timing resonances within a pulse-spacing interval."""
    frequency = float(target_frequency_hz)
    lower = float(tau_min_s)
    upper = float(tau_max_s)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("target_frequency_hz must be positive and finite.")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("The pulse-spacing interval must be finite and ordered.")

    output = []
    order = 0
    while True:
        tau = (2.0 * order + 1.0) / (4.0 * frequency)
        if tau > upper:
            break
        if tau >= lower:
            output.append((order, tau))
        order += 1
    return tuple(output)


def build_resonance_refined_tau_grid(
    target_frequency_hz: float,
    tau_min_s: float,
    tau_max_s: float,
    *,
    background_count: int = 201,
    resonance_halfwidth_fraction: float = 0.08,
    resonance_point_count: int = 401,
    dedupe_rtol: float = 1e-12,
) -> np.ndarray:
    """Build a global timing grid with dense brackets at every resonance.

    An odd number of bracket points guarantees that each theoretical
    resonance is sampled exactly.  The coarse background retains observations
    in the gaps, which are needed to establish that separate regions are truly
    disconnected on the sampled grid.
    """
    lower = float(tau_min_s)
    upper = float(tau_max_s)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or lower >= upper:
        raise ValueError("The timing range must be positive, finite, and increasing.")
    if int(background_count) < 2:
        raise ValueError("background_count must be at least two.")
    if int(resonance_point_count) < 3 or int(resonance_point_count) % 2 == 0:
        raise ValueError("resonance_point_count must be an odd integer of at least three.")
    halfwidth = float(resonance_halfwidth_fraction)
    if not np.isfinite(halfwidth) or not 0.0 < halfwidth < 1.0:
        raise ValueError("resonance_halfwidth_fraction must lie between zero and one.")
    if not np.isfinite(dedupe_rtol) or dedupe_rtol < 0.0:
        raise ValueError("dedupe_rtol must be finite and non-negative.")

    pieces = [np.linspace(lower, upper, int(background_count))]
    for _, resonance_tau in resonant_taus(target_frequency_hz, lower, upper):
        bracket_lower = max(lower, resonance_tau * (1.0 - halfwidth))
        bracket_upper = min(upper, resonance_tau * (1.0 + halfwidth))
        pieces.append(
            np.linspace(bracket_lower, bracket_upper, int(resonance_point_count))
        )
        pieces.append(np.array([resonance_tau], dtype=float))

    candidates = np.sort(np.concatenate(pieces))
    kept = [float(candidates[0])]
    for value in candidates[1:]:
        value = float(value)
        tolerance = dedupe_rtol * max(abs(value), abs(kept[-1]), 1e-30)
        if value - kept[-1] > tolerance:
            kept.append(value)
    return np.asarray(kept, dtype=float)


def characterize_regions(
    result: ConnectedRegionResult,
    *,
    target_frequency_hz: float,
    resonance_rtol: float = 1e-9,
    resonance_atol_s: float = 1e-15,
) -> tuple[RegionCharacterization, ...]:
    """Describe every connected component in deterministic performance order.

    A component contains a timing resonance only when a grid row at that exact
    resonance belongs to the component.  Merely spanning a resonance between
    the component's minimum and maximum timing is not sufficient.
    """
    frequency = float(target_frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("target_frequency_hz must be positive and finite.")
    if resonance_rtol < 0.0 or resonance_atol_s < 0.0:
        raise ValueError("Resonance tolerances must be non-negative.")

    surface = result.surface
    possible_resonances = resonant_taus(
        frequency,
        float(surface.taus_s[0]),
        float(surface.taus_s[-1]),
    )
    global_best_index = np.unravel_index(np.argmin(surface.eta), surface.eta.shape)
    provisional = []
    for label in range(1, result.component_count + 1):
        row_indices, column_indices = np.nonzero(result.labels == label)
        if row_indices.size == 0:
            raise RuntimeError(f"Connected-component label {label} has no cells.")

        component_eta = surface.eta[row_indices, column_indices]
        local_order = np.lexsort(
            (
                surface.depths[column_indices],
                surface.taus_s[row_indices],
                component_eta,
            )
        )
        best_local = int(local_order[0])
        best_row = int(row_indices[best_local])
        best_column = int(column_indices[best_local])
        unique_rows = np.unique(row_indices)
        unique_columns = np.unique(column_indices)
        contained_resonances = [
            (order, tau)
            for order, tau in possible_resonances
            if any(
                np.isclose(
                    surface.taus_s[row],
                    tau,
                    rtol=resonance_rtol,
                    atol=resonance_atol_s,
                )
                for row in unique_rows
            )
        ]
        provisional.append(
            {
                "label": label,
                "point_count": int(row_indices.size),
                "sampled_tau_count": int(unique_rows.size),
                "sampled_depth_count": int(unique_columns.size),
                "eta_min": float(surface.eta[best_row, best_column]),
                "relative_min": float(surface.relative_score[best_row, best_column]),
                "best_tau_s": float(surface.taus_s[best_row]),
                "best_depth": int(surface.depths[best_column]),
                "tau_min_s": float(np.min(surface.taus_s[unique_rows])),
                "tau_max_s": float(np.max(surface.taus_s[unique_rows])),
                "depth_min": int(np.min(surface.depths[unique_columns])),
                "depth_max": int(np.max(surface.depths[unique_columns])),
                "contains_global_optimum": bool(
                    result.labels[global_best_index] == label
                ),
                "resonance_orders": tuple(item[0] for item in contained_resonances),
                "resonance_taus_s": tuple(item[1] for item in contained_resonances),
                "touches_tau_lower_boundary": bool(np.any(row_indices == 0)),
                "touches_tau_upper_boundary": bool(
                    np.any(row_indices == surface.taus_s.size - 1)
                ),
                "touches_depth_lower_boundary": bool(np.any(column_indices == 0)),
                "touches_depth_upper_boundary": bool(
                    np.any(column_indices == surface.depths.size - 1)
                ),
                "contains_depth_16": bool(
                    np.any(surface.depths[column_indices] == 16)
                ),
                "contains_depth_20": bool(
                    np.any(surface.depths[column_indices] == 20)
                ),
            }
        )

    provisional.sort(
        key=lambda row: (
            row["relative_min"],
            row["best_tau_s"],
            row["best_depth"],
            row["label"],
        )
    )
    characterized = []
    for index, row in enumerate(provisional, start=1):
        characterized.append(
            RegionCharacterization(
                component_id=f"R{index}",
                label=row["label"],
                relative_limit=result.relative_limit,
                point_count=row["point_count"],
                sampled_tau_count=row["sampled_tau_count"],
                sampled_depth_count=row["sampled_depth_count"],
                eta_min=row["eta_min"],
                relative_min=row["relative_min"],
                best_tau_s=row["best_tau_s"],
                best_depth=row["best_depth"],
                tau_min_s=row["tau_min_s"],
                tau_max_s=row["tau_max_s"],
                tau_width_s=row["tau_max_s"] - row["tau_min_s"],
                depth_min=row["depth_min"],
                depth_max=row["depth_max"],
                depth_span_blocks=row["depth_max"] - row["depth_min"],
                contains_global_optimum=row["contains_global_optimum"],
                resonance_orders=row["resonance_orders"],
                resonance_taus_s=row["resonance_taus_s"],
                touches_tau_lower_boundary=row["touches_tau_lower_boundary"],
                touches_tau_upper_boundary=row["touches_tau_upper_boundary"],
                touches_depth_lower_boundary=row["touches_depth_lower_boundary"],
                touches_depth_upper_boundary=row["touches_depth_upper_boundary"],
                contains_depth_16=row["contains_depth_16"],
                contains_depth_20=row["contains_depth_20"],
            )
        )
    return tuple(characterized)
