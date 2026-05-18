from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NVParams:
    signal_freq_hz: float = 50_000.0
    t2_base_s: float = 400e-6
    t2_alpha: float = 0.6
    t2_beta: float = 0.0
    t2_tau0_s: float = 1e-6
    stretching_p: float = 2.0
    readout_contrast: float = 0.25
    contrast_decay_per_pulse: float = 0.0
    photon_rate_hz: float = 2e5
    pi_pulse_fidelity: float = 0.995
    pulse_error: float | None = None
    t_init_s: float = 5e-6
    t_read_s: float = 300e-9
    t_dead_s: float = 0.0
    duty_overhead_s: float = 0.0


@dataclass(frozen=True)
class SweepResult:
    tau_grid: np.ndarray
    n_pulses_grid: np.ndarray
    sensitivity: np.ndarray
    best_tau: float
    best_n_pulses: int
    best_value: float


class XY8Model:
    def __init__(self, params: NVParams):
        self.params = params

    @staticmethod
    def sequence_duration(tau: float, n_pulses: int) -> float:
        return float(2.0 * max(int(n_pulses), 1) * max(float(tau), 1e-12))

    @staticmethod
    def _pulse_times(tau: float, n_pulses: int) -> np.ndarray:
        n_pulses = max(int(n_pulses), 1)
        tau = max(float(tau), 1e-12)
        pulse_index = np.arange(1, n_pulses + 1, dtype=float)
        return (2.0 * pulse_index - 1.0) * tau

    def _switching_response(self, freq_hz: float, tau: float, n_pulses: int) -> complex:
        total_time = self.sequence_duration(tau, n_pulses)
        pulse_times = self._pulse_times(tau, n_pulses)
        edges = np.concatenate(([0.0], pulse_times, [total_time]))
        durations = np.diff(edges)
        signs = (-1.0) ** np.arange(durations.size, dtype=float)
        omega = 2.0 * np.pi * max(float(freq_hz), 0.0)
        if np.isclose(omega, 0.0):
            return complex(np.sum(signs * durations))
        phases = np.exp(1j * omega * edges)
        return complex(np.sum(signs * (phases[1:] - phases[:-1]) / (1j * omega)))

    def sensitivity_kernel(self, tau: float, n_pulses: int, freq_hz: float | None = None) -> float:
        freq_hz = self.params.signal_freq_hz if freq_hz is None else float(freq_hz)
        total_time = self.sequence_duration(tau, n_pulses)
        return float(abs(self._switching_response(freq_hz, tau, n_pulses)) / max(total_time, 1e-30))

    def filter_gain(self, tau: float, n_pulses: int, freq_hz: float | None = None) -> float:
        return self.sensitivity_kernel(tau, n_pulses, freq_hz=freq_hz)

    def coherence_time(self, n_pulses: int, tau: float) -> float:
        n_pulses = max(int(n_pulses), 1)
        tau = max(float(tau), 1e-12)
        tau0 = max(float(self.params.t2_tau0_s), 1e-12)
        return float(
            self.params.t2_base_s
            * (n_pulses ** self.params.t2_alpha)
            * ((tau / tau0) ** self.params.t2_beta)
        )

    def coherence_factor(self, tau: float, n_pulses: int) -> float:
        total_time = self.sequence_duration(tau, n_pulses)
        t2_eff = max(self.coherence_time(n_pulses, tau), 1e-12)
        return float(np.exp(-((total_time / t2_eff) ** self.params.stretching_p)))

    def pulse_fidelity(self, n_pulses: int) -> float:
        n_pulses = max(int(n_pulses), 1)
        if self.params.pulse_error is not None:
            return float(np.exp(-self.params.pulse_error * n_pulses))
        return float(self.params.pi_pulse_fidelity ** n_pulses)

    def readout_contrast(self, n_pulses: int) -> float:
        n_pulses = max(int(n_pulses), 1)
        return float(self.params.readout_contrast * np.exp(-self.params.contrast_decay_per_pulse * n_pulses))

    def cycle_time(self, tau: float, n_pulses: int) -> float:
        total_time = self.sequence_duration(tau, n_pulses)
        return float(
            total_time
            + self.params.t_init_s
            + self.params.t_read_s
            + self.params.t_dead_s
            + max(int(n_pulses), 1) * self.params.duty_overhead_s
        )

    def sensitivity_proxy(self, tau: float, n_pulses: int) -> float:
        seq_time = self.sequence_duration(tau, n_pulses)
        gain = max(self.filter_gain(tau, n_pulses), 1e-30)
        contrast = max(self.readout_contrast(n_pulses), 1e-30)
        coherence = max(self.coherence_factor(tau, n_pulses), 1e-30)
        fidelity = max(self.pulse_fidelity(n_pulses), 1e-30)
        photon_rate = max(self.params.photon_rate_hz, 1e-30)
        cycle_time = max(self.cycle_time(tau, n_pulses), 1e-30)
        effective_contrast = gain * contrast * coherence * fidelity
        return float(np.sqrt(cycle_time / photon_rate) / max(effective_contrast * seq_time, 1e-30))

    def factor_breakdown(self, tau: float, n_pulses: int, freq_hz: float | None = None) -> dict[str, float]:
        freq_hz = self.params.signal_freq_hz if freq_hz is None else float(freq_hz)
        return {
            "frequency_hz": freq_hz,
            "sequence_time_s": self.sequence_duration(tau, n_pulses),
            "cycle_time_s": self.cycle_time(tau, n_pulses),
            "filter_gain": self.filter_gain(tau, n_pulses, freq_hz=freq_hz),
            "coherence_time_s": self.coherence_time(n_pulses, tau),
            "coherence_factor": self.coherence_factor(tau, n_pulses),
            "pulse_factor": self.pulse_fidelity(n_pulses),
            "contrast": self.readout_contrast(n_pulses),
            "photon_rate_hz": self.params.photon_rate_hz,
            "sensitivity_proxy": self.sensitivity_proxy(tau, n_pulses),
        }

    def sweep(self, tau_values: np.ndarray, n_pulse_values: np.ndarray) -> SweepResult:
        tau_values = np.asarray(tau_values, dtype=float)
        n_pulse_values = np.asarray(n_pulse_values, dtype=int)
        surface = np.zeros((len(n_pulse_values), len(tau_values)), dtype=float)

        for i, n_pulses in enumerate(n_pulse_values):
            for j, tau in enumerate(tau_values):
                surface[i, j] = self.sensitivity_proxy(float(tau), int(n_pulses))

        idx = np.unravel_index(np.argmin(surface), surface.shape)
        best_n = int(n_pulse_values[idx[0]])
        best_tau = float(tau_values[idx[1]])
        return SweepResult(
            tau_grid=tau_values,
            n_pulses_grid=n_pulse_values,
            sensitivity=surface,
            best_tau=best_tau,
            best_n_pulses=best_n,
            best_value=float(surface[idx]),
        )
