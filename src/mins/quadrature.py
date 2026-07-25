"""Pure log-space nested-sampling quadrature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from .exceptions import NumericalInvariantError


@dataclass(frozen=True, slots=True)
class QuadratureSummary:
    """Final evidence, posterior weights, information, and uncertainty."""

    logz: float
    information: float
    logzerr: float
    log_contributions: NDArray[np.float64]
    log_posterior_weights: NDArray[np.float64]


def logdiffexp(log_a: float, log_b: float) -> float:
    """Return ``log(exp(log_a) - exp(log_b))`` for ``log_a > log_b``.

    Raises
    ------
    ValueError
        If the interval is not strictly positive.
    """
    if not log_a > log_b:
        raise ValueError("log_a must be greater than log_b")
    return float(log_a + np.log1p(-np.exp(log_b - log_a)))


def dead_log_contribution(
    iteration: int,
    n_live: int,
    dead_log_psi: float,
) -> tuple[float, float, float]:
    """Return ``(log_x, log_delta_x, log_weight)`` for one dead point."""
    if iteration < 1:
        raise ValueError("iteration must be >= 1")
    if n_live < 2:
        raise ValueError("n_live must be >= 2")
    log_x_prev = -(iteration - 1) / n_live
    log_x = -iteration / n_live
    log_delta_x = logdiffexp(log_x_prev, log_x)
    return log_x, log_delta_x, log_delta_x + dead_log_psi


def live_log_contributions(
    log_x: float,
    live_log_psi: ArrayLike,
) -> NDArray[np.float64]:
    """Return individual final-live log evidence contributions."""
    values = np.asarray(live_log_psi, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("live_log_psi must be one-dimensional with length >= 2")
    return np.asarray(log_x - np.log(len(values)) + values, dtype=np.float64)


def estimated_live_logz(log_x: float, live_log_psi: ArrayLike) -> float:
    """Return the mean-live estimate of remaining log evidence."""
    contributions = live_log_contributions(log_x, live_log_psi)
    return float(logsumexp(contributions))


def finalize_quadrature(
    dead_log_weights: ArrayLike,
    dead_log_psi: ArrayLike,
    log_x: float,
    live_log_psi: ArrayLike,
    n_live: int,
) -> QuadratureSummary:
    """Combine dead and live contributions and calculate ``Z``, ``H``, error.

    All inputs and outputs are in natural-log units except ``information``,
    which is in nats and is non-negative.
    """
    dead_weights = np.asarray(dead_log_weights, dtype=float)
    dead_psi = np.asarray(dead_log_psi, dtype=float)
    live_psi = np.asarray(live_log_psi, dtype=float)
    if dead_weights.ndim != 1 or dead_psi.shape != dead_weights.shape:
        raise ValueError("dead arrays must be one-dimensional with equal shape")
    if live_psi.shape != (n_live,):
        raise ValueError(f"live_log_psi must have shape ({n_live},)")

    live_weights = live_log_contributions(log_x, live_psi)
    contributions = np.concatenate((dead_weights, live_weights))
    all_log_psi = np.concatenate((dead_psi, live_psi))
    logz = float(logsumexp(contributions))
    if not np.isfinite(logz):
        raise NumericalInvariantError("final evidence is not finite")
    log_posterior_weights = contributions - logz
    posterior_weights = np.exp(log_posterior_weights)
    if not np.isclose(np.sum(posterior_weights), 1.0, rtol=1e-12, atol=1e-12):
        raise NumericalInvariantError("posterior weights do not sum to one")
    positive = posterior_weights > 0.0
    information = float(
        np.sum(posterior_weights[positive] * (all_log_psi[positive] - logz))
    )
    if information < -1.0e-10:
        raise NumericalInvariantError(
            f"materially negative information estimate: {information}"
        )
    information = max(0.0, information)
    return QuadratureSummary(
        logz=logz,
        information=information,
        logzerr=float(np.sqrt(information / n_live)),
        log_contributions=contributions,
        log_posterior_weights=log_posterior_weights,
    )
