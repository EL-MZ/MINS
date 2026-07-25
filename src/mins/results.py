"""Immutable result and run-history containers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from .config import MINSConfig
from .quadrature import live_log_contributions


def _readonly(
    values: ArrayLike,
    *,
    dtype: Any = np.float64,
) -> NDArray[Any]:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class RunHistory:
    """Per-completed-iteration diagnostic arrays.

    Every field has shape ``(niter,)``. Logarithms are natural logarithms.
    """

    iteration: NDArray[np.int64]
    discarded_log_psi: NDArray[np.float64]
    log_x: NDArray[np.float64]
    log_delta_x: NDArray[np.float64]
    logz_dead: NDArray[np.float64]
    logz_live: NDArray[np.float64]
    logz_total: NDArray[np.float64]
    live_min_log_psi: NDArray[np.float64]
    live_median_log_psi: NDArray[np.float64]
    live_max_log_psi: NDArray[np.float64]
    proposals: NDArray[np.int64]
    likelihood_calls: NDArray[np.int64]
    acceptance_fraction: NDArray[np.float64]
    elapsed_seconds: NDArray[np.float64]

    def __post_init__(self) -> None:
        lengths: set[int] = set()
        integer_fields = {"iteration", "proposals", "likelihood_calls"}
        for field in fields(self):
            values = getattr(self, field.name)
            dtype = np.int64 if field.name in integer_fields else np.float64
            array = _readonly(values, dtype=dtype)
            if array.ndim != 1:
                raise ValueError(f"history {field.name} must be one-dimensional")
            lengths.add(len(array))
            object.__setattr__(self, field.name, array)
        if len(lengths) > 1:
            raise ValueError("all run-history arrays must have equal length")


@dataclass(frozen=True, slots=True)
class MINSResult:
    """Complete immutable output of a fixed-Morph MINS run.

    Arrays contain enough cached information to recompute the evidence,
    posterior weights, and information without reevaluating the model.
    """

    logz: float
    logzerr: float
    information: float
    success: bool
    termination_reason: str
    niter: int
    nlive: int
    n_likelihood_calls: int
    n_prior_calls: int
    n_proposals: int
    dead_points: NDArray[np.float64]
    dead_log_likelihood: NDArray[np.float64]
    dead_log_prior: NDArray[np.float64]
    dead_log_q: NDArray[np.float64]
    dead_log_psi: NDArray[np.float64]
    dead_tie_breakers: NDArray[np.float64]
    dead_log_x: NDArray[np.float64]
    dead_log_weights: NDArray[np.float64]
    final_live_points: NDArray[np.float64]
    final_live_log_likelihood: NDArray[np.float64]
    final_live_log_prior: NDArray[np.float64]
    final_live_log_q: NDArray[np.float64]
    final_live_log_psi: NDArray[np.float64]
    final_live_tie_breakers: NDArray[np.float64]
    log_posterior_weights: NDArray[np.float64]
    history: RunHistory
    config: MINSConfig
    rng_bit_generator: str
    rng_state_initial: str
    rng_state_final: str
    proposal_description: str
    nonfinite_counts: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.nlive != self.config.n_live:
            raise ValueError("nlive must equal config.n_live")
        if self.niter < 0 or self.niter != len(self.dead_log_psi):
            raise ValueError("niter must equal the number of dead points")
        if len(self.history.iteration) != self.niter:
            raise ValueError("history length must equal niter")
        ndim = (
            self.final_live_points.shape[1] if self.final_live_points.ndim == 2 else -1
        )
        expected_dead_matrix = (self.niter, ndim)
        expected_live_matrix = (self.nlive, ndim)

        matrix_fields = ("dead_points", "final_live_points")
        dead_fields = (
            "dead_log_likelihood",
            "dead_log_prior",
            "dead_log_q",
            "dead_log_psi",
            "dead_tie_breakers",
            "dead_log_x",
            "dead_log_weights",
        )
        live_fields = (
            "final_live_log_likelihood",
            "final_live_log_prior",
            "final_live_log_q",
            "final_live_log_psi",
            "final_live_tie_breakers",
        )
        for name in matrix_fields:
            array = _readonly(getattr(self, name))
            expected = (
                expected_dead_matrix if name == "dead_points" else expected_live_matrix
            )
            if array.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {array.shape}"
                )
            object.__setattr__(self, name, array)
        for name in dead_fields:
            array = _readonly(getattr(self, name))
            if array.shape != (self.niter,):
                raise ValueError(f"{name} must have shape ({self.niter},)")
            object.__setattr__(self, name, array)
        for name in live_fields:
            array = _readonly(getattr(self, name))
            if array.shape != (self.nlive,):
                raise ValueError(f"{name} must have shape ({self.nlive},)")
            object.__setattr__(self, name, array)
        log_weights = _readonly(self.log_posterior_weights)
        if log_weights.shape != (self.niter + self.nlive,):
            raise ValueError("log_posterior_weights must have shape (niter + nlive,)")
        object.__setattr__(self, "log_posterior_weights", log_weights)

        if self.niter and np.any(np.diff(self.dead_log_psi) < 0.0):
            raise ValueError("dead pseudo-likelihood thresholds must be monotone")
        if not np.isclose(
            logsumexp(self.log_posterior_weights), 0.0, rtol=0.0, atol=1e-11
        ):
            raise ValueError("log posterior weights are not normalized")
        log_x = -self.niter / self.nlive
        recomputed = float(
            logsumexp(
                np.concatenate(
                    (
                        self.dead_log_weights,
                        live_log_contributions(log_x, self.final_live_log_psi),
                    )
                )
            )
        )
        if not np.isclose(recomputed, self.logz, rtol=1e-12, atol=1e-12):
            raise ValueError("stored logz cannot be recomputed from result arrays")
        if self.success != (self.termination_reason == "remaining_evidence"):
            raise ValueError(
                "success must correspond to remaining_evidence termination"
            )

    @property
    def posterior_weights(self) -> NDArray[np.float64]:
        """Return normalized quadrature weights with shape ``(niter + nlive,)``."""
        values = np.exp(self.log_posterior_weights)
        values.setflags(write=False)
        return values

    @property
    def all_points(self) -> NDArray[np.float64]:
        """Return dead then final-live parameter points."""
        values = np.concatenate((self.dead_points, self.final_live_points))
        values.setflags(write=False)
        return values

    @property
    def all_log_psi(self) -> NDArray[np.float64]:
        """Return dead then final-live pseudo-likelihood values."""
        values = np.concatenate((self.dead_log_psi, self.final_live_log_psi))
        values.setflags(write=False)
        return values
