"""Immutable result and run-history containers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from .config import MINSConfig
from .proposals import MorphMetadata
from .quadrature import live_log_contributions
from .stopping import SCIENTIFIC_TERMINATION_REASONS


def _readonly(
    values: ArrayLike,
    *,
    dtype: Any = np.float64,
) -> NDArray[Any]:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ProposalUpdateRecord:
    """Outcome of one scheduled adaptive proposal-Morph refit."""

    iteration: int
    success: bool
    active_revision: int
    n_training: int
    proposal_metadata: MorphMetadata | None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.iteration < 1:
            raise ValueError("proposal update iteration must be positive")
        if self.active_revision < 0:
            raise ValueError("active proposal revision must be nonnegative")
        if self.n_training < 2:
            raise ValueError("proposal update requires at least two training rows")
        if self.success and (
            self.error_type is not None or self.error_message is not None
        ):
            raise ValueError("successful proposal update cannot contain an error")
        if not self.success and (not self.error_type or self.error_message is None):
            raise ValueError("failed proposal update must describe its error")


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
    information: NDArray[np.float64]
    logzerr: NDArray[np.float64]
    remaining_fraction: NDArray[np.float64]
    live_ess: NDArray[np.float64]
    live_mean_rse: NDArray[np.float64]
    live_logz_error: NDArray[np.float64]
    logz_stability: NDArray[np.float64]
    stopping_streak: NDArray[np.int64]
    live_min_log_psi: NDArray[np.float64]
    live_median_log_psi: NDArray[np.float64]
    live_max_log_psi: NDArray[np.float64]
    proposals: NDArray[np.int64]
    likelihood_calls: NDArray[np.int64]
    acceptance_fraction: NDArray[np.float64]
    elapsed_seconds: NDArray[np.float64]
    proposal_revision: NDArray[np.int64]
    proposal_update_attempts: NDArray[np.int64]
    proposal_update_failures: NDArray[np.int64]

    def __post_init__(self) -> None:
        lengths: set[int] = set()
        integer_fields = {
            "iteration",
            "proposals",
            "likelihood_calls",
            "stopping_streak",
            "proposal_revision",
            "proposal_update_attempts",
            "proposal_update_failures",
        }
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
    """Complete immutable output of a fixed-importance MINS run.

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
    dead_log_q0: NDArray[np.float64]
    dead_log_psi0: NDArray[np.float64]
    dead_tie_breakers: NDArray[np.float64]
    dead_log_x: NDArray[np.float64]
    dead_log_weights: NDArray[np.float64]
    final_live_points: NDArray[np.float64]
    final_live_log_likelihood: NDArray[np.float64]
    final_live_log_prior: NDArray[np.float64]
    final_live_log_q0: NDArray[np.float64]
    final_live_log_psi0: NDArray[np.float64]
    final_live_tie_breakers: NDArray[np.float64]
    log_posterior_weights: NDArray[np.float64]
    history: RunHistory
    config: MINSConfig
    rng_bit_generator: str
    rng_state_initial: str
    rng_state_final: str
    importance_morph_description: str
    proposal_updates: tuple[ProposalUpdateRecord, ...]
    nonfinite_counts: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.nlive != self.config.n_live:
            raise ValueError("nlive must equal config.n_live")
        if self.niter < 0 or self.niter != len(self.dead_log_psi0):
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
            "dead_log_q0",
            "dead_log_psi0",
            "dead_tie_breakers",
            "dead_log_x",
            "dead_log_weights",
        )
        live_fields = (
            "final_live_log_likelihood",
            "final_live_log_prior",
            "final_live_log_q0",
            "final_live_log_psi0",
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

        if self.niter and np.any(np.diff(self.dead_log_psi0) < 0.0):
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
                        live_log_contributions(log_x, self.final_live_log_psi0),
                    )
                )
            )
        )
        if not np.isclose(recomputed, self.logz, rtol=1e-12, atol=1e-12):
            raise ValueError("stored logz cannot be recomputed from result arrays")
        expected_success = self.termination_reason in SCIENTIFIC_TERMINATION_REASONS
        if self.success != expected_success:
            raise ValueError(
                "success must correspond to scientific stopping termination"
            )
        updates = tuple(self.proposal_updates)
        if any(
            current.iteration <= previous.iteration
            for previous, current in pairwise(updates)
        ):
            raise ValueError("proposal updates must have increasing iterations")
        object.__setattr__(self, "proposal_updates", updates)

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
    def all_log_psi0(self) -> NDArray[np.float64]:
        """Return dead then final-live fixed-importance pseudo-likelihoods."""
        values = np.concatenate((self.dead_log_psi0, self.final_live_log_psi0))
        values.setflags(write=False)
        return values

    def resample_equal(
        self,
        rng: int | np.random.Generator,
        n_samples: int | None = None,
    ) -> NDArray[np.float64]:
        """Return equal-weight posterior samples by systematic resampling.

        Parameters
        ----------
        rng
            Explicit NumPy generator or integer seed.
        n_samples
            Number of returned draws. By default, preserve the number of
            weighted dead-plus-live points.

        Returns
        -------
        numpy.ndarray
            New output array with shape ``(n_samples, ndim)``. Points may
            repeat, as required when resampling discrete quadrature mass.

        Notes
        -----
        Weighted samples are the primary MINS posterior representation.
        Resampling is convenient for tools that require equal weights but adds
        Monte Carlo variation.
        """
        if isinstance(rng, np.random.Generator):
            generator = rng
        elif isinstance(rng, bool) or not isinstance(rng, (int, np.integer)):
            raise TypeError("rng must be an integer seed or numpy.random.Generator")
        else:
            generator = np.random.default_rng(int(rng))

        weighted_count = self.niter + self.nlive
        if n_samples is None:
            sample_count = weighted_count
        elif (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, (int, np.integer))
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer")
        else:
            sample_count = int(n_samples)

        weights = self.posterior_weights
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        positions = (
            np.arange(sample_count, dtype=float) + generator.random()
        ) / sample_count
        indices = np.searchsorted(cumulative, positions, side="right")
        generator.shuffle(indices)
        return np.array(self.all_points[indices], copy=True)
