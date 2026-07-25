"""Validated immutable sampler configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .exceptions import ConfigurationError

TiePolicy = Literal["strict", "randomized_plateau"]


@dataclass(frozen=True, slots=True)
class MINSConfig:
    """Numerical and resource settings for one MINS run.

    Parameters
    ----------
    n_live
        Number of live points. Must be at least two.
    dlogz
        Relative estimated-live-evidence stopping tolerance in ``(0, 1)``.
    proposal_batch_size
        Number of independent Morph draws evaluated per rejection batch.
    max_iterations
        Maximum number of completed dead-point replacements.
    max_proposals_per_replacement
        Maximum proposal points evaluated for one replacement.
    max_likelihood_calls
        Optional run-wide likelihood-evaluation limit.
    max_wall_time
        Optional run-wide wall-time limit in seconds.
    tie_policy
        Strict likelihood ordering or lexicographic randomized plateau ordering.
    """

    n_live: int
    dlogz: float = 1.0e-3
    proposal_batch_size: int = 64
    max_iterations: int = 10_000
    max_proposals_per_replacement: int = 100_000
    max_likelihood_calls: int | None = None
    max_wall_time: float | None = None
    tie_policy: TiePolicy = "strict"

    def __post_init__(self) -> None:
        """Validate all settings eagerly."""
        if (
            isinstance(self.n_live, bool)
            or not isinstance(self.n_live, int)
            or self.n_live < 2
        ):
            raise ConfigurationError("n_live must be an integer >= 2")
        if not 0.0 < self.dlogz < 1.0:
            raise ConfigurationError("dlogz must satisfy 0 < dlogz < 1")
        for name in (
            "proposal_batch_size",
            "max_iterations",
            "max_proposals_per_replacement",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        if self.max_likelihood_calls is not None and (
            isinstance(self.max_likelihood_calls, bool)
            or not isinstance(self.max_likelihood_calls, int)
            or self.max_likelihood_calls < self.n_live
        ):
            raise ConfigurationError(
                "max_likelihood_calls must be an integer >= n_live"
            )
        if self.max_wall_time is not None and (
            isinstance(self.max_wall_time, bool)
            or not isinstance(self.max_wall_time, (int, float))
            or self.max_wall_time <= 0.0
        ):
            raise ConfigurationError("max_wall_time must be positive")
        if self.tie_policy not in ("strict", "randomized_plateau"):
            raise ConfigurationError(f"unsupported tie_policy: {self.tie_policy!r}")
