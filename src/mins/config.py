"""Validated immutable sampler configuration."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from .exceptions import ConfigurationError
from .stopping import (
    StoppingCriterionConfig,
    StoppingPolicy,
    validate_stopping_policy_for_n_live,
)

TiePolicy = Literal["strict", "randomized_plateau"]


def _validate_dlogz(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError("dlogz must satisfy 0 < dlogz < 1")
    number = float(value)
    if not np.isfinite(number) or not 0.0 < number < 1.0:
        raise ConfigurationError("dlogz must satisfy 0 < dlogz < 1")
    return number


@dataclass(frozen=True, slots=True)
class MINSConfig:
    """Numerical and resource settings for one MINS run.

    Parameters
    ----------
    n_live
        Number of live points. Must be at least two.
    dlogz
        Optional legacy remaining-evidence-fraction tolerance in ``(0, 1)``.
    stopping
        Resolved immutable stopping policy. Supplying this together with
        ``dlogz`` is invalid.
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
    dlogz: float | None = None
    stopping: StoppingPolicy | None = None
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
        if self.dlogz is not None and self.stopping is not None:
            raise ConfigurationError("dlogz and stopping cannot both be supplied")
        dlogz = self.dlogz
        stopping = self.stopping
        if dlogz is None and stopping is None:
            dlogz = 1.0e-3
        if dlogz is not None:
            dlogz = _validate_dlogz(dlogz)
            stopping = StoppingPolicy(
                criteria=(
                    StoppingCriterionConfig(
                        name="remaining_fraction",
                        tolerance=dlogz,
                    ),
                )
            )
        elif not isinstance(stopping, StoppingPolicy):
            raise ConfigurationError("stopping must be a StoppingPolicy")
        if stopping is None:  # pragma: no cover - resolution above is exhaustive
            raise ConfigurationError("a stopping policy could not be resolved")
        validate_stopping_policy_for_n_live(stopping, self.n_live)
        object.__setattr__(self, "dlogz", dlogz)
        object.__setattr__(self, "stopping", stopping)
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
