from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class UniformProposal:
    ndim = 1

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.uniform(0.0, 1.0, size=(n, 1))

    def log_prob(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        points = np.asarray(theta)
        inside = (points[:, 0] >= 0.0) & (points[:, 0] <= 1.0)
        return np.where(inside, 0.0, -np.inf)


class StandardNormalProposal:
    ndim = 1

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.normal(size=(n, 1))

    def log_prob(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        points = np.asarray(theta)
        return -0.5 * points[:, 0] ** 2 - 0.5 * np.log(2.0 * np.pi)
