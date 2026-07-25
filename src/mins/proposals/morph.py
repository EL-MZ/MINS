"""MorphZ ``GroupKDE`` adapter implementing the normalized proposal contract."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import InvalidProposalOutput, MissingOptionalDependency


@dataclass(frozen=True, slots=True)
class MorphMetadata:
    """Immutable description of a fitted MorphZ proposal."""

    n_training: int
    ndim: int
    parameter_names: tuple[str, ...]
    kde_bw: str
    min_tc: float | None
    top_k_greedy: int
    group_source: str
    morphz_version: str
    rng_note: str


class MorphProposal:
    """Adapt a fixed normalized ``morphZ.GroupKDE`` to MINS.

    Instances must be constructed with :meth:`fit`.

    Notes
    -----
    MorphZ 0.4.1 accepts an integer seed in ``resample``. MINS derives that
    integer from the run's explicit :class:`numpy.random.Generator`. The
    inspected MorphZ implementation temporarily seeds NumPy's legacy global
    RNG and restores its previous state before returning. MINS never directly
    reseeds global state.
    """

    def __init__(
        self,
        backend: Any,
        metadata: MorphMetadata,
    ) -> None:
        self._backend = backend
        self.metadata = metadata
        self.ndim = metadata.ndim

    @classmethod
    def fit(
        cls,
        posterior_samples: ArrayLike,
        *,
        group_file: str | Path | None = None,
        groups: Sequence[Sequence[object]] | None = None,
        param_names: Sequence[str] | None = None,
        kde_bw: str | float | dict[str, float] = "silverman",
        min_tc: float | None = None,
        top_k_greedy: int = 1,
    ) -> MorphProposal:
        """Fit one fixed MorphZ ``GroupKDE``.

        Parameters
        ----------
        posterior_samples
            Finite training values with shape ``(n_samples, ndim)``. The input
            is copied and is never used as the initial live set.
        group_file
            JSON grouping definition accepted by MorphZ. MINS reads the file
            and passes its contents in memory to prevent MorphZ from writing a
            sibling selection file.
        groups
            In-memory MorphZ grouping definition. Specify at most one of
            ``group_file`` and ``groups``. An empty sequence selects independent
            one-dimensional KDEs.
        param_names
            Names corresponding to sample columns.
        kde_bw, min_tc, top_k_greedy
            Values passed unchanged to ``morphZ.GroupKDE``.

        Returns
        -------
        MorphProposal
            A proposal whose sampling and log density use the same fitted
            normalized GroupKDE.

        Raises
        ------
        MissingOptionalDependency
            If MorphZ is unavailable.
        ValueError
            If training values or grouping inputs are invalid.
        """
        samples = np.array(posterior_samples, dtype=float, copy=True)
        if samples.ndim != 2:
            raise ValueError("posterior_samples must have shape (n_samples, ndim)")
        if samples.shape[0] < 2 or samples.shape[1] < 1:
            raise ValueError(
                "posterior_samples must contain at least two rows and one column"
            )
        if not np.all(np.isfinite(samples)):
            raise ValueError("posterior_samples must contain only finite values")
        if group_file is not None and groups is not None:
            raise ValueError("specify only one of group_file and groups")
        if param_names is None:
            names = tuple(f"param_{index}" for index in range(samples.shape[1]))
        else:
            names = tuple(str(name) for name in param_names)
        if len(names) != samples.shape[1]:
            raise ValueError("param_names length must match posterior sample columns")
        if len(set(names)) != len(names):
            raise ValueError("param_names must be unique")
        if isinstance(top_k_greedy, bool) or top_k_greedy < 1:
            raise ValueError("top_k_greedy must be a positive integer")

        if group_file is not None:
            group_path = Path(group_file)
            with group_path.open(encoding="utf-8") as stream:
                group_definition = json.load(stream)
            group_source = str(group_path)
        elif groups is not None:
            group_definition = list(groups)
            group_source = "in-memory"
        else:
            group_definition = []
            group_source = "independent-default"

        try:
            import morphZ
        except ImportError as error:  # pragma: no cover - environment dependent
            raise MissingOptionalDependency(
                "MorphProposal requires MorphZ; install MINS with the 'morph' extra"
            ) from error

        backend = morphZ.GroupKDE(
            samples,
            param_tc=group_definition,
            param_names=list(names),
            kde_bw=kde_bw,
            min_tc=min_tc,
            verbose=False,
            top_k_greedy=top_k_greedy,
        )
        metadata = MorphMetadata(
            n_training=samples.shape[0],
            ndim=samples.shape[1],
            parameter_names=names,
            kde_bw=repr(kde_bw),
            min_tc=min_tc,
            top_k_greedy=top_k_greedy,
            group_source=group_source,
            morphz_version=str(getattr(morphZ, "__version__", "unknown")),
            rng_note=(
                "MINS derives an integer seed from its Generator for each MorphZ "
                "resample call; MorphZ 0.4.1 restores legacy global RNG state."
            ),
        )
        return cls(backend, metadata)

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw independent points from the fitted MorphZ density.

        Parameters
        ----------
        n
            Positive number of points.
        rng
            Run-owned random generator used to derive MorphZ's integer seed.

        Returns
        -------
        numpy.ndarray
            Finite points with shape ``(n, ndim)``.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        points = np.asarray(
            self._backend.resample(n, random_state=seed),
            dtype=float,
        )
        if points.shape != (n, self.ndim):
            raise InvalidProposalOutput(
                f"MorphZ resample returned {points.shape}, expected {(n, self.ndim)}"
            )
        if not np.all(np.isfinite(points)):
            raise InvalidProposalOutput("MorphZ resample returned NaN or infinity")
        return points

    def log_prob(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate the normalized fitted MorphZ log density.

        MorphZ 0.4.1's public ``logpdf`` supports single points reliably. The
        adapter evaluates rows individually to preserve the MINS ``(n, ndim)``
        convention without guessing a different batch orientation.
        """
        points = np.asarray(theta, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != self.ndim:
            raise InvalidProposalOutput(
                f"theta must have shape (n, {self.ndim}), got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise InvalidProposalOutput("theta contains NaN or infinity")
        values = np.asarray(
            [self._backend.logpdf(point) for point in points],
            dtype=float,
        )
        if values.shape != (len(points),):
            raise InvalidProposalOutput(
                f"MorphZ logpdf returned {values.shape}, expected {(len(points),)}"
            )
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise InvalidProposalOutput("MorphZ logpdf returned NaN or +infinity")
        return values
