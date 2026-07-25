"""Pure diagnostic summaries for stored MINS results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .results import MINSResult


@dataclass(frozen=True, slots=True)
class RunDiagnostics:
    """Small run-health summary derived without mutating the result."""

    posterior_ess: float
    relative_posterior_ess: float
    proposal_acceptance_fraction: float
    maximum_proposals_per_replacement: int
    thresholds_monotone: bool
    conservative_log_remaining: float


def posterior_ess(result: MINSResult) -> float:
    """Return Kish effective sample size of normalized quadrature weights."""
    weights = result.posterior_weights
    return float(1.0 / np.sum(weights**2))


def summarize(result: MINSResult) -> RunDiagnostics:
    """Calculate minimal evidence-run health diagnostics."""
    ess = posterior_ess(result)
    n_weighted = result.niter + result.nlive
    acceptance = result.niter / result.n_proposals if result.n_proposals else 0.0
    maximum_proposals = int(np.max(result.history.proposals)) if result.niter else 0
    log_x = -result.niter / result.nlive
    conservative = log_x + float(np.max(result.final_live_log_psi))
    return RunDiagnostics(
        posterior_ess=ess,
        relative_posterior_ess=ess / n_weighted,
        proposal_acceptance_fraction=acceptance,
        maximum_proposals_per_replacement=maximum_proposals,
        thresholds_monotone=bool(np.all(np.diff(result.dead_log_psi) >= 0.0)),
        conservative_log_remaining=conservative,
    )
