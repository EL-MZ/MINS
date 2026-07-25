from __future__ import annotations

import numpy as np
import pytest
from tests.helpers import StandardNormalProposal

from mins import CallableModel, MINSampler
from mins.diagnostics import summarize

pytestmark = pytest.mark.integration


def _constant_problem() -> tuple[CallableModel, StandardNormalProposal]:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.full(len(x), np.log(2.5)),
        log_prior_fn=proposal.log_prob,
    )
    return model, proposal


def test_constant_integrand_end_to_end_with_randomized_plateau() -> None:
    model, proposal = _constant_problem()
    result = MINSampler(
        model,
        proposal,
        n_live=20,
        rng=8,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.2,
        max_iterations=200,
        max_proposals_per_replacement=10_000,
    )
    assert result.success
    assert result.termination_reason == "remaining_evidence"
    assert result.logz == pytest.approx(np.log(2.5), abs=1e-12)
    assert result.information == pytest.approx(0.0, abs=1e-12)
    assert result.niter == 33
    assert np.sum(result.posterior_weights) == pytest.approx(1.0)
    assert not result.dead_points.flags.writeable
    assert summarize(result).thresholds_monotone


def test_same_seed_reproduces_scientific_result() -> None:
    model, proposal = _constant_problem()
    first = MINSampler(
        model,
        proposal,
        n_live=12,
        rng=123,
        tie_policy="randomized_plateau",
    ).run(dlogz=0.25, max_iterations=100)
    second = MINSampler(
        model,
        proposal,
        n_live=12,
        rng=123,
        tie_policy="randomized_plateau",
    ).run(dlogz=0.25, max_iterations=100)
    assert first.logz == second.logz
    np.testing.assert_array_equal(first.dead_points, second.dead_points)
    np.testing.assert_array_equal(first.dead_tie_breakers, second.dead_tie_breakers)
    np.testing.assert_array_equal(
        first.log_posterior_weights, second.log_posterior_weights
    )


def test_strict_plateau_returns_partial_failed_result() -> None:
    model, proposal = _constant_problem()
    result = MINSampler(
        model,
        proposal,
        n_live=10,
        rng=4,
        tie_policy="strict",
        proposal_batch_size=4,
    ).run(
        dlogz=0.1,
        max_iterations=100,
        max_proposals_per_replacement=12,
    )
    assert not result.success
    assert result.termination_reason == "plateau_stall"
    assert result.niter == 0
    assert result.n_proposals == 12
    assert result.logz == pytest.approx(np.log(2.5), abs=1e-12)


def test_iteration_limit_is_not_scientific_success() -> None:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -0.5 * (x[:, 0] - 1.0) ** 2,
        log_prior_fn=proposal.log_prob,
    )
    result = MINSampler(model, proposal, n_live=15, rng=2).run(
        dlogz=1e-8,
        max_iterations=3,
        max_proposals_per_replacement=1_000,
    )
    assert not result.success
    assert result.termination_reason == "max_iterations"
    assert result.niter == 3
    assert result.history.iteration.tolist() == [1, 2, 3]


def test_likelihood_call_limit_is_not_scientific_success() -> None:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -0.5 * (x[:, 0] - 0.5) ** 2,
        log_prior_fn=proposal.log_prob,
    )
    result = MINSampler(
        model,
        proposal,
        n_live=10,
        rng=19,
        proposal_batch_size=4,
    ).run(
        dlogz=1e-8,
        max_iterations=100,
        max_likelihood_calls=14,
        max_proposals_per_replacement=100,
    )
    assert not result.success
    assert result.termination_reason == "max_likelihood_calls"
    assert result.n_likelihood_calls == 14


def test_wall_time_limit_returns_initialized_partial_result() -> None:
    model, proposal = _constant_problem()
    result = MINSampler(
        model,
        proposal,
        n_live=10,
        rng=7,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.1,
        max_iterations=100,
        max_wall_time=1e-12,
    )
    assert not result.success
    assert result.termination_reason == "max_wall_time"
    assert result.niter == 0
    assert result.logz == pytest.approx(np.log(2.5), abs=1e-12)
