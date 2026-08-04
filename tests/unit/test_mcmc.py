from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.helpers import StandardNormalProposal

from mins import (
    CallableModel,
    ConfigurationError,
    EnsembleRWalkSettings,
    MINSConfig,
    RWalkSettings,
    SRWalkSettings,
)
from mins.constrained import BatchEvaluator, passes_constraint
from mins.mcmc import (
    RWalkSampler,
    SRWalkSampler,
    _random_unit_ball,
    accepts_log_q0_metropolis,
    bounding_ellipsoid_axes,
    covariance_factor,
    draw_ensemble_rwalk_constrained,
    draw_rwalk_constrained,
    draw_srwalk_constrained,
    eligible_survivor_indices,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "proposal_scheme",
    ["fixed_morph", "adaptive_morph", "rwalk", "s-rwalk", "en-rwalk"],
)
def test_all_proposal_schemes_are_configured(proposal_scheme: str) -> None:
    settings = EnsembleRWalkSettings(n_walkers=4)
    config = MINSConfig(
        n_live=5,
        proposal_scheme=proposal_scheme,  # type: ignore[arg-type]
        ensemble_rwalk_settings=settings,
    )
    assert config.proposal_scheme == proposal_scheme


@pytest.mark.parametrize(
    "kwargs",
    [
        {"walks": 0},
        {"walks": True},
        {"facc": True},
        {"facc": np.inf},
        {"facc": np.nan},
        {"ncdim": 0},
        {"ncdim": True},
    ],
)
def test_rwalk_settings_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        RWalkSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_steps": 4},
        {"scale": 1.0},
        {"covariance_shrinkage": 0.1},
        {"covariance_jitter": 1.0e-10},
    ],
)
def test_removed_rwalk_settings_are_not_accepted(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        RWalkSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_steps": 0},
        {"n_steps": True},
        {"scale": 0.0},
        {"scale": np.inf},
        {"scale": True},
        {"facc": True},
        {"facc": np.nan},
        {"covariance_shrinkage": -0.1},
        {"covariance_shrinkage": 1.1},
        {"covariance_jitter": 0.0},
        {"covariance_jitter": np.inf},
    ],
)
def test_srwalk_settings_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        SRWalkSettings(**kwargs)


def test_srwalk_sampler_uses_gaussian_default_and_rwalk_adaptation() -> None:
    sampler = SRWalkSampler(settings=SRWalkSettings(n_steps=4), ndim=2)
    assert sampler.n_steps == 4
    assert sampler.facc == 0.5
    assert sampler.scale == pytest.approx(2.38 / np.sqrt(2.0))

    initial_scale = sampler.scale
    sampler.record_completed_walk(accept=4, scale=initial_scale)
    assert sampler.scale == pytest.approx(initial_scale * np.exp(0.5))
    sampler.record_completed_walk(accept=0, scale=sampler.scale)
    assert sampler.scale == pytest.approx(initial_scale)

    explicit = SRWalkSampler(
        settings=SRWalkSettings(n_steps=5, scale=0.25, facc=0.2),
        ndim=1,
    )
    assert explicit.scale == 0.25
    assert explicit.facc == 0.2


def test_rwalk_sampler_resolves_dynesty_defaults_and_clamps_controls() -> None:
    default = RWalkSampler(settings=RWalkSettings(), ndim=4)
    assert default.walks == 24
    assert default.facc == 0.5
    assert default.ncdim == 4
    assert default.scale == 1.0
    assert default.update_bound_interval_ratio == 24
    assert default.citations == [
        ("Skilling (2006)", "projecteuclid.org/euclid.ba/1340370944")
    ]

    low = RWalkSampler(settings=RWalkSettings(walks=1, facc=-2.0), ndim=2)
    assert low.walks == 2
    assert low.facc == 0.5
    high = RWalkSampler(settings=RWalkSettings(walks=8, facc=3.0), ndim=2)
    assert high.facc == 1.0

    with pytest.raises(ConfigurationError, match="model dimension"):
        RWalkSampler(settings=RWalkSettings(ncdim=1), ndim=2)


def test_rwalk_tuning_accumulates_updates_and_resets_history() -> None:
    sampler = RWalkSampler(
        settings=RWalkSettings(walks=4, facc=0.5),
        ndim=2,
    )
    sampler.tune({"scale": 1.0, "accept": 4, "reject": 0}, update=False)
    assert sampler.scale == 1.0
    assert sampler.rwalk_history == {"n_accept": 4, "n_reject": 0}
    sampler.tune({"scale": 1.0, "accept": 0, "reject": 4})
    assert sampler.scale == pytest.approx(1.0)
    assert sampler.rwalk_history == {"n_accept": 0, "n_reject": 0}

    sampler.tune({"scale": 1.0, "accept": 4, "reject": 0})
    assert sampler.scale == pytest.approx(np.exp(0.5))
    sampler.tune({"scale": 1.0, "accept": 0, "reject": 4})
    assert sampler.scale == pytest.approx(np.exp(-0.5))


def test_rwalk_bound_cache_refreshes_after_walks_times_nlive_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counted_axes(points: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return np.ones((points.shape[1], points.shape[1]))

    monkeypatch.setattr("mins.mcmc.bounding_ellipsoid_axes", counted_axes)
    sampler = RWalkSampler(settings=RWalkSettings(walks=2), ndim=1)
    live = np.array([[0.0], [1.0], [2.0]])
    sampler.axes_for(live)
    for _ in range(3):
        sampler.record_completed_walk(accept=1, scale=sampler.scale)
        sampler.axes_for(live)
    assert calls == 2


def test_random_ball_draws_stay_inside_the_unit_ball() -> None:
    rng = np.random.default_rng(917)
    draws = np.array([_random_unit_ball(3, rng) for _ in range(2_000)])
    radii = np.linalg.norm(draws, axis=1)
    assert np.all(radii <= 1.0)
    assert np.mean(radii) == pytest.approx(0.75, abs=0.02)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_walkers": 2},
        {"n_walkers": 5},
        {"n_walkers": True},
        {"n_sweeps": 0},
        {"gamma": 0.0},
        {"gamma": np.nan},
        {"jitter_scale": 0.0},
        {"jitter_scale": np.inf},
        {"covariance_shrinkage": -0.1},
        {"covariance_shrinkage": 1.1},
        {"covariance_jitter": 0.0},
    ],
)
def test_ensemble_settings_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        EnsembleRWalkSettings(**kwargs)


def test_ensemble_size_is_checked_against_live_survivors() -> None:
    with pytest.raises(ConfigurationError, match="n_live"):
        MINSConfig(
            n_live=8,
            proposal_scheme="en-rwalk",
            ensemble_rwalk_settings=EnsembleRWalkSettings(n_walkers=8),
        )


def test_constraint_helper_implements_strict_and_plateau_ordering() -> None:
    values = np.array([0.9, 1.0, 1.0, 1.1])
    ties = np.array([0.9, 0.4, 0.6, 0.1])
    strict = passes_constraint(
        values,
        ties,
        threshold=1.0,
        threshold_tie_breaker=0.5,
        tie_policy="strict",
    )
    randomized = passes_constraint(
        values,
        ties,
        threshold=1.0,
        threshold_tie_breaker=0.5,
        tie_policy="randomized_plateau",
    )
    np.testing.assert_array_equal(strict, [False, False, False, True])
    np.testing.assert_array_equal(randomized, [False, False, True, True])
    assert not passes_constraint(
        1.0,
        0.9,
        threshold=1.0,
        threshold_tie_breaker=0.5,
        tie_policy="strict",
    )


def test_covariance_factor_handles_one_point_and_rank_deficiency() -> None:
    one_dimensional = covariance_factor(
        np.array([[2.0]]),
        shrinkage=0.1,
        jitter=1.0e-8,
    )
    assert one_dimensional.shape == (1, 1)
    assert one_dimensional[0, 0] == pytest.approx(1.0e-4)

    rank_deficient = covariance_factor(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        shrinkage=0.0,
        jitter=1.0e-10,
    )
    assert rank_deficient.shape == (3, 3)
    assert np.all(np.isfinite(rank_deficient))
    assert np.linalg.matrix_rank(rank_deficient) == 3


def test_dynesty_ellipsoid_contains_rank_deficient_live_points() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
    )
    axes = bounding_ellipsoid_axes(points)
    assert axes.shape == (3, 3)
    assert np.all(np.isfinite(axes))
    assert np.linalg.matrix_rank(axes) == 3
    inverse = np.linalg.inv(axes)
    transformed = (points - np.mean(points, axis=0)) @ inverse.T
    assert np.all(np.linalg.norm(transformed, axis=1) < 1.0)


def test_fixed_q0_ratio_can_reject_a_point_above_the_constraint() -> None:
    rejected_rng = np.random.default_rng(1)
    assert not accepts_log_q0_metropolis(
        current_log_q0=0.0,
        proposed_log_q0=-1.0,
        rng=rejected_rng,
    )
    accepted_rng = np.random.default_rng(1)
    assert accepts_log_q0_metropolis(
        current_log_q0=-1.0,
        proposed_log_q0=0.0,
        rng=accepted_rng,
    )


def _all_rejected_problem() -> tuple[
    BatchEvaluator,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: np.full(len(theta), -np.inf),
        log_prior_fn=proposal.log_prob,
    )
    theta = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    log_q0 = proposal.log_prob(theta)
    return (
        BatchEvaluator(model, proposal),
        theta,
        np.arange(6.0),
        np.array(log_q0, copy=True),
        log_q0,
        np.arange(6.0),
        np.linspace(0.1, 0.6, 6),
    )


def test_rwalk_starts_uniformly_from_an_eligible_survivor_and_can_stay_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        evaluator,
        theta,
        log_likelihood,
        log_prior,
        log_q0,
        log_psi0,
        ties,
    ) = _all_rejected_problem()
    seed = 20260731
    expected_rng = np.random.default_rng(seed)
    eligible = eligible_survivor_indices(
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
    )
    expected_index = int(expected_rng.choice(eligible))
    bound_calls = 0

    def counted_bound(
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        nonlocal bound_calls
        bound_calls += 1
        assert points.shape == (6, 1)
        return np.ones((1, 1))

    monkeypatch.setattr("mins.mcmc.bounding_ellipsoid_axes", counted_bound)
    sampler = RWalkSampler(settings=RWalkSettings(walks=7), ndim=1)
    attempt = draw_rwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=sampler,
        rng=np.random.default_rng(seed),
        max_proposals=7,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    assert bound_calls == 1
    np.testing.assert_array_equal(attempt.draw.point.theta, theta[expected_index])
    assert attempt.draw.point.log_likelihood == log_likelihood[expected_index]
    assert attempt.draw.point.tie_breaker == ties[expected_index]
    assert expected_index != 0
    assert attempt.n_proposed == 7
    assert evaluator.n_likelihood_calls == 7
    assert attempt.n_valid == 0
    assert attempt.n_accepted == 0
    assert attempt.n_moved == 0
    assert attempt.n_completed == 7


def test_srwalk_uses_frozen_survivor_covariance_and_can_stay_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        evaluator,
        theta,
        log_likelihood,
        log_prior,
        log_q0,
        log_psi0,
        ties,
    ) = _all_rejected_problem()
    seed = 20260804
    expected_rng = np.random.default_rng(seed)
    eligible = eligible_survivor_indices(
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
    )
    expected_index = int(expected_rng.choice(eligible))
    covariance_calls = 0

    def counted_covariance(
        points: NDArray[np.float64],
        *,
        shrinkage: float,
        jitter: float,
    ) -> NDArray[np.float64]:
        nonlocal covariance_calls
        covariance_calls += 1
        np.testing.assert_array_equal(points, theta[1:])
        assert shrinkage == 0.2
        assert jitter == 1.0e-8
        return np.ones((1, 1))

    monkeypatch.setattr("mins.mcmc.covariance_factor", counted_covariance)
    sampler = SRWalkSampler(
        settings=SRWalkSettings(
            n_steps=7,
            scale=1.0,
            covariance_shrinkage=0.2,
            covariance_jitter=1.0e-8,
        ),
        ndim=1,
    )
    attempt = draw_srwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=sampler,
        rng=np.random.default_rng(seed),
        max_proposals=7,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    assert covariance_calls == 1
    np.testing.assert_array_equal(attempt.draw.point.theta, theta[expected_index])
    assert attempt.draw.point.log_likelihood == log_likelihood[expected_index]
    assert attempt.draw.point.tie_breaker == ties[expected_index]
    assert expected_index != 0
    assert attempt.n_proposed == 7
    assert evaluator.n_likelihood_calls == 7
    assert attempt.n_valid == 0
    assert attempt.n_accepted == 0
    assert attempt.n_moved == 0
    assert attempt.n_completed == 7
    assert sampler.scale == pytest.approx(np.exp(-1.0))


class _ScriptedGenerator:
    def __init__(
        self,
        *,
        start: int,
        normal_values: list[float],
        random_values: list[float],
    ):
        self.start = start
        self.normal_values = iter(normal_values)
        self.random_values = iter(random_values)

    def choice(
        self,
        values: NDArray[np.int64],
        size: int | None = None,
        replace: bool = True,
    ) -> int:
        assert size is None
        assert replace
        assert self.start in values
        return self.start

    def standard_normal(self, *, size: int) -> NDArray[np.float64]:
        return np.full(size, next(self.normal_values))

    def random(self, size: int | None = None) -> float:
        assert size is None
        return next(self.random_values)


def test_rejected_mh_proposal_retains_every_cached_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LinearLogQ:
        ndim = 1

        def log_prob(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
            return -2.0 * theta[:, 0]

    proposal = LinearLogQ()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: 4.0 + theta[:, 0],
        log_prior_fn=proposal.log_prob,
    )
    theta = np.array([[-2.0], [0.0], [2.0]])
    log_q0 = proposal.log_prob(theta)
    log_likelihood = 4.0 + theta[:, 0]
    log_psi0 = np.array(log_likelihood, copy=True)
    ties = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(
        "mins.mcmc.bounding_ellipsoid_axes",
        lambda *args: np.ones((1, 1)),
    )
    sampler = RWalkSampler(settings=RWalkSettings(walks=2), ndim=1)
    attempt = draw_rwalk_constrained(
        evaluator=BatchEvaluator(model, proposal),
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_q0,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=1.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=sampler,
        rng=_ScriptedGenerator(  # type: ignore[arg-type]
            start=1,
            normal_values=[1.0, 1.0],
            random_values=[1.0, 0.9, 0.9, 1.0, 0.9, 0.9],
        ),
        max_proposals=2,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    point = attempt.draw.point
    np.testing.assert_array_equal(point.theta, theta[1])
    assert point.log_likelihood == log_likelihood[1]
    assert point.log_prior == log_q0[1]
    assert point.log_q0 == log_q0[1]
    assert point.log_psi0 == log_psi0[1]
    assert point.tie_breaker == ties[1]
    assert attempt.n_valid == 2
    assert attempt.n_accepted == 0


def test_accepted_mh_proposal_updates_every_cached_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LinearLogQ:
        ndim = 1

        def log_prob(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
            return -2.0 * theta[:, 0]

    proposal = LinearLogQ()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: 4.0 + theta[:, 0],
        log_prior_fn=proposal.log_prob,
    )
    theta = np.array([[-2.0], [1.0], [2.0]])
    log_q0 = proposal.log_prob(theta)
    log_likelihood = 4.0 + theta[:, 0]
    log_psi0 = np.array(log_likelihood, copy=True)
    ties = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(
        "mins.mcmc.bounding_ellipsoid_axes",
        lambda *args: np.full((1, 1), 10.0),
    )
    sampler = RWalkSampler(settings=RWalkSettings(walks=2), ndim=1)
    attempt = draw_rwalk_constrained(
        evaluator=BatchEvaluator(model, proposal),
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_q0,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=1.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=sampler,
        rng=_ScriptedGenerator(  # type: ignore[arg-type]
            start=1,
            normal_values=[-1.0, -10.0],
            random_values=[0.1, 0.8, 0.999, 1.0, 0.5],
        ),
        max_proposals=2,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    point = attempt.draw.point
    np.testing.assert_array_equal(point.theta, [0.0])
    assert point.log_likelihood == 4.0
    assert point.log_prior == 0.0
    assert point.log_q0 == 0.0
    assert point.log_psi0 == 4.0
    assert point.tie_breaker == 0.8
    assert attempt.n_valid == 1
    assert attempt.n_accepted == 1
    assert attempt.n_moved == 1


def test_ensemble_walk_has_exact_counts_and_reproducible_output() -> None:
    first = _all_rejected_problem()
    second = _all_rejected_problem()
    settings = EnsembleRWalkSettings(n_walkers=4, n_sweeps=3)
    attempts = []
    for problem in (first, second):
        evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = problem
        attempts.append(
            draw_ensemble_rwalk_constrained(
                evaluator=evaluator,
                live_theta=theta,
                live_log_likelihood=log_likelihood,
                live_log_prior=log_prior,
                live_log_q0=log_q0,
                live_log_psi0=log_psi0,
                live_tie_breakers=ties,
                worst=0,
                threshold=0.0,
                threshold_tie_breaker=ties[0],
                tie_policy="strict",
                settings=settings,
                rng=np.random.default_rng(73),
                max_proposals=12,
                max_likelihood_calls=None,
                deadline=None,
            )
        )
        assert evaluator.n_likelihood_calls == 12
    assert attempts[0].draw is not None
    assert attempts[1].draw is not None
    np.testing.assert_array_equal(
        attempts[0].draw.point.theta,
        attempts[1].draw.point.theta,
    )
    assert attempts[0].n_proposed == 12
    assert attempts[0].n_completed == 3
    assert attempts[0].n_valid == 0
    assert attempts[0].n_accepted == 0
    assert attempts[0].n_moved == 0


def test_mcmc_preflight_does_not_start_a_shortened_evolution() -> None:
    evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = (
        _all_rejected_problem()
    )
    attempt = draw_rwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=RWalkSampler(settings=RWalkSettings(walks=5), ndim=1),
        rng=np.random.default_rng(9),
        max_proposals=4,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is None
    assert attempt.reason == "max_proposals_per_replacement"
    assert attempt.n_proposed == 0
    assert evaluator.n_likelihood_calls == 0


def test_srwalk_preflight_does_not_start_a_shortened_evolution() -> None:
    evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = (
        _all_rejected_problem()
    )
    attempt = draw_srwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=SRWalkSampler(settings=SRWalkSettings(n_steps=5), ndim=1),
        rng=np.random.default_rng(9),
        max_proposals=4,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is None
    assert attempt.reason == "max_proposals_per_replacement"
    assert attempt.n_proposed == 0
    assert evaluator.n_likelihood_calls == 0
