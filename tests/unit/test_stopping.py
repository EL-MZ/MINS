from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mins import (
    ConfigurationError,
    MINSConfig,
    NumericalInvariantError,
    StoppingCriterionConfig,
    StoppingPolicy,
)
from mins.stopping import (
    StoppingMetrics,
    calculate_stopping_metrics,
    evaluate_stopping_policy,
)

pytestmark = pytest.mark.unit


def _metrics(
    *,
    remaining_fraction: float = 0.1,
    live_ess: float = 8.0,
    live_logz_error: float = 0.01,
    logz_stability: float = 0.02,
    logzerr: float = 0.03,
) -> StoppingMetrics:
    return StoppingMetrics(
        remaining_fraction=remaining_fraction,
        live_ess=live_ess,
        live_mean_rse=0.1,
        live_logz_error=live_logz_error,
        logz_stability=logz_stability,
        logzerr=logzerr,
    )


def _calculate(
    live_log_psi: list[float],
    *,
    remaining_fraction: float = 0.25,
    history: list[float] | None = None,
    stability_window: int = 3,
) -> StoppingMetrics:
    logz_live = (
        -np.inf if np.all(np.isneginf(live_log_psi)) else np.log(remaining_fraction)
    )
    return calculate_stopping_metrics(
        live_log_psi=live_log_psi,
        logz_live=float(logz_live),
        logz_total=0.0,
        logz_history=[0.0] if history is None else history,
        logzerr=0.2,
        stability_window=stability_window,
    )


@pytest.mark.parametrize("offset", [0.0, 1.0e200, -1.0e200])
def test_equal_finite_live_values_have_full_ess_and_zero_error(offset: float) -> None:
    metrics = _calculate([offset] * 5)
    assert metrics.live_ess == 5.0
    assert metrics.live_mean_rse == 0.0
    assert metrics.live_logz_error == 0.0


def test_one_dominant_live_value_has_unit_ess_and_scaled_error() -> None:
    first = _calculate([0.0, -1_000.0, -1_000.0, -1_000.0], remaining_fraction=0.4)
    second = _calculate(
        [0.0, -1_000.0, -1_000.0, -1_000.0],
        remaining_fraction=0.2,
    )
    assert first.live_ess == pytest.approx(1.0)
    assert first.live_mean_rse > 0.0
    assert first.live_logz_error == pytest.approx(
        first.remaining_fraction * first.live_mean_rse
    )
    assert second.live_logz_error == pytest.approx(first.live_logz_error / 2.0)


def test_all_zero_live_contributions_have_defined_zero_uncertainty() -> None:
    metrics = calculate_stopping_metrics(
        live_log_psi=[-np.inf] * 6,
        logz_live=-np.inf,
        logz_total=-np.inf,
        logz_history=[],
        logzerr=0.0,
        stability_window=3,
    )
    assert metrics.remaining_fraction == 0.0
    assert metrics.live_ess == 6.0
    assert metrics.live_mean_rse == 0.0
    assert metrics.live_logz_error == 0.0
    assert not np.isnan(metrics.live_ess)


def test_ess_is_stable_under_large_common_log_offsets() -> None:
    reference = _calculate([0.0, -2.0, -4.0, -6.0])
    positive = _calculate([1.0e200, 1.0e200 - 2.0, 1.0e200 - 4.0, 1.0e200 - 6.0])
    negative = _calculate([-1.0e200, -1.0e200 - 2.0, -1.0e200 - 4.0, -1.0e200 - 6.0])
    assert np.isfinite(positive.live_ess)
    assert np.isfinite(negative.live_ess)
    assert 1.0 <= reference.live_ess <= 4.0
    assert 1.0 <= positive.live_ess <= 4.0
    assert 1.0 <= negative.live_ess <= 4.0


def test_remaining_fraction_matches_log_evidence_ratio() -> None:
    metrics = calculate_stopping_metrics(
        live_log_psi=[-1.0, -2.0, -3.0],
        logz_live=-4.0,
        logz_total=-3.25,
        logz_history=[-3.25],
        logzerr=0.1,
        stability_window=2,
    )
    assert metrics.remaining_fraction == pytest.approx(np.exp(-4.0 + 3.25))


def test_stability_requires_the_exact_full_window() -> None:
    unavailable = _calculate(
        [0.0, 0.0],
        history=[1.0, 1.1],
        stability_window=3,
    )
    available = _calculate(
        [0.0, 0.0],
        history=[100.0, 1.0, 1.4, 1.2],
        stability_window=3,
    )
    assert np.isnan(unavailable.logz_stability)
    assert available.logz_stability == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("name", "tolerance", "metrics"),
    [
        ("remaining_fraction", 0.1, _metrics(remaining_fraction=0.1)),
        ("live_logz_error", 0.1, _metrics(live_logz_error=0.1)),
        ("logz_stability", 0.1, _metrics(logz_stability=0.1)),
        ("logzerr", 0.1, _metrics(logzerr=0.1)),
        ("live_ess", 1.0, _metrics(live_ess=1.0)),
    ],
)
def test_each_criterion_passes_at_equality(
    name: str,
    tolerance: float,
    metrics: StoppingMetrics,
) -> None:
    decision = evaluate_stopping_policy(
        metrics=metrics,
        policy=StoppingPolicy(
            criteria=(
                StoppingCriterionConfig(name=name, tolerance=tolerance),  # type: ignore[arg-type]
            )
        ),
        niter=1,
        previous_streak=0,
    )
    assert decision.evaluations[0].met
    assert decision.should_stop


def test_live_ess_uses_greater_than_direction() -> None:
    policy = StoppingPolicy(criteria=(StoppingCriterionConfig("live_ess", 5.0),))
    below = evaluate_stopping_policy(
        metrics=_metrics(live_ess=4.9),
        policy=policy,
        niter=1,
        previous_streak=0,
    )
    above = evaluate_stopping_policy(
        metrics=_metrics(live_ess=5.1),
        policy=policy,
        niter=1,
        previous_streak=0,
    )
    assert not below.combined_met
    assert above.combined_met


def test_all_and_any_modes_combine_criterion_results() -> None:
    criteria = (
        StoppingCriterionConfig("remaining_fraction", 0.2),
        StoppingCriterionConfig("live_logz_error", 0.001),
    )
    metrics = _metrics(remaining_fraction=0.1, live_logz_error=0.01)
    all_decision = evaluate_stopping_policy(
        metrics=metrics,
        policy=StoppingPolicy(criteria=criteria, mode="all"),
        niter=1,
        previous_streak=0,
    )
    any_decision = evaluate_stopping_policy(
        metrics=metrics,
        policy=StoppingPolicy(criteria=criteria, mode="any"),
        niter=1,
        previous_streak=0,
    )
    assert not all_decision.combined_met
    assert any_decision.combined_met


def test_minimum_iterations_and_consecutive_streak_are_applied_exactly() -> None:
    policy = StoppingPolicy(
        criteria=(StoppingCriterionConfig("remaining_fraction", 0.2),),
        consecutive=2,
        min_iterations=3,
    )
    metrics = _metrics(remaining_fraction=0.1)
    too_early = evaluate_stopping_policy(
        metrics=metrics,
        policy=policy,
        niter=2,
        previous_streak=4,
    )
    first = evaluate_stopping_policy(
        metrics=metrics,
        policy=policy,
        niter=3,
        previous_streak=too_early.streak,
    )
    second = evaluate_stopping_policy(
        metrics=metrics,
        policy=policy,
        niter=4,
        previous_streak=first.streak,
    )
    assert too_early.streak == 0
    assert first.streak == 1
    assert not first.should_stop
    assert second.streak == 2
    assert second.should_stop


def test_streak_resets_immediately_after_combined_failure() -> None:
    policy = StoppingPolicy(
        criteria=(StoppingCriterionConfig("remaining_fraction", 0.2),),
        consecutive=3,
    )
    decision = evaluate_stopping_policy(
        metrics=_metrics(remaining_fraction=0.3),
        policy=policy,
        niter=5,
        previous_streak=2,
    )
    assert not decision.combined_met
    assert decision.streak == 0
    assert not decision.should_stop


def test_unavailable_stability_is_unmet() -> None:
    decision = evaluate_stopping_policy(
        metrics=_metrics(logz_stability=np.nan),
        policy=StoppingPolicy(
            criteria=(StoppingCriterionConfig("logz_stability", 0.1),)
        ),
        niter=100,
        previous_streak=0,
    )
    assert not decision.evaluations[0].met


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: StoppingPolicy(criteria=()), "at least one"),
        (
            lambda: StoppingPolicy(
                criteria=(
                    StoppingCriterionConfig("logzerr", 0.1),
                    StoppingCriterionConfig("logzerr", 0.2),
                )
            ),
            "unique",
        ),
        (
            lambda: StoppingCriterionConfig("unknown", 0.1),  # type: ignore[arg-type]
            "unsupported",
        ),
        (
            lambda: StoppingPolicy(
                criteria=(StoppingCriterionConfig("logzerr", 0.1),),
                mode="neither",  # type: ignore[arg-type]
            ),
            "mode",
        ),
        (
            lambda: StoppingPolicy(
                criteria=(StoppingCriterionConfig("logzerr", 0.1),),
                consecutive=0,
            ),
            "consecutive",
        ),
        (
            lambda: StoppingPolicy(
                criteria=(StoppingCriterionConfig("logzerr", 0.1),),
                consecutive=True,
            ),
            "consecutive",
        ),
        (
            lambda: StoppingPolicy(
                criteria=(StoppingCriterionConfig("logzerr", 0.1),),
                min_iterations=-1,
            ),
            "min_iterations",
        ),
        (
            lambda: StoppingPolicy(
                criteria=(StoppingCriterionConfig("logzerr", 0.1),),
                stability_window=1,
            ),
            "stability_window",
        ),
    ],
)
def test_policy_validation_rejects_invalid_configuration(
    factory: Any,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        factory()


@pytest.mark.parametrize(
    ("name", "tolerance"),
    [
        ("remaining_fraction", 0.0),
        ("remaining_fraction", 1.0),
        ("remaining_fraction", np.nan),
        ("live_logz_error", 0.0),
        ("live_logz_error", np.inf),
        ("logz_stability", np.nan),
        ("live_ess", 0.9),
        ("logzerr", -1.0),
        ("logzerr", np.inf),
        ("logzerr", True),
    ],
)
def test_criterion_validation_rejects_invalid_tolerances(
    name: str,
    tolerance: Any,
) -> None:
    with pytest.raises(ConfigurationError, match="tolerance"):
        StoppingCriterionConfig(name=name, tolerance=tolerance)  # type: ignore[arg-type]


def test_config_rejects_live_ess_target_above_n_live() -> None:
    policy = StoppingPolicy(criteria=(StoppingCriterionConfig("live_ess", 11.0),))
    with pytest.raises(ConfigurationError, match=r"live_ess.*n_live"):
        MINSConfig(n_live=10, stopping=policy)


def test_config_resolves_legacy_and_rejects_ambiguous_stopping() -> None:
    default = MINSConfig(n_live=10)
    explicit = MINSConfig(n_live=10, dlogz=0.02)
    policy = StoppingPolicy(criteria=(StoppingCriterionConfig("logzerr", 0.1),))
    assert default.dlogz == 1.0e-3
    assert default.stopping is not None
    assert default.stopping.criteria[0].tolerance == 1.0e-3
    assert explicit.stopping is not None
    assert explicit.stopping.criteria[0].tolerance == 0.02
    with pytest.raises(ConfigurationError, match="dlogz and stopping"):
        MINSConfig(n_live=10, dlogz=0.1, stopping=policy)
    with pytest.raises(ConfigurationError, match="dlogz"):
        MINSConfig(n_live=10, dlogz=True)


def test_inconsistent_nonfinite_metric_state_raises_typed_error() -> None:
    with pytest.raises(NumericalInvariantError, match="all-zero"):
        calculate_stopping_metrics(
            live_log_psi=[-np.inf, -np.inf],
            logz_live=0.0,
            logz_total=0.0,
            logz_history=[0.0],
            logzerr=0.1,
            stability_window=2,
        )
