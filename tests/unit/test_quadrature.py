from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from mins.quadrature import (
    dead_log_contribution,
    finalize_quadrature,
    live_log_contributions,
    logdiffexp,
)

pytestmark = pytest.mark.unit


def test_logdiffexp_matches_safe_direct_arithmetic() -> None:
    for log_a, log_b in [(0.0, -0.1), (-3.0, -7.0), (12.0, 11.0)]:
        expected = np.log(np.exp(log_a) - np.exp(log_b))
        assert logdiffexp(log_a, log_b) == pytest.approx(expected)


def test_logdiffexp_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError, match="greater"):
        logdiffexp(-1.0, -1.0)


def test_dead_and_live_contributions_cover_unit_volume() -> None:
    nlive = 10
    niter = 17
    log_c = np.log(3.25)
    dead = [dead_log_contribution(i, nlive, log_c)[2] for i in range(1, niter + 1)]
    live = live_log_contributions(-niter / nlive, np.full(nlive, log_c))
    assert logsumexp(np.concatenate((dead, live))) == pytest.approx(log_c, abs=1e-14)


def test_finalize_constant_integrand_identity_and_weights() -> None:
    nlive = 20
    niter = 41
    log_c = np.log(7.0)
    dead = np.array(
        [dead_log_contribution(i, nlive, log_c)[2] for i in range(1, niter + 1)]
    )
    summary = finalize_quadrature(
        dead,
        np.full(niter, log_c),
        -niter / nlive,
        np.full(nlive, log_c),
        nlive,
    )
    assert summary.logz == pytest.approx(log_c, abs=1e-13)
    assert summary.information == pytest.approx(0.0, abs=1e-13)
    assert summary.logzerr == pytest.approx(0.0, abs=1e-13)
    assert np.sum(np.exp(summary.log_posterior_weights)) == pytest.approx(1.0)
