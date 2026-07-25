from __future__ import annotations

import numpy as np
import pytest

from mins import MorphProposal
from mins.exceptions import InvalidProposalOutput

pytestmark = pytest.mark.unit


def test_morph_adapter_sampling_density_and_seed_reproducibility() -> None:
    training = np.random.default_rng(4).normal(size=(250, 1))
    proposal = MorphProposal.fit(
        training,
        groups=[],
        param_names=("x",),
        kde_bw="silverman",
    )
    first = proposal.sample(20, np.random.default_rng(91))
    second = proposal.sample(20, np.random.default_rng(91))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 1)
    log_q = proposal.log_prob(first)
    assert log_q.shape == (20,)
    assert np.all(np.isfinite(log_q))
    assert proposal.metadata.n_training == 250
    assert proposal.metadata.morphz_version != "unknown"


def test_morph_adapter_copies_and_validates_training_data() -> None:
    training = np.random.default_rng(5).normal(size=(100, 1))
    proposal = MorphProposal.fit(training, groups=[])
    training[:] = np.nan
    assert np.isfinite(proposal.log_prob(np.array([[0.0]]))).all()
    with pytest.raises(ValueError, match="finite"):
        MorphProposal.fit(training, groups=[])


def test_morph_adapter_rejects_bad_log_prob_shape() -> None:
    training = np.random.default_rng(6).normal(size=(100, 1))
    proposal = MorphProposal.fit(training, groups=[])
    with pytest.raises(InvalidProposalOutput, match="shape"):
        proposal.log_prob(np.zeros((2, 2)))
