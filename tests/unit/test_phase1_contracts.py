from __future__ import annotations

import numpy as np
import pytest

import mins
from mins import CallableModel, ConfigurationError, InvalidModelOutput, MINSConfig

pytestmark = pytest.mark.unit


def test_public_import_and_version() -> None:
    assert mins.__version__ == "0.1.0.dev3"
    assert mins.Model is not None
    assert mins.Proposal is not None


def test_callable_model_vectorized_and_scalar_agree() -> None:
    points = np.array([[-1.0], [0.5], [2.0]])
    vectorized = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -(x[:, 0] ** 2),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    scalar = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -(x[0] ** 2),
        log_prior_fn=lambda x: 0.0,
        vectorized=False,
    )
    np.testing.assert_allclose(
        vectorized.log_likelihood(points), scalar.log_likelihood(points)
    )
    np.testing.assert_allclose(vectorized.log_prior(points), scalar.log_prior(points))


def test_model_rejects_invalid_shape_and_nonfinite_points() -> None:
    model = CallableModel(
        ndim=2,
        parameter_names=("x", "y"),
        log_likelihood_fn=lambda x: np.zeros(len(x)),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    with pytest.raises(InvalidModelOutput, match="shape"):
        model.log_likelihood(np.zeros((3, 1)))
    with pytest.raises(InvalidModelOutput, match="NaN or infinity"):
        model.log_prior(np.array([[0.0, np.nan]]))


def test_model_rejects_wrong_output_shape() -> None:
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.zeros((len(x), 1)),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    with pytest.raises(InvalidModelOutput, match="must return shape"):
        model.log_likelihood(np.zeros((2, 1)))


def test_rng_reproducibility_is_explicit() -> None:
    first = np.random.default_rng(10).normal(size=8)
    second = np.random.default_rng(10).normal(size=8)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_live": 1}, "n_live"),
        ({"n_live": 2, "dlogz": 0.0}, "dlogz"),
        ({"n_live": 2, "proposal_batch_size": 0}, "proposal_batch_size"),
        ({"n_live": 5, "max_likelihood_calls": 4}, "max_likelihood_calls"),
        ({"n_live": 2, "tie_policy": "jitter"}, "tie_policy"),
    ],
)
def test_config_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        MINSConfig(**kwargs)
