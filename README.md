# MINS

MINS is an experimental implementation of Morph-assisted nested importance
sampling for post-processing Bayesian posterior samples.

> **Development status:** pre-alpha research software. The Phase 2 method uses
> a fixed, non-defensive Morph density. Missing proposal support can bias the
> evidence estimate, and the API is not yet stable. No PyPI release has been
> made.

## Installation

For development, install the repository and its quality tools:

```bash
python -m pip install -e ".[dev]"
```

The scientific Morph adapter requires MorphZ 0.4.1.dev2 or newer. If MorphZ is
already installed from its source repository, the base package can be installed
without resolving the optional extra:

```bash
python -m pip install -e . --no-deps
```

For a normal user installation with Morph and the terminal/notebook progress
bar:

```bash
python -m pip install -e ".[morph,progress]"
```

## Minimal API sketch

```python
import numpy as np

from mins import CallableModel, MINSampler, MorphProposal

model = CallableModel(
    ndim=1,
    parameter_names=("x",),
    log_likelihood_fn=lambda x: -0.5 * x[:, 0] ** 2,
    log_prior_fn=lambda x: -0.5 * x[:, 0] ** 2 - 0.5 * np.log(2.0 * np.pi),
)

posterior_samples = np.random.default_rng(7).normal(size=(2_000, 1))
proposal = MorphProposal.fit(
    posterior_samples,
    param_names=model.parameter_names,
    groups=[],
)
sampler = MINSampler(model, proposal, n_live=100, rng=42)
result = sampler.run(
    dlogz=1e-3,
    max_iterations=5_000,
    progress=True,
)

print(result.logz, result.logzerr, result.termination_reason)

equal_samples = result.resample_equal(
    rng=43,
    n_samples=10_000,
)
```

The progress bar reports iteration, live-point count, likelihood calls,
proposal efficiency, `logZ`, current theoretical `logZerr`, information,
remaining-evidence fraction, and the current `logPsi` threshold.

`result.all_points` and `result.posterior_weights` are the primary weighted
posterior representation. `result.resample_equal(...)` provides reproducible
equal-weight draws for tools that do not accept weights.

The normalized prior, Morph target transformation, evidence quadrature, and
stopping semantics are defined in
[the mathematical contract](docs/mathematical_contract.md). See the
[API guide](docs/api.md) and [Phase 2 tutorial](docs/tutorial.md) before using
the estimator.

## Development

```bash
python -m pytest -m "not slow"
python -m ruff format --check .
python -m ruff check .
python -m build
python -m twine check dist/*
```

Phase reports and exact validation commands are stored under
[`docs/phases/`](docs/phases/).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
