# MINS

MINS is an experimental implementation of Morph-assisted nested importance
sampling for post-processing Bayesian posterior samples.

> **Development status:** pre-alpha research software. The importance Morph is
> fixed and non-defensive. Missing importance support can bias the evidence
> estimate. The optional adaptive proposal scheme is heuristic and can add
> further bias; the API is not yet stable and no PyPI release has been made.

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

from mins import (
    CallableModel,
    MINSampler,
    MorphProposal,
    StoppingCriterionConfig,
    StoppingPolicy,
)

model = CallableModel(
    ndim=1,
    parameter_names=("x",),
    log_likelihood_fn=lambda x: -0.5 * x[:, 0] ** 2,
    log_prior_fn=lambda x: -0.5 * x[:, 0] ** 2 - 0.5 * np.log(2.0 * np.pi),
)

posterior_samples = np.random.default_rng(7).normal(size=(2_000, 1))
importance_morph = MorphProposal.fit(
    posterior_samples,
    param_names=model.parameter_names,
    groups=[],
)
sampler = MINSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="fixed_morph",
    n_live=100,
    rng=42,
)
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

The `dlogz` interface remains the default scientific behavior. It bounds the
estimated log-evidence increment from adding the mean-live remainder,
`log(Z_dead + Z_live) - log(Z_dead)`. A multi-criterion policy is available as
an opt-in experimental alternative:

```python
n_live = sampler.n_live
stopping = StoppingPolicy(
    criteria=(
        StoppingCriterionConfig("remaining_dlogz", 1e-2),
        StoppingCriterionConfig("live_logz_error", 2e-3),
        StoppingCriterionConfig("logz_stability", 5e-3),
    ),
    mode="all",
    consecutive=3,
    min_iterations=n_live,
    stability_window=n_live,
)
result = sampler.run(stopping=stopping, max_iterations=20_000)
```

These thresholds are illustrative calibration choices, not universal
constants; calibrate them with repeated runs for the target family.
`remaining_dlogz` is the estimated log-evidence correction,
`live_logz_error` estimates uncertainty transmitted from the finite live-set
mean, and `remaining_fraction` remains available separately for the live
evidence share. See the [stopping guide](docs/stopping.md) for the complete API
and limitations.

Set `proposal_scheme="adaptive_morph"` to refit a separate proposal Morph from
the complete equal-weight live set every `proposal_update_interval=25`
completed iterations. The original `importance_morph` remains fixed and still
defines `log_q0` and `log_psi0`. Adaptive candidates are accepted directly
after the `log_psi0` constraint, so this mode is a heuristic: it does not in
general draw from the constrained importance Morph and its `logZ` may be
biased.

The terminal live display reports iteration, live-point count, likelihood
calls, proposal efficiency, `logZ`, theoretical `logZerr`, and the stopping
streak without treating the hard iteration limit as a convergence percentage.
Criterion-specific metrics appear only when enabled. Proposal revision fields
appear only after adaptive proposal updates are used. The remaining-evidence
fraction remains available in callbacks and history rather than the terminal
display.

`result.all_points` and `result.posterior_weights` are the primary weighted
posterior representation. `result.resample_equal(...)` provides reproducible
equal-weight draws for tools that do not accept weights.

The normalized prior, Morph target transformation, evidence quadrature, and
stopping semantics are defined in
[the mathematical contract](docs/mathematical_contract.md). See the
[API guide](docs/api.md), [stopping guide](docs/stopping.md), and
[Phase 2 tutorial](docs/tutorial.md) before using the estimator.

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

The non-CI eggbox comparison in
[`benchmarks/compare_stopping_policies.py`](benchmarks/compare_stopping_policies.py)
records repeated-seed accuracy, cost, failure-rate, and speed-up summaries for
two `remaining_dlogz` tolerances and a hybrid policy.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
