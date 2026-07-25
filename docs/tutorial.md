# Fixed-Morph evidence tutorial

Phase 2 estimates evidence after posterior sampling. It does not discover a
posterior from the original prior.

## 1. Supply representative posterior samples

Use an array whose columns match the model coordinates:

```python
posterior_samples = np.load("posterior_samples.npy")
```

Every material posterior mode must be represented. The non-defensive Phase 2
proposal cannot repair omitted support.

## 2. Define normalized model densities

```python
model = CallableModel(
    ndim=2,
    parameter_names=("x", "y"),
    log_likelihood_fn=log_likelihood,
    log_prior_fn=normalized_log_prior,
)
```

Both functions receive `(n, 2)` batches and return `(n,)`. Include all prior
normalization constants.

## 3. Fit MorphZ once

```python
proposal = MorphProposal.fit(
    posterior_samples,
    group_file="params_2-order_TC.json",
    param_names=model.parameter_names,
    kde_bw="silverman",
)
```

Sampling and normalized log-density evaluation come from this same fixed
`GroupKDE`.

## 4. Run

```python
sampler = MINSampler(
    model,
    proposal,
    n_live=500,
    rng=np.random.default_rng(42),
)
result = sampler.run(
    dlogz=1e-4,
    max_iterations=20_000,
    max_proposals_per_replacement=200_000,
    progress=True,
)
```

The progress bar's `rem` field is the estimated live-evidence fraction. The
scientific stop occurs when `rem < dlogz`; hard resource-limit stops are shown
separately.

Inspect termination before interpreting evidence:

```python
print(result.success, result.termination_reason)
print(result.logz, result.logzerr)
print(summarize(result))
```

`logzerr` is the theoretical \(\sqrt{H/N_{\rm live}}\) approximation and does
not include all Morph-fit uncertainty. Repeat complete runs and compare with
direct importance sampling under the same fixed Morph proposal.

Every displayed statistic remains available after the run:

```python
result.history.logz_total
result.history.logzerr
result.history.information
result.history.remaining_fraction
result.history.acceptance_fraction
result.history.likelihood_calls
```

## 5. Obtain equal-weight posterior samples

The native result is weighted:

```python
points = result.all_points
weights = result.posterior_weights
```

When a downstream tool requires equal weights, resample directly from the
result:

```python
equal_samples = result.resample_equal(
    rng=123,
    n_samples=10_000,
)
```

The method uses systematic resampling and returns a new `(n_samples, ndim)`
array. Duplicate rows are normal. Weighted estimates from `all_points` and
`posterior_weights` should be preferred when the consumer supports them.

## 6. Plot stored results

```python
figure, axes = plot_run(result)
figure.savefig("run.png")
```

Plots consume only the result. The full executable Gaussian example is
[`examples/phase2_gaussian.py`](../examples/phase2_gaussian.py).
