# Fixed-importance Morph evidence tutorial

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

## 3. Fit the importance Morph once

```python
importance_morph = MorphProposal.fit(
    posterior_samples,
    morph_type="2_group",
    param_names=model.parameter_names,
    kde_bw="silverman",
)
```

MorphZ computes all second-order total correlations, greedily selects disjoint
groups, and fits one fixed `GroupKDE`. This object remains the importance
density `q0` for the complete run.

## 4. Run

```python
sampler = MINSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="fixed_morph",
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
result.history.proposal_revision
```

To experiment with periodic live-set proposal fits, select:

```python
sampler = MINSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="adaptive_morph",
    proposal_update_interval=25,
    n_live=500,
    rng=np.random.default_rng(42),
)
```

The proposal is refitted from all current live rows before replacements 26,
51, 76, and so on. The original importance Morph still computes every
`log_q0` and `log_psi0`. Threshold-only adaptive acceptance is heuristic,
however: it samples from the refitted proposal under the constraint rather
than from constrained `q0`, so the resulting `logZ` can be biased.

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
