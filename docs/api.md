# Phase 2 API

MINS uses natural logarithms throughout and expects point batches with shape
`(n, ndim)`.

## Model

`Model` defines:

```python
ndim: int
parameter_names: Sequence[str]
log_likelihood(theta) -> ndarray shape (n,)
log_prior(theta) -> ndarray shape (n,)
```

`log_prior` must be normalized and must include all constants. Values of
`-inf` represent zero density. NaN and positive infinity are rejected.

`CallableModel` adapts explicit vectorized functions. Set `vectorized=False`
only when both functions accept individual `(ndim,)` rows; the adapter does not
catch user exceptions to guess vectorization.

## MorphProposal

```python
proposal = MorphProposal.fit(
    posterior_samples,
    morph_type="2_group",
    param_names=("x", "y"),
    kde_bw=0.02,
)
```

`morph_type="{k}_group"` follows MorphZ's automatic grouped workflow:
`Nth_TC.compute_and_save_tc` computes the k-order total correlations, then
`GroupKDE` greedily selects non-overlapping groups. For example, use
`morph_type="2_group"` or `morph_type="3_group"`; the literal string
`"n_group"` is not valid. Intermediate TC files are held in a temporary
directory and removed after fitting.

Alternatively, load precomputed MorphZ entries with `group_file=...`, or pass
them directly through `groups=...`; `groups=[]` selects independent
one-dimensional components. These three grouping inputs are mutually
exclusive. Training data must be finite with shape `(n_samples, ndim)`.

The selected structure is recorded in metadata:

```python
proposal.metadata.selected_groups
proposal.metadata.single_parameters
```

The adapter uses the installed `morphZ.GroupKDE` directly. It copies training
data, resolves grouping data in memory, calls `GroupKDE.resample` for draws,
and calls the same fitted object's normalized `logpdf` for `log_prob`. MorphZ
0.4.1 uses integer seeds, so the adapter derives one from the sampler's
explicit NumPy Generator per resample. The inspected MorphZ implementation
restores legacy global RNG state.

## MINSampler

```python
sampler = MINSampler(
    model,
    proposal,
    n_live=500,
    rng=np.random.default_rng(42),
    proposal_batch_size=64,
    tie_policy="strict",
)
result = sampler.run(
    dlogz=1e-4,
    max_iterations=20_000,
    max_proposals_per_replacement=200_000,
    max_likelihood_calls=None,
    max_wall_time=None,
    progress=True,
)
```

`dlogz` and `stopping` are mutually exclusive. Omitting both preserves the
legacy `dlogz=1e-3` behavior. The multi-criterion API uses immutable
`StoppingCriterionConfig` and `StoppingPolicy` values:

```python
from mins import StoppingCriterionConfig, StoppingPolicy

stopping = StoppingPolicy(
    criteria=(
        StoppingCriterionConfig("live_logz_error", 5e-3),
        StoppingCriterionConfig("remaining_fraction", 5e-2),
        StoppingCriterionConfig("logz_stability", 5e-3),
    ),
    mode="all",
    consecutive=3,
    min_iterations=10,
    stability_window=10,
)
result = sampler.run(
    stopping=stopping,
    max_iterations=20_000,
)
```

The resolved policy is retained in `result.config.stopping`. See
[Stopping criteria](stopping.md) for metric definitions, custom `"all"` and
`"any"` policies, persistence rules, and limitations.

The constructor validates dimensions and owns the supplied generator. Live
points are new proposal draws. At every iteration, ordering and rejection use
`log_psi = log_likelihood + log_prior - log_q`.

`tie_policy="strict"` is correct for ordinary continuous pseudo-likelihoods.
Use `"randomized_plateau"` for targets with exact nonzero-probability ties. It
stores an independent uniform auxiliary value for every dead and live point.

Progress is silent by default. `progress=True` enables an optional
`tqdm.auto` bar suitable for terminals and notebooks. Install it through
`.[progress]`. The bar displays:

- iteration and `n_live`;
- likelihood calls and constrained-proposal efficiency;
- current total `logZ` and theoretical `logZerr`;
- live-set log-evidence error and remaining-evidence fraction;
- live-point ESS;
- the current stopping streak and required consecutive count.

A custom callable can be passed instead of `True`. It receives a mapping after
every completed iteration with the displayed quantities plus
`live_mean_rse`, `logz_stability`, dead/live evidence, per-iteration proposal
counts, live `logPsi` range, and elapsed time. It also receives integer met
flags named `criterion_<name>_met` for each enabled criterion. Library code
does not otherwise print, display, or write files.

## Result

`MINSResult` is frozen and its arrays are read-only. Key scalar fields are:

- `logz`, `logzerr`, `information`;
- `success`, `termination_reason`;
- `niter`, `nlive`, likelihood/prior/proposal counts;
- complete `config` and initial/final RNG state representations.

Dead and final-live arrays separately store points, `log_likelihood`,
`log_prior`, `log_q`, `log_psi`, volume/weight values, and tie breakers.
`log_posterior_weights` covers dead then live points. Convenience properties
`all_points`, `all_log_psi`, and `posterior_weights` use that same ordering.

For consumers that require unweighted samples:

```python
equal_samples = result.resample_equal(
    rng=123,
    n_samples=10_000,
)
```

`resample_equal` uses randomized systematic resampling and has no Dynesty
dependency. The explicit seed or NumPy Generator is required for
reproducibility. If `n_samples` is omitted, it returns `niter + nlive` rows.
Repeated points are expected. Retain the original points and weights for
highest-fidelity posterior summaries.

`RunHistory` stores every completed threshold, volume interval, cumulative
dead/live/total log evidence, information, `logzerr`, remaining fraction, live
pseudo-likelihood ESS, live-mean RSE, live log-evidence error, log-evidence
stability, stopping streak, live pseudo-likelihood range, calls, proposal
counts, acceptance fraction, and elapsed time. Every array is read-only and
has shape `(niter,)`; early `logz_stability` entries are `NaN` until its exact
window is available.

## Failure semantics

Scientific termination reasons are:

- `remaining_evidence` for the legacy `dlogz` path;
- `stopping_criteria` for an explicit `StoppingPolicy`.

Both have `success=True`. Hard stops include:

- `max_iterations`;
- `max_likelihood_calls`;
- `max_wall_time`;
- `constrained_sampling_exhausted`;
- `plateau_stall`.

Hard stops return a valid partial result with the current final-live correction.
Malformed model/proposal output and proposal support failure raise typed
exceptions because no statistically coherent result can be formed.

## Diagnostics and plots

`mins.diagnostics.summarize(result)` reports posterior ESS, proposal acceptance,
maximum proposals per replacement, threshold monotonicity, the separate
conservative max-live remainder, and the final values of every stopping
diagnostic and streak.

`plot_run`, `plot_weight_health`, and `plot_posterior_1d` return Matplotlib
objects. They never call `show` or save files.
