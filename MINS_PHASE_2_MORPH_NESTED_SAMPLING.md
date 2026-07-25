# MINS Phase 2: Morph-Assisted Nested Importance Sampling

## Agent implementation specification

This document is the authoritative implementation brief for Phase 2 of the
**Morphing Importance Nested Sampling (MINS)** project.

Phase 1 is assumed to have already created the installable, PyPI-ready package
skeleton. In this phase, implement and test the smallest statistically coherent
sampler that uses a fixed Morph approximation as the pseudo-prior in a standard
nested-sampling calculation.

Read this entire document before changing code. Implement only the scope stated
here. At the end of Phase 2, stop and wait for the project owner to test and
approve the sampler.

---

## 1. Phase objective

Given:

- posterior samples supplied by the user;
- a log-likelihood function \(\log L(\theta)\);
- a normalized log-prior density \(\log \pi(\theta)\);
- a Morph grouping definition and KDE configuration;

the sampler must:

1. fit one fixed normalized Morph density \(q(\theta)\);
2. treat \(q\) as the nested-sampling pseudo-prior;
3. use

   \[
   \log\Psi(\theta)
   =
   \log L(\theta)
   +
   \log\pi(\theta)
   -
   \log q(\theta)
   \]

   as the nested-sampling log pseudo-likelihood;
4. sample from \(q\) subject to the current \(\log\Psi\) constraint;
5. estimate the original marginal likelihood in log space;
6. return the dead points, final live points, evidence estimate, uncertainty,
   run history, and minimal diagnostics.

This phase is a post-processing evidence estimator. The user supplies posterior
samples to train Morph. The sampler does not yet discover the posterior by
starting from the original prior.

---

## 2. Statistical definition

The original marginal likelihood is

\[
Z
=
\int_\Theta L(\theta)\pi(\theta)\,\mathrm d\theta.
\]

In the general tempered construction,

\[
\widetilde g_\beta(\theta)=q(\theta)^\beta,
\qquad
g_\beta(\theta)=\frac{\widetilde g_\beta(\theta)}{Z_\beta}.
\]

Phase 2 fixes

\[
\beta=1.
\]

Because \(q\) must be a normalized density,

\[
g_1(\theta)=q(\theta),
\qquad
Z_1=1.
\]

Therefore,

\[
Z
=
\int_\Theta
\frac{L(\theta)\pi(\theta)}{q(\theta)}
q(\theta)\,\mathrm d\theta
=
\int_\Theta \Psi(\theta)q(\theta)\,\mathrm d\theta.
\]

The transformed nested-sampling problem is consequently:

| Original problem | Phase 2 transformed problem |
|---|---|
| Prior | \(q(\theta)\) |
| Likelihood | \(\Psi(\theta)=L(\theta)\pi(\theta)/q(\theta)\) |
| Log-likelihood | \(\log\Psi=\log L+\log\pi-\log q\) |
| Evidence | The same \(Z\) as the original model |

The information relative to the pseudo-prior is

\[
H
=
D_{\mathrm{KL}}\!\left(
P(\theta\mid X)\,\|\,q(\theta)
\right)
=
\int P(\theta\mid X)
\left[\log\Psi(\theta)-\log Z\right]\,\mathrm d\theta,
\]

and the usual nested-sampling error approximation is

\[
\operatorname{SD}(\log Z)\approx\sqrt{\frac{H}{N_{\rm live}}}.
\]

### Required assumptions

The implementation and API documentation must state these assumptions:

1. `log_prior(theta)` is a normalized log density, including every prior
   normalization constant.
2. The Morph adapter's `log_prob(theta)` is the log of the same normalized
   density from which `sample(n, rng)` draws.
3. The support of \(q\) covers every region with material
   \(L(\theta)\pi(\theta)\).
4. The posterior training samples are representative enough for a
   non-defensive Morph approximation to be usable.
5. The constrained sampler returns draws distributed as
   \(q(\theta\mid\log\Psi(\theta)>\lambda)\), not merely points satisfying the
   inequality.

The non-defensive support assumption is intentionally strong. It is a known
Phase 2 limitation, not something to hide or silently repair.

---

## 3. Scope boundaries

### Implement now

- One fixed Morph proposal trained before nested sampling starts.
- \(\beta=1\) only.
- Independent rejection sampling from Morph under a strict pseudo-likelihood
  constraint.
- Deterministic expected-volume shrinkage:

  \[
  X_i=\exp(-i/N_{\rm live}).
  \]

- Rectangular nested-sampling quadrature.
- Final live-point evidence correction.
- Information and theoretical log-evidence error.
- Minimal histories, diagnostics, checkpoint-ready result data, and plots.
- Serial execution with optional vectorized model evaluation.
- Explicit random-number control where supported by MorphZ.

### Do not implement in Phase 2

- Defensive Morph–prior mixtures.
- Power tempering or estimation of \(Z_\beta\).
- Adaptive Morph refitting during a run.
- Multiple Morph components or meta-proposals.
- Reuse of exploratory points with importance-nested-sampling mixture weights.
- Dynamic allocation of live points.
- MCMC, HMC, slice sampling, normalizing flows, MPI, or GPU backends.
- Automatic prior-to-posterior learning.
- Advanced convergence diagnostics or publication-ready calibration claims.

Leave clean extension points, but do not add speculative machinery.

---

## 4. What the existing prototype proves

The supplied prototype contains useful working components:

- scalar or vectorized likelihood evaluation;
- MorphZ `GroupKDE` training;
- batched proposal generation;
- strict threshold checking;
- iterative replacement of the lowest-valued live point;
- a minimum-value trace;
- successful empirical tests on peak–plateau and Gaussian-shell targets.

These behaviors should become regression tests.

The prototype is evidence that Morph can generate points that climb nested
constraints. It is not yet a complete evidence sampler because it does not
evaluate \(\log q\), form \(\log\Psi\), accumulate quadrature weights, add the
final live-point contribution, or calculate uncertainty.

---

## 5. Mandatory corrections to the prototype

These corrections are required for a statistically meaningful evidence path.

### 5.1 Constrain on the pseudo-likelihood

Rename ambiguous variables such as `likelihood_fn`,
`minimum_likelihood`, and `proposal_likelihoods` in the sampler internals.

Nested sampling must order points using:

```python
log_psi = log_likelihood + log_prior - log_q
```

not the raw likelihood alone.

Raw \(\log L\) and \(\log\Psi\) can have different orderings. Preserve both
values in the result.

### 5.2 Draw initial live points from Morph

Do not initialize the live set by selecting a subset of the posterior samples
used to train Morph. Training samples are input data for constructing \(q\);
they are not guaranteed to be independent draws from the fitted density.

Instead:

```python
live_theta = proposal.sample(n_live, rng)
```

Then calculate `log_likelihood`, `log_prior`, `log_q`, and `log_psi` for those
new points.

### 5.3 Do not select the maximum proposal in a batch

The current helper selects the highest-likelihood proposal from the first batch
containing an acceptable point. This produces a greedy order-statistic draw,
not a draw from the required constrained density.

The correct evidence mode must use one of these equivalent strategies:

1. scan independently drawn proposals in generation order and accept the first
   point satisfying `log_psi > threshold`; or
2. from a batch of independent proposals, choose one acceptable point uniformly
   at random.

Never use `argmax(log_psi)` in the evidence path.

If the existing maximum-of-batch behavior is retained for experimentation,
name it `selection="greedy_max"`, mark it as invalid for evidence estimation,
emit a prominent warning, and exclude it from the default public API.

### 5.4 Avoid an unbounded retry loop

The prototype retries forever when no proposal clears the threshold. Replace
this with explicit resource limits:

- `max_proposals_per_replacement`;
- optional `max_likelihood_calls`;
- optional `max_wall_time`.

When a limit is reached, terminate cleanly with a typed reason such as
`"constrained_sampling_exhausted"`. Return a partial result with
`success=False`; do not hang and do not fabricate a replacement.

### 5.5 Include the live-point remainder

The reported evidence must include both dead-point quadrature and the final
live-point contribution. Returning the dead-point sum alone is incomplete.

---

## 6. Reference algorithm

### 6.1 Definitions

Let:

- \(N=N_{\rm live}\);
- \(K\) be the number of discarded points;
- \(X_0=1\);
- \(X_i=\exp(-i/N)\);
- \(\lambda_i\) be the discarded point's pseudo-likelihood;
- \(\ell_i=\log\lambda_i=\log\Psi(\theta_i)\).

For the Phase 2 rectangular quadrature,

\[
\Delta X_i=X_{i-1}-X_i
\]

and

\[
\widehat Z_{\rm dead}
=
\sum_{i=1}^{K}\Delta X_i\,\lambda_i.
\]

The remaining live contribution is

\[
\widehat Z_{\rm live}
=
\frac{X_K}{N}
\sum_{j=1}^{N}\Psi(\theta_j^{\rm live}).
\]

The final estimate is

\[
\widehat Z
=
\widehat Z_{\rm dead}
+
\widehat Z_{\rm live}.
\]

### 6.2 Pseudocode

```text
INPUT:
    posterior_training_samples
    model.log_likelihood
    model.log_prior
    n_live
    Morph configuration
    stopping configuration
    rng

FIT:
    q = fit one fixed Morph density to posterior_training_samples
    validate that q supports sample(n, rng) and normalized log_prob(theta)

INITIALIZE:
    live_theta = q.sample(n_live, rng)
    evaluate logL, logPrior, logQ for every live point
    live_logPsi = logL + logPrior - logQ
    reject or fail clearly on NaN values
    logZ_dead = -infinity
    logX_prev = 0
    iteration = 0

REPEAT:
    iteration += 1

    worst = argmin(live_logPsi)
    threshold = live_logPsi[worst]

    logX = -iteration / n_live
    logDeltaX = log(exp(logX_prev) - exp(logX))
    logWt = logDeltaX + threshold
    logZ_dead = logaddexp(logZ_dead, logWt)

    store the complete worst point as a dead point

    draw proposals independently from q in batches
    evaluate logPsi for all proposals
    find indices where logPsi > threshold
    if at least one is valid:
        choose one valid proposal without favoring larger logPsi
        replace the worst live point and reuse its cached evaluations
    else:
        draw another batch unless a resource limit has been reached

    logX_prev = logX

    estimate the current live remainder:
        logZ_live =
            logX + logsumexp(live_logPsi) - log(n_live)
        logZ_total = logaddexp(logZ_dead, logZ_live)

    update history and stopping diagnostics

    if scientific stopping rule passes:
        termination_reason = "remaining_evidence"
        break

    if a hard resource limit is reached:
        termination_reason = the relevant hard-limit reason
        success = False
        break

FINALIZE:
    calculate the final live-point contributions individually
    combine dead and live contributions with logsumexp
    normalize all quadrature contributions into posterior weights
    calculate H and sqrt(H / n_live)
    construct and return a MINSResult
```

### 6.3 Strict inequality and ties

Use:

```python
candidate_log_psi > threshold
```

for ordinary continuous targets.

Peak–plateau targets can contain exact ties with nonzero probability. A strict
constraint can then stall on a plateau. Do not silently change the scientific
algorithm globally. Implement an explicit tie policy:

```python
tie_policy: Literal["strict", "randomized_plateau"]
```

- `"strict"` is the default nested-sampling rule.
- `"randomized_plateau"` augments every point with an independent
  \(u\sim\operatorname{Uniform}(0,1)\) and orders points lexicographically by
  \((\log\Psi,u)\). A candidate clears the constraint if its
  `log_psi` is larger, or if `log_psi` is exactly tied and its `u` is larger.
  This is nested sampling on the augmented pseudo-prior
  \(q(\theta)\operatorname{Uniform}(u)\); because \(\Psi\) does not depend on
  \(u\), the evidence remains unchanged.

At minimum, detect repeated threshold ties and report a clear
`"plateau_stall"` diagnostic rather than looping forever. Never perturb
`log_psi` with an arbitrary floating-point jitter in general sampler code. If
the randomized policy is enabled, store the auxiliary tie-breaking values for
reproducibility.

### 6.4 Mapping from the naïve R sampler

The initial R implementation contains the correct rectangular expected-volume
pattern:

| R expression | Phase 2 equivalent |
|---|---|
| `width1 <- log(1-exp(-1/n))` | \(\log(1-e^{-1/N})\) |
| `logwidth <- width1-(iter-1)/n` | \(\log(X_{i-1}-X_i)\) |
| `log.plus(logZb, logwidth+logL[worst])` | `np.logaddexp(logz_dead, log_delta_x + dead_log_psi)` |
| `max(logL)-iter/n` | conservative `log_x + max(live_log_psi)` remainder proxy |
| `log(mean(exp(logL)))-iter/n` | `log_x + logsumexp(live_log_psi) - log(n_live)` |

In MINS, replace every R `logL` used for ordering or quadrature with
`log_psi`.

The R interval width is a rectangular nested-sampling weight. It is not divided
by two. Do not label it as a trapezoidal weight. A genuine trapezoidal rule
requires neighbouring likelihood ordinates and explicit endpoint treatment;
that is outside the Phase 2 reference implementation.

The R value called `Corrected_Evidence` includes the remaining live points. Its
log-space equivalent is the primary final `result.logz`. The dead-point-only
value may be stored as a diagnostic, but must not be presented as the final
evidence.

---

## 7. Log-space numerical implementation

All evidence calculations must remain in log space.

Use `numpy.logaddexp` and `scipy.special.logsumexp`. Implement the shrinking
interval width stably:

```python
def logdiffexp(log_a: float, log_b: float) -> float:
    """Return log(exp(log_a) - exp(log_b)) for log_a > log_b."""
    if not log_a > log_b:
        raise ValueError("log_a must be greater than log_b")
    return log_a + np.log1p(-np.exp(log_b - log_a))
```

At iteration \(i\):

```python
log_x_prev = -(i - 1) / n_live
log_x = -i / n_live
log_delta_x = logdiffexp(log_x_prev, log_x)
log_dead_contribution = log_delta_x + dead_log_psi
```

For the live remainder:

```python
log_z_live = (
    log_x
    + logsumexp(live_log_psi)
    - np.log(n_live)
)
```

For final normalized posterior weights, concatenate:

- every dead-point `log_delta_x + dead_log_psi`;
- every live-point `log_x - log(n_live) + live_log_psi`.

Then:

```python
log_z = logsumexp(log_contributions)
posterior_weights = np.exp(log_contributions - log_z)
```

Compute the information as:

```python
information = np.sum(
    posterior_weights * (all_log_psi - log_z)
)
```

Allow only a tiny negative value caused by floating-point error to be clipped to
zero. A materially negative value is a failed invariant and must produce a
warning or exception.

Do not exponentiate `log_likelihood`, `log_psi`, or `log_z` inside the main
sampler loop.

---

## 8. Stopping rule

The Phase 2 scientific stopping rule mirrors the simple reference nested
sampler. Let:

\[
\log Z_{\rm live}
=
\log X_K
+
\log\left[
\frac{1}{N}
\sum_{j=1}^{N}\Psi(\theta_j^{\rm live})
\right].
\]

Stop when the estimated remaining evidence is small relative to the current
total:

```python
log_z_live - log_z_total < np.log(dlogz)
```

where `dlogz` is a positive relative remaining-evidence tolerance.

Also support hard limits:

```text
max_iterations
max_likelihood_calls
max_proposals_per_replacement
max_wall_time
```

The result must record exactly one termination reason. Reaching
`max_iterations`, `max_likelihood_calls`, a proposal limit, or a wall-time limit
does not count as scientific convergence.

For additional monitoring, store the conservative upper proxy

\[
\log Z_{\rm remaining,max}
=
\log X_K+\max_j\log\Psi_j^{\rm live},
\]

but do not confuse it with the mean-live estimate used above.

---

## 9. Required package boundaries

Use the Phase 1 package namespace and keep responsibilities disentangled. The
exact filenames may follow the established skeleton, but preserve these
boundaries:

```text
src/mins/
├── model.py
├── proposals/
│   ├── base.py
│   └── morph.py
├── constrained.py
├── quadrature.py
├── sampler.py
├── results.py
├── diagnostics.py
└── plotting.py
```

### `model.py`

- Define the model protocol.
- Validate output shapes.
- Adapt scalar and vectorized callables without hiding arbitrary user-code
  exceptions.
- Count likelihood and prior evaluations.

Suggested public protocol:

```python
class Model(Protocol):
    ndim: int
    parameter_names: Sequence[str]

    def log_likelihood(self, theta: NDArray) -> NDArray: ...
    def log_prior(self, theta: NDArray) -> NDArray: ...
```

### `proposals/base.py`

Define the proposal contract independently of MorphZ:

```python
class Proposal(Protocol):
    ndim: int

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray: ...

    def log_prob(self, theta: NDArray) -> NDArray: ...
```

Sampling and density evaluation must describe the same normalized
distribution.

### `proposals/morph.py`

- Own every direct interaction with `morphZ.GroupKDE`.
- Train from a copied, validated 2D sample array.
- Adapt the actual installed MorphZ sampling and density APIs.
- Do not guess method names or density conventions. Inspect the installed API
  and write adapter tests.
- Convert density output to log density once, handling zero density as
  `-np.inf`.
- Validate dimensions and finite training input.
- Record Morph configuration as immutable metadata.
- Make RNG limitations explicit. Do not silently reseed NumPy's global RNG.

If MorphZ cannot provide a normalized density evaluation consistent with
`resample`, stop and report the blocker. Sampling alone is insufficient for
nested importance sampling.

### `constrained.py`

- Implement independent batched rejection sampling.
- Accept the first valid proposal or choose uniformly among valid proposals.
- Cache and return `theta`, `log_likelihood`, `log_prior`, `log_q`, and
  `log_psi`.
- Return proposal counts and rejection counts.
- Enforce proposal and call limits.
- Contain no evidence quadrature.

Suggested return type:

```python
@dataclass(frozen=True)
class EvaluatedPoint:
    theta: NDArray
    log_likelihood: float
    log_prior: float
    log_q: float
    log_psi: float


@dataclass(frozen=True)
class ConstrainedDraw:
    point: EvaluatedPoint
    n_proposed: int
    n_valid: int
```

### `quadrature.py`

- Implement `logdiffexp`.
- Compute dead and live log-contributions.
- Normalize posterior weights.
- Compute \(\log Z\), \(H\), and \(\sqrt{H/N}\).
- Be pure: no Morph calls, plotting, logging, or mutable sampler state.

### `sampler.py`

- Coordinate the state machine.
- Own live-point replacement and stopping.
- Use the proposal, model evaluator, constrained sampler, and quadrature through
  their interfaces.
- Contain no MorphZ-specific code and no plotting.

### `results.py`

Return a typed result object. At minimum:

```python
@dataclass(frozen=True)
class MINSResult:
    logz: float
    logzerr: float
    information: float
    success: bool
    termination_reason: str
    niter: int
    nlive: int
    n_likelihood_calls: int
    n_proposals: int
    dead_points: NDArray
    dead_log_likelihood: NDArray
    dead_log_prior: NDArray
    dead_log_q: NDArray
    dead_log_psi: NDArray
    dead_log_x: NDArray
    dead_log_weights: NDArray
    final_live_points: NDArray
    final_live_log_likelihood: NDArray
    final_live_log_prior: NDArray
    final_live_log_q: NDArray
    final_live_log_psi: NDArray
    log_posterior_weights: NDArray
    history: RunHistory
    config: MINSConfig
```

The final arrays must be sufficient to recompute \(\log Z\), posterior weights,
and \(H\) without rerunning the likelihood.

### `diagnostics.py` and `plotting.py`

Diagnostic functions consume stored results; they do not alter the sampler.
Plot functions must return Matplotlib figure and axes objects and must not call
`plt.show()` internally.

---

## 10. Public API for Phase 2

Aim for a small interface:

```python
proposal = MorphProposal.fit(
    posterior_samples,
    group_file=group_file,
    kde_bw=0.02,
)

sampler = MINSampler(
    model=model,
    proposal=proposal,
    n_live=500,
    rng=np.random.default_rng(42),
)

result = sampler.run(
    dlogz=1e-4,
    max_iterations=20_000,
    max_proposals_per_replacement=200_000,
)
```

Also provide a convenience constructor if it remains thin:

```python
sampler = MINSampler.from_posterior_samples(
    model=model,
    posterior_samples=posterior_samples,
    morph_config=morph_config,
    n_live=500,
    rng=42,
)
```

Do not make plotting, progress bars, or filesystem output mandatory. Use:

- `progress=False` by default in library code;
- structured logging rather than `print`;
- an optional progress callback or optional `tqdm` integration.

---

## 11. Input validation and failure behavior

Validate before the expensive run:

- training samples have shape `(n_samples, ndim)`;
- `n_samples > 0`;
- all training values are finite;
- `n_live >= 2`;
- batch sizes and limits are positive integers;
- `0 < dlogz < 1`;
- model and proposal dimensions agree;
- model evaluation returns one value per point;
- no NaN is returned by `log_likelihood`, `log_prior`, or `log_q`.

Interpret values as follows:

- `log_prior == -np.inf`: outside prior support; hence `log_psi == -np.inf`;
- `log_likelihood == -np.inf`: a valid zero-likelihood point;
- `log_q == -np.inf` while `log_likelihood + log_prior` is finite: proposal
  support failure; stop with a clear error;
- any NaN: invalid model or proposal output; stop immediately;
- positive infinity: reject unless a documented benchmark genuinely requires
  it, because evidence arithmetic will otherwise be undefined.

Never catch broad `Exception` merely to retry scalar evaluation. The current
prototype's vectorization fallback should be narrowed so real errors raised by
user likelihood code remain visible.

---

## 12. Minimal diagnostics

Store these at every iteration:

- iteration number;
- discarded `log_psi`;
- `log_x`;
- `log_delta_x`;
- cumulative dead-point `logz`;
- estimated live remainder;
- current total `logz`;
- minimum, median, and maximum live `log_psi`;
- proposals attempted for the replacement;
- cumulative likelihood calls;
- cumulative proposal acceptance fraction;
- elapsed wall time.

Return these final summaries:

- \(\log\widehat Z\);
- \(\sqrt{H/N_{\rm live}}\);
- \(H\);
- iteration and evaluation counts;
- total proposal acceptance fraction;
- maximum proposals needed for one replacement;
- termination reason;
- whether the threshold sequence is monotone;
- number of non-finite evaluations by category;
- posterior effective sample size from quadrature weights:

  \[
  \operatorname{ESS}
  =
  \frac{1}{\sum_i \widetilde w_i^2}.
  \]

Provide one minimal run plot with three aligned panels:

1. dead and live-range `log_psi` versus iteration;
2. cumulative `logz` and estimated live remainder versus iteration;
3. cumulative likelihood calls and per-iteration proposal count.

Keep peak–plateau-specific and Gaussian-shell-specific visualizations in
examples or benchmarks, not in the core sampler.

---

## 13. Test plan

The test suite must distinguish deterministic unit tests from repeated
statistical tests. Mark slow statistical tests appropriately.

### 13.1 Unit tests

Test:

- `logdiffexp` against direct arithmetic in a safe numerical range;
- dead-point and live-point log contributions;
- posterior weights sum to one;
- recomputed `logz` equals the stored result;
- scalar and vectorized model evaluation agree;
- proposal and model shape validation;
- NaN, infinity, and support-failure handling;
- deterministic behavior for the same seed where MorphZ permits it;
- result serialization round-trip if Phase 1 already defined serialization.

### 13.2 Constrained-sampler invariants

For every replacement:

- accepted `log_psi` is strictly greater than the discarded threshold;
- the accepted point and its cached `log_psi` agree on reevaluation;
- dead-point thresholds are non-decreasing;
- the number of live points remains constant;
- a proposal-limit failure returns a partial failed result;
- no greedy `argmax` selection is used in evidence mode.

### 13.3 Statistical test that detects maximum-of-batch bias

Construct a one-dimensional proposal \(q=\operatorname{Uniform}(0,1)\) with
`log_psi = log(theta)` and threshold \(\theta>0.5\).

Correct constrained draws are uniform on \((0.5,1)\), with mean \(0.75\).
Generate many constrained draws and test the empirical mean and/or a
distributional statistic against the analytic target.

This test must fail for maximum-of-batch selection and pass for first-valid or
uniform-among-valid selection.

### 13.4 Constant-integrand identity test

Choose \(L(\theta)\pi(\theta)=Cq(\theta)\), so:

\[
\Psi(\theta)=C
\quad\text{and}\quad
Z=C.
\]

This is a crucial bookkeeping test. Because every pseudo-likelihood value is
equal, use the documented plateau policy or test quadrature directly. Confirm
that dead plus live volume recovers \(C\) to numerical precision.

### 13.5 Analytic Gaussian test

Use a normalized Gaussian prior and a Gaussian likelihood for which the
evidence is available analytically. Train Morph from independent posterior
draws, run multiple seeds, and test:

- mean \(\log Z\) bias;
- empirical spread across runs;
- approximate consistency with reported `logzerr`;
- improvement or stability as `n_live` increases.

### 13.6 Peak–plateau regression benchmark

Port the already successful peak–plateau experiment into a documented benchmark.
Record:

- exact target definition;
- analytic evidence;
- plateau/tie policy;
- seed;
- `n_live`, bandwidth, grouping, and stopping configuration;
- estimated \(\log Z\), error, calls, and termination reason.

The test must guard against stalling as well as evidence error.

### 13.7 Gaussian-shell regression benchmark

Port the working Gaussian-shell experiment without embedding a hard-coded path
such as:

```text
./gaussian_shell/params_2-order_TC.json
```

Test at least:

- correct dimensionality and group-file resolution;
- monotone constraints;
- finite Morph log densities;
- finite evidence estimate;
- agreement with the analytic or high-accuracy reference evidence within a
  predeclared repeated-run tolerance.

Keep a small version in CI and a larger accuracy study under a slow benchmark
marker.

### 13.8 Direct importance-sampling cross-check

For the same fixed normalized \(q\), independently estimate:

\[
\widehat Z_{\rm IS}
=
\frac{1}{M}\sum_{j=1}^{M}
\frac{L(\theta_j)\pi(\theta_j)}{q(\theta_j)},
\qquad
\theta_j\sim q.
\]

With sufficiently large \(M\), the direct importance result should agree with
the nested-importance result within Monte Carlo uncertainty. This is a
high-value end-to-end check of `log_prior`, `log_q`, and `log_psi`
bookkeeping.

---

## 14. Documentation required in this phase

Write documentation alongside implementation, not afterward.

Required:

1. **Mathematical method page**
   - the transformation from \(Z=\int L\pi\) to \(Z=\int\Psi q\);
   - the \(\beta=1\) restriction;
   - quadrature and final live-point correction;
   - information and uncertainty;
   - assumptions and non-defensive support limitation.

2. **API page**
   - model protocol;
   - Morph proposal construction;
   - sampler configuration;
   - result fields;
   - failure and termination semantics.

3. **Minimal tutorial**
   - load posterior samples;
   - fit Morph;
   - define `log_likelihood` and normalized `log_prior`;
   - run the sampler;
   - inspect `logz`, `logzerr`, diagnostics, and the run plot.

4. **Benchmark pages**
   - peak–plateau;
   - Gaussian shells;
   - exact/reference evidence and full reproducibility configuration.

5. **Developer note**
   - map each function from the supplied prototype to its new module;
   - state which prototype behaviors were corrected and why;
   - explain how later defensive/adaptive phases can extend the interfaces
     without changing Phase 2 result semantics.

Every public class and function needs a NumPy-style docstring with shapes,
units/conventions, exceptions, and return values.

---

## 15. Implementation sequence

Follow this order:

1. Add the `Proposal` and `Model` protocols.
2. Implement and test `MorphProposal.sample` and `MorphProposal.log_prob`.
3. Implement the evaluated-point batch function and exact `log_psi`
   bookkeeping.
4. Implement unbiased constrained rejection sampling and its resource limits.
5. Implement pure log-space quadrature utilities.
6. Implement the sampler state machine with deterministic shrinkage.
7. Add final live-point correction, posterior weights, \(H\), and `logzerr`.
8. Implement the typed immutable result and run history.
9. Add minimal diagnostics and plotting.
10. Port peak–plateau and Gaussian-shell experiments into benchmarks.
11. Add the analytic and direct-importance cross-checks.
12. Complete the required documentation and examples.
13. Run the full Phase 2 test suite and produce a short validation report.
14. Stop.

Do not optimize parallelism before the reference serial implementation passes
the analytic and statistical tests.

---

## 16. Phase 2 acceptance gate

Phase 2 is complete only when all of the following are true:

- The package builds and installs using the Phase 1 workflow.
- Morph sampling and normalized log-density evaluation are both available and
  tested through the proposal interface.
- Live points are newly drawn from Morph.
- All ordering and constraints use `log_psi`, not raw `log_likelihood`.
- Evidence-mode constrained draws are unbiased with respect to the restricted
  Morph density.
- The maximum-of-batch statistical test passes by rejecting that behavior from
  evidence mode.
- Evidence arithmetic is entirely in log space.
- The final live-point contribution is included.
- `logz`, `H`, `logzerr`, posterior weights, histories, and termination reason
  can be recomputed from the stored result.
- Constant, analytic Gaussian, peak–plateau, and Gaussian-shell tests pass.
- The direct importance-sampling cross-check agrees within the declared
  tolerance.
- Hard resource limits terminate cleanly without infinite retry loops.
- Core code contains no plotting, hard-coded benchmark paths, or direct
  MorphZ calls outside the Morph adapter.
- Public documentation and the validation report are complete.

## Mandatory stop

After satisfying the acceptance gate:

1. summarize the files changed;
2. report tests and benchmark results, including any limitations;
3. provide exact commands for the project owner to reproduce the tests;
4. stop and wait for approval.

Do **not** begin defensive mixtures, power tempering, adaptive Morph training,
prior-started exploration, or full importance nested sampling until the project
owner has independently tested and approved this Phase 2 sampler.
