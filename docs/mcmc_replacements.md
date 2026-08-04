# Constrained MCMC replacements

For standalone equation-by-equation audits, see the current standard
[`rwalk` implementation](rwalk_implementation.tex) and the separate
[`s-rwalk` implementation](s_rwalk_implementation.tex).

MINS always defines its pseudo-likelihood with one fixed importance density:

\[
\log\Psi_0(\theta)
=\log L(\theta)+\log\pi(\theta)-\log q_0(\theta).
\]

At discarded threshold \(\lambda\), the statistically correct replacement
target is

\[
p_\lambda(\theta)\propto
q_0(\theta)\mathbf 1[\log\Psi_0(\theta)>\lambda].
\]

This is a constrained prior only in the special case \(q_0=\pi\). The
`"rwalk"`, `"s-rwalk"`, and `"en-rwalk"` proposal schemes are
Metropolis-Hastings kernels invariant under this constrained fixed-\(q_0\)
density.

## Constraint and acceptance

The sampler first identifies the worst point using stored `live_log_psi0`
ordering. That point defines the threshold. It is not an MCMC start: under
strict ordering it has `log_psi0 == threshold` and lies outside the support.
Standard random walk starts from one uniformly selected eligible survivor.
The statistically specified random walk does the same.
Ensemble random walk starts from distinct eligible survivors selected uniformly
without replacement.

Every proposed parameter point is evaluated once against the model and the
original `importance_morph`. If it passes the pseudo-likelihood constraint,
the symmetric-proposal log acceptance ratio is

```python
log_alpha = min(0.0, proposed.log_q0 - current.log_q0)
accept = np.log(rng.random()) < log_alpha
```

Likelihood, prior, and pseudo-likelihood values affect the constraint. They are
not the MH density ratio. In particular, passing the constraint does not imply
acceptance.

For `tie_policy="randomized_plateau"`, the state is
\((\theta,t)\), with \(t\sim U(0,1)\). Every proposal draws a new tie breaker.
Equal-pseudo-likelihood candidates pass only if their proposed tie exceeds the
discarded threshold tie. Acceptance updates both fields; rejection retains
both.

## Proposal geometry

Standard `rwalk` constructs Dynesty's single bounding ellipsoid from the
complete live set, including the point that defines the current threshold.
Dynesty's internal eigenvalue and condition-number repair produces finite axes
for rank-deficient live sets. The axes are cached and rebuilt after roughly
`walks * n_live` random-walk calls.

`s-rwalk` and `en-rwalk` use a regularized covariance built from frozen
survivors:

\[
C_{\rm reg}
=(1-\rho)C+\rho\,\operatorname{diag}(C)+\epsilon I.
\]

Geometry remains fixed during each complete replacement. Parameters are
proposed in their full physical space. MINS does not clip to a prior boundary
or redraw until a point enters the prior, because either operation would
destroy proposal symmetry. A proposal with zero prior or likelihood is
rejected through the ordinary constraint.

## Standard random walk

```python
from mins import MINSampler, RWalkSettings

sampler = MINSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="rwalk",
    rwalk_settings=RWalkSettings(
        walks=50,
        facc=0.5,
    ),
    n_live=200,
    rng=42,
)
```

For dimension \(D\), each transition proposes

\[
\theta'=\theta+sAr,\qquad r\sim\operatorname{Uniform}(B_D),
\]

where \(A\) contains the bounding-ellipsoid axes and \(B_D\) is the unit
\(D\)-ball. Scale starts at 1 and, after each completed replacement, follows
Dynesty's update

\[
s_{k+1}=s_k\exp\!\left(\frac{f_k-f_0}{D f_0}\right),
\]

where \(f_0\) is `facc`. The configured value is clamped to
`[1 / walks, 1]`. Omitting `walks` uses `D + 20`; explicit values have a
minimum of two. MINS attempts exactly `walks` transitions and returns the
actual final state. If every proposal is rejected, that final state is the
valid starting survivor; movement is never forced.

Dynesty can refresh dimensions beyond `ncdim` from a unit-cube prior. MINS
supports arbitrary correlated importance Morphs without such a transform, so
`ncdim` must be omitted or equal to the complete model dimension.

## Statistically specified Gaussian random walk

```python
from mins import MINSampler, SRWalkSettings

sampler = MINSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="s-rwalk",
    srwalk_settings=SRWalkSettings(
        n_steps=50,
        facc=0.5,
    ),
    n_live=200,
    rng=42,
)
```

For dimension \(D\), `s-rwalk` freezes the regularized survivor covariance
factor \(LL^\mathsf{T}=C_{\rm reg}\) for a complete replacement and proposes

\[
\theta'=\theta+sLz,\qquad z\sim\mathcal N(0,I).
\]

The initial scale defaults to \(2.38/\sqrt D\), or to an explicitly configured
positive `scale`. After each complete chain it uses the same acceptance-target
recursion as `rwalk`; `facc` defaults to 0.5 and is clamped to
`[1 / n_steps, 1]`. Exactly `n_steps` transitions are attempted. Covariance
shrinkage and jitter are configurable, and rank-deficient or
lower-sample-than-dimension live sets are handled by deterministic eigenvalue
flooring.

## Ensemble differential-evolution walk

```python
from mins import EnsembleRWalkSettings, MINSampler

sampler = MINSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="en-rwalk",
    ensemble_rwalk_settings=EnsembleRWalkSettings(
        n_walkers=8,
        n_sweeps=6,
    ),
    n_live=200,
    rng=42,
)
```

`n_walkers` must be even, at least four, and no greater than
`n_live - 1`. Each sweep randomly splits the ensemble into equal halves.
Walkers in one half propose using two distinct, ordered references from the
frozen complementary half:

\[
\theta_i'=\theta_i+\gamma(\theta_j-\theta_k)
          +\sigma_\epsilon Lz.
\]

The default is \(\gamma=2.38/\sqrt{2D}\). Nonzero symmetric jitter is always
present. MINS batch-evaluates a half, independently accepts or rejects its
walkers, then updates the other half against the now-current but
frozen-during-that-update complement. After every sweep, one walker is selected
uniformly from the complete final ensemble. Unchanged and rejected walkers
remain eligible for output.

## Limits and diagnostics

One random-walk transition and one ensemble walker candidate each count as one
proposal. A successful replacement therefore costs:

- `rwalk`: `walks` (default `ndim + 20`);
- `s-rwalk`: `n_steps` (default 25);
- `en-rwalk`: `n_walkers * n_sweeps`.

MINS checks obvious proposal and likelihood-call incompatibilities before
starting. It checks the deadline before every scalar transition or ensemble
half-batch. An interrupted evolution fails the replacement and leaves the
worst live point untouched; it never returns a shortened chain.

`RunHistory` separates nested replacement efficiency from MCMC behavior:

- `acceptance_fraction` retains cumulative completed-replacements/proposals;
- `constraint_pass_fraction` is constraint passes divided by all candidates;
- `mh_acceptance_fraction` is MH acceptances divided by all candidates;
- `mcmc_accepted` counts accepted transitions;
- `mcmc_moved` is one for a single chain that moved, or the number of
  ensemble walkers that moved at least once;
- `mcmc_completed` counts transitions for `"rwalk"` and `"s-rwalk"`, and
  complete sweeps for `"en-rwalk"`.

The two fractions are `NaN` and counts are zero for non-MCMC schemes.

## Mixing limitations

A finite-length Metropolis kernel is invariant under the constrained target,
but its replacement is generally correlated with existing live points. It is
not an independent constrained draw, and invariance alone does not establish
adequate mixing. Calibrate `walks`, `n_steps`, `n_sweeps`, walker count, and
`n_live` with repeated complete runs and compare evidence, posterior summaries,
and cost across seeds.

The `rwalk` implementation is adapted from Dynesty 3.1.0. Its runtime
`citations` entry remains Skilling (2006); source attribution and Dynesty's MIT
license are distributed with MINS.

Low MH acceptance, low constraint-pass fractions, or many unchanged walkers
can indicate poor mixing. Do not repair those symptoms by returning only moved
walkers, requiring at least one acceptance, or selecting a high-likelihood
state; all such conditioning changes the target.
