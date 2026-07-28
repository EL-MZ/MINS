# Mathematical contract

This page is the authoritative statistical contract for Phase 2. It is written
before the sampler loop.

## Base measure and transformed target

The original evidence is

\[
Z=\int_\Theta L(\theta)\pi(\theta)\,\mathrm d\theta,
\]

where \(\pi\) is a normalized density with respect to Lebesgue measure on the
declared parameterization. A fixed MorphZ `GroupKDE`, trained from user-supplied
posterior samples, represents a normalized importance density \(q_0\) on the
same measure.

Phase 2 fixes \(\beta=1\), uses \(q_0\) as its pseudo-prior, and defines

\[
\log\Psi_0(\theta)
=\log L(\theta)+\log\pi(\theta)-\log q_0(\theta).
\]

Consequently \(Z=\int\Psi_0(\theta)q_0(\theta)\,\mathrm d\theta\).

## Morph training and support

The importance Morph is fit exactly once before a run from a copied finite
array with shape
`(n_samples, ndim)`. The supplied grouping definition and bandwidth settings
determine `GroupKDE`; it is not adapted from live or dead points.

`importance_morph.sample(n, rng)` creates the initial live set and
`importance_morph.log_prob(theta)` supplies every stored `log_q0`. This object
is never mutated or replaced.

The required non-defensive assumption is that \(q_0(\theta)>0\) everywhere
\(L(\theta)\pi(\theta)\) has material mass. A finite numerator with
`log_q0 == -inf` is a fatal importance-support error. Phase 2 neither repairs nor
hides missing support.

## Replacement distribution

Initial live points are independent new draws from \(q_0\), never reused
training samples.

With `proposal_scheme="fixed_morph"`, candidates are also drawn from \(q_0\).
At threshold \(\lambda\), independent draws are scanned in generation order
and the first point satisfying `candidate_log_psi0 > threshold` is accepted.
This rejection sampler targets

\[
q_0(\theta\mid\log\Psi_0(\theta)>\lambda).
\]

The optional `randomized_plateau` policy augments every point with independent
\(u\sim U(0,1)\) and applies lexicographic ordering to
\((\log\Psi_0,u)\).

With `proposal_scheme="adaptive_morph"`, MINS initially proposes from \(q_0\).
After every 25 completed iterations by default, it fits a new proposal Morph
\(r_c\) to a copy of all current live points using the original Morph fit
configuration. Candidates then come from \(r_c\), while their `log_q0`,
`log_psi0`, ordering, and constraint continue to use the original importance
Morph. A failed refit leaves the previous proposal active and is retried at the
next interval.

The adaptive scheme directly accepts the first \(r_c\) draw above the
\(\Psi_0\) constraint. It therefore targets
\(r_c(\theta\mid\log\Psi_0(\theta)>\lambda)\), not constrained \(q_0\).
There is no Metropolis-Hastings or rejection-ratio correction. Consequently,
the deterministic \(q_0\)-volume quadrature below is heuristic in adaptive
mode and its evidence estimate may be biased.

## Evidence estimator

For \(N\) live points and completed discard \(i\), deterministic expected
volumes are \(X_i=\exp(-i/N)\). Rectangular dead-point contributions are

\[
\log w_i=\log(X_{i-1}-X_i)+\log\Psi_{0,i}.
\]

After \(K\) completed replacements, every remaining point contributes

\[
\log w_j^{\rm live}=\log X_K-\log N+\log\Psi_{0,j}^{\rm live}.
\]

The reported `logz` is `logsumexp` of all dead and final-live contributions.
Normalized quadrature contributions are posterior weights. Information is

\[
H=\sum_a \widetilde w_a(\log\Psi_{0,a}-\log Z)
\]

and the reported theoretical error is \(\sqrt{H/N}\). This is an approximation,
not a calibration guarantee.

## Stopping semantics

At every completed replacement the mean-live estimate is

\[
\log Z_{\rm live}=\log X_K+
\operatorname{logsumexp}(\log\Psi_0^{\rm live})-\log N
\]

The remaining-fraction diagnostic is

\[
f_{\rm live} = Z_{\rm live}/Z_{\rm total}.
\]

The default `dlogz` API instead succeeds when

\[
\Delta\log Z_{\rm rem}
=\log Z_{\rm total}-\log Z_{\rm dead}
=\log\left(1+\frac{Z_{\rm live}}{Z_{\rm dead}}\right)
\leq \mathtt{dlogz}.
\]

This is the estimated change in accumulated log evidence from adding the
mean-live remainder. The exact relationship is
\(\Delta\log Z_{\rm rem}=-\log(1-f_{\rm live})\); the two diagnostics are
separately named and selectable.

An explicit stopping policy may combine that diagnostic with the Kish live ESS

\[
N_{\rm eff,live}
=\frac{(\sum_j\Psi_j)^2}{\sum_j\Psi_j^2},
\]

the live-mean relative standard error

\[
\operatorname{RSE}_{\rm live}
=\sqrt{\max\left(
\frac{N/N_{\rm eff,live}-1}{N-1},0\right)},
\]

and its first-order contribution to total log-evidence uncertainty

\[
\sigma_{\log Z,\rm live}
\approx f_{\rm live}\operatorname{RSE}_{\rm live}.
\]

This last quantity estimates Monte Carlo uncertainty from representing the
remaining integral with the finite live set. It excludes stochastic shrinkage,
Morph-fitting uncertainty, missing proposal support, undiscovered modes, and
correlated or invalid constrained draws. Policies may also use the range of
recent `logZ` values and the theoretical nested-sampling approximation
\(\sqrt{H/N}\). Stability can hold for a biased estimate, and theoretical
`logzerr` is not a complete calibration guarantee.

Enabled conditions combine with `"all"` or `"any"` and may require consecutive
passes after a minimum iteration. An explicit policy succeeds with
`termination_reason="stopping_criteria"`; legacy stopping retains
`"remaining_evidence"`.

Iteration, likelihood-call, per-replacement proposal, and wall-time limits are
hard failures, not evidence of convergence. The max-live remainder remains a
separate conservative diagnostic. Deterministic shrinkage
\(X_i=\exp(-i/N)\) remains the central quadrature trajectory regardless of
stopping policy.

## User obligations and limitations

- `log_prior` includes every normalizing constant.
- `log_likelihood`, `log_prior`, and importance-Morph `log_prob` use the same
  coordinates.
- Training samples are representative of every material posterior region.
- Fixed rejection or heuristic adaptive draws are practical at the requested
  final constraint.
- Adaptive threshold-only draws do not preserve constrained \(q_0\), so their
  evidence and posterior-weight interpretation is approximate.
- Deterministic shrinkage and \(\sqrt{H/N}\) do not account for all Monte Carlo
  or Morph-fitting uncertainty.
- Phase 2 is post-processing evidence estimation and does not discover the
  posterior from the original prior.
