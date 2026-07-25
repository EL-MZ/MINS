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
posterior samples, represents a normalized density \(q\) on the same measure.

Phase 2 fixes \(\beta=1\), uses \(q\) as its pseudo-prior, and defines

\[
\log\Psi(\theta)
=\log L(\theta)+\log\pi(\theta)-\log q(\theta).
\]

Consequently \(Z=\int\Psi(\theta)q(\theta)\,\mathrm d\theta\).

## Morph training and support

Morph is fit exactly once before a run from a copied finite array with shape
`(n_samples, ndim)`. The supplied grouping definition and bandwidth settings
determine `GroupKDE`; it is not adapted from live or dead points.

`MorphProposal.sample(n, rng)` calls the fitted GroupKDE resampler.
`MorphProposal.log_prob(theta)` calls that same object's normalized log density.
Thus sampling and density evaluation describe one fixed \(q\).

The required non-defensive assumption is that \(q(\theta)>0\) everywhere
\(L(\theta)\pi(\theta)\) has material mass. A finite numerator with
`log_q == -inf` is a fatal proposal-support error. Phase 2 neither repairs nor
hides missing support.

## Replacement distribution

Initial live points are independent new draws from \(q\), never reused training
samples. At threshold \(\lambda\), independent proposal draws are scanned in
generation order and the first point satisfying
`candidate_log_psi > threshold` is accepted. This rejection sampler targets

\[
q(\theta\mid\log\Psi(\theta)>\lambda).
\]

The optional `randomized_plateau` policy augments every point with independent
\(u\sim U(0,1)\) and applies lexicographic ordering to
\((\log\Psi,u)\). No Metropolis-Hastings or additional importance ratio is used.

## Evidence estimator

For \(N\) live points and completed discard \(i\), deterministic expected
volumes are \(X_i=\exp(-i/N)\). Rectangular dead-point contributions are

\[
\log w_i=\log(X_{i-1}-X_i)+\log\Psi_i.
\]

After \(K\) completed replacements, every remaining point contributes

\[
\log w_j^{\rm live}=\log X_K-\log N+\log\Psi_j^{\rm live}.
\]

The reported `logz` is `logsumexp` of all dead and final-live contributions.
Normalized quadrature contributions are posterior weights. Information is

\[
H=\sum_a \widetilde w_a(\log\Psi_a-\log Z)
\]

and the reported theoretical error is \(\sqrt{H/N}\). This is an approximation,
not a calibration guarantee.

## Stopping semantics

The scientific stop uses the mean-live estimate

\[
\log Z_{\rm live}=\log X_K+
\operatorname{logsumexp}(\log\Psi^{\rm live})-\log N
\]

and succeeds when
`log_z_live - log_z_total < log(dlogz)`. Iteration, likelihood-call,
per-replacement proposal, and wall-time limits are hard failures, not evidence
of convergence. The max-live remainder is stored only as a conservative
diagnostic.

## User obligations and limitations

- `log_prior` includes every normalizing constant.
- `log_likelihood`, `log_prior`, and Morph `log_prob` use the same coordinates.
- Training samples are representative of every material posterior region.
- Rejection draws are practical at the requested final constraint.
- Deterministic shrinkage and \(\sqrt{H/N}\) do not account for all Monte Carlo
  or Morph-fitting uncertainty.
- Phase 2 is post-processing evidence estimation and does not discover the
  posterior from the original prior.

