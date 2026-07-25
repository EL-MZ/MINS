# Prototype-to-package developer note

The supplied repository contained specifications but no prototype source.
The mapping below therefore maps documented prototype responsibilities rather
than copied functions:

| Documented behavior | Package location | Correction |
|---|---|---|
| MorphZ GroupKDE construction | `proposals/morph.py` | One fixed adapter; normalized sampling/log density kept together |
| Initial points | `sampler.py` | New Morph draws, never training rows |
| Likelihood evaluation | `model.py`, `constrained.py` | Explicit scalar/vectorized adapter and cached batch validation |
| Constraint helper | `constrained.py` | First valid draw, never batch maximum; hard limits |
| Minimum trace | `results.py` | Full threshold and live-range history |
| Evidence accumulation | `quadrature.py` | Log-space rectangular weights and final-live correction |
| Run loop | `sampler.py` | Orders on transformed `log_psi`, not raw `log_likelihood` |
| Plotting | `plotting.py` | Result-only functions with no display or hidden write |

Later defensive or adaptive proposals can implement the existing normalized
`Proposal` interface. They must not change the meaning or ordering of Phase 2
result fields. Meta-proposal bookkeeping, refitting, and prior-started
exploration remain intentionally absent pending owner approval.

