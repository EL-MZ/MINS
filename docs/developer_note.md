# Prototype-to-package developer note

The supplied repository contained specifications but no prototype source.
The mapping below therefore maps documented prototype responsibilities rather
than copied functions:

| Documented behavior | Package location | Correction |
|---|---|---|
| MorphZ GroupKDE construction | `proposals/morph.py` | Fixed importance adapter plus non-mutating proposal refits |
| Initial points | `sampler.py` | New Morph draws, never training rows |
| Likelihood evaluation | `model.py`, `constrained.py` | Explicit scalar/vectorized adapter and cached batch validation |
| Constraint helper | `constrained.py` | Proposal draws evaluated against fixed `log_q0`; first valid draw and hard limits |
| Minimum trace | `results.py` | Full threshold and live-range history |
| Evidence accumulation | `quadrature.py` | Log-space rectangular weights and final-live correction |
| Run loop | `sampler.py` | Orders on fixed transformed `log_psi0`, not raw `log_likelihood` |
| Plotting | `plotting.py` | Result-only functions with no display or hidden write |

The fixed scheme proposes from the importance Morph. The adaptive scheme
periodically refits a separate Morph from the current live set while leaving
all `q0` evaluations unchanged. Its threshold-only acceptance is heuristic and
is not a constrained-`q0` transition.
