# Peak–plateau regression

The Phase 2 regression uses:

- prior and pseudo-prior: normalized `Uniform(-1, 1)`;
- likelihood: `2` for `|x| <= 0.25`, otherwise `0.5`;
- analytic evidence: `0.875`, so `logz = -0.1335313926`;
- tie policy: `randomized_plateau`;
- seed: `2026`;
- `n_live=40`, remaining-log-evidence tolerance `dlogz=0.05`, proposal batch
  size `32`;
- at most 500 iterations and 20,000 proposals per replacement.

This deliberately exercises exact pseudo-likelihood ties. The strict policy is
tested separately to stop with `plateau_stall` rather than loop forever.

Run:

```bash
PYTHONPATH=src python -m pytest \
  tests/statistical/test_benchmarks.py::test_peak_plateau_regression_uses_explicit_tie_policy
```

The supplied repository contained no previous prototype or stored regression
summary, so this documented analytic target is a new Phase 2 fixture rather
than a claim of equivalence to an unavailable experiment.
