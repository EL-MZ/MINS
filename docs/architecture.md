# Architecture

MINS keeps the transformed statistical calculation separate from MorphZ and
from presentation concerns.

```text
user model ──> model validation ─┐
                                ├─> sampler state machine ─> immutable result
MorphZ ──> MorphProposal ────────┘            │
                    constrained rejection <──┤
                    log-space quadrature <────┘

immutable result ──> diagnostics / plotting / user persistence
```

Dependency direction is inward toward small protocols:

- `model.py` defines batched model behavior and callable adaptation.
- `proposals/base.py` defines a normalized proposal contract.
- `proposals/morph.py` is the only module permitted to import or call MorphZ.
- `constrained.py` performs unbiased constrained rejection and caches model
  evaluations.
- `quadrature.py` contains pure expected-volume evidence arithmetic.
- `sampler.py` coordinates iteration and owns no density or plotting details.
- `results.py` validates and freezes complete run outputs.
- `diagnostics.py` and `plotting.py` consume results without changing them.

The sampler accepts any proposal satisfying the same normalized
`sample`/`log_prob` contract, which makes statistical components independently
testable. No library module writes files or displays plots during a run.

