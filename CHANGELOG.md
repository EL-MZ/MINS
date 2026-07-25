# Changelog

All notable changes to this project are documented here.

## 0.1.0.dev3 - 2026-07-25

- Added automatic MorphZ total-correlation and greedy group selection through
  `MorphProposal.fit(..., morph_type="{k}_group")`.

## 0.1.0.dev2 - 2026-07-25

- Added reproducible equal-weight posterior resampling on `MINSResult`.

## 0.1.0.dev1 - 2026-07-25

- Added optional tqdm progress reporting with standard nested-sampling state.
- Persisted per-iteration information, theoretical log-evidence error, and
  remaining-evidence fraction.

## 0.1.0.dev0 - 2026-07-25

- Created the Phase 1 package and quality-control skeleton.
- Added the experimental Phase 2 fixed-Morph nested-importance estimator.
