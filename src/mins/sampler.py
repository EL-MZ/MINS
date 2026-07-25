"""Serial fixed-Morph nested-importance sampler state machine."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import MINSConfig, TiePolicy
from .constrained import (
    BatchEvaluator,
    draw_constrained,
    validate_proposal_sample,
)
from .model import Model
from .progress import ProgressOption, create_progress_reporter
from .proposals import MorphProposal, Proposal
from .quadrature import (
    dead_log_contribution,
    estimate_information,
    estimated_live_logz,
    finalize_quadrature,
    update_log_weighted_mean,
)
from .results import MINSResult, RunHistory


def _as_generator(
    rng: int | np.random.Generator,
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, bool) or not isinstance(rng, int):
        raise TypeError("rng must be an integer seed or numpy.random.Generator")
    return np.random.default_rng(rng)


class MINSampler:
    """Fixed-pseudo-prior serial nested-importance sampler.

    Parameters
    ----------
    model
        Batch model with a normalized ``log_prior``.
    proposal
        Fixed normalized proposal. Phase 2 normally uses
        :class:`~mins.MorphProposal`.
    n_live
        Static live-point count, at least two.
    rng
        Explicit generator or integer seed. A supplied generator is consumed
        in place.
    proposal_batch_size
        Independent proposal points evaluated per constrained-rejection batch.
    tie_policy
        ``"strict"`` or lexicographic ``"randomized_plateau"``.
    """

    def __init__(
        self,
        model: Model,
        proposal: Proposal,
        *,
        n_live: int,
        rng: int | np.random.Generator,
        proposal_batch_size: int = 64,
        tie_policy: TiePolicy = "strict",
    ) -> None:
        if model.ndim != proposal.ndim:
            raise ValueError(
                f"model ndim {model.ndim} does not match proposal ndim {proposal.ndim}"
            )
        self.model = model
        self.proposal = proposal
        self.n_live = n_live
        self.rng = _as_generator(rng)
        self.proposal_batch_size = proposal_batch_size
        self.tie_policy = tie_policy
        MINSConfig(
            n_live=n_live,
            proposal_batch_size=proposal_batch_size,
            tie_policy=tie_policy,
        )

    @classmethod
    def from_posterior_samples(
        cls,
        *,
        model: Model,
        posterior_samples: NDArray[np.float64],
        morph_config: Mapping[str, Any],
        n_live: int,
        rng: int | np.random.Generator,
        proposal_batch_size: int = 64,
        tie_policy: TiePolicy = "strict",
    ) -> MINSampler:
        """Fit MorphZ once and construct a sampler.

        ``morph_config`` is passed as keyword arguments to
        :meth:`MorphProposal.fit`.
        """
        proposal = MorphProposal.fit(
            posterior_samples,
            param_names=model.parameter_names,
            **dict(morph_config),
        )
        return cls(
            model,
            proposal,
            n_live=n_live,
            rng=rng,
            proposal_batch_size=proposal_batch_size,
            tie_policy=tie_policy,
        )

    def run(
        self,
        *,
        dlogz: float = 1.0e-3,
        max_iterations: int = 10_000,
        max_proposals_per_replacement: int = 100_000,
        max_likelihood_calls: int | None = None,
        max_wall_time: float | None = None,
        progress: ProgressOption = False,
    ) -> MINSResult:
        """Run deterministic-shrinkage nested importance sampling.

        Parameters
        ----------
        dlogz
            Relative estimated-live-evidence stopping tolerance.
        max_iterations
            Hard limit on completed replacements.
        max_proposals_per_replacement
            Hard proposal limit for one constrained draw.
        max_likelihood_calls
            Optional run-wide likelihood-evaluation limit.
        max_wall_time
            Optional wall-time limit in seconds.
        progress
            ``False`` for silence, ``True`` for the standard tqdm bar, or a
            callable receiving a progress mapping after every iteration.

        Returns
        -------
        MINSResult
            A complete result. Hard limits yield ``success=False`` and preserve
            the valid partial quadrature state.

        Raises
        ------
        InvalidModelOutput
            If model values are malformed, NaN, or positive infinity.
        InvalidProposalOutput
            If proposal samples or densities are malformed.
        ProposalSupportError
            If ``q`` is zero at a finite target-integrand point.
        MissingOptionalDependency
            If ``progress=True`` is requested without the optional tqdm
            dependency.
        """
        config = MINSConfig(
            n_live=self.n_live,
            dlogz=dlogz,
            proposal_batch_size=self.proposal_batch_size,
            max_iterations=max_iterations,
            max_proposals_per_replacement=max_proposals_per_replacement,
            max_likelihood_calls=max_likelihood_calls,
            max_wall_time=max_wall_time,
            tie_policy=self.tie_policy,
        )
        progress_reporter = create_progress_reporter(
            progress,
            max_iterations=config.max_iterations,
            n_live=config.n_live,
        )
        start = time.monotonic()
        deadline = None if max_wall_time is None else start + max_wall_time
        initial_state = copy.deepcopy(self.rng.bit_generator.state)
        evaluator = BatchEvaluator(self.model, self.proposal)

        try:
            live_theta = np.array(
                validate_proposal_sample(
                    self.proposal.sample(self.n_live, self.rng),
                    n=self.n_live,
                    ndim=self.model.ndim,
                ),
                copy=True,
            )
            initial = evaluator.evaluate(live_theta)
        except BaseException:
            progress_reporter.close("error")
            raise
        live_log_likelihood = np.array(initial.log_likelihood, copy=True)
        live_log_prior = np.array(initial.log_prior, copy=True)
        live_log_q = np.array(initial.log_q, copy=True)
        live_log_psi = np.array(initial.log_psi, copy=True)
        live_tie_breakers = self.rng.random(self.n_live)

        dead_points: list[NDArray[np.float64]] = []
        dead_log_likelihood: list[float] = []
        dead_log_prior: list[float] = []
        dead_log_q: list[float] = []
        dead_log_psi: list[float] = []
        dead_tie_breakers: list[float] = []
        dead_log_x: list[float] = []
        dead_log_delta_x: list[float] = []
        dead_log_weights: list[float] = []

        history_logz_dead: list[float] = []
        history_logz_live: list[float] = []
        history_logz_total: list[float] = []
        history_information: list[float] = []
        history_logzerr: list[float] = []
        history_remaining_fraction: list[float] = []
        history_live_min: list[float] = []
        history_live_median: list[float] = []
        history_live_max: list[float] = []
        history_proposals: list[int] = []
        history_likelihood_calls: list[int] = []
        history_acceptance: list[float] = []
        history_elapsed: list[float] = []

        niter = 0
        n_proposals = 0
        logz_dead = -np.inf
        dead_log_psi_mean = 0.0
        termination_reason = ""
        while not termination_reason:
            if niter >= config.max_iterations:
                termination_reason = "max_iterations"
                break
            if (
                config.max_likelihood_calls is not None
                and evaluator.n_likelihood_calls >= config.max_likelihood_calls
            ):
                termination_reason = "max_likelihood_calls"
                break
            if deadline is not None and time.monotonic() >= deadline:
                termination_reason = "max_wall_time"
                break

            if config.tie_policy == "randomized_plateau":
                worst = int(np.lexsort((live_tie_breakers, live_log_psi))[0])
            else:
                worst = int(np.argmin(live_log_psi))
            threshold = float(live_log_psi[worst])
            threshold_tie = float(live_tie_breakers[worst])
            try:
                attempt = draw_constrained(
                    evaluator=evaluator,
                    proposal=self.proposal,
                    threshold=threshold,
                    threshold_tie_breaker=threshold_tie,
                    tie_policy=config.tie_policy,
                    rng=self.rng,
                    batch_size=config.proposal_batch_size,
                    max_proposals=config.max_proposals_per_replacement,
                    max_likelihood_calls=config.max_likelihood_calls,
                    deadline=deadline,
                )
            except BaseException:
                progress_reporter.close("error")
                raise
            n_proposals += attempt.n_proposed
            if attempt.draw is None:
                termination_reason = attempt.reason or "constrained_sampling_exhausted"
                if (
                    termination_reason == "constrained_sampling_exhausted"
                    and config.tie_policy == "strict"
                    and np.count_nonzero(live_log_psi == threshold) > 1
                ):
                    termination_reason = "plateau_stall"
                break

            point = attempt.draw.point
            iteration = niter + 1
            log_x, log_delta_x, log_weight = dead_log_contribution(
                iteration, self.n_live, threshold
            )
            dead_points.append(np.array(live_theta[worst], copy=True))
            dead_log_likelihood.append(float(live_log_likelihood[worst]))
            dead_log_prior.append(float(live_log_prior[worst]))
            dead_log_q.append(float(live_log_q[worst]))
            dead_log_psi.append(threshold)
            dead_tie_breakers.append(threshold_tie)
            dead_log_x.append(log_x)
            dead_log_delta_x.append(log_delta_x)
            dead_log_weights.append(log_weight)

            live_theta[worst] = point.theta
            live_log_likelihood[worst] = point.log_likelihood
            live_log_prior[worst] = point.log_prior
            live_log_q[worst] = point.log_q
            live_log_psi[worst] = point.log_psi
            live_tie_breakers[worst] = point.tie_breaker
            niter = iteration

            logz_dead, dead_log_psi_mean = update_log_weighted_mean(
                logz_dead,
                dead_log_psi_mean,
                log_weight,
                threshold,
            )
            logz_live = estimated_live_logz(log_x, live_log_psi)
            logz_total = float(np.logaddexp(logz_dead, logz_live))
            information = estimate_information(
                logz_dead=logz_dead,
                dead_log_psi_mean=dead_log_psi_mean,
                logz_live=logz_live,
                live_log_psi=live_log_psi,
                logz_total=logz_total,
            )
            logzerr = float(np.sqrt(information / self.n_live))
            remaining_fraction = float(np.exp(logz_live - logz_total))
            elapsed = time.monotonic() - start
            history_logz_dead.append(logz_dead)
            history_logz_live.append(logz_live)
            history_logz_total.append(logz_total)
            history_information.append(information)
            history_logzerr.append(logzerr)
            history_remaining_fraction.append(remaining_fraction)
            history_live_min.append(float(np.min(live_log_psi)))
            history_live_median.append(float(np.median(live_log_psi)))
            history_live_max.append(float(np.max(live_log_psi)))
            history_proposals.append(attempt.n_proposed)
            history_likelihood_calls.append(evaluator.n_likelihood_calls)
            history_acceptance.append(niter / n_proposals)
            history_elapsed.append(elapsed)

            progress_reporter.update(
                {
                    "iteration": niter,
                    "max_iterations": config.max_iterations,
                    "nlive": config.n_live,
                    "likelihood_calls": evaluator.n_likelihood_calls,
                    "proposals": n_proposals,
                    "proposals_iteration": attempt.n_proposed,
                    "efficiency_percent": 100.0 * niter / n_proposals,
                    "logz": logz_total,
                    "logzerr": logzerr,
                    "information": information,
                    "logz_dead": logz_dead,
                    "logz_live": logz_live,
                    "remaining_fraction": remaining_fraction,
                    "stopping_tolerance": config.dlogz,
                    "threshold": threshold,
                    "live_min_log_psi": history_live_min[-1],
                    "live_median_log_psi": history_live_median[-1],
                    "live_max_log_psi": history_live_max[-1],
                    "elapsed_seconds": elapsed,
                }
            )
            if logz_live - logz_total < np.log(config.dlogz):
                termination_reason = "remaining_evidence"

        progress_reporter.close(termination_reason)
        final_log_x = -niter / self.n_live
        quadrature = finalize_quadrature(
            dead_log_weights,
            dead_log_psi,
            final_log_x,
            live_log_psi,
            self.n_live,
        )
        history = RunHistory(
            iteration=np.arange(1, niter + 1, dtype=np.int64),
            discarded_log_psi=np.asarray(dead_log_psi),
            log_x=np.asarray(dead_log_x),
            log_delta_x=np.asarray(dead_log_delta_x),
            logz_dead=np.asarray(history_logz_dead),
            logz_live=np.asarray(history_logz_live),
            logz_total=np.asarray(history_logz_total),
            information=np.asarray(history_information),
            logzerr=np.asarray(history_logzerr),
            remaining_fraction=np.asarray(history_remaining_fraction),
            live_min_log_psi=np.asarray(history_live_min),
            live_median_log_psi=np.asarray(history_live_median),
            live_max_log_psi=np.asarray(history_live_max),
            proposals=np.asarray(history_proposals, dtype=np.int64),
            likelihood_calls=np.asarray(history_likelihood_calls, dtype=np.int64),
            acceptance_fraction=np.asarray(history_acceptance),
            elapsed_seconds=np.asarray(history_elapsed),
        )
        success = termination_reason == "remaining_evidence"
        warnings = [
            "Phase 2 uses a fixed non-defensive Morph pseudo-prior; missing "
            "support can bias logz and is not automatically repaired."
        ]
        if not success:
            warnings.append(
                f"Run stopped by {termination_reason!r}; logz is a partial-run "
                "estimate with the current live remainder."
            )
        if config.tie_policy == "randomized_plateau":
            warnings.append(
                "randomized_plateau augments the pseudo-prior with stored "
                "Uniform(0, 1) tie breakers."
            )
        proposal_metadata = getattr(self.proposal, "metadata", None)
        final_state = copy.deepcopy(self.rng.bit_generator.state)
        ndim = self.model.ndim
        return MINSResult(
            logz=quadrature.logz,
            logzerr=quadrature.logzerr,
            information=quadrature.information,
            success=success,
            termination_reason=termination_reason,
            niter=niter,
            nlive=self.n_live,
            n_likelihood_calls=evaluator.n_likelihood_calls,
            n_prior_calls=evaluator.n_prior_calls,
            n_proposals=n_proposals,
            dead_points=np.asarray(dead_points, dtype=float).reshape(niter, ndim),
            dead_log_likelihood=np.asarray(dead_log_likelihood),
            dead_log_prior=np.asarray(dead_log_prior),
            dead_log_q=np.asarray(dead_log_q),
            dead_log_psi=np.asarray(dead_log_psi),
            dead_tie_breakers=np.asarray(dead_tie_breakers),
            dead_log_x=np.asarray(dead_log_x),
            dead_log_weights=np.asarray(dead_log_weights),
            final_live_points=live_theta,
            final_live_log_likelihood=live_log_likelihood,
            final_live_log_prior=live_log_prior,
            final_live_log_q=live_log_q,
            final_live_log_psi=live_log_psi,
            final_live_tie_breakers=live_tie_breakers,
            log_posterior_weights=quadrature.log_posterior_weights,
            history=history,
            config=config,
            rng_bit_generator=self.rng.bit_generator.__class__.__name__,
            rng_state_initial=repr(initial_state),
            rng_state_final=repr(final_state),
            proposal_description=repr(proposal_metadata),
            nonfinite_counts=(
                ("outside_prior", evaluator.outside_prior),
                ("zero_likelihood", evaluator.zero_likelihood),
            ),
            warnings=tuple(warnings),
        )
