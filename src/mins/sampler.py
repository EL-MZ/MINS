"""Serial fixed-importance nested sampler with selectable Morph proposals."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .adaptive import AdaptiveMorphController
from .config import (
    EnsembleRWalkSettings,
    MINSConfig,
    ProposalScheme,
    RWalkSettings,
    SRWalkSettings,
    TiePolicy,
)
from .constrained import (
    BatchEvaluator,
    ConstrainedAttempt,
    draw_constrained,
    validate_proposal_sample,
)
from .mcmc import (
    RWALK_CITATIONS,
    RWalkSampler,
    SRWalkSampler,
    draw_ensemble_rwalk_constrained,
    draw_rwalk_constrained,
    draw_srwalk_constrained,
)
from .model import Model
from .progress import ProgressOption, create_progress_reporter
from .proposals import MorphProposal, Proposal, RefittableProposal
from .quadrature import (
    dead_log_contribution,
    estimate_information,
    estimated_live_logz,
    finalize_quadrature,
    update_log_weighted_mean,
)
from .results import MINSResult, RunHistory
from .stopping import (
    SCIENTIFIC_TERMINATION_REASONS,
    StoppingPolicy,
    calculate_stopping_metrics,
    evaluate_stopping_policy,
)


def _as_generator(
    rng: int | np.random.Generator,
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, bool) or not isinstance(rng, int):
        raise TypeError("rng must be an integer seed or numpy.random.Generator")
    return np.random.default_rng(rng)


def _draw_replacement(
    *,
    config: MINSConfig,
    evaluator: BatchEvaluator,
    proposal_morph: Proposal,
    live_theta: NDArray[np.float64],
    live_log_likelihood: NDArray[np.float64],
    live_log_prior: NDArray[np.float64],
    live_log_q0: NDArray[np.float64],
    live_log_psi0: NDArray[np.float64],
    live_tie_breakers: NDArray[np.float64],
    worst: int,
    threshold: float,
    threshold_tie_breaker: float,
    rng: np.random.Generator,
    deadline: float | None,
    rwalk_sampler: RWalkSampler | None,
    srwalk_sampler: SRWalkSampler | None,
) -> ConstrainedAttempt:
    """Dispatch one replacement without mixing proposal-specific mechanics."""
    if config.proposal_scheme in ("fixed_morph", "adaptive_morph"):
        return draw_constrained(
            evaluator=evaluator,
            proposal_morph=proposal_morph,
            threshold=threshold,
            threshold_tie_breaker=threshold_tie_breaker,
            tie_policy=config.tie_policy,
            rng=rng,
            batch_size=config.proposal_batch_size,
            max_proposals=config.max_proposals_per_replacement,
            max_likelihood_calls=config.max_likelihood_calls,
            deadline=deadline,
        )
    if config.proposal_scheme == "rwalk":
        if rwalk_sampler is None:
            raise RuntimeError("rwalk controller is not initialized")
        return draw_rwalk_constrained(
            evaluator=evaluator,
            live_theta=live_theta,
            live_log_likelihood=live_log_likelihood,
            live_log_prior=live_log_prior,
            live_log_q0=live_log_q0,
            live_log_psi0=live_log_psi0,
            live_tie_breakers=live_tie_breakers,
            worst=worst,
            threshold=threshold,
            threshold_tie_breaker=threshold_tie_breaker,
            tie_policy=config.tie_policy,
            sampler=rwalk_sampler,
            rng=rng,
            max_proposals=config.max_proposals_per_replacement,
            max_likelihood_calls=config.max_likelihood_calls,
            deadline=deadline,
        )
    if config.proposal_scheme == "s-rwalk":
        if srwalk_sampler is None:
            raise RuntimeError("s-rwalk controller is not initialized")
        return draw_srwalk_constrained(
            evaluator=evaluator,
            live_theta=live_theta,
            live_log_likelihood=live_log_likelihood,
            live_log_prior=live_log_prior,
            live_log_q0=live_log_q0,
            live_log_psi0=live_log_psi0,
            live_tie_breakers=live_tie_breakers,
            worst=worst,
            threshold=threshold,
            threshold_tie_breaker=threshold_tie_breaker,
            tie_policy=config.tie_policy,
            sampler=srwalk_sampler,
            rng=rng,
            max_proposals=config.max_proposals_per_replacement,
            max_likelihood_calls=config.max_likelihood_calls,
            deadline=deadline,
        )
    if config.proposal_scheme == "en-rwalk":
        return draw_ensemble_rwalk_constrained(
            evaluator=evaluator,
            live_theta=live_theta,
            live_log_likelihood=live_log_likelihood,
            live_log_prior=live_log_prior,
            live_log_q0=live_log_q0,
            live_log_psi0=live_log_psi0,
            live_tie_breakers=live_tie_breakers,
            worst=worst,
            threshold=threshold,
            threshold_tie_breaker=threshold_tie_breaker,
            tie_policy=config.tie_policy,
            settings=config.ensemble_rwalk_settings,
            rng=rng,
            max_proposals=config.max_proposals_per_replacement,
            max_likelihood_calls=config.max_likelihood_calls,
            deadline=deadline,
        )
    raise RuntimeError(f"unsupported proposal scheme: {config.proposal_scheme!r}")


class MINSampler:
    """Fixed-importance sampler with selectable Morph proposal schemes.

    Parameters
    ----------
    model
        Batch model with a normalized ``log_prior``.
    importance_morph
        Fixed normalized importance distribution. Phase 2 normally uses
        :class:`~mins.MorphProposal`.
    proposal_scheme
        ``"fixed_morph"`` draws from the importance Morph.
        ``"adaptive_morph"`` periodically refits a separate proposal Morph.
        ``"rwalk"``, ``"s-rwalk"``, and ``"en-rwalk"`` apply constrained
        Metropolis kernels invariant under the fixed importance density.
    proposal_update_interval
        Completed iterations between adaptive proposal refits.
    n_live
        Static live-point count, at least two.
    rng
        Explicit generator or integer seed. A supplied generator is consumed
        in place.
    proposal_batch_size
        Independent proposal points evaluated per constrained-rejection batch.
    tie_policy
        ``"strict"`` or lexicographic ``"randomized_plateau"``.
    rwalk_settings
        Optional immutable Dynesty-style random-walk settings.
    srwalk_settings
        Optional immutable Gaussian-covariance random-walk settings.
    ensemble_rwalk_settings
        Optional immutable ensemble random-walk settings.
    """

    def __init__(
        self,
        *,
        model: Model,
        importance_morph: Proposal,
        proposal_scheme: ProposalScheme = "fixed_morph",
        proposal_update_interval: int = 25,
        n_live: int,
        rng: int | np.random.Generator,
        proposal_batch_size: int = 64,
        tie_policy: TiePolicy = "strict",
        rwalk_settings: RWalkSettings | None = None,
        srwalk_settings: SRWalkSettings | None = None,
        ensemble_rwalk_settings: EnsembleRWalkSettings | None = None,
    ) -> None:
        if model.ndim != importance_morph.ndim:
            raise ValueError(
                f"model ndim {model.ndim} does not match importance Morph ndim "
                f"{importance_morph.ndim}"
            )
        if proposal_scheme == "adaptive_morph" and not isinstance(
            importance_morph, RefittableProposal
        ):
            raise TypeError(
                "adaptive_morph requires an importance Morph with a refit method"
            )
        self.model = model
        self.importance_morph = importance_morph
        self.proposal_morph = importance_morph
        self.adaptive_proposal_controller: AdaptiveMorphController | None = None
        self.proposal_scheme = proposal_scheme
        self.proposal_update_interval = proposal_update_interval
        self.n_live = n_live
        self.rng = _as_generator(rng)
        self.proposal_batch_size = proposal_batch_size
        self.tie_policy = tie_policy
        self.rwalk_settings = (
            RWalkSettings() if rwalk_settings is None else rwalk_settings
        )
        self.srwalk_settings = (
            SRWalkSettings() if srwalk_settings is None else srwalk_settings
        )
        self.ensemble_rwalk_settings = (
            EnsembleRWalkSettings()
            if ensemble_rwalk_settings is None
            else ensemble_rwalk_settings
        )
        MINSConfig(
            n_live=n_live,
            proposal_batch_size=proposal_batch_size,
            proposal_scheme=proposal_scheme,
            proposal_update_interval=proposal_update_interval,
            tie_policy=tie_policy,
            rwalk_settings=self.rwalk_settings,
            srwalk_settings=self.srwalk_settings,
            ensemble_rwalk_settings=self.ensemble_rwalk_settings,
        )
        if proposal_scheme == "rwalk":
            RWalkSampler(settings=self.rwalk_settings, ndim=model.ndim)
        if proposal_scheme == "s-rwalk":
            SRWalkSampler(settings=self.srwalk_settings, ndim=model.ndim)

    @property
    def citations(self) -> list[tuple[str, str]]:
        """Return citations declared by the configured sampling method."""
        if self.proposal_scheme in ("rwalk", "s-rwalk"):
            return list(RWALK_CITATIONS)
        return []

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
        proposal_scheme: ProposalScheme = "fixed_morph",
        proposal_update_interval: int = 25,
        tie_policy: TiePolicy = "strict",
        rwalk_settings: RWalkSettings | None = None,
        srwalk_settings: SRWalkSettings | None = None,
        ensemble_rwalk_settings: EnsembleRWalkSettings | None = None,
    ) -> MINSampler:
        """Fit MorphZ once and construct a sampler.

        ``morph_config`` is passed as keyword arguments to
        :meth:`MorphProposal.fit`.
        """
        importance_morph = MorphProposal.fit(
            posterior_samples,
            param_names=model.parameter_names,
            **dict(morph_config),
        )
        return cls(
            model=model,
            importance_morph=importance_morph,
            n_live=n_live,
            rng=rng,
            proposal_batch_size=proposal_batch_size,
            proposal_scheme=proposal_scheme,
            proposal_update_interval=proposal_update_interval,
            tie_policy=tie_policy,
            rwalk_settings=rwalk_settings,
            srwalk_settings=srwalk_settings,
            ensemble_rwalk_settings=ensemble_rwalk_settings,
        )

    def run(
        self,
        *,
        dlogz: float | None = None,
        stopping: StoppingPolicy | None = None,
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
            Maximum estimated change in log evidence caused by adding the
            current mean-live remaining-evidence estimate. When both ``dlogz``
            and ``stopping`` are omitted, this defaults to ``1e-3``.
        stopping
            Optional multi-criterion scientific stopping policy. It cannot be
            supplied together with ``dlogz``.
        max_iterations
            Hard limit on completed replacements.
        max_proposals_per_replacement
            Hard proposal limit for one constrained draw.
        max_likelihood_calls
            Optional run-wide likelihood-evaluation limit.
        max_wall_time
            Optional wall-time limit in seconds.
        progress
            ``False`` for silence, ``True`` for the standard tqdm live display,
            or a callable receiving a progress mapping after every iteration.

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
            If fixed ``q0`` is zero at a finite target-integrand point.
        MissingOptionalDependency
            If ``progress=True`` is requested without the optional tqdm
            dependency.
        """
        config = MINSConfig(
            n_live=self.n_live,
            dlogz=dlogz,
            stopping=stopping,
            proposal_batch_size=self.proposal_batch_size,
            proposal_scheme=self.proposal_scheme,
            proposal_update_interval=self.proposal_update_interval,
            rwalk_settings=self.rwalk_settings,
            srwalk_settings=self.srwalk_settings,
            ensemble_rwalk_settings=self.ensemble_rwalk_settings,
            max_iterations=max_iterations,
            max_proposals_per_replacement=max_proposals_per_replacement,
            max_likelihood_calls=max_likelihood_calls,
            max_wall_time=max_wall_time,
            tie_policy=self.tie_policy,
        )
        stopping_policy = config.stopping
        if stopping_policy is None:  # pragma: no cover - config always resolves it
            raise RuntimeError("MINSConfig did not resolve a stopping policy")
        progress_reporter = create_progress_reporter(
            progress,
            max_iterations=config.max_iterations,
            n_live=config.n_live,
        )
        start = time.monotonic()
        deadline = None if max_wall_time is None else start + max_wall_time
        initial_state = copy.deepcopy(self.rng.bit_generator.state)
        evaluator = BatchEvaluator(self.model, self.importance_morph)
        self.proposal_morph = self.importance_morph
        self.adaptive_proposal_controller = None
        rwalk_sampler = (
            RWalkSampler(settings=config.rwalk_settings, ndim=self.model.ndim)
            if config.proposal_scheme == "rwalk"
            else None
        )
        srwalk_sampler = (
            SRWalkSampler(settings=config.srwalk_settings, ndim=self.model.ndim)
            if config.proposal_scheme == "s-rwalk"
            else None
        )
        if config.proposal_scheme == "adaptive_morph":
            # Constructor validation establishes this runtime protocol.
            if not isinstance(self.importance_morph, RefittableProposal):
                raise RuntimeError("adaptive importance Morph lost its refit contract")
            self.adaptive_proposal_controller = AdaptiveMorphController(
                importance_morph=self.importance_morph,
                update_interval=config.proposal_update_interval,
            )

        try:
            live_theta = np.array(
                validate_proposal_sample(
                    self.importance_morph.sample(self.n_live, self.rng),
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
        live_log_q0 = np.array(initial.log_q0, copy=True)
        live_log_psi0 = np.array(initial.log_psi0, copy=True)
        live_tie_breakers = self.rng.random(self.n_live)

        dead_points: list[NDArray[np.float64]] = []
        dead_log_likelihood: list[float] = []
        dead_log_prior: list[float] = []
        dead_log_q0: list[float] = []
        dead_log_psi0: list[float] = []
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
        history_remaining_dlogz: list[float] = []
        history_live_ess: list[float] = []
        history_live_mean_rse: list[float] = []
        history_live_logz_error: list[float] = []
        history_logz_stability: list[float] = []
        history_stopping_streak: list[int] = []
        history_live_min: list[float] = []
        history_live_median: list[float] = []
        history_live_max: list[float] = []
        history_proposals: list[int] = []
        history_likelihood_calls: list[int] = []
        history_acceptance: list[float] = []
        history_mh_acceptance: list[float] = []
        history_constraint_pass: list[float] = []
        history_mcmc_accepted: list[int] = []
        history_mcmc_moved: list[int] = []
        history_mcmc_completed: list[int] = []
        history_elapsed: list[float] = []
        history_proposal_revision: list[int] = []
        history_proposal_update_attempts: list[int] = []
        history_proposal_update_failures: list[int] = []

        niter = 0
        n_proposals = 0
        logz_dead = -np.inf
        dead_log_psi_mean = 0.0
        stopping_streak = 0
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

            if self.adaptive_proposal_controller is not None:
                self.proposal_morph = self.adaptive_proposal_controller.update_if_due(
                    iteration=niter,
                    live_theta=live_theta,
                )
                if deadline is not None and time.monotonic() >= deadline:
                    termination_reason = "max_wall_time"
                    break

            if config.tie_policy == "randomized_plateau":
                worst = int(np.lexsort((live_tie_breakers, live_log_psi0))[0])
            else:
                worst = int(np.argmin(live_log_psi0))
            threshold = float(live_log_psi0[worst])
            threshold_tie = float(live_tie_breakers[worst])
            try:
                attempt = _draw_replacement(
                    config=config,
                    evaluator=evaluator,
                    proposal_morph=self.proposal_morph,
                    live_theta=live_theta,
                    live_log_likelihood=live_log_likelihood,
                    live_log_prior=live_log_prior,
                    live_log_q0=live_log_q0,
                    live_log_psi0=live_log_psi0,
                    live_tie_breakers=live_tie_breakers,
                    worst=worst,
                    threshold=threshold,
                    threshold_tie_breaker=threshold_tie,
                    rng=self.rng,
                    deadline=deadline,
                    rwalk_sampler=rwalk_sampler,
                    srwalk_sampler=srwalk_sampler,
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
                    and np.count_nonzero(live_log_psi0 == threshold) > 1
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
            dead_log_q0.append(float(live_log_q0[worst]))
            dead_log_psi0.append(threshold)
            dead_tie_breakers.append(threshold_tie)
            dead_log_x.append(log_x)
            dead_log_delta_x.append(log_delta_x)
            dead_log_weights.append(log_weight)

            live_theta[worst] = point.theta
            live_log_likelihood[worst] = point.log_likelihood
            live_log_prior[worst] = point.log_prior
            live_log_q0[worst] = point.log_q0
            live_log_psi0[worst] = point.log_psi0
            live_tie_breakers[worst] = point.tie_breaker
            niter = iteration

            logz_dead, dead_log_psi_mean = update_log_weighted_mean(
                logz_dead,
                dead_log_psi_mean,
                log_weight,
                threshold,
            )
            logz_live = estimated_live_logz(log_x, live_log_psi0)
            logz_total = float(np.logaddexp(logz_dead, logz_live))
            information = estimate_information(
                logz_dead=logz_dead,
                dead_log_psi_mean=dead_log_psi_mean,
                logz_live=logz_live,
                live_log_psi=live_log_psi0,
                logz_total=logz_total,
            )
            logzerr = float(np.sqrt(information / self.n_live))
            stability_history = (
                *history_logz_total[-(stopping_policy.stability_window - 1) :],
                logz_total,
            )
            stopping_metrics = calculate_stopping_metrics(
                live_log_psi=live_log_psi0,
                logz_dead=logz_dead,
                logz_live=logz_live,
                logz_total=logz_total,
                logz_history=stability_history,
                logzerr=logzerr,
                stability_window=stopping_policy.stability_window,
            )
            stopping_decision = evaluate_stopping_policy(
                metrics=stopping_metrics,
                policy=stopping_policy,
                niter=niter,
                previous_streak=stopping_streak,
            )
            stopping_streak = stopping_decision.streak
            elapsed = time.monotonic() - start
            history_logz_dead.append(logz_dead)
            history_logz_live.append(logz_live)
            history_logz_total.append(logz_total)
            history_information.append(information)
            history_logzerr.append(logzerr)
            history_remaining_fraction.append(stopping_metrics.remaining_fraction)
            history_remaining_dlogz.append(stopping_metrics.remaining_dlogz)
            history_live_ess.append(stopping_metrics.live_ess)
            history_live_mean_rse.append(stopping_metrics.live_mean_rse)
            history_live_logz_error.append(stopping_metrics.live_logz_error)
            history_logz_stability.append(stopping_metrics.logz_stability)
            history_stopping_streak.append(stopping_streak)
            history_live_min.append(float(np.min(live_log_psi0)))
            history_live_median.append(float(np.median(live_log_psi0)))
            history_live_max.append(float(np.max(live_log_psi0)))
            history_proposals.append(attempt.n_proposed)
            history_likelihood_calls.append(evaluator.n_likelihood_calls)
            history_acceptance.append(niter / n_proposals)
            if config.proposal_scheme in ("rwalk", "s-rwalk", "en-rwalk"):
                history_mh_acceptance.append(attempt.n_accepted / attempt.n_proposed)
                history_constraint_pass.append(attempt.n_valid / attempt.n_proposed)
                history_mcmc_accepted.append(attempt.n_accepted)
                history_mcmc_moved.append(attempt.n_moved)
                history_mcmc_completed.append(attempt.n_completed)
            else:
                history_mh_acceptance.append(float("nan"))
                history_constraint_pass.append(float("nan"))
                history_mcmc_accepted.append(0)
                history_mcmc_moved.append(0)
                history_mcmc_completed.append(0)
            history_elapsed.append(elapsed)
            if self.adaptive_proposal_controller is None:
                proposal_revision = 0
                proposal_update_attempts = 0
                proposal_update_failures = 0
            else:
                proposal_revision = self.adaptive_proposal_controller.revision
                proposal_update_attempts = (
                    self.adaptive_proposal_controller.update_attempts
                )
                proposal_update_failures = (
                    self.adaptive_proposal_controller.update_failures
                )
            history_proposal_revision.append(proposal_revision)
            history_proposal_update_attempts.append(proposal_update_attempts)
            history_proposal_update_failures.append(proposal_update_failures)

            progress_info: dict[str, float | int] = {
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
                "remaining_fraction": stopping_metrics.remaining_fraction,
                "remaining_dlogz": stopping_metrics.remaining_dlogz,
                "live_ess": stopping_metrics.live_ess,
                "live_mean_rse": stopping_metrics.live_mean_rse,
                "live_logz_error": stopping_metrics.live_logz_error,
                "logz_stability": stopping_metrics.logz_stability,
                "stopping_streak": stopping_streak,
                "stopping_consecutive": stopping_policy.consecutive,
                "threshold": threshold,
                "live_min_log_psi": history_live_min[-1],
                "live_median_log_psi": history_live_median[-1],
                "live_max_log_psi": history_live_max[-1],
                "elapsed_seconds": elapsed,
                "proposal_revision": proposal_revision,
                "proposal_update_attempts": proposal_update_attempts,
                "proposal_update_failures": proposal_update_failures,
                "mh_acceptance_fraction": history_mh_acceptance[-1],
                "constraint_pass_fraction": history_constraint_pass[-1],
                "mcmc_accepted": history_mcmc_accepted[-1],
                "mcmc_moved": history_mcmc_moved[-1],
                "mcmc_completed": history_mcmc_completed[-1],
            }
            if config.dlogz is not None:
                progress_info["stopping_tolerance"] = config.dlogz
            for evaluation in stopping_decision.evaluations:
                progress_info[f"criterion_{evaluation.name}_met"] = int(evaluation.met)
            progress_reporter.update(progress_info)
            if stopping_decision.should_stop:
                termination_reason = (
                    "remaining_evidence"
                    if config.dlogz is not None
                    else "stopping_criteria"
                )

        progress_reporter.close(termination_reason)
        final_log_x = -niter / self.n_live
        quadrature = finalize_quadrature(
            dead_log_weights,
            dead_log_psi0,
            final_log_x,
            live_log_psi0,
            self.n_live,
        )
        history = RunHistory(
            iteration=np.arange(1, niter + 1, dtype=np.int64),
            discarded_log_psi=np.asarray(dead_log_psi0),
            log_x=np.asarray(dead_log_x),
            log_delta_x=np.asarray(dead_log_delta_x),
            logz_dead=np.asarray(history_logz_dead),
            logz_live=np.asarray(history_logz_live),
            logz_total=np.asarray(history_logz_total),
            information=np.asarray(history_information),
            logzerr=np.asarray(history_logzerr),
            remaining_fraction=np.asarray(history_remaining_fraction),
            remaining_dlogz=np.asarray(history_remaining_dlogz),
            live_ess=np.asarray(history_live_ess),
            live_mean_rse=np.asarray(history_live_mean_rse),
            live_logz_error=np.asarray(history_live_logz_error),
            logz_stability=np.asarray(history_logz_stability),
            stopping_streak=np.asarray(history_stopping_streak, dtype=np.int64),
            live_min_log_psi=np.asarray(history_live_min),
            live_median_log_psi=np.asarray(history_live_median),
            live_max_log_psi=np.asarray(history_live_max),
            proposals=np.asarray(history_proposals, dtype=np.int64),
            likelihood_calls=np.asarray(history_likelihood_calls, dtype=np.int64),
            acceptance_fraction=np.asarray(history_acceptance),
            mh_acceptance_fraction=np.asarray(history_mh_acceptance),
            constraint_pass_fraction=np.asarray(history_constraint_pass),
            mcmc_accepted=np.asarray(history_mcmc_accepted, dtype=np.int64),
            mcmc_moved=np.asarray(history_mcmc_moved, dtype=np.int64),
            mcmc_completed=np.asarray(history_mcmc_completed, dtype=np.int64),
            elapsed_seconds=np.asarray(history_elapsed),
            proposal_revision=np.asarray(history_proposal_revision, dtype=np.int64),
            proposal_update_attempts=np.asarray(
                history_proposal_update_attempts, dtype=np.int64
            ),
            proposal_update_failures=np.asarray(
                history_proposal_update_failures, dtype=np.int64
            ),
        )
        success = termination_reason in SCIENTIFIC_TERMINATION_REASONS
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
        importance_metadata = getattr(self.importance_morph, "metadata", None)
        proposal_updates = (
            ()
            if self.adaptive_proposal_controller is None
            else self.adaptive_proposal_controller.records
        )
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
            dead_log_q0=np.asarray(dead_log_q0),
            dead_log_psi0=np.asarray(dead_log_psi0),
            dead_tie_breakers=np.asarray(dead_tie_breakers),
            dead_log_x=np.asarray(dead_log_x),
            dead_log_weights=np.asarray(dead_log_weights),
            final_live_points=live_theta,
            final_live_log_likelihood=live_log_likelihood,
            final_live_log_prior=live_log_prior,
            final_live_log_q0=live_log_q0,
            final_live_log_psi0=live_log_psi0,
            final_live_tie_breakers=live_tie_breakers,
            log_posterior_weights=quadrature.log_posterior_weights,
            history=history,
            config=config,
            rng_bit_generator=self.rng.bit_generator.__class__.__name__,
            rng_state_initial=repr(initial_state),
            rng_state_final=repr(final_state),
            importance_morph_description=repr(importance_metadata),
            proposal_updates=proposal_updates,
            nonfinite_counts=(
                ("outside_prior", evaluator.outside_prior),
                ("zero_likelihood", evaluator.zero_likelihood),
            ),
            warnings=tuple(warnings),
        )
