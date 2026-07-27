"""Public Phase 2 interface for MINS."""

from ._version import __version__
from .config import MINSConfig
from .exceptions import (
    ConfigurationError,
    InvalidModelOutput,
    InvalidProposalOutput,
    MINSError,
    MissingOptionalDependency,
    NumericalInvariantError,
    ProposalSupportError,
)
from .model import CallableModel, Model
from .proposals import MorphMetadata, MorphProposal, Proposal
from .results import MINSResult, RunHistory
from .sampler import MINSampler
from .stopping import StoppingCriterionConfig, StoppingPolicy

__all__ = [
    "CallableModel",
    "ConfigurationError",
    "InvalidModelOutput",
    "InvalidProposalOutput",
    "MINSConfig",
    "MINSError",
    "MINSResult",
    "MINSampler",
    "MissingOptionalDependency",
    "Model",
    "MorphMetadata",
    "MorphProposal",
    "NumericalInvariantError",
    "Proposal",
    "ProposalSupportError",
    "RunHistory",
    "StoppingCriterionConfig",
    "StoppingPolicy",
    "__version__",
]
