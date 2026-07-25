"""Typed exceptions raised by MINS."""


class MINSError(Exception):
    """Base class for package-specific errors."""


class ConfigurationError(MINSError, ValueError):
    """Raised when sampler configuration is inconsistent or out of range."""


class InvalidModelOutput(MINSError, ValueError):
    """Raised when a model returns invalid values or array shapes."""


class InvalidProposalOutput(MINSError, ValueError):
    """Raised when a proposal returns invalid values or array shapes."""


class ProposalSupportError(MINSError):
    """Raised when Morph has no density where the target integrand is finite."""


class MissingOptionalDependency(MINSError, ImportError):
    """Raised when a requested optional integration is unavailable."""


class NumericalInvariantError(MINSError, ArithmeticError):
    """Raised when evidence arithmetic violates a required invariant."""
