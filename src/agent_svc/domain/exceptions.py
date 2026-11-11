"""Domain exception hierarchy. Each error carries the HTTP semantics the API layer renders."""

from __future__ import annotations

from typing import Any


class DomainError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    title: str = "Internal Server Error"
    retryable: bool = False

    def __init__(
        self, detail: str | None = None, *, retryable: bool | None = None, **extra: Any
    ) -> None:
        super().__init__(detail or self.title)
        self.detail = detail or self.title
        self.extra = extra
        if retryable is not None:
            self.retryable = retryable


class BadRequestError(DomainError):
    status_code, code, title = 400, "bad_request", "Bad Request"


class UnauthorizedError(DomainError):
    status_code, code, title = 401, "unauthorized", "Unauthorized"


class ForbiddenError(DomainError):
    status_code, code, title = 403, "forbidden", "Forbidden"


class NotFoundError(DomainError):
    status_code, code, title = 404, "not_found", "Not Found"


class ConflictError(DomainError):
    status_code, code, title = 409, "conflict", "Conflict"


class PayloadTooLargeError(DomainError):
    status_code, code, title = 413, "payload_too_large", "Payload Too Large"


class PolicyViolationError(DomainError):
    """A guardrail blocked the request or response."""

    status_code, code, title = 422, "policy_violation", "Unprocessable Content"


class RateLimitedError(DomainError):
    status_code, code, title = 429, "rate_limited", "Too Many Requests"
    retryable = True

    def __init__(self, detail: str | None = None, *, retry_after: int = 1, **extra: Any) -> None:
        super().__init__(detail, **extra)
        self.retry_after = retry_after


class BudgetExceededError(RateLimitedError):
    code, title = "token_budget_exceeded", "Token Budget Exceeded"


class ClientClosedError(DomainError):
    status_code, code, title = 499, "client_closed_request", "Client Closed Request"


class ConfigurationError(DomainError):
    """Raised at startup for bad config or a missing optional extra."""

    status_code, code, title = 500, "configuration_error", "Configuration Error"


class UpstreamError(DomainError):
    """Dependency (LLM, tool, store) failed."""

    status_code, code, title = 502, "upstream_error", "Bad Gateway"
    retryable = True


class UnavailableError(DomainError):
    """Circuit open, overloaded, draining or dependency down."""

    status_code, code, title = 503, "service_unavailable", "Service Unavailable"
    retryable = True


class CircuitOpenError(UnavailableError):
    """Fail-fast: breaker is open. Not worth retrying against the same provider."""

    code, title = "circuit_open", "Circuit Open"
    retryable = False


class UpstreamTimeoutError(DomainError):
    status_code, code, title = 504, "upstream_timeout", "Gateway Timeout"
    retryable = True
