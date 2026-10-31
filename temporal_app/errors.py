from __future__ import annotations

from temporalio.exceptions import ApplicationError

from temporal_app.models import NON_RETRYABLE_ERROR_TYPES


def app_error(message: str, error_type: str, retryable: bool | None = None) -> ApplicationError:
    """Build an ApplicationError whose retry behaviour matches the pipeline classification."""
    if retryable is None:
        retryable = error_type not in NON_RETRYABLE_ERROR_TYPES
    return ApplicationError(message, type=error_type, non_retryable=not retryable)
