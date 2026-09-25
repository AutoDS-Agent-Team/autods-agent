import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ApplicationError(Exception):
    status_code = 400

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class AuthenticationError(ApplicationError):
    status_code = 401


class DatasetValidationError(ApplicationError):
    """Raised when an uploaded file is not a usable CSV dataset."""


class DatasetTooLargeError(ApplicationError):
    status_code = 413


class DatasetNotFoundError(ApplicationError):
    status_code = 404


class DatasetInUseError(ApplicationError):
    status_code = 409


class ExperimentNotFoundError(ApplicationError):
    status_code = 404


class JobNotFoundError(ApplicationError):
    status_code = 404


class ExperimentValidationError(ApplicationError):
    """Raised when an experiment definition is not safe or usable."""


class PlanningNotReadyError(ApplicationError):
    status_code = 409


class PlanningFailureError(ApplicationError):
    status_code = 502


class TrainingNotReadyError(ApplicationError):
    status_code = 409


class TrainingFailureError(ApplicationError):
    """Raised when trusted preprocessing cannot prepare a dataset for training."""


async def application_error_handler(
    _: Request, exception: ApplicationError
) -> JSONResponse:
    return JSONResponse(
        status_code=exception.status_code,
        content={"detail": exception.detail},
    )


async def unhandled_exception_handler(
    request: Request, exception: Exception
) -> JSONResponse:
    logger.exception(
        "Unhandled exception while processing %s %s",
        request.method,
        request.url.path,
        exc_info=exception,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."},
    )


def register_exception_handlers(application: FastAPI) -> None:
    application.add_exception_handler(ApplicationError, application_error_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)
