"""Errors that may cross group boundaries."""


class RuntimePrototypeError(Exception):
    """Base error for the prototype."""


class UnsupportedBackendError(RuntimePrototypeError):
    pass


class BackendExecutionError(RuntimePrototypeError):
    pass


class RequestCancelledError(RuntimePrototypeError):
    pass


class RequestDeadlineExceededError(RuntimePrototypeError):
    pass


class ModelPackageError(RuntimePrototypeError):
    pass
