"""Errors shared by the StreamVLN runtime components."""


class StreamVLNLoadError(RuntimeError):
    """The official repository, dependencies, or checkpoint could not load."""


class StreamVLNInferenceError(RuntimeError):
    """The recurrent evaluator failed after it had been loaded."""


__all__ = ["StreamVLNInferenceError", "StreamVLNLoadError"]
