"""Cloud-session protocol and remote failure types."""


class CloudSessionProtocolError(RuntimeError):
    """The shared cloud service returned an invalid session response."""


class CloudSessionRemoteError(RuntimeError):
    """The shared cloud service rejected a well-formed request."""

    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        self.remote_message = message
        super().__init__(f"{error_type}: {message}")
