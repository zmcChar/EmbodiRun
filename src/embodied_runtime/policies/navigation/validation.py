"""Input validation shared by navigation-policy adapters."""

from embodied_runtime.tasks.navigation import NavigationRequest


def navigation_request(payload: object) -> NavigationRequest:
    if not isinstance(payload, NavigationRequest):
        raise TypeError("navigation policy payload must be a NavigationRequest")
    return payload


__all__ = ["navigation_request"]
