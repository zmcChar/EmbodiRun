import os
import subprocess
import sys

from embodied_runtime.policies import NavigationPolicy as PackageNavigationPolicy
from embodied_runtime.policies.navigation import (
    EpisodeCursor,
    InternVLANavigationPolicy,
    NavigationPolicy,
    NavigationPolicyError,
    NaVILANavigationPolicy,
    QwenNavigationPolicy,
    StreamVLNNavigationPolicy,
)
from embodied_runtime.policies.navigation.episode import EpisodeCursor as DirectEpisodeCursor
from embodied_runtime.policies.navigation.errors import (
    NavigationPolicyError as DirectNavigationPolicyError,
)
from embodied_runtime.policies.navigation.internvla import (
    InternVLANavigationPolicy as DirectInternVLAPolicy,
)
from embodied_runtime.policies.navigation.navila import (
    NaVILANavigationPolicy as DirectNaVILAPolicy,
)
from embodied_runtime.policies.navigation.qwen.policy import (
    QwenNavigationPolicy as DirectQwenPolicy,
)
from embodied_runtime.policies.navigation.streamvln import (
    StreamVLNNavigationPolicy as DirectStreamVLNPolicy,
)
from embodied_runtime.tasks.navigation.interfaces import (
    NavigationPolicy as TaskNavigationPolicy,
)


def test_navigation_package_exports_canonical_symbols() -> None:
    assert PackageNavigationPolicy is TaskNavigationPolicy
    assert NavigationPolicy is TaskNavigationPolicy
    assert EpisodeCursor is DirectEpisodeCursor
    assert NavigationPolicyError is DirectNavigationPolicyError
    assert QwenNavigationPolicy is DirectQwenPolicy
    assert StreamVLNNavigationPolicy is DirectStreamVLNPolicy
    assert InternVLANavigationPolicy is DirectInternVLAPolicy
    assert NaVILANavigationPolicy is DirectNaVILAPolicy


def test_navigation_package_keeps_concrete_policies_lazy() -> None:
    script = """
import sys
import embodied_runtime.policies.navigation as navigation

concrete = (
    'embodied_runtime.policies.navigation.qwen.policy',
    'embodied_runtime.policies.navigation.streamvln.policy',
    'embodied_runtime.policies.navigation.internvla.policy',
    'embodied_runtime.policies.navigation.navila.policy',
)
assert not any(name in sys.modules for name in concrete)
assert navigation.QwenNavigationPolicy.__name__ == 'QwenNavigationPolicy'
assert concrete[0] in sys.modules
assert concrete[1] not in sys.modules
assert concrete[2] not in sys.modules
assert concrete[3] not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
    )


def test_all_navigation_policies_implement_task_policy_structurally() -> None:
    assert isinstance(object.__new__(QwenNavigationPolicy), NavigationPolicy)
    assert isinstance(object.__new__(StreamVLNNavigationPolicy), NavigationPolicy)
    assert isinstance(object.__new__(InternVLANavigationPolicy), NavigationPolicy)
    assert isinstance(object.__new__(NaVILANavigationPolicy), NavigationPolicy)
