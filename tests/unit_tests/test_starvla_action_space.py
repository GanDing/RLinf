# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for starVLA action-space helpers across starVLA API layouts.

starVLA moved ``unnormalize_actions`` / ``get_action_stats`` off
``baseframework`` onto ``starVLA.model.tools.FrameworkTools``. Both layouts are
stubbed here so the resolution logic is covered without installing starVLA.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# Loaded by path: importing the package would pull in the starVLA third-party
# dependencies, which are only present in the starvla venv.
_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "rlinf"
    / "models"
    / "embodiment"
    / "starvla"
    / "utils"
    / "action_space.py"
)
_spec = importlib.util.spec_from_file_location("starvla_action_space", _MODULE_PATH)
action_space = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(action_space)


def _unnormalize_actions(normalized_actions, action_norm_stats, gripper_channel_idx=6):
    """Stand-in for starVLA's helper, matching its documented formula."""
    mask = action_norm_stats["mask"]
    high = np.asarray(action_norm_stats["q99"])
    low = np.asarray(action_norm_stats["q01"])
    normalized_actions = np.clip(normalized_actions, -1, 1)
    normalized_actions[:, gripper_channel_idx] = np.where(
        normalized_actions[:, gripper_channel_idx] < 0.5, 0, 1
    )
    return np.where(
        mask, 0.5 * (normalized_actions + 1) * (high - low) + low, normalized_actions
    )


def _install_module(monkeypatch, name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture(autouse=True)
def _clear_resolver_cache(monkeypatch):
    monkeypatch.setattr(action_space, "_UNNORMALIZE_ACTIONS_FN", None, raising=True)
    # Both module paths are absent unless a test installs a stub.
    for name in ("starVLA.model.tools", "starVLA.model.framework.base_framework"):
        monkeypatch.setitem(sys.modules, name, None)


def _install_current_starvla(monkeypatch):
    """starVLA revisions that expose the helpers on FrameworkTools."""

    class FrameworkTools:
        @staticmethod
        def unnormalize_actions(
            normalized_actions, action_norm_stats, gripper_channel_idx=6
        ):
            return _unnormalize_actions(
                normalized_actions, action_norm_stats, gripper_channel_idx
            )

        @staticmethod
        def get_action_stats(norm_stats, unnorm_key=None):
            return norm_stats[unnorm_key]["action"]

    _install_module(monkeypatch, "starVLA.model.tools", FrameworkTools=FrameworkTools)

    class baseframework:  # noqa: N801 - mirrors starVLA's own class name
        """The current layout keeps the class but drops the action helpers."""

    _install_module(
        monkeypatch,
        "starVLA.model.framework.base_framework",
        baseframework=baseframework,
    )
    return FrameworkTools


def _install_legacy_starvla(monkeypatch):
    """starVLA up to the pinned starVLA-1.2 tag: helpers on baseframework."""

    class baseframework:  # noqa: N801 - mirrors starVLA's own class name
        @staticmethod
        def unnormalize_actions(normalized_actions, action_norm_stats):
            return _unnormalize_actions(normalized_actions, action_norm_stats)

        def get_action_stats(self, unnorm_key=None):
            return self.norm_stats[unnorm_key]["action"]

    _install_module(
        monkeypatch,
        "starVLA.model.framework.base_framework",
        baseframework=baseframework,
    )
    return baseframework


def test_resolves_unnormalize_from_current_starvla(monkeypatch):
    tools = _install_current_starvla(monkeypatch)
    assert action_space.resolve_unnormalize_actions_fn() is tools.unnormalize_actions


def test_resolves_unnormalize_from_legacy_starvla(monkeypatch):
    legacy = _install_legacy_starvla(monkeypatch)
    assert action_space.resolve_unnormalize_actions_fn() is legacy.unnormalize_actions


def test_unnormalize_raises_when_no_layout_provides_the_helper(monkeypatch):
    _install_module(monkeypatch, "starVLA.model.tools", FrameworkTools=object())
    with pytest.raises(ModuleNotFoundError, match="unnormalize_actions"):
        action_space.resolve_unnormalize_actions_fn()


@pytest.mark.parametrize("layout", ["current", "legacy"])
def test_unnormalize_actions_for_env_matches_starvla_formula(monkeypatch, layout):
    if layout == "current":
        _install_current_starvla(monkeypatch)
    else:
        _install_legacy_starvla(monkeypatch)

    stats = {
        "q99": np.ones(7, dtype=np.float32),
        "q01": -np.ones(7, dtype=np.float32),
        "mask": np.ones(7, dtype=bool),
    }
    normalized = np.zeros((1, 2, 7), dtype=np.float32)
    normalized[..., 0] = 0.5
    normalized[..., 6] = 1.0

    env_actions = action_space.unnormalize_actions_for_env(normalized, stats)

    assert env_actions.shape == normalized.shape
    np.testing.assert_allclose(env_actions[..., 0], 0.5, rtol=1e-6)
    # No policy_setup and no ROBOT_PLATFORM: the gripper stays in 0/1 space.
    np.testing.assert_allclose(env_actions[..., 6], 1.0, rtol=1e-6)


def test_unnormalize_actions_for_env_applies_libero_gripper_mapping(monkeypatch):
    _install_current_starvla(monkeypatch)

    stats = {
        "q99": np.ones(7, dtype=np.float32),
        "q01": -np.ones(7, dtype=np.float32),
        "mask": np.ones(7, dtype=bool),
    }
    normalized = np.zeros((2, 7), dtype=np.float32)
    normalized[0, 6] = 1.0
    normalized[1, 6] = 0.0

    env_actions = action_space.unnormalize_actions_for_env(
        normalized, stats, policy_setup="libero"
    )

    # LIBERO expects the gripper as -1 (closed) / +1 (open).
    np.testing.assert_allclose(env_actions[0, 6], -1.0, rtol=1e-6)
    np.testing.assert_allclose(env_actions[1, 6], 1.0, rtol=1e-6)


def test_action_stats_resolve_via_framework_tools(monkeypatch):
    """Newer starVLA drops model.get_action_stats; the static helper stands in."""
    _install_current_starvla(monkeypatch)

    stats_block = {
        "q99": [1.0] * 7,
        "q01": [-1.0] * 7,
        "mask": [True] * 7,
    }
    # norm_stats present but keyed differently, so the direct lookup misses.
    model = SimpleNamespace(norm_stats={"libero_spatial": {"action": stats_block}})

    resolved = action_space.resolve_action_norm_stats(
        model, "libero_spatial", action_dim=7
    )

    np.testing.assert_allclose(resolved["q99"], np.ones(7, dtype=np.float32))
    np.testing.assert_allclose(resolved["q01"], -np.ones(7, dtype=np.float32))
    assert resolved["mask"].all()


def test_action_stats_getter_ignores_unbound_legacy_method(monkeypatch):
    """A legacy baseframework.get_action_stats needs a model instance, not stats."""
    _install_legacy_starvla(monkeypatch)

    model = SimpleNamespace(norm_stats={"other_key": {"action": {}}})

    assert action_space._resolve_action_stats_getter(model) is None
    with pytest.raises(RuntimeError, match="no usable 'norm_stats' mapping"):
        action_space.resolve_action_norm_stats(model, "libero_spatial", action_dim=7)
