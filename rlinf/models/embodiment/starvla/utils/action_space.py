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

"""Action normalization helpers for starVLA-backed RLinf policies."""

from __future__ import annotations

import importlib
import inspect
import os
from collections.abc import Callable, Mapping
from typing import Any, Optional

import numpy as np

# starVLA moved its model-agnostic action helpers off ``baseframework`` onto
# ``starVLA.model.tools.FrameworkTools``. Releases up to the ``starVLA-1.2``
# tag (the revision requirements/install.sh pins) only carry the former, newer
# revisions only the latter, so both locations are probed in order.
_ACTION_HELPER_OWNERS = (
    ("starVLA.model.tools", "FrameworkTools"),
    ("starVLA.model.framework.base_framework", "baseframework"),
)

_UNNORMALIZE_ACTIONS_FN: Optional[Callable[..., np.ndarray]] = None


def _iter_action_helper_owners():
    """Yield ``(dotted_name, owner)`` for the starVLA action-helper holders."""
    for module_name, attr_name in _ACTION_HELPER_OWNERS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # ImportError, or a broken optional dep
            yield f"{module_name}.{attr_name}", exc
            continue
        owner = getattr(module, attr_name, None)
        yield f"{module_name}.{attr_name}", owner


def resolve_unnormalize_actions_fn() -> Callable[..., np.ndarray]:
    """Return starVLA's ``unnormalize_actions`` helper, whichever version is installed.

    Returns:
        The static helper mapping normalized actions back to the env action
        space, resolved once and cached.

    Raises:
        ModuleNotFoundError: If starVLA is not importable, or exposes the
            helper at neither the current nor the legacy location.
    """
    global _UNNORMALIZE_ACTIONS_FN
    if _UNNORMALIZE_ACTIONS_FN is not None:
        return _UNNORMALIZE_ACTIONS_FN

    problems: list[str] = []
    for name, owner in _iter_action_helper_owners():
        if isinstance(owner, Exception):
            problems.append(f"{name}: {type(owner).__name__}: {owner}")
            continue
        fn = getattr(owner, "unnormalize_actions", None)
        if callable(fn):
            _UNNORMALIZE_ACTIONS_FN = fn
            return fn
        problems.append(f"{name}: no 'unnormalize_actions' attribute")

    raise ModuleNotFoundError(
        "starVLA is required for action unnormalization, but none of its known "
        "locations provide 'unnormalize_actions'. Checked "
        + "; ".join(problems)
        + ". Set cfg.unnorm_key=None to use normalized actions directly."
    )


def _resolve_action_stats_getter(
    starvla_model: Any,
) -> Optional[Callable[[Optional[str]], Any]]:
    """Return a callable mapping an ``unnorm_key`` to a stats block, if any.

    Older starVLA bound ``get_action_stats`` to the model; newer revisions
    moved it to ``FrameworkTools``, where it takes the stats dict explicitly.
    """
    getter = getattr(starvla_model, "get_action_stats", None)
    if callable(getter):
        return getter

    norm_stats = getattr(starvla_model, "norm_stats", None)
    if not isinstance(norm_stats, Mapping):
        return None
    for _name, owner in _iter_action_helper_owners():
        if isinstance(owner, Exception):
            continue
        tools_getter = getattr(owner, "get_action_stats", None)
        if callable(tools_getter) and _takes_stats_first(tools_getter):
            return lambda key, _getter=tools_getter: _getter(norm_stats, key)
    return None


def _takes_stats_first(getter: Callable[..., Any]) -> bool:
    """Whether ``getter`` takes the stats dict first, not an unbound ``self``.

    Legacy starVLA declared ``get_action_stats`` as a method reading
    ``self.norm_stats``; only the ``FrameworkTools`` version is a static helper
    that accepts the stats dict.
    """
    try:
        parameters = list(inspect.signature(getter).parameters)
    except (TypeError, ValueError):
        return False
    return bool(parameters) and parameters[0] != "self"


def resolve_action_norm_stats(
    starvla_model: Any,
    unnorm_key: Optional[str],
    action_dim: int,
    action_stats_source: Optional[str] = None,
) -> Optional[dict[str, np.ndarray]]:
    """Resolve action normalization stats from starVLA.

    Strict contract:
      - If unnorm_key is None, return None (caller opted out of unnormalization).
      - If unnorm_key is not None, return valid stats or raise an exception.
    """
    if unnorm_key is None:
        return None

    raw_stats: Any = None
    norm_stats = getattr(starvla_model, "norm_stats", None)
    if isinstance(norm_stats, Mapping) and unnorm_key in norm_stats:
        raw_stats = norm_stats.get(unnorm_key)
    else:
        getter = _resolve_action_stats_getter(starvla_model)
        if getter is None:
            raise RuntimeError(
                "starVLA action unnormalization requires action norm stats, but the "
                f"loaded model exposes no stats for unnorm_key={unnorm_key!r}: it has "
                "no usable 'norm_stats' mapping, no 'get_action_stats()' method, and "
                "starVLA provides no 'get_action_stats' helper."
            )
        try:
            raw_stats = getter(unnorm_key)
        except Exception as exc:
            raise RuntimeError(
                "starVLA get_action_stats failed; cannot unnormalize actions for env. "
                f"unnorm_key={unnorm_key!r}, error={type(exc).__name__}: {exc}."
            ) from exc

    if raw_stats is None or not isinstance(raw_stats, Mapping):
        raise RuntimeError(
            "starVLA action norm stats payload is missing or invalid; cannot "
            f"unnormalize actions for env. unnorm_key={unnorm_key!r}."
        )

    stats_payload = raw_stats.get("action", raw_stats)
    if not isinstance(stats_payload, Mapping):
        raise RuntimeError(
            "starVLA action stats payload is invalid; cannot unnormalize actions for env. "
            f"unnorm_key={unnorm_key!r}."
        )

    source = (
        "q01q99"
        if action_stats_source is None
        else str(action_stats_source).strip().lower()
    )
    if source in ("minmax", "maxmin"):
        high_src = stats_payload.get("max")
        low_src = stats_payload.get("min")
        missing_label = "max/min"
    elif source in ("q01q99", "q99q01", "percentile"):
        high_src = stats_payload.get("q99", stats_payload.get("max"))
        low_src = stats_payload.get("q01", stats_payload.get("min"))
        missing_label = "q99/q01 (or max/min fallback)"
    else:
        raise ValueError(
            "Unsupported starVLA action_stats_source. "
            f"Expected 'q01q99' or 'minmax', got {action_stats_source!r}."
        )

    if high_src is None or low_src is None:
        raise RuntimeError(
            f"starVLA action norm stats missing {missing_label}; cannot unnormalize "
            f"actions for env. unnorm_key={unnorm_key!r}, action_stats_source={source!r}."
        )

    try:
        high = np.asarray(high_src, dtype=np.float32).reshape(-1)
        low = np.asarray(low_src, dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise RuntimeError(
            "starVLA action norm stats are not numeric arrays; cannot unnormalize actions "
            f"for env. unnorm_key={unnorm_key!r}, error={type(exc).__name__}: {exc}."
        ) from exc

    if high.shape[0] != action_dim or low.shape[0] != action_dim:
        raise RuntimeError(
            "starVLA action norm stats dim mismatch with RLinf action_dim; cannot "
            f"unnormalize actions for env. stats_dim={high.shape[0]}, action_dim={action_dim}, "
            f"unnorm_key={unnorm_key!r}."
        )

    mask_src = stats_payload.get("mask")
    if mask_src is None:
        mask = np.ones((action_dim,), dtype=bool)
    else:
        try:
            mask = np.asarray(mask_src, dtype=bool).reshape(-1)
        except Exception:
            mask = np.ones((action_dim,), dtype=bool)
        if mask.shape[0] != action_dim:
            mask = np.ones((action_dim,), dtype=bool)

    return {
        "q99": high,
        "q01": low,
        "mask": mask,
    }


_LIBERO_PLATFORMS = {"libero"}


def _gripper_mapping(
    actions: np.ndarray,
    policy_setup: Optional[str] = None,
) -> np.ndarray:
    """Apply LIBERO gripper mapping aligned with starVLA eval pipeline.

    Converts gripper dim (index 6) from 0/1 (as output by starVLA's
    ``unnormalize_actions``) to +1/-1 as expected by the LIBERO env.

    The mapping is activated when *policy_setup* resolves to a LIBERO
    platform.  When *policy_setup* is ``None`` we fall back to the
    ``ROBOT_PLATFORM`` env-var for backward compatibility.
    """
    resolved_platform: str
    if policy_setup is not None:
        resolved_platform = str(policy_setup).strip().lower()
    else:
        resolved_platform = str(os.environ.get("ROBOT_PLATFORM", "")).strip().lower()

    if resolved_platform not in _LIBERO_PLATFORMS:
        return actions
    if actions.shape[-1] < 7:
        return actions

    out = actions.astype(np.float32, copy=True)
    open_gripper = np.where(out[..., 6] < 0.5, 0.0, 1.0).astype(np.float32)
    out[..., 6] = (1.0 - 2.0 * (open_gripper > 0.5).astype(np.float32)).astype(
        np.float32
    )
    return out


def unnormalize_actions_for_env(
    normalized_actions: np.ndarray,
    action_norm_stats: dict[str, np.ndarray],
    policy_setup: Optional[str] = None,
) -> np.ndarray:
    """Map model normalized actions to env action space (strict).

    Args:
        normalized_actions: Actions in normalized [-1, 1] space from the model.
        action_norm_stats: Dict with ``q99``, ``q01``, ``mask`` arrays.
        policy_setup: Robot platform identifier (e.g. ``"libero"``).
            When provided, the gripper mapping is decided by this value
            instead of the ``ROBOT_PLATFORM`` environment variable.
    """
    if action_norm_stats is None:
        raise RuntimeError(
            "Missing action_norm_stats: cannot unnormalize actions for env. "
            "Set cfg.unnorm_key=None to use normalized actions directly."
        )

    unnormalize_actions = resolve_unnormalize_actions_fn()

    actions = np.asarray(normalized_actions, dtype=np.float32)
    flat = actions.reshape(-1, actions.shape[-1]).astype(np.float32, copy=False)
    starvla_stats = {
        "q99": np.asarray(action_norm_stats["q99"], dtype=np.float32),
        "q01": np.asarray(action_norm_stats["q01"], dtype=np.float32),
        "mask": np.asarray(action_norm_stats["mask"], dtype=bool),
    }
    env_flat = unnormalize_actions(flat, starvla_stats)
    env_actions = np.asarray(env_flat, dtype=np.float32).reshape(actions.shape)
    return _gripper_mapping(env_actions, policy_setup=policy_setup)
