# Copyright 2025 The RLinf Authors.
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

"""Opt-in accelerator memory probes for diagnosing OOMs.

Every helper here is a no-op unless ``RLINF_MEM_DEBUG`` is set to a truthy
value (``1``, ``true``, ``yes``, ``on``), so the probes can live permanently at
the interesting call sites without costing anything in normal runs.

The probes go through :attr:`Worker.torch_platform`, so they work on any
accelerator the scheduler supports (CUDA, Ascend NPU, MUSA, ...).

Example::

    RLINF_MEM_DEBUG=1 bash examples/embodiment/run_embodiment.sh maniskill_grpo_openvla

Each line is prefixed with ``[mem]`` so a run can be filtered with
``grep '\\[mem\\]'``.
"""

from typing import Any, Iterable, Optional

import torch

_ENV_VAR = "RLINF_MEM_DEBUG"
_TRUTHY = ("1", "true", "yes", "on")

_MIB = 1024.0 * 1024.0


def mem_debug_enabled() -> bool:
    """Whether the memory probes are enabled via ``RLINF_MEM_DEBUG``."""
    import os

    return os.environ.get(_ENV_VAR, "").strip().lower() in _TRUTHY


def _logger():
    from rlinf.utils.logging import get_logger

    return get_logger()


def _platform():
    """Return the accelerator platform module, or ``None`` if unavailable."""
    from rlinf.scheduler import Worker

    return Worker.torch_platform


def _stat(platform, name: str) -> Optional[float]:
    fn = getattr(platform, name, None)
    if fn is None:
        return None
    try:
        return fn() / _MIB
    except Exception:  # pragma: no cover - platform-specific failures
        return None


def mem_snapshot(
    tag: str,
    logger: Any = None,
    reset_peak: bool = False,
) -> None:
    """Log the current accelerator memory usage under ``tag``.

    No-op unless ``RLINF_MEM_DEBUG`` is enabled.

    Args:
        tag: Short label identifying the call site, e.g.
            ``"sync_model_to_rollout/enter"``.
        logger: Logger to use. Defaults to the current worker's logger.
        reset_peak: Reset the platform's peak-memory counters after logging, so
            the next snapshot's peak covers only the region in between.
    """
    if not mem_debug_enabled():
        return

    platform = _platform()
    if platform is None:
        return

    logger = logger if logger is not None else _logger()
    parts = []
    for label, name in (
        ("alloc", "memory_allocated"),
        ("reserved", "memory_reserved"),
        ("peak_alloc", "max_memory_allocated"),
        ("peak_reserved", "max_memory_reserved"),
    ):
        value = _stat(platform, name)
        if value is not None:
            parts.append(f"{label}={value:.0f}MiB")

    logger.info(f"[mem] {tag} " + " ".join(parts))

    if reset_peak:
        reset_fn = getattr(platform, "reset_peak_memory_stats", None)
        if reset_fn is not None:
            try:
                reset_fn()
            except Exception:  # pragma: no cover - platform-specific failures
                pass


def tensor_bytes(tensor: Optional[torch.Tensor]) -> int:
    """Storage size of ``tensor`` in bytes (0 for ``None``)."""
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def log_tensor_sizes(
    tag: str,
    tensors: Iterable[tuple[str, Optional[torch.Tensor]]],
    logger: Any = None,
    max_items: int = 12,
) -> None:
    """Log ``dtype``/``numel``/bytes for each named tensor, plus the total.

    No-op unless ``RLINF_MEM_DEBUG`` is enabled.

    Args:
        tag: Short label identifying the call site.
        tensors: ``(name, tensor)`` pairs. ``None`` tensors are reported as
            absent so a payload's optional fields stay visible.
        logger: Logger to use. Defaults to the current worker's logger.
        max_items: Cap on how many tensors are listed individually. Beyond it
            only the largest ``max_items`` are shown, so bucket payloads with
            hundreds of entries stay readable. The total always covers all of
            them.
    """
    if not mem_debug_enabled():
        return

    logger = logger if logger is not None else _logger()
    entries = list(tensors)
    total = sum(tensor_bytes(tensor) for _, tensor in entries)

    shown = entries
    elided = 0
    if len(entries) > max_items:
        shown = sorted(entries, key=lambda kv: -tensor_bytes(kv[1]))[:max_items]
        elided = len(entries) - max_items

    parts = []
    for name, tensor in shown:
        if tensor is None:
            parts.append(f"{name}=None")
            continue
        parts.append(
            f"{name}[{tensor.dtype}, n={tensor.numel()}, "
            f"{tensor_bytes(tensor) / _MIB:.1f}MiB]"
        )
    if elided:
        parts.append(f"(+{elided} more)")

    logger.info(
        f"[mem] {tag} count={len(entries)} total={total / _MIB:.1f}MiB "
        + " ".join(parts)
    )
