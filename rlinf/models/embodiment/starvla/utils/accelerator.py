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

"""Accelerator portability helpers for the starVLA RLinf wrapper.

starVLA upstream targets CUDA, so its execution paths hardcode
``torch.autocast("cuda", ...)`` and rely on CUDA kernel coverage. RLinf also
runs on non-CUDA accelerators (Ascend NPU via ``torch_npu``, Intel XPU, ...),
where a ``"cuda"`` autocast context silently does nothing: ops then keep the
backbone's bfloat16 dtype instead of being upcast, and reach kernels that only
exist for float32 on that backend. The helpers here resolve the autocast device
type from the tensors/modules actually being used, and build Gaussian policy
distributions in float32.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, ContextManager, Optional

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

logger = logging.getLogger(__name__)

# Autocast is only requested for accelerator backends; CPU runs simply execute
# without it.
_AUTOCAST_DEVICE_TYPES = frozenset({"cuda", "npu", "xpu", "musa", "hpu"})

# The only dtypes a backend can autocast *to*. Asking for float32 does not mean
# "compute in float32": no backend accepts it, so torch warns on every context
# entry and falls back to running the block with autocast switched off.
_AUTOCAST_DTYPES = frozenset({torch.float16, torch.bfloat16})

_AUTOCAST_UNUSABLE_DEVICE_TYPES: set[str] = set()


def resolve_device_type(source: Any) -> str:
    """Return the torch device type backing ``source``.

    Args:
        source: A tensor, module, device, device string, or ``None``.

    Returns:
        The device type (e.g. ``"cuda"``, ``"npu"``), or ``"cpu"`` when it
        cannot be determined (an empty module, or ``None``).
    """
    if isinstance(source, torch.Tensor):
        return source.device.type
    if isinstance(source, torch.device):
        return source.type
    if isinstance(source, str):
        try:
            return torch.device(source).type
        except RuntimeError:
            # A backend name this torch build does not know, e.g. "npu" before
            # torch_npu registers it. Keep the name, drop the device index.
            return source.split(":", 1)[0]
    if isinstance(source, nn.Module):
        for tensor in source.parameters():
            return tensor.device.type
        for tensor in source.buffers():
            return tensor.device.type
    return "cpu"


def is_autocast_available(device_type: str) -> bool:
    """Return whether torch exposes an autocast backend for ``device_type``."""
    if device_type not in _AUTOCAST_DEVICE_TYPES:
        return False
    checker = getattr(torch.amp, "is_autocast_available", None)
    if not callable(checker):  # torch < 2.4
        return True
    try:
        return bool(checker(device_type))
    except RuntimeError:
        # Raised for device types torch does not know about at all.
        return False


def autocast_ctx(dtype: torch.dtype, *, device: Any) -> ContextManager:
    """Return an autocast context on ``device``'s backend, else a no-op context.

    Args:
        dtype: Autocast dtype. ``torch.float16``/``torch.bfloat16`` cast the
            block; ``torch.float32`` means "not autocast", matching what
            upstream starVLA's ``torch.autocast("cuda", dtype=torch.float32)``
            blocks actually do — they suspend any enclosing autocast so the
            block runs in the parameters' own precision.
        device: Tensor, module, device, or device string naming the backend to
            autocast on.

    Returns:
        ``torch.autocast`` bound to the resolved device type: casting for a
        reduced-precision *dtype*, explicitly disabled for any other, and
        ``contextlib.nullcontext()`` when the backend has no autocast at all.
    """
    device_type = resolve_device_type(device)
    if device_type in _AUTOCAST_UNUSABLE_DEVICE_TYPES or not is_autocast_available(
        device_type
    ):
        return nullcontext()
    try:
        if dtype not in _AUTOCAST_DTYPES:
            # Passing the dtype through would make torch warn ("the target
            # dtype is not supported. Disabling autocast") once per entry, on
            # every rank, and then do exactly this.
            return torch.autocast(device_type, enabled=False)
        return torch.autocast(device_type, dtype=dtype)
    except (AssertionError, RuntimeError) as error:
        # torch reports the backend as autocast-capable, but its device module
        # is incomplete (e.g. no 'get_amp_supported_dtype'). Run uncasted
        # rather than failing the forward pass, and warn once per backend.
        _AUTOCAST_UNUSABLE_DEVICE_TYPES.add(device_type)
        logger.warning(
            "Autocast is unusable on device type '%s' (%s); "
            "running starVLA without autocast on this backend.",
            device_type,
            error,
        )
        return nullcontext()


def build_gaussian(
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = torch.float32,
) -> Normal:
    """Build a Gaussian action distribution with both arguments in ``dtype``.

    ``mean`` inherits the backbone dtype (bfloat16 when autocast did not upcast
    it) and ``std`` inherits the policy parameter dtype. Sampling from a
    bfloat16 ``Normal`` calls ``torch.normal(mean, std)``, which has no
    bfloat16 kernel on Ascend ("tensor mean not implemented for DT_BFLOAT16"),
    so both are cast up front. Downstream log-probs, entropies and values are
    consumed as float32 anyway, so this costs nothing on CUDA.

    Args:
        mean: Distribution location, any floating dtype.
        std: Distribution scale, broadcastable against ``mean``.
        dtype: Dtype to build the distribution in; ``None`` keeps the inputs
            unchanged.

    Returns:
        The ``Normal`` distribution over the (broadcast) action shape.
    """
    if dtype is not None:
        mean = mean.to(dtype=dtype)
        std = std.to(dtype=dtype)
    return Normal(mean, std)
