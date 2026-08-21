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

"""Ascend NPU patches for the Qwen VL backbones starVLA builds on.

The Qwen2-VL / Qwen2.5-VL / Qwen3-VL vision towers embed patches with a
``Conv3d`` whose kernel and stride both equal the patch extent, applied to an
input of exactly that extent::

    proj = nn.Conv3d(3, embed_dim, kernel_size=[2, 14, 14], stride=[2, 14, 14])
    proj(hidden_states.view(-1, 3, 2, 14, 14))  # -> [N, embed_dim, 1, 1, 1]

Every output cell is therefore a plain dot product over one patch, and the op
is equivalent to a matmul. On Ascend the backward of that convolution crashes:
``aclnnConvolutionBackwardGetWorkspaceSize`` segfaults on the whole rank set
during ``ConvolutionBackward``, with the patch count as the batch dimension.
Rewriting the embedding as ``F.linear`` keeps the numerics identical (same
weights, same flattening order) while routing both directions through matmul
kernels the backend handles.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Patch-embedding classes across the Qwen VL generations starVLA can load.
# Each holds the convolution as ``self.proj`` and takes flattened patches.
_PATCH_EMBED_TARGETS = (
    ("transformers.models.qwen2_vl.modeling_qwen2_vl", "PatchEmbed"),
    ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VisionPatchEmbed"),
    ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLVisionPatchEmbed"),
)

# Set to 1/0 to force the patch on or off regardless of the detected device.
PATCH_EMBED_ENV_VAR = "RLINF_STARVLA_PATCH_EMBED_LINEAR"

_PATCHED_MARKER = "_rlinf_patch_embed_linear"


def patch_embed_forward_linear(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Compute the Qwen VL patch embedding as a matmul.

    Args:
        hidden_states: Packed patches, ``[num_patches, in_channels * temporal *
            patch_h * patch_w]`` (the convolution's flattened receptive field).

    Returns:
        Patch embeddings, ``[num_patches, embed_dim]`` — the same values the
        ``Conv3d`` produces.
    """
    weight = self.proj.weight
    flat_weight = weight.reshape(weight.shape[0], -1)
    hidden_states = hidden_states.reshape(-1, flat_weight.shape[1]).to(
        dtype=weight.dtype
    )
    return F.linear(hidden_states, flat_weight, self.proj.bias)


def _is_npu() -> bool:
    """Whether this worker runs on an Ascend NPU, per the Worker device API."""
    try:
        from rlinf.scheduler import AcceleratorType, Worker
    except Exception:  # scheduler unavailable (e.g. plain unit-test process)
        return False
    return getattr(Worker, "accelerator_type", None) == AcceleratorType.NPU


def _patch_requested(force: Optional[bool]) -> bool:
    """Resolve the patch decision from the caller, the env var, then the device."""
    if force is not None:
        return force
    override = os.environ.get(PATCH_EMBED_ENV_VAR)
    if override is not None and override.strip():
        return override.strip().lower() not in ("0", "false", "no", "off")
    return _is_npu()


def apply_starvla_npu_patches(force: Optional[bool] = None) -> list[str]:
    """Replace the Qwen VL patch-embedding convolution with its matmul form.

    Patches classes, so it applies to models that are already built. No-op off
    Ascend unless *force* or :data:`PATCH_EMBED_ENV_VAR` says otherwise, and
    idempotent across repeated calls.

    Args:
        force: ``True``/``False`` to override device detection and the env var.

    Returns:
        The names of the classes patched by this call (empty when disabled, or
        when every target was already patched or is not installed).
    """
    if not _patch_requested(force):
        return []

    patched: list[str] = []
    for module_name, class_name in _PATCH_EMBED_TARGETS:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # transformers version without this backbone
            continue
        patch_embed_cls = getattr(module, class_name, None)
        if patch_embed_cls is None or getattr(patch_embed_cls, _PATCHED_MARKER, False):
            continue
        patch_embed_cls.forward = patch_embed_forward_linear
        setattr(patch_embed_cls, _PATCHED_MARKER, True)
        patched.append(f"{module_name}.{class_name}")

    if patched:
        logger.info(
            "Ascend NPU: computing the Qwen VL patch embedding as a matmul "
            "instead of Conv3d (%s), whose backward crashes in CANN.",
            ", ".join(patched),
        )
    return patched
