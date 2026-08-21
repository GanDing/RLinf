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

"""Tests for the Ascend patch-embedding workaround in the starVLA wrapper."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest
import torch
import torch.nn as nn

# Loaded by path: importing the package would pull in the starVLA third-party
# dependencies, which are only present in the starvla venv.
_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "rlinf"
    / "models"
    / "embodiment"
    / "starvla"
    / "utils"
    / "npu_patches.py"
)
_spec = importlib.util.spec_from_file_location("starvla_npu_patches", _MODULE_PATH)
npu_patches = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(npu_patches)

_PATCH_SIZE = 14
_TEMPORAL_PATCH_SIZE = 2
_IN_CHANNELS = 3
_EMBED_DIM = 32


class _ReferencePatchEmbed(nn.Module):
    """Mirror of the Qwen VL patch embedding (Conv3d over one patch extent)."""

    def __init__(self, bias: bool = False) -> None:
        super().__init__()
        self.patch_size = _PATCH_SIZE
        self.temporal_patch_size = _TEMPORAL_PATCH_SIZE
        self.in_channels = _IN_CHANNELS
        self.embed_dim = _EMBED_DIM
        kernel_size = [_TEMPORAL_PATCH_SIZE, _PATCH_SIZE, _PATCH_SIZE]
        self.proj = nn.Conv3d(
            _IN_CHANNELS,
            _EMBED_DIM,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


def _packed_patches(num_patches: int = 5) -> torch.Tensor:
    flat = _IN_CHANNELS * _TEMPORAL_PATCH_SIZE * _PATCH_SIZE * _PATCH_SIZE
    generator = torch.Generator().manual_seed(0)
    return torch.randn(num_patches, flat, generator=generator)


@pytest.mark.parametrize("bias", [False, True])
def test_linear_patch_embed_matches_conv_forward(bias):
    module = _ReferencePatchEmbed(bias=bias)
    hidden_states = _packed_patches()

    expected = module(hidden_states)
    actual = npu_patches.patch_embed_forward_linear(module, hidden_states)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_linear_patch_embed_matches_conv_gradients():
    module = _ReferencePatchEmbed()
    hidden_states = _packed_patches()

    conv_input = hidden_states.clone().requires_grad_(True)
    module(conv_input).square().sum().backward()
    conv_input_grad = conv_input.grad.clone()
    conv_weight_grad = module.proj.weight.grad.clone()

    module.zero_grad(set_to_none=True)
    linear_input = hidden_states.clone().requires_grad_(True)
    npu_patches.patch_embed_forward_linear(
        module, linear_input
    ).square().sum().backward()

    torch.testing.assert_close(linear_input.grad, conv_input_grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        module.proj.weight.grad, conv_weight_grad, rtol=1e-4, atol=1e-4
    )


def test_linear_patch_embed_casts_input_to_weight_dtype():
    module = _ReferencePatchEmbed().to(dtype=torch.bfloat16)
    out = npu_patches.patch_embed_forward_linear(module, _packed_patches())
    assert out.dtype == torch.bfloat16


def _install_fake_backbone(monkeypatch, module_name: str, class_name: str):
    module = ModuleType(module_name)
    setattr(module, class_name, type(class_name, (_ReferencePatchEmbed,), {}))
    monkeypatch.setitem(sys.modules, module_name, module)
    return getattr(module, class_name)


def test_apply_patches_is_a_noop_off_npu(monkeypatch):
    monkeypatch.delenv(npu_patches.PATCH_EMBED_ENV_VAR, raising=False)
    monkeypatch.setattr(npu_patches, "_is_npu", lambda: False)
    assert npu_patches.apply_starvla_npu_patches() == []


def test_apply_patches_rebinds_forward_once(monkeypatch):
    monkeypatch.delenv(npu_patches.PATCH_EMBED_ENV_VAR, raising=False)
    monkeypatch.setattr(npu_patches, "_is_npu", lambda: True)
    module_name, class_name = npu_patches._PATCH_EMBED_TARGETS[1]
    patch_embed_cls = _install_fake_backbone(monkeypatch, module_name, class_name)
    # Absent modules are skipped rather than failing the whole call.
    for other_module, _other_class in (
        npu_patches._PATCH_EMBED_TARGETS[0],
        npu_patches._PATCH_EMBED_TARGETS[2],
    ):
        monkeypatch.setitem(sys.modules, other_module, None)

    patched = npu_patches.apply_starvla_npu_patches()

    assert patched == [f"{module_name}.{class_name}"]
    assert patch_embed_cls.forward is npu_patches.patch_embed_forward_linear
    # An instance built before the call also picks the patched forward up.
    instance = patch_embed_cls()
    hidden_states = _packed_patches()
    torch.testing.assert_close(
        instance(hidden_states),
        npu_patches.patch_embed_forward_linear(instance, hidden_states),
    )
    # Idempotent: a second call reports nothing left to patch.
    assert npu_patches.apply_starvla_npu_patches() == []


def test_env_var_forces_the_patch_off_npu(monkeypatch):
    monkeypatch.setattr(npu_patches, "_is_npu", lambda: False)
    monkeypatch.setenv(npu_patches.PATCH_EMBED_ENV_VAR, "1")
    assert npu_patches._patch_requested(None) is True
    monkeypatch.setenv(npu_patches.PATCH_EMBED_ENV_VAR, "0")
    assert npu_patches._patch_requested(None) is False


def test_env_var_can_disable_the_patch_on_npu(monkeypatch):
    monkeypatch.setattr(npu_patches, "_is_npu", lambda: True)
    monkeypatch.setenv(npu_patches.PATCH_EMBED_ENV_VAR, "off")
    assert npu_patches._patch_requested(None) is False
    assert npu_patches._patch_requested(True) is True


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not installed",
)
def test_matches_the_real_qwen2_5_vl_patch_embedding():
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionPatchEmbed,
    )

    module = Qwen2_5_VisionPatchEmbed(
        patch_size=_PATCH_SIZE,
        temporal_patch_size=_TEMPORAL_PATCH_SIZE,
        in_channels=_IN_CHANNELS,
        embed_dim=_EMBED_DIM,
    )
    hidden_states = _packed_patches()

    torch.testing.assert_close(
        npu_patches.patch_embed_forward_linear(module, hidden_states),
        module(hidden_states),
        rtol=1e-5,
        atol=1e-5,
    )
