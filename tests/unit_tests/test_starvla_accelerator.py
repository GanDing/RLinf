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

"""Tests for the starVLA accelerator portability helpers."""

from __future__ import annotations

import importlib.util
import pathlib
import warnings
from contextlib import nullcontext

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
    / "accelerator.py"
)
_spec = importlib.util.spec_from_file_location("starvla_accelerator", _MODULE_PATH)
accelerator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(accelerator)


def test_resolve_device_type_from_tensor_module_and_device():
    assert accelerator.resolve_device_type(torch.zeros(2)) == "cpu"
    assert accelerator.resolve_device_type(nn.Linear(2, 2)) == "cpu"
    assert accelerator.resolve_device_type(torch.device("cuda:3")) == "cuda"
    assert accelerator.resolve_device_type("npu:0") == "npu"


def test_resolve_device_type_falls_back_to_cpu():
    assert accelerator.resolve_device_type(None) == "cpu"
    # A module with no parameters and no buffers carries no device.
    assert accelerator.resolve_device_type(nn.Identity()) == "cpu"


def test_autocast_ctx_is_a_noop_on_cpu():
    # CPU autocast rejects float32, so the fp32 paths must run uncasted there.
    assert isinstance(
        accelerator.autocast_ctx(torch.float32, device=torch.zeros(2)), nullcontext
    )


def test_autocast_ctx_uses_the_device_backend(monkeypatch):
    recorded = {}

    def fake_autocast(device_type, dtype=None, enabled=True):
        recorded.update(device_type=device_type, dtype=dtype, enabled=enabled)
        return nullcontext()

    monkeypatch.setattr(accelerator.torch, "autocast", fake_autocast)
    monkeypatch.setattr(accelerator, "is_autocast_available", lambda _: True)

    accelerator.autocast_ctx(torch.bfloat16, device="npu:0")

    # Not "cuda": on Ascend a cuda autocast context silently does nothing and
    # leaves the backbone's bfloat16 in place.
    assert recorded == {
        "device_type": "npu",
        "dtype": torch.bfloat16,
        "enabled": True,
    }


def test_autocast_ctx_disables_autocast_for_full_precision(monkeypatch):
    """float32 is not an autocast dtype; torch warns per entry if asked for it."""
    recorded = {}

    def fake_autocast(device_type, dtype=None, enabled=True):
        recorded.update(device_type=device_type, dtype=dtype, enabled=enabled)
        return nullcontext()

    monkeypatch.setattr(accelerator.torch, "autocast", fake_autocast)
    monkeypatch.setattr(accelerator, "is_autocast_available", lambda _: True)

    accelerator.autocast_ctx(torch.float32, device="npu:0")

    assert recorded == {"device_type": "npu", "dtype": None, "enabled": False}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_autocast_ctx_never_warns_on_a_registered_backend(monkeypatch, dtype):
    monkeypatch.setattr(accelerator, "is_autocast_available", lambda _: True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        accelerator.autocast_ctx(dtype, device=torch.zeros(1))

    assert [str(w.message) for w in caught] == []


def test_autocast_ctx_skips_backends_without_autocast(monkeypatch):
    monkeypatch.setattr(accelerator, "is_autocast_available", lambda _: False)
    assert isinstance(
        accelerator.autocast_ctx(torch.bfloat16, device="npu:0"), nullcontext
    )


def test_autocast_ctx_falls_back_when_the_backend_module_is_incomplete(monkeypatch):
    """A backend can advertise autocast yet reject ``torch.autocast(...)``."""
    monkeypatch.setattr(accelerator, "is_autocast_available", lambda _: True)
    monkeypatch.setattr(
        accelerator, "_AUTOCAST_UNUSABLE_DEVICE_TYPES", set(), raising=True
    )

    calls = []

    def failing_autocast(device_type, dtype=None, enabled=True):
        calls.append(device_type)
        raise AssertionError("the backend has not registered a module")

    monkeypatch.setattr(accelerator.torch, "autocast", failing_autocast)

    assert isinstance(
        accelerator.autocast_ctx(torch.float32, device="npu:0"), nullcontext
    )
    # The backend is remembered, so the failure is not retried every forward.
    assert isinstance(
        accelerator.autocast_ctx(torch.float32, device="npu:0"), nullcontext
    )
    assert calls == ["npu"]


def test_is_autocast_available_rejects_unknown_device_types():
    assert not accelerator.is_autocast_available("cpu")
    assert not accelerator.is_autocast_available("not_a_device")


def test_build_gaussian_upcasts_bfloat16_inputs():
    mean = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    std = torch.full((1, 1, 4), 0.5, dtype=torch.bfloat16)

    dist = accelerator.build_gaussian(mean, std)

    assert dist.loc.dtype == torch.float32
    assert dist.scale.dtype == torch.float32
    assert dist.sample().dtype == torch.float32
    assert dist.log_prob(torch.zeros(2, 3, 4)).dtype == torch.float32


def test_build_gaussian_samples_where_bfloat16_normal_has_no_kernel(monkeypatch):
    """Regression: Ascend has no bfloat16 ``aten::normal`` (tensor mean)."""
    torch_normal = torch.normal

    def normal_without_bfloat16_support(mean, std, *args, **kwargs):
        if isinstance(mean, torch.Tensor) and mean.dtype == torch.bfloat16:
            raise RuntimeError("tensor mean not implemented for DT_BFLOAT16")
        return torch_normal(mean, std, *args, **kwargs)

    monkeypatch.setattr(torch, "normal", normal_without_bfloat16_support)

    mean = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    std = torch.full((1, 1, 4), 0.5, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="DT_BFLOAT16"):
        torch.distributions.normal.Normal(mean, std.expand_as(mean)).sample()

    assert accelerator.build_gaussian(mean, std).sample().shape == (2, 3, 4)


def test_build_gaussian_keeps_dtype_when_disabled():
    mean = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    std = torch.full((1, 1, 4), 0.5, dtype=torch.bfloat16)

    dist = accelerator.build_gaussian(mean, std, dtype=None)

    assert dist.loc.dtype == torch.bfloat16
