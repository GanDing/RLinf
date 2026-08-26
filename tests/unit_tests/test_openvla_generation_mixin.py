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

"""Regression tests for OpenVLA generation compatibility.

``transformers>=4.50`` dropped ``GenerationMixin`` from ``PreTrainedModel`` and
pre-allocates an empty cache before the prefill step, which used to break
``OpenVLAForRLActionPrediction.predict_action_batch`` with
``AttributeError: ... object has no attribute 'generate'``.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.generation import GenerationMixin  # noqa: E402

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "rlinf/models/embodiment/openvla/openvla_action_model.py"
)


class _FakePrismaticModel(torch.nn.Module):
    """Stand-in for upstream ``OpenVLAForActionPrediction``.

    Like the real class it does **not** provide ``generate()`` on its own, so a
    loaded module only exposes ``generate()`` if it mixes ``GenerationMixin`` in
    explicitly.
    """


def _install_prismatic_stubs(monkeypatch):
    for name in ("prismatic", "prismatic.extern", "prismatic.extern.hf"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    modeling = ModuleType("prismatic.extern.hf.modeling_prismatic")
    modeling.IGNORE_INDEX = -100
    modeling.OpenVLAForActionPrediction = _FakePrismaticModel
    modeling.PrismaticCausalLMOutputWithPast = type(
        "PrismaticCausalLMOutputWithPast", (), {}
    )
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)

    processing = ModuleType("prismatic.extern.hf.processing_prismatic")
    processing.PrismaticImageProcessor = type("PrismaticImageProcessor", (), {})
    processing.PrismaticProcessor = type("PrismaticProcessor", (), {})
    monkeypatch.setitem(sys.modules, processing.__name__, processing)


def _load_openvla_module(monkeypatch):
    _install_prismatic_stubs(monkeypatch)

    spec = importlib.util.spec_from_file_location(
        "openvla_action_model_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    try:
        spec.loader.exec_module(module)
    except ImportError as err:  # pragma: no cover - depends on installed deps
        pytest.skip(f"openvla_action_model is not importable here: {err}")
    return module


@pytest.fixture
def openvla_module(monkeypatch):
    return _load_openvla_module(monkeypatch)


def test_generate_is_available_on_action_models(openvla_module):
    for cls in (
        openvla_module.OpenVLAForBatchActionPrediction,
        openvla_module.OpenVLAForRLActionPrediction,
    ):
        assert issubclass(cls, GenerationMixin)
        assert hasattr(cls, "generate")


class _FakeCache:
    def __init__(self, length):
        self._length = length

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._length


def test_get_cache_length(openvla_module):
    get_cache_length = openvla_module.get_cache_length

    assert get_cache_length(None) == 0
    assert get_cache_length(()) == 0
    assert get_cache_length(_FakeCache(0)) == 0
    assert get_cache_length(_FakeCache(7)) == 7

    legacy_cache = ((torch.zeros(2, 4, 5, 8), torch.zeros(2, 4, 5, 8)),)
    assert get_cache_length(legacy_cache) == 5


def test_prepare_inputs_keeps_full_prompt_on_prefill(openvla_module):
    # `prepare_inputs_for_generation` does not touch `self`, so it can be
    # exercised without building a full VLA model.
    prepare = (
        openvla_module.OpenVLAForBatchActionPrediction.prepare_inputs_for_generation
    )
    input_ids = torch.arange(12).reshape(2, 6)
    attention_mask = torch.ones_like(input_ids)

    prefill = prepare(
        None,
        input_ids=input_ids,
        past_key_values=_FakeCache(0),
        attention_mask=attention_mask,
    )
    assert prefill["input_ids"].shape == input_ids.shape
    assert prefill["past_key_values"] is None

    decode = prepare(
        None,
        input_ids=input_ids,
        past_key_values=_FakeCache(6),
        attention_mask=attention_mask,
    )
    assert decode["input_ids"].shape == (2, 1)
    assert torch.equal(decode["input_ids"], input_ids[:, -1:])
    assert decode["past_key_values"] is not None
