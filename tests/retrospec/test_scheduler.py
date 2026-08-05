# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.v1.core.utils import create_scheduler
from vllm.config import DeviceConfig, SpeculativeConfig

pytestmark = pytest.mark.cpu_test


def test_retrospec_reserves_speculative_lookahead_tokens():
    speculative_config = SpeculativeConfig(
        method="retrospec",
        num_speculative_tokens=64,
    )

    scheduler = create_scheduler(
        speculative_config=speculative_config,
        device_config=DeviceConfig(device="cpu"),
    )

    assert scheduler.num_spec_tokens == 64
    assert scheduler.num_lookahead_tokens == 64


def test_retrospec_scheduler_change_does_not_affect_ngram():
    scheduler = create_scheduler(
        num_speculative_tokens=4,
        device_config=DeviceConfig(device="cpu"),
    )

    assert scheduler.num_spec_tokens == 4
    assert scheduler.num_lookahead_tokens == 0
