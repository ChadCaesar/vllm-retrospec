# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import ArgumentError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import _xxhash


@pytest.mark.parametrize(
    ("speculative_config", "expected_max_num_seqs"),
    [
        ({"method": "retrospec", "num_speculative_tokens": 64}, 8),
        ({"method": "ngram", "num_speculative_tokens": 4}, 256),
        (None, 256),
    ],
)
def test_retrospec_uses_resource_constrained_default_max_num_seqs(
    speculative_config, expected_max_num_seqs
):
    engine_args = EngineArgs(
        model="unused",
        speculative_config=speculative_config,
        max_num_batched_tokens=512,
    )
    engine_args.get_batch_defaults = Mock(return_value=({None: 512}, {None: 256}))

    engine_args._set_default_max_num_seqs_and_batched_tokens_args(
        None,
        SimpleNamespace(max_model_len=64),
    )

    assert engine_args.max_num_seqs == expected_max_num_seqs


def test_retrospec_preserves_explicit_max_num_seqs():
    engine_args = EngineArgs(
        model="unused",
        speculative_config={"method": "retrospec", "num_speculative_tokens": 64},
        max_num_batched_tokens=128,
        max_num_seqs=3,
    )
    engine_args.get_batch_defaults = Mock(return_value=({None: 128}, {None: 256}))

    engine_args._set_default_max_num_seqs_and_batched_tokens_args(
        None,
        SimpleNamespace(max_model_len=64),
    )

    assert engine_args.max_num_seqs == 3


def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching, (
        "V1 turns on prefix caching by default."
    )

    # Turn it off possible with flag.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert not vllm_config.cache_config.enable_prefix_caching

    # Turn it on with flag.
    args = parser.parse_args(["--enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching

    # default hash algorithm is "builtin"
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # set hash algorithm to sha256_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256_cbor"

    # set hash algorithm to sha256
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # an invalid hash algorithm raises an error
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        args = parser.parse_args(["--prefix-caching-hash-algo", "invalid"])


@pytest.mark.skipif(_xxhash is None, reason="xxhash not installed")
def test_prefix_caching_xxhash_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # set hash algorithm to xxhash (pickle)
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash"

    # set hash algorithm to xxhash_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash_cbor"


def test_defaults_with_usage_context():
    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config: VllmConfig = engine_args.create_engine_config(UsageContext.LLM_CLASS)

    from vllm.platforms import current_platform
    from vllm.utils.mem_constants import GiB_bytes

    device_memory = current_platform.get_device_total_memory()
    device_name = current_platform.get_device_name().lower()
    if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
        # For GPUs like H100, H200, and MI300x with >= 70GB memory
        default_llm_tokens = 16384
        default_server_tokens = 8192
        default_max_num_seqs = 1024
    else:
        default_llm_tokens = 8192
        default_server_tokens = 2048
        default_max_num_seqs = 256

    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_llm_tokens  # noqa: E501

    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_server_tokens  # noqa: E501
