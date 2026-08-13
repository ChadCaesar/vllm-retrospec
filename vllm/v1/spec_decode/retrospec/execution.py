# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _pack_primary_exact_kernel(
    key_cache,
    value_cache,
    block_table,
    token_indices,
    token_mask,
    token_prefix,
    cu_seqlens_k,
    output_keys,
    output_values,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    key_stride_3,
    value_stride_0,
    value_stride_1,
    value_stride_2,
    value_stride_3,
    block_table_stride_0,
    block_table_stride_1,
    NUM_KV_HEADS: tl.constexpr,
    MAX_PRIMARY_TOKENS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    sequence_idx = tl.program_id(0)
    source_token_idx = tl.program_id(1)

    batch_idx = sequence_idx // NUM_KV_HEADS
    kv_head_idx = sequence_idx % NUM_KV_HEADS

    metadata_offset = sequence_idx * MAX_PRIMARY_TOKENS + source_token_idx
    valid_token = tl.load(token_mask + metadata_offset).to(tl.int1)

    logical_token_idx = tl.load(
        token_indices + metadata_offset,
        mask=valid_token,
        other=0,
    )
    logical_block_idx = logical_token_idx // PAGE_SIZE
    block_offset = logical_token_idx % PAGE_SIZE

    physical_block_idx = tl.load(
        block_table
        + batch_idx * block_table_stride_0
        + logical_block_idx * block_table_stride_1,
        mask=valid_token,
        other=0,
    ).to(tl.int64)

    # token_prefix is inclusive, so subtract one to obtain the zero-based
    # destination position of this valid source token.
    packed_token_idx = (
        tl.load(
            token_prefix + metadata_offset,
            mask=valid_token,
            other=1,
        )
        - 1
    )
    sequence_start = tl.load(cu_seqlens_k + sequence_idx)
    destination_token_idx = sequence_start + packed_token_idx

    head_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < HEAD_SIZE
    copy_mask = valid_token & head_mask

    key_source_offsets = (
        physical_block_idx * key_stride_0
        + block_offset * key_stride_1
        + kv_head_idx * key_stride_2
        + head_offsets * key_stride_3
    )
    value_source_offsets = (
        physical_block_idx * value_stride_0
        + block_offset * value_stride_1
        + kv_head_idx * value_stride_2
        + head_offsets * value_stride_3
    )
    destination_offsets = destination_token_idx * HEAD_SIZE + head_offsets

    key_vector = tl.load(
        key_cache + key_source_offsets,
        mask=copy_mask,
        other=0.0,
    )
    value_vector = tl.load(
        value_cache + value_source_offsets,
        mask=copy_mask,
        other=0.0,
    )

    tl.store(
        output_keys + destination_offsets,
        key_vector,
        mask=copy_mask,
    )
    tl.store(
        output_values + destination_offsets,
        value_vector,
        mask=copy_mask,
    )


@triton.jit
def _pack_cluster_pages_kernel(
    cluster_key_pages,
    cluster_value_pages,
    page_ids,
    page_token_counts,
    page_token_prefix,
    primary_token_counts,
    cu_seqlens_k,
    output_keys,
    output_values,
    key_page_stride_0,
    key_page_stride_1,
    key_page_stride_2,
    value_page_stride_0,
    value_page_stride_1,
    value_page_stride_2,
    MAX_PAGE_SLOTS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    sequence_idx = tl.program_id(0)
    page_slot_idx = tl.program_id(1)
    page_token_idx = tl.program_id(2)

    metadata_offset = sequence_idx * MAX_PAGE_SLOTS + page_slot_idx

    page_id = tl.load(page_ids + metadata_offset)
    page_token_count = tl.load(page_token_counts + metadata_offset)

    valid_token = (page_id >= 0) & (page_token_idx < page_token_count)
    safe_page_id = tl.maximum(page_id, 0).to(tl.int64)

    # page_token_prefix is inclusive. Subtract this page's count to obtain
    # the number of cluster tokens stored before this page.
    previous_page_tokens = (
        tl.load(page_token_prefix + metadata_offset) - page_token_count
    )
    primary_token_count = tl.load(primary_token_counts + sequence_idx)
    sequence_start = tl.load(cu_seqlens_k + sequence_idx)

    destination_token_idx = (
        sequence_start + primary_token_count + previous_page_tokens + page_token_idx
    )

    head_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < HEAD_SIZE
    copy_mask = valid_token & head_mask

    key_source_offsets = (
        safe_page_id * key_page_stride_0
        + page_token_idx * key_page_stride_1
        + head_offsets * key_page_stride_2
    )
    value_source_offsets = (
        safe_page_id * value_page_stride_0
        + page_token_idx * value_page_stride_1
        + head_offsets * value_page_stride_2
    )
    destination_offsets = destination_token_idx * HEAD_SIZE + head_offsets

    key_vector = tl.load(
        cluster_key_pages + key_source_offsets,
        mask=copy_mask,
        other=0.0,
    )
    value_vector = tl.load(
        cluster_value_pages + value_source_offsets,
        mask=copy_mask,
        other=0.0,
    )

    tl.store(
        output_keys + destination_offsets,
        key_vector,
        mask=copy_mask,
    )
    tl.store(
        output_values + destination_offsets,
        value_vector,
        mask=copy_mask,
    )


@dataclass(frozen=True)
class RetroSpecExactExecution:
    keys: torch.Tensor
    values: torch.Tensor

    exact_seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor

    batch_size: int
    num_kv_heads: int
    head_size: int
    max_exact_seq_len: int


class RetroSpecExactExecutionBuffer:
    """Reusable exact-KV execution workspace shared by model layers."""

    _MIN_TOKEN_CAPACITY = 1024
    _MIN_METADATA_CAPACITY = 64

    def __init__(self, page_size: int) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        self.page_size = page_size

        self._dtype: torch.dtype | None = None
        self._device: torch.device | None = None
        self._head_size: int | None = None

        self._token_capacity = 0
        self._key_buffer: torch.Tensor | None = None
        self._value_buffer: torch.Tensor | None = None

        self._sequence_capacity = 0
        self._cu_seqlens_q: torch.Tensor | None = None
        self._cu_seqlens_k: torch.Tensor | None = None

        self._prefix_capacity = 0
        self._prefix_buffer: torch.Tensor | None = None

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 << (max(value, 1) - 1).bit_length()

    def _check_or_set_configuration(
        self,
        dtype: torch.dtype,
        device: torch.device,
        head_size: int,
    ) -> None:
        if self._dtype is None:
            self._dtype = dtype
            self._device = device
            self._head_size = head_size
            return

        if self._dtype != dtype:
            raise ValueError("Execution-buffer dtype changed")
        if self._device != device:
            raise ValueError("Execution-buffer device changed")
        if self._head_size != head_size:
            raise ValueError("Execution-buffer head size changed")

    def _ensure_token_capacity(self, required_tokens: int) -> None:
        if required_tokens <= self._token_capacity:
            return

        assert self._dtype is not None
        assert self._device is not None
        assert self._head_size is not None

        self._token_capacity = max(
            self._MIN_TOKEN_CAPACITY,
            self._next_power_of_two(required_tokens),
        )
        shape = (
            self._token_capacity,
            1,
            self._head_size,
        )
        self._key_buffer = torch.empty(
            shape,
            dtype=self._dtype,
            device=self._device,
        )
        self._value_buffer = torch.empty_like(self._key_buffer)

    def _ensure_sequence_capacity(
        self,
        num_sequences: int,
    ) -> None:
        required = num_sequences + 1
        if required <= self._sequence_capacity:
            return

        assert self._device is not None

        self._sequence_capacity = max(
            self._MIN_METADATA_CAPACITY,
            self._next_power_of_two(required),
        )
        self._cu_seqlens_q = torch.arange(
            self._sequence_capacity,
            dtype=torch.int32,
            device=self._device,
        )
        self._cu_seqlens_k = torch.empty(
            self._sequence_capacity,
            dtype=torch.int32,
            device=self._device,
        )

    def _ensure_prefix_capacity(
        self,
        required_items: int,
    ) -> None:
        if required_items <= self._prefix_capacity:
            return

        assert self._device is not None

        self._prefix_capacity = max(
            self._MIN_METADATA_CAPACITY,
            self._next_power_of_two(required_items),
        )
        self._prefix_buffer = torch.empty(
            self._prefix_capacity,
            dtype=torch.int32,
            device=self._device,
        )

    def pack(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        primary_token_indices: torch.Tensor,
        primary_token_mask: torch.Tensor,
        page_ids: torch.Tensor,
        page_token_counts: torch.Tensor,
        cluster_key_pages: torch.Tensor | None,
        cluster_value_pages: torch.Tensor | None,
    ) -> RetroSpecExactExecution:
        """Pack primary and clustered exact KV into one reusable buffer."""
        if key_cache.shape != value_cache.shape:
            raise ValueError("Primary key and value cache shapes must match")
        if key_cache.ndim != 4:
            raise ValueError(
                "Primary KV cache must have shape "
                "[blocks, block_size, kv_heads, head_size]"
            )
        if key_cache.shape[1] != self.page_size:
            raise ValueError("Primary KV block size does not match page size")
        if primary_token_indices.shape != primary_token_mask.shape:
            raise ValueError("Primary token indices and mask shapes must match")
        if primary_token_indices.ndim != 3:
            raise ValueError(
                "Primary token metadata must have shape [batch, kv_heads, tokens]"
            )
        if page_ids.shape != page_token_counts.shape:
            raise ValueError("Page ID and token-count shapes must match")
        if page_ids.ndim != 4:
            raise ValueError(
                "Page metadata must have shape [batch, kv_heads, clusters, pages]"
            )
        if primary_token_mask.dtype != torch.bool:
            raise ValueError("Primary token mask must be boolean")
        if primary_token_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Primary token indices must be integral")
        if page_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Page IDs must be integral")
        if page_token_counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("Page token counts must be integral")
        if block_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("Block table entries must be integral")
        if key_cache.device.type != "cuda":
            raise ValueError("Triton exact execution requires CUDA tensors")

        batch_size, num_kv_heads, max_primary_tokens = primary_token_indices.shape
        if page_ids.shape[:2] != (batch_size, num_kv_heads):
            raise ValueError("Primary and clustered metadata batch shapes must match")
        if block_table.shape[0] != batch_size:
            raise ValueError("Block table batch size does not match exact metadata")
        if key_cache.shape[2] != num_kv_heads:
            raise ValueError("Exact metadata KV-head count does not match KV cache")

        head_size = key_cache.shape[3]
        device = key_cache.device
        dtype = key_cache.dtype

        for tensor in (
            value_cache,
            block_table,
            primary_token_indices,
            primary_token_mask,
            page_ids,
            page_token_counts,
        ):
            if tensor.device != device:
                raise ValueError("All exact execution tensors must be on one device")

        max_page_slots = page_ids.shape[2] * page_ids.shape[3]
        if max_page_slots:
            if cluster_key_pages is None or cluster_value_pages is None:
                raise RuntimeError("Cluster page storage is required by the selection")
            if cluster_key_pages.shape != cluster_value_pages.shape:
                raise ValueError("Cluster key and value page shapes must match")
            if cluster_key_pages.ndim != 3:
                raise ValueError(
                    "Cluster pages must have shape [pages, page_size, head_size]"
                )
            if cluster_key_pages.shape[1:] != (
                self.page_size,
                head_size,
            ):
                raise ValueError("Cluster page shape does not match execution layout")
            if cluster_key_pages.dtype != dtype or cluster_value_pages.dtype != dtype:
                raise ValueError("Cluster page dtype does not match primary KV")
            if (
                cluster_key_pages.device != device
                or cluster_value_pages.device != device
            ):
                raise ValueError("Cluster pages must be on the execution device")

        self._check_or_set_configuration(dtype, device, head_size)

        # The Triton kernels address selection metadata as flat arrays. The
        # index normally produces contiguous tensors, but making that contract
        # explicit here also keeps direct callers correct.
        primary_token_indices = primary_token_indices.contiguous()
        primary_token_mask = primary_token_mask.contiguous()
        page_ids = page_ids.contiguous()
        page_token_counts = page_token_counts.contiguous()

        num_sequences = batch_size * num_kv_heads
        max_exact_seq_len = max_primary_tokens + max_page_slots * self.page_size
        required_tokens = num_sequences * max_exact_seq_len

        self._ensure_token_capacity(max(required_tokens, 1))
        self._ensure_sequence_capacity(num_sequences)
        self._ensure_prefix_capacity(
            max(
                primary_token_mask.numel(),
                page_token_counts.numel(),
                1,
            )
        )

        assert self._key_buffer is not None
        assert self._value_buffer is not None
        assert self._cu_seqlens_q is not None
        assert self._cu_seqlens_k is not None
        assert self._prefix_buffer is not None

        primary_token_counts = primary_token_mask.sum(
            dim=2,
            dtype=torch.int32,
        ).contiguous()

        flat_page_token_counts = page_token_counts.reshape(
            batch_size,
            num_kv_heads,
            max_page_slots,
        )
        clustered_token_counts = flat_page_token_counts.sum(
            dim=2,
            dtype=torch.int32,
        )
        exact_seq_lens = (
            (primary_token_counts + clustered_token_counts).reshape(-1).contiguous()
        )

        cu_seqlens_q = self._cu_seqlens_q[: num_sequences + 1]
        cu_seqlens_k = self._cu_seqlens_k[: num_sequences + 1]
        cu_seqlens_k[0].zero_()
        torch.cumsum(
            exact_seq_lens,
            dim=0,
            out=cu_seqlens_k[1:],
        )

        execution_keys = self._key_buffer[:required_tokens]
        execution_values = self._value_buffer[:required_tokens]

        block_d = triton.next_power_of_2(head_size)

        if num_sequences and max_primary_tokens:
            primary_prefix = self._prefix_buffer[: primary_token_mask.numel()].view_as(
                primary_token_mask
            )
            torch.cumsum(
                primary_token_mask,
                dim=2,
                dtype=torch.int32,
                out=primary_prefix,
            )

            _pack_primary_exact_kernel[(num_sequences, max_primary_tokens)](
                key_cache,
                value_cache,
                block_table,
                primary_token_indices,
                primary_token_mask,
                primary_prefix,
                cu_seqlens_k,
                execution_keys,
                execution_values,
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                key_cache.stride(3),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                value_cache.stride(3),
                block_table.stride(0),
                block_table.stride(1),
                NUM_KV_HEADS=num_kv_heads,
                MAX_PRIMARY_TOKENS=max_primary_tokens,
                PAGE_SIZE=self.page_size,
                HEAD_SIZE=head_size,
                BLOCK_D=block_d,
            )

        if num_sequences and max_page_slots:
            assert cluster_key_pages is not None
            assert cluster_value_pages is not None

            flat_page_ids = page_ids.reshape(
                batch_size,
                num_kv_heads,
                max_page_slots,
            )
            page_prefix = self._prefix_buffer[: page_token_counts.numel()].view_as(
                flat_page_token_counts
            )
            torch.cumsum(
                flat_page_token_counts,
                dim=2,
                dtype=torch.int32,
                out=page_prefix,
            )

            _pack_cluster_pages_kernel[
                (
                    num_sequences,
                    max_page_slots,
                    self.page_size,
                )
            ](
                cluster_key_pages,
                cluster_value_pages,
                flat_page_ids,
                flat_page_token_counts,
                page_prefix,
                primary_token_counts,
                cu_seqlens_k,
                execution_keys,
                execution_values,
                cluster_key_pages.stride(0),
                cluster_key_pages.stride(1),
                cluster_key_pages.stride(2),
                cluster_value_pages.stride(0),
                cluster_value_pages.stride(1),
                cluster_value_pages.stride(2),
                MAX_PAGE_SLOTS=max_page_slots,
                PAGE_SIZE=self.page_size,
                HEAD_SIZE=head_size,
                BLOCK_D=block_d,
            )

        return RetroSpecExactExecution(
            keys=execution_keys,
            values=execution_values,
            exact_seq_lens=exact_seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            max_exact_seq_len=max_exact_seq_len,
        )
