# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError
from contextlib import suppress

import torch

from vllm.v1.spec_decode.retrospec.legacy.cluster_store import (
    RetroSpecFullVerificationDescriptor,
    RetroSpecFullVerificationStaging,
)
from vllm.v1.spec_decode.retrospec.legacy.segmented.types import (
    RetroSpecFullVerificationPlan,
    _PrefetchedFullVerificationLayer,
    _PrimedFullVerificationPipeline,
)


class _RetroSpecSegmentedTokenIndexVerificationMixin:
    def _get_full_verification_descriptors(
        self,
        layer_name: str,
        request_ids: Sequence[str],
        num_kv_heads: int,
    ) -> tuple[RetroSpecFullVerificationDescriptor, ...]:
        layer_indices = self._indices.get(layer_name, {})
        descriptors: list[RetroSpecFullVerificationDescriptor] = []
        for request_id in request_ids:
            record = layer_indices.get(request_id)
            descriptor = None if record is None else record.full_verification_descriptor
            if descriptor is None:
                descriptor = RetroSpecFullVerificationDescriptor.empty(num_kv_heads)
            elif descriptor.num_kv_heads != num_kv_heads:
                raise RuntimeError("Full-verification descriptor changed KV-head count")
            descriptors.append(descriptor)
        return tuple(descriptors)

    @staticmethod
    def _canonical_full_verification_device(device: torch.device) -> torch.device:
        if device.type != "cuda":
            raise ValueError("Full-verification pipeline requires a CUDA device")
        if device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    @staticmethod
    def _normalize_full_verification_layers(
        layer_num_kv_heads: Mapping[str, int],
    ) -> tuple[tuple[str, int], ...]:
        if not layer_num_kv_heads:
            raise ValueError("Full-verification pipeline requires model layers")
        layers = tuple(
            (layer_name, int(num_kv_heads))
            for layer_name, num_kv_heads in layer_num_kv_heads.items()
        )
        if any(num_kv_heads <= 0 for _, num_kv_heads in layers):
            raise ValueError("Full-verification KV-head counts must be positive")
        return layers

    def _get_full_verification_revisions(
        self,
        request_ids: Sequence[str],
        layers: Sequence[tuple[str, int]],
    ) -> tuple[tuple[int, ...], ...]:
        revisions: list[tuple[int, ...]] = []
        for layer_name, _ in layers:
            layer_indices = self._indices.get(layer_name, {})
            revisions.append(
                tuple(
                    -1
                    if (record := layer_indices.get(request_id)) is None
                    else record.revision
                    for request_id in request_ids
                )
            )
        return tuple(revisions)

    def _record_full_verification_prime_outcome(self, adopted: bool) -> None:
        sample = 1.0 if adopted else 0.0
        previous = self._full_verify_prime_adoption_ema
        if previous is None:
            self._full_verify_prime_adoption_ema = sample
        else:
            alpha = self._FULL_VERIFY_PRIME_EMA_ALPHA
            self._full_verify_prime_adoption_ema = (
                previous * (1.0 - alpha) + sample * alpha
            )
        self._full_verify_prime_outcomes += 1
        if adopted:
            self._full_verify_prime_skipped_opportunities = 0

        if self.performance_stats is not None:
            counter = (
                "full_verify_prime_adopted"
                if adopted
                else "full_verify_prime_discarded"
            )
            self.performance_stats.add_counter(counter)

    def _should_submit_full_verification_prime(self) -> bool:
        if (
            self._full_verify_prime_outcomes
            < self._FULL_VERIFY_PRIME_BOOTSTRAP_OUTCOMES
        ):
            return True

        adoption_ema = self._full_verify_prime_adoption_ema
        if (
            adoption_ema is None
            or adoption_ema >= self._FULL_VERIFY_PRIME_MIN_ADOPTION_RATE
        ):
            self._full_verify_prime_skipped_opportunities = 0
            return True

        self._full_verify_prime_skipped_opportunities += 1
        if (
            self._full_verify_prime_skipped_opportunities
            >= self._FULL_VERIFY_PRIME_REPROBE_INTERVAL
        ):
            self._full_verify_prime_skipped_opportunities = 0
            return True

        if self.performance_stats is not None:
            self.performance_stats.add_counter("full_verify_prime_policy_skipped")
        return False

    @staticmethod
    def _cancel_full_verification_prefetches(
        prefetched: Sequence[_PrefetchedFullVerificationLayer],
        wait: bool,
    ) -> None:
        tickets = tuple(
            layer.ticket for layer in prefetched if layer.ticket is not None
        )
        for ticket in tickets:
            ticket.cancel()
        if not wait:
            return
        for ticket in tickets:
            with suppress(CancelledError):
                ticket.result()

    def _discard_primed_full_verification(
        self,
        wait: bool,
        record_outcome: bool = True,
    ) -> None:
        primed = self._primed_full_verification
        if primed is None:
            return

        self._primed_full_verification = None
        self._cancel_full_verification_prefetches(primed.prefetched, wait)
        if record_outcome:
            self._record_full_verification_prime_outcome(adopted=False)

    def _discard_primed_full_verification_for_requests(
        self,
        request_ids: Sequence[str],
        wait: bool,
    ) -> None:
        primed = self._primed_full_verification
        if primed is None or set(request_ids).isdisjoint(primed.request_ids):
            return
        self._discard_primed_full_verification(wait=wait)

    def _submit_full_verification_layer(
        self,
        request_ids: Sequence[str],
        layer: tuple[str, int],
        primed: bool,
    ) -> _PrefetchedFullVerificationLayer:
        layer_name, num_kv_heads = layer
        descriptors = self._get_full_verification_descriptors(
            layer_name,
            request_ids,
            num_kv_heads,
        )
        ticket = None
        if any(descriptor.num_tokens for descriptor in descriptors):
            ticket = self.cluster_store.submit_full_verification_tokens(
                layer_name=layer_name,
                descriptors=descriptors,
            )
        return _PrefetchedFullVerificationLayer(
            layer_name=layer_name,
            ticket=ticket,
            primed=primed,
        )

    def _prefetch_full_verification_layer(
        self,
        layer_index: int,
    ) -> _PrefetchedFullVerificationLayer:
        if self._full_verification_device is None:
            raise RuntimeError("Full-verification pipeline has no CUDA device")
        return self._submit_full_verification_layer(
            request_ids=self._full_verification_request_ids,
            layer=self._full_verification_layers[layer_index],
            primed=False,
        )

    def prime_full_verification_pipeline(
        self,
        request_ids: Sequence[str],
        layer_num_kv_heads: Mapping[str, int],
        device: torch.device,
    ) -> bool:
        if not self._proposal_active:
            raise RuntimeError(
                "Full-verification priming must run inside an active proposal"
            )
        if self._full_verification_pipeline_active:
            raise RuntimeError(
                "Cannot prime while a full-verification pipeline is active"
            )

        request_ids = tuple(request_ids)
        if request_ids != self._proposal_request_ids:
            raise ValueError(
                "Full-verification prime requests do not match the proposal batch"
            )

        layers = self._normalize_full_verification_layers(layer_num_kv_heads)
        device = self._canonical_full_verification_device(device)
        revisions = self._get_full_verification_revisions(request_ids, layers)
        current = self._primed_full_verification
        if (
            current is not None
            and current.request_ids == request_ids
            and current.layers == layers
            and current.revisions == revisions
            and current.device == device
        ):
            if self.performance_stats is not None:
                self.performance_stats.add_counter("full_verify_prime_coalesced")
            return True

        if current is not None:
            self._discard_primed_full_verification(wait=False)
        if not self._should_submit_full_verification_prime():
            return False

        prime_depth = min(self._FULL_VERIFY_PRIME_DEPTH, len(layers))
        prefetched: list[_PrefetchedFullVerificationLayer] = []
        try:
            for layer in layers[:prime_depth]:
                prefetched.append(
                    self._submit_full_verification_layer(
                        request_ids=request_ids,
                        layer=layer,
                        primed=True,
                    )
                )
        except BaseException:
            self._cancel_full_verification_prefetches(prefetched, wait=False)
            raise

        if not any(layer.ticket is not None for layer in prefetched):
            if self.performance_stats is not None:
                self.performance_stats.add_counter("full_verify_prime_empty")
            return False

        self._primed_full_verification = _PrimedFullVerificationPipeline(
            request_ids=request_ids,
            layers=layers,
            revisions=revisions,
            device=device,
            prefetched=tuple(prefetched),
        )
        if self.performance_stats is not None:
            self.performance_stats.add_counter("full_verify_prime_submitted")
            self.performance_stats.add_counter(
                "full_verify_prime_layers",
                sum(layer.ticket is not None for layer in prefetched),
            )
        return True

    def _adopt_primed_full_verification(
        self,
        request_ids: tuple[str, ...],
        layers: tuple[tuple[str, int], ...],
        device: torch.device,
    ) -> bool:
        primed = self._primed_full_verification
        if primed is None:
            return False

        revisions = self._get_full_verification_revisions(request_ids, layers)
        matches = (
            primed.request_ids == request_ids
            and primed.layers == layers
            and primed.revisions == revisions
            and primed.device == device
        )
        if not matches:
            self._discard_primed_full_verification(wait=False)
            return False

        self._primed_full_verification = None
        self._full_verification_prefetched.extend(primed.prefetched)
        self._full_verification_next_layer_index = len(primed.prefetched)
        self._record_full_verification_prime_outcome(adopted=True)
        return True

    def begin_full_verification_pipeline(
        self,
        request_ids: Sequence[str],
        layer_num_kv_heads: Mapping[str, int],
        device: torch.device,
    ) -> None:
        if self._full_verification_pipeline_active:
            raise RuntimeError("Full-verification pipeline is already active")

        request_ids = tuple(request_ids)
        layers = self._normalize_full_verification_layers(layer_num_kv_heads)
        device = self._canonical_full_verification_device(device)

        self._full_verification_pipeline_active = True
        self._full_verification_request_ids = request_ids
        self._full_verification_layers = layers
        self._full_verification_layer_cursor = 0
        self._full_verification_next_layer_index = 0
        self._full_verification_device = device
        self._full_verification_prefetched.clear()
        try:
            adopted = self._adopt_primed_full_verification(
                request_ids,
                layers,
                device,
            )
            if not adopted:
                self._full_verification_prefetched.append(
                    self._prefetch_full_verification_layer(0)
                )
                self._full_verification_next_layer_index = 1
        except BaseException:
            self.end_full_verification_pipeline()
            raise

    def consume_full_verification_layer(
        self,
        layer_name: str,
    ) -> RetroSpecFullVerificationStaging | None:
        if not self._full_verification_pipeline_active:
            raise RuntimeError("Full-verification pipeline is not active")
        if not self._full_verification_prefetched:
            raise RuntimeError("Full-verification pipeline has no remaining layer")

        expected_layer_name = self._full_verification_layers[
            self._full_verification_layer_cursor
        ][0]
        if expected_layer_name != layer_name:
            raise RuntimeError(
                "Full-verification layer order differs from the installed model"
            )

        prefetched = self._full_verification_prefetched.popleft()
        if prefetched.layer_name != layer_name:
            raise RuntimeError(
                "Prefetched full-verification layer does not match the model layer"
            )
        if prefetched.primed and prefetched.ticket is not None:
            counter = (
                "full_verify_prime_ready"
                if prefetched.ticket.ready()
                else "full_verify_prime_late"
            )
            if self.performance_stats is not None:
                self.performance_stats.add_counter(counter)

        clustered_kv = None if prefetched.ticket is None else prefetched.ticket.result()
        self._full_verification_layer_cursor += 1
        if (
            not self._full_verification_prefetched
            and self._full_verification_next_layer_index
            < len(self._full_verification_layers)
        ):
            next_layer = self._prefetch_full_verification_layer(
                self._full_verification_next_layer_index
            )
            self._full_verification_prefetched.append(next_layer)
            self._full_verification_next_layer_index += 1
        return clustered_kv

    def end_full_verification_pipeline(self) -> None:
        prefetched = tuple(self._full_verification_prefetched)
        self._full_verification_prefetched.clear()
        self._cancel_full_verification_prefetches(prefetched, wait=False)

        self._full_verification_pipeline_active = False
        self._full_verification_request_ids = ()
        self._full_verification_layers = ()
        self._full_verification_layer_cursor = 0
        self._full_verification_next_layer_index = 0
        self._full_verification_device = None

    def build_full_verification_plan(
        self,
        request_ids: Sequence[str],
        layer_name: str,
        seq_lens: Sequence[int],
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> RetroSpecFullVerificationPlan:
        """Build an exact full-verification view over existing KV storage.

        Complete indexed segments are represented by every cluster page owned
        by the requests. Tokens outside those segments remain primary logical
        token references into the active vLLM KV cache.
        """
        request_ids = tuple(request_ids)
        seq_lens = tuple(int(seq_len) for seq_len in seq_lens)

        if any(staged.layer_name == layer_name for staged in self._staged_segments):
            raise RuntimeError(
                "Cannot build full verification because index updates are staged "
                "for this layer"
            )
        if key_cache.ndim != 4:
            raise ValueError(
                "KV cache must have shape [num_blocks, block_size, kv_heads, head_size]"
            )
        if key_cache.shape[1] != self.block_size:
            raise ValueError("KV cache block size does not match the index")
        if block_table.ndim != 2:
            raise ValueError("block_table must be two-dimensional")
        if block_table.shape[0] != len(request_ids):
            raise ValueError("block_table batch size does not match request_ids")
        if len(seq_lens) != len(request_ids):
            raise ValueError("request_ids and seq_lens must have equal length")
        if block_table.device != key_cache.device:
            raise ValueError("block_table and KV cache must use one device")
        if block_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("block_table entries must be integral")

        max_num_tokens = block_table.shape[1] * self.block_size
        if any(seq_len < 0 for seq_len in seq_lens):
            raise ValueError("Full-verification context lengths must be non-negative")
        if any(seq_len > max_num_tokens for seq_len in seq_lens):
            raise ValueError(
                "Full-verification sequence length exceeds the block table"
            )

        layer_indices = self._indices.get(layer_name, {})
        primary_token_counts: list[int] = []

        for request_id, seq_len in zip(request_ids, seq_lens):
            record = layer_indices.get(request_id)

            if record is None or not record.segments:
                indexed_token_count = 0
            else:
                if record.indexed_end > seq_len:
                    raise RuntimeError(
                        "Full verification requires rolled-back cluster state "
                        "to be rebuilt first"
                    )
                indexed_token_count = (
                    record.indexed_end - record.segments[0].indexed_start
                )

            primary_token_counts.append(seq_len - indexed_token_count)

        view = self._get_resident_view(layer_name, request_ids, key_cache)

        seq_lens_tensor = torch.tensor(
            seq_lens,
            dtype=torch.int64,
            device=block_table.device,
        )
        logical_token_ids, valid_token_mask, _ = self._build_token_layout(
            block_table,
            seq_lens_tensor,
        )

        # Every committed token not owned by a complete clustered segment is
        # part of the exact primary/steady zone.
        indexed_starts, indexed_ends, indexed_requests = (
            self._get_resident_indexed_bounds(view, block_table.device)
        )
        primary_token_mask = valid_token_mask & (
            ~indexed_requests.unsqueeze(1)
            | (logical_token_ids.unsqueeze(0) < indexed_starts.unsqueeze(1))
            | (logical_token_ids.unsqueeze(0) >= indexed_ends.unsqueeze(1))
        )
        num_kv_heads = key_cache.shape[2]
        per_head_primary_mask = primary_token_mask.unsqueeze(1).expand(
            -1,
            num_kv_heads,
            -1,
        )

        (
            primary_exact_token_indices,
            primary_exact_token_mask,
        ) = self._pack_bounded_mask_indices(
            per_head_primary_mask,
            max(primary_token_counts, default=0),
        )

        primary_exact_token_counts = primary_exact_token_mask.sum(
            dim=2,
            dtype=torch.int32,
        )
        descriptors = self._get_full_verification_descriptors(
            layer_name, request_ids, num_kv_heads
        )
        if self._full_verification_pipeline_active:
            clustered_kv = self.consume_full_verification_layer(layer_name)
            if clustered_kv is not None and clustered_kv.token_counts.shape != (
                len(request_ids),
                num_kv_heads,
            ):
                raise RuntimeError(
                    "Prefetched full-verification staging changed batch shape"
                )
        else:
            clustered_kv = None
        clustered_exact_token_counts = (
            torch.stack(
                tuple(
                    descriptor.head_token_counts_tensor for descriptor in descriptors
                ),
                dim=0,
            )
            .to(
                device=primary_exact_token_counts.device,
                dtype=torch.int32,
            )
            .contiguous()
        )
        if clustered_exact_token_counts.shape != primary_exact_token_counts.shape:
            raise RuntimeError(
                "Full-verification clustered and native counts have different shapes"
            )
        exact_token_counts = (
            primary_exact_token_counts + clustered_exact_token_counts
        ).contiguous()
        return RetroSpecFullVerificationPlan(
            layer_name=layer_name,
            primary_exact_token_indices=primary_exact_token_indices,
            primary_exact_token_mask=primary_exact_token_mask,
            clustered_descriptors=descriptors,
            clustered_kv=clustered_kv,
            exact_token_counts=exact_token_counts,
        )
