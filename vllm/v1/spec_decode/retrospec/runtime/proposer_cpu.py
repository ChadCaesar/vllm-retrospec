# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.retrospec.cpu_state import RetroSpecCPUWindow
from vllm.v1.spec_decode.retrospec.decision import RetroSpecMetrics, RetroSpecReason
from vllm.v1.spec_decode.retrospec.state import RetroSpecStage


class RetroSpecCPUProposalMixin:
    _CPU_PLAN_ROWS = (
        "valid_rows",
        "primary_exact_token_indices",
        "primary_exact_token_mask",
        "ranked_cluster_indices",
        "candidate_counts",
        "sparse_attn",
        "expanded_attn",
    )

    def _cpu_save_plans(self, start, end):
        index = getattr(self.sparse_attention, "index", None)
        return {
            name: {
                **{
                    field: getattr(table, field)[start:end, :1].clone()
                    for field in self._CPU_PLAN_ROWS
                },
                "request_slot_ids": table.request_slot_ids[:1].clone(),
                "request_slot_generations": table.request_slot_generations[:1].clone(),
            }
            for name, table in getattr(index, "_selection_plan_tables", {}).items()
        }

    def _cpu_restore_plans(self, plans):
        index = self.sparse_attention.index
        for name, fields in plans.items():
            table = index._selection_plan_tables[name]
            for field, saved in fields.items():
                target = getattr(table, field)
                if field in self._CPU_PLAN_ROWS:
                    target[: saved.shape[0], :1].copy_(saved)
                else:
                    target[:1].copy_(saved)
            index._selection_plan_written_layers.add(name)

    def _cpu_request_supported(self, request_ids, sampling_metadata):
        if getattr(self, "cpu_coordinator", None) is None or len(request_ids) != 1:
            return False
        return self._cpu_risk_request_supported(request_ids, sampling_metadata)

    def _cpu_risk_request_supported(self, request_ids, sampling_metadata):
        if len(request_ids) != 1:
            return False
        if not self._can_use_raw_greedy_verification_sampling(sampling_metadata):
            return False
        request = self.runner.requests[request_ids[0]]
        params = request.sampling_params
        return (
            params is not None
            and params.min_tokens == 0
            and params.logprobs is None
            and params.prompt_logprobs is None
            and params.structured_outputs is None
            and not params.logit_bias
            and not params.logits_processors
            and request.lora_request is None
            and request.prompt_embeds is None
        )

    def _cpu_refresh_index(self, request_id, length, metadata):
        index = self.sparse_attention.index
        for name, layer in self.cpu_attention_layers.items():
            keys, values = layer.kv_cache[0].unbind(0)
            index.build_or_update(
                layer_name=name,
                request_ids=(request_id,),
                seq_lens=(length,),
                is_prefill=(False,),
                rows=(0,),
                key_cache=keys,
                value_cache=values,
                block_table=metadata.block_table_tensor,
            )
        index.flush_staged_updates()
        self.index_update_state.begin_batch((request_id,), (length,))

    def _propose_with_cpu(
        self,
        request_ids,
        committed_positions,
        metadata,
        sampling_metadata,
        remaining_generation_budget=None,
    ):
        cpu = self.cpu_coordinator
        request_id = request_ids[0]
        budget = int(self._proposal_token_budgets.gpu[0].item())
        if remaining_generation_budget is not None:
            self._proposal_token_budgets.gpu[0] = remaining_generation_budget
        base = int(self.proposal_start_positions[0].item())
        seed = int(self.proposal_input_ids[0].item())
        params = self.runner.requests[request_id].sampling_params
        carried = getattr(cpu, "deferred_window", None)
        carried_plans = getattr(cpu, "deferred_plans", None)
        cpu.deferred_window = None
        cpu.deferred_plans = None
        cpu.begin(request_id, base, seed, budget, metadata, params)
        window = RetroSpecCPUWindow(base)
        if carried is not None:
            carried_request, carried_base, tokens, sparse = carried
            if (carried_request, carried_base) != (request_id, base):
                raise RuntimeError(
                    "CPU deferred proposal lost its authoritative prefix"
                )
            window.tokens = list(tokens)
            window.sparse = sparse
        active = self.state.active_mask
        force_sparse = False
        require_full = False
        proposal_round = 0
        index_boundary = int(self.index_update_state.next_update_positions[0].item())
        execution_end = base + budget
        generation_end = min(
            base
            + (
                budget
                if remaining_generation_budget is None
                else remaining_generation_budget
            ),
            self.max_model_len - 1,
        )
        defer_window = False
        stopped = seed in cpu.stop_ids

        def materialize():
            self._draft_token_ids[0].fill_(-1)
            if window.tokens:
                self._draft_token_ids[0, : window.draft_end] = torch.tensor(
                    window.tokens,
                    dtype=torch.int32,
                    device=self.device,
                )

        with self.sparse_attention.proposal_context(request_ids, committed_positions):
            materialize()
            if carried_plans:
                self._cpu_restore_plans(carried_plans)
            while not stopped:
                changed = cpu.poll(window)
                if not window.sparse_pending:
                    force_sparse = False
                if (
                    window.confirmed
                    and window.tokens[window.confirmed - 1] in cpu.stop_ids
                ):
                    stopped = True
                    break
                if changed:
                    materialize()
                if self.speculative_config.retrospec_cpu_kv_background_upload:
                    cpu.synchronize(metadata, background=True)

                end_position = base + window.draft_end
                terminal_draft = (
                    window.full_pending > 0 and window.tokens[-1] in cpu.stop_ids
                )
                execution_boundary = (
                    window.draft_end >= self.num_speculative_tokens
                    or end_position >= execution_end
                )
                if (
                    execution_boundary
                    and window.confirmed > 0
                    and window.full_pending < self.policy.pending_limit
                    and end_position < generation_end
                    and end_position < index_boundary
                    and not terminal_draft
                    and not force_sparse
                ):
                    # Yield confirmed outputs so the scheduler can reserve
                    # more native slots. Keep candidates and sparse progress;
                    # a physical workspace boundary is not a verify trigger.
                    defer_window = True
                    break
                hard_boundary = (
                    window.full_pending >= self.policy.pending_limit
                    or execution_boundary
                    or end_position >= generation_end
                    or end_position >= index_boundary
                    or terminal_draft
                )
                should_verify = window.sparse_pending and (
                    force_sparse
                    or hard_boundary
                    or window.sparse_pending >= self.policy.max_draft_tokens
                )
                if should_verify:
                    cpu.synchronize(metadata, background=False)
                    start = max(window.confirmed, window.sparse)
                    snapshot_version = window.version
                    snapshot_end = window.draft_end
                    materialize()
                    starts = torch.tensor(
                        [start], dtype=torch.int32, device=self.device
                    )
                    self.state.draft_counts.fill_(snapshot_end - start)
                    self.state.pending_counts.fill_(start)
                    self.state.set_stage(active, RetroSpecStage.SPARSE_VERIFY)
                    verification = self._verify_draft_tokens(
                        1,
                        request_ids,
                        proposal_round + 1,
                        starts,
                        metadata,
                        sampling_metadata,
                        confirmed_counts=torch.tensor(
                            [window.confirmed], dtype=torch.int32, device=self.device
                        ),
                    )
                    count = int(verification.verified_counts[0].item())
                    verified_tokens = (
                        self._draft_token_ids[0, start : start + count].cpu().tolist()
                    )
                    cpu.poll(window)
                    applied = window.apply_sparse(
                        snapshot_version, start, verified_tokens
                    )
                    force_sparse = False
                    if not applied:
                        self.performance_stats.add_counter("cpu_stale_verifications")
                        materialize()
                        cpu.synchronize(metadata, background=False)
                        continue
                    require_full = bool(verification.require_full[0].item())
                    reasons = getattr(verification, "full_reasons", None)
                    if (
                        require_full
                        and reasons is not None
                        and int(reasons[0].item()) == int(RetroSpecReason.PENDING_LIMIT)
                        and window.full_pending < self.policy.pending_limit
                    ):
                        # The sparse snapshot stays fixed. Only a full verify
                        # not yet dispatched may be delayed by released quota.
                        require_full = False
                    require_full |= (
                        window.full_pending > 0 and window.tokens[-1] in cpu.stop_ids
                    )
                    self.performance_stats.add_gpu_counter("verified_tokens", count)
                    if require_full:
                        break
                    continue
                if hard_boundary:
                    break

                proposal_round += 1
                if self.sparse_attention.selection_provenance_enabled:
                    self.sparse_attention.set_proposal_round(proposal_round)
                slot = window.draft_end
                self.positions[0] = base + slot
                self.input_ids[0] = seed if slot == 0 else window.tokens[-1]
                self.state.set_stage(active, RetroSpecStage.DRAFT)
                self.state.draft_counts.fill_(window.sparse_pending)
                sampled, margin, mass = self._run_draft_step(
                    1,
                    slot,
                    metadata,
                    active,
                    sampling_metadata,
                )
                token = int(sampled[0].item())
                window.tokens.append(token)
                self._draft_token_ids[0, slot] = token
                self.performance_stats.add_gpu_counter("draft_tokens", 1)
                self.state.draft_counts.fill_(window.sparse_pending)
                self.state.pending_counts.fill_(window.full_pending)
                decision = self.policy.evaluate(
                    current_stage=RetroSpecStage.DRAFT,
                    request_stages=self.state.stage,
                    metrics=RetroSpecMetrics(draft_margin=margin, hit_attn=mass),
                    draft_counts=self.state.draft_counts,
                    pending_counts=self.state.pending_counts,
                    active_mask=active,
                )
                # Length limits are recomputed after CPU progress, so they can
                # be refilled. Quality triggers remain latched for verification.
                quality_reasons = int(
                    RetroSpecReason.DRAFT_MARGIN | RetroSpecReason.HIT_ATTN
                )
                force_sparse = bool(int(decision.reasons[0].item()) & quality_reasons)
                if window.confirmed:
                    self.performance_stats.add_counter("cpu_sliding_draft_tokens")

        # Freeze the last already committed prefix without waiting for CPU work.
        cpu.freeze(window, metadata)
        if stopped:
            window.tokens[window.confirmed :] = []
        committed_length = base + window.confirmed
        if not window.full_pending and committed_length >= index_boundary:
            self._cpu_refresh_index(request_id, committed_length, metadata)
        pending = window.tokens[window.confirmed :]
        if defer_window and pending:
            cpu.deferred_plans = self._cpu_save_plans(
                window.confirmed, window.draft_end
            )
            cpu.deferred_window = (
                request_id,
                committed_length,
                pending,
                max(0, window.sparse - window.confirmed),
            )
            self.performance_stats.add_counter("cpu_deferred_proposal_batches")
        if len(pending) > self.policy.pending_limit:
            raise RuntimeError("CPU-assisted full-verification window overflowed")
        self.state.pending_counts.fill_(len(pending))
        self.state.set_stage(
            active, RetroSpecStage.FULL_VERIFY if pending else RetroSpecStage.IDLE
        )
        self.performance_stats.add_gpu_counter("proposed_tokens", len(pending))
        self.performance_stats.maybe_log()
        return [pending]
