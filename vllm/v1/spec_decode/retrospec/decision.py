# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import IntFlag

import torch

from vllm.config import SpeculativeConfig

from .state import RetroSpecStage


class RetroSpecReason(IntFlag):
    NONE = 0

    DRAFT_MARGIN = 1 << 0
    HIT_ATTN = 1 << 1
    MAX_DRAFT_TOKENS = 1 << 2

    SPARSE_MARGIN = 1 << 3
    RETRIEVAL_ATTN = 1 << 4
    SPARSE_TOKEN_CHANGED = 1 << 5

    EXPANDED_MARGIN = 1 << 6
    EXPANDED_ATTN = 1 << 7
    EXPANDED_TOKEN_CHANGED = 1 << 8

    PENDING_LIMIT = 1 << 9
    GENERATION_LIMIT = 1 << 10
    INDEX_UPDATE = 1 << 11


@dataclass(frozen=True)
class RetroSpecMetrics:
    draft_margin: torch.Tensor | None = None
    sparse_margin: torch.Tensor | None = None
    expanded_margin: torch.Tensor | None = None

    hit_attn: torch.Tensor | None = None
    retrieval_attn: torch.Tensor | None = None
    expanded_attn: torch.Tensor | None = None


@dataclass(frozen=True)
class RetroSpecDecision:
    stop_draft: torch.Tensor
    require_expanded: torch.Tensor
    require_full: torch.Tensor
    reasons: torch.Tensor
    next_stage: torch.Tensor


@dataclass(frozen=True)
class _HardTriggers:
    pending_limit_reached: torch.Tensor | None = None
    generation_limit_reached: torch.Tensor | None = None
    index_update_required: torch.Tensor | None = None


@dataclass
class _EvalContext:
    metrics: RetroSpecMetrics
    draft_counts: torch.Tensor
    pending_counts: torch.Tensor
    stage_mask: torch.Tensor
    stop_draft: torch.Tensor
    require_expanded: torch.Tensor
    require_full: torch.Tensor
    reasons: torch.Tensor

    @classmethod
    def create(
        cls,
        metrics: RetroSpecMetrics,
        draft_counts: torch.Tensor,
        pending_counts: torch.Tensor,
        stage_mask: torch.Tensor,
    ) -> "_EvalContext":
        return cls(
            metrics=metrics,
            draft_counts=draft_counts,
            pending_counts=pending_counts,
            stage_mask=stage_mask,
            stop_draft=torch.zeros_like(draft_counts, dtype=torch.bool),
            require_expanded=torch.zeros_like(draft_counts, dtype=torch.bool),
            require_full=torch.zeros_like(draft_counts, dtype=torch.bool),
            reasons=torch.zeros_like(draft_counts, dtype=torch.int32),
        )


class RetroSpecDecisionPolicy:
    def __init__(self, config: SpeculativeConfig) -> None:
        assert config.method == "retrospec"
        assert config.num_speculative_tokens is not None

        self.min_draft_tokens = config.retrospec_min_draft_tokens
        self.max_draft_tokens = config.retrospec_max_draft_tokens
        self.pending_limit = config.num_speculative_tokens

        self.draft_margin_threshold = config.retrospec_draft_margin_threshold
        self.sparse_margin_threshold = config.retrospec_sparse_margin_threshold
        self.expanded_margin_threshold = config.retrospec_expanded_margin_threshold

        self.hit_attn_threshold = config.retrospec_hit_attn_threshold
        self.retrieval_attn_threshold = config.retrospec_retrieval_attn_threshold
        self.expanded_attn_threshold = config.retrospec_expanded_attn_threshold

    @staticmethod
    def _validate_vector(
        name: str,
        value: torch.Tensor,
        reference: torch.Tensor,
        require_bool: bool = False,
    ) -> None:
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional tensor")

        if value.shape != reference.shape:
            raise ValueError(
                f"{name} shape {value.shape} does not match "
                f"reference shape {reference.shape}"
            )

        if value.device != reference.device:
            raise ValueError(
                f"{name} must be on device {reference.device}, but is on {value.device}"
            )

        if require_bool and value.dtype != torch.bool:
            raise ValueError(f"{name} must have dtype torch.bool")

    @classmethod
    def _validate_counts(
        cls,
        name: str,
        value: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        cls._validate_vector(name, value, reference)
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must use an integer dtype")

    @staticmethod
    def _record_reason(
        ctx: _EvalContext,
        triggered: torch.Tensor,
        reason: RetroSpecReason,
    ) -> None:
        ctx.reasons.bitwise_or_(triggered.to(torch.int32) * int(reason))

    def _below_threshold(
        self,
        name: str,
        value: torch.Tensor | None,
        threshold: float | None,
        ctx: _EvalContext,
    ) -> torch.Tensor | None:
        if threshold is None:
            return None
        if value is None:
            raise ValueError(f"{name} is required when its threshold is configured")

        self._validate_vector(name, value, ctx.draft_counts)
        return (value < threshold) & ctx.stage_mask

    def _mask_external_trigger(
        self, name: str, value: torch.Tensor | None, ctx: _EvalContext
    ) -> torch.Tensor | None:
        if value is None:
            return None
        self._validate_vector(name, value, ctx.draft_counts, require_bool=True)
        return value & ctx.stage_mask

    def _build_hard_triggers(
        self,
        ctx: _EvalContext,
        generation_limit_reached: torch.Tensor | None,
        index_update_required: torch.Tensor | None,
    ) -> _HardTriggers:
        self._validate_counts("pending_counts", ctx.pending_counts, ctx.draft_counts)
        pending_limit_reached = (
            ctx.pending_counts >= self.pending_limit
        ) & ctx.stage_mask
        generation_limit_reached = self._mask_external_trigger(
            "generation_limit_reached", generation_limit_reached, ctx
        )
        index_update_required = self._mask_external_trigger(
            "index_update_required", index_update_required, ctx
        )
        return _HardTriggers(
            pending_limit_reached,
            generation_limit_reached,
            index_update_required,
        )

    def _apply_hard_triggers(
        self,
        ctx: _EvalContext,
        target: torch.Tensor,
        hard_triggers: _HardTriggers,
    ) -> None:
        trigger_reasons = (
            (hard_triggers.pending_limit_reached, RetroSpecReason.PENDING_LIMIT),
            (hard_triggers.generation_limit_reached, RetroSpecReason.GENERATION_LIMIT),
            (hard_triggers.index_update_required, RetroSpecReason.INDEX_UPDATE),
        )
        for triggered, reason in trigger_reasons:
            if triggered is None:
                continue
            target |= triggered
            self._record_reason(ctx, triggered, reason)

    def _evaluate_draft(self, ctx: _EvalContext, hard_triggers: _HardTriggers) -> None:
        min_reached = ctx.draft_counts >= self.min_draft_tokens
        max_reached = (ctx.draft_counts >= self.max_draft_tokens) & ctx.stage_mask
        self._record_reason(ctx, max_reached, RetroSpecReason.MAX_DRAFT_TOKENS)
        ctx.stop_draft |= max_reached

        draft_margin_low = self._below_threshold(
            "draft_margin", ctx.metrics.draft_margin, self.draft_margin_threshold, ctx
        )
        if draft_margin_low is not None:
            draft_margin_low &= min_reached
            ctx.stop_draft |= draft_margin_low
            self._record_reason(ctx, draft_margin_low, RetroSpecReason.DRAFT_MARGIN)

        hit_attn_low = self._below_threshold(
            "hit_attn", ctx.metrics.hit_attn, self.hit_attn_threshold, ctx
        )
        if hit_attn_low is not None:
            hit_attn_low &= min_reached
            ctx.stop_draft |= hit_attn_low
            self._record_reason(ctx, hit_attn_low, RetroSpecReason.HIT_ATTN)

        self._apply_hard_triggers(ctx, ctx.stop_draft, hard_triggers)

    def _evaluate_sparse_verify(
        self,
        ctx: _EvalContext,
        sparse_token_changed: torch.Tensor | None,
        hard_triggers: _HardTriggers,
    ) -> None:
        sparse_margin_low = self._below_threshold(
            "sparse_margin",
            ctx.metrics.sparse_margin,
            self.sparse_margin_threshold,
            ctx,
        )
        if sparse_margin_low is not None:
            ctx.require_expanded |= sparse_margin_low
            self._record_reason(ctx, sparse_margin_low, RetroSpecReason.SPARSE_MARGIN)

        retrieval_attn_low = self._below_threshold(
            "retrieval_attn",
            ctx.metrics.retrieval_attn,
            self.retrieval_attn_threshold,
            ctx,
        )
        if retrieval_attn_low is not None:
            ctx.require_expanded |= retrieval_attn_low
            self._record_reason(ctx, retrieval_attn_low, RetroSpecReason.RETRIEVAL_ATTN)

        if sparse_token_changed is not None:
            self._validate_vector(
                "sparse_token_changed",
                sparse_token_changed,
                ctx.draft_counts,
                require_bool=True,
            )
            triggered = sparse_token_changed & ctx.stage_mask
            ctx.require_expanded |= triggered
            self._record_reason(ctx, triggered, RetroSpecReason.SPARSE_TOKEN_CHANGED)

        self._apply_hard_triggers(ctx, ctx.require_full, hard_triggers)

    def _evaluate_expanded_verify(
        self,
        ctx: _EvalContext,
        expanded_token_changed: torch.Tensor | None,
        hard_triggers: _HardTriggers,
    ) -> None:
        expanded_margin_low = self._below_threshold(
            "expanded_margin",
            ctx.metrics.expanded_margin,
            self.expanded_margin_threshold,
            ctx,
        )
        if expanded_margin_low is not None:
            ctx.require_full |= expanded_margin_low
            self._record_reason(
                ctx, expanded_margin_low, RetroSpecReason.EXPANDED_MARGIN
            )

        expanded_attn_low = self._below_threshold(
            "expanded_attn",
            ctx.metrics.expanded_attn,
            self.expanded_attn_threshold,
            ctx,
        )
        if expanded_attn_low is not None:
            ctx.require_full |= expanded_attn_low
            self._record_reason(ctx, expanded_attn_low, RetroSpecReason.EXPANDED_ATTN)

        if expanded_token_changed is not None:
            self._validate_vector(
                "expanded_token_changed",
                expanded_token_changed,
                ctx.draft_counts,
                require_bool=True,
            )
            triggered = expanded_token_changed & ctx.stage_mask
            ctx.require_full |= triggered
            self._record_reason(ctx, triggered, RetroSpecReason.EXPANDED_TOKEN_CHANGED)

        self._apply_hard_triggers(ctx, ctx.require_full, hard_triggers)

    def evaluate(
        self,
        *,
        current_stage: RetroSpecStage,
        request_stages: torch.Tensor,
        metrics: RetroSpecMetrics,
        draft_counts: torch.Tensor,
        pending_counts: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        sparse_token_changed: torch.Tensor | None = None,
        expanded_token_changed: torch.Tensor | None = None,
        generation_limit_reached: torch.Tensor | None = None,
        index_update_required: torch.Tensor | None = None,
    ) -> RetroSpecDecision:
        """
        Evaluate transitions for one execution phase.
        ``current_stage`` selects the only rule set evaluated by this call;
        ``request_stages`` records the current stage of every request.
        Their intersection with ``active_mask`` forms ``stage_mask``,
        which selects the active batch rows belonging to this execution phase.
        Metrics from every other stage are intentionally untouched.
        """
        if draft_counts.ndim != 1:
            raise ValueError("draft_counts must be one-dimensional")
        self._validate_counts("draft_counts", draft_counts, draft_counts)
        self._validate_vector("request_stages", request_stages, draft_counts)
        if request_stages.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("request_stages must use an integer dtype")
        if not isinstance(current_stage, RetroSpecStage):
            raise ValueError("current_stage must be a RetroSpecStage")

        if active_mask is None:
            active_mask = torch.ones_like(draft_counts, dtype=torch.bool)
        else:
            self._validate_vector(
                "active_mask", active_mask, draft_counts, require_bool=True
            )

        stage_mask = active_mask & (request_stages == int(current_stage))
        ctx = _EvalContext.create(metrics, draft_counts, pending_counts, stage_mask)
        hard_triggers = _HardTriggers()
        if current_stage in (
            RetroSpecStage.DRAFT,
            RetroSpecStage.SPARSE_VERIFY,
            RetroSpecStage.EXPANDED_VERIFY,
        ):
            hard_triggers = self._build_hard_triggers(
                ctx, generation_limit_reached, index_update_required
            )
        next_stage = request_stages.clone()

        if current_stage == RetroSpecStage.DRAFT:
            self._evaluate_draft(ctx, hard_triggers)
            next_stage.masked_fill_(ctx.stop_draft, int(RetroSpecStage.SPARSE_VERIFY))
        elif current_stage == RetroSpecStage.SPARSE_VERIFY:
            self._evaluate_sparse_verify(ctx, sparse_token_changed, hard_triggers)
            ctx.require_expanded &= ~ctx.require_full
            next_stage.masked_fill_(stage_mask, int(RetroSpecStage.DRAFT))
            next_stage.masked_fill_(
                ctx.require_expanded, int(RetroSpecStage.EXPANDED_VERIFY)
            )
            next_stage.masked_fill_(ctx.require_full, int(RetroSpecStage.FULL_VERIFY))
        elif current_stage == RetroSpecStage.EXPANDED_VERIFY:
            self._evaluate_expanded_verify(ctx, expanded_token_changed, hard_triggers)
            next_stage.masked_fill_(stage_mask, int(RetroSpecStage.DRAFT))
            next_stage.masked_fill_(ctx.require_full, int(RetroSpecStage.FULL_VERIFY))

        next_stage.masked_fill_(~active_mask, int(RetroSpecStage.IDLE))

        return RetroSpecDecision(
            ctx.stop_draft,
            ctx.require_expanded,
            ctx.require_full,
            ctx.reasons,
            next_stage,
        )
