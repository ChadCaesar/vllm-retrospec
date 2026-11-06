# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded full-target CPU inference and shared KV views."""

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import create_causal_mask


class CPUForwardState:
    """One Llama/Qwen2 decode token, resumable between decoder layers.

    Keep the same masks, RoPE tensors, cache and hidden state for the entire
    token. A layer pre-hook can yield control by raising before the next layer;
    calling run() again starts at that layer without repeating completed work.
    """

    def __init__(self, model, input_ids, cache, position, serial=0, started_at=0.0):
        if model.config.model_type not in ("llama", "qwen2"):
            raise ValueError("CPU continuation requires a Llama/Qwen2 model")
        self.model = model
        self.cache = cache
        self.serial = serial
        self.started_at = started_at
        self.resumes = 0
        self.layer_index = 0
        self.hidden_states = model.model.embed_tokens(input_ids)
        self.cache_position = torch.tensor([position], dtype=torch.int64)
        self.position_ids = self.cache_position.unsqueeze(0)
        self.causal_mask = create_causal_mask(
            config=model.config,
            input_embeds=self.hidden_states,
            attention_mask=None,
            cache_position=self.cache_position,
            past_key_values=cache,
            position_ids=self.position_ids,
        )
        if getattr(model.model, "has_sliding_layers", False):
            raise ValueError("CPU continuation does not support sliding attention")
        self.position_embeddings = model.model.rotary_emb(
            self.hidden_states, self.position_ids
        )

    def run(self):
        # Mirrors the single-token HF LlamaModel/Qwen2Model forward. Store the
        # cursor only after a layer has completed, so a pre-hook suspension
        # leaves its input and all preceding layers intact.
        layers = self.model.model.layers
        while self.layer_index < self.model.config.num_hidden_layers:
            kwargs = dict(
                attention_mask=self.causal_mask,
                position_ids=self.position_ids,
                past_key_values=self.cache,
                cache_position=self.cache_position,
                position_embeddings=self.position_embeddings,
            )
            if self.model.config.model_type == "qwen2":
                kwargs["use_cache"] = True
            self.hidden_states = layers[self.layer_index](self.hidden_states, **kwargs)
            self.layer_index += 1
        hidden_states = self.model.model.norm(self.hidden_states)
        return self.model.lm_head(hidden_states)[0, -1]


def grouped_decode_attention(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs
):
    if (
        query.device.type != "cpu"
        or query.shape[-2] != 1
        or attention_mask is not None
        or dropout
        or kwargs.get("is_causal")
    ):
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )
    batch, heads, _, dimension = query.shape
    kv_heads, length = key.shape[1:3]
    # All query rows have the same absolute position. No causal mask is used.
    output = torch.nn.functional.scaled_dot_product_attention(
        query.reshape(batch * kv_heads, 1, heads // kv_heads, dimension),
        key.reshape(batch * kv_heads, 1, length, dimension),
        value.reshape(batch * kv_heads, 1, length, dimension),
        is_causal=False,
        scale=scaling,
    )
    return output.reshape(batch, heads, 1, dimension).transpose(1, 2).contiguous(), None


class AppendOnlyLayer(DynamicLayer):
    def __init__(self, capacity: int):
        super().__init__()
        if capacity <= 0:
            raise ValueError("KV capacity must be positive")
        self.capacity = capacity
        self.length = 0

    def lazy_initialization(self, key_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        shape = (*key_states.shape[:-2], self.capacity, key_states.shape[-1])
        self.keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        self.values = torch.empty_like(self.keys)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        end = self.length + key_states.shape[-2]
        if end > self.capacity:
            raise ValueError(f"KV capacity exceeded: {end} > {self.capacity}")
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        self.keys[..., self.length : end, :].copy_(key_states)
        self.values[..., self.length : end, :].copy_(value_states)
        self.length = end
        return self.keys[..., :end, :], self.values[..., :end, :]

    def get_seq_length(self):
        return self.length

    def crop(self, max_length):
        if max_length < 0:
            max_length = max(0, self.length + max_length)
        self.length = min(self.length, max_length)

    def reset(self):
        self.length = 0


def make_cache(config, capacity: int) -> DynamicCache:
    cache = DynamicCache(config=config)
    cache.layers = [AppendOnlyLayer(capacity) for _ in cache.layers]
    return cache


def attach_cache(config, storage: torch.Tensor, length: int) -> DynamicCache:
    cache = make_cache(config, storage.shape[-2])
    for index, layer in enumerate(cache.layers):
        layer.keys = storage[0, index].unsqueeze(0)
        layer.values = storage[1, index].unsqueeze(0)
        layer.dtype, layer.device = storage.dtype, storage.device
        layer.is_initialized = True
        layer.length = length
    return cache
