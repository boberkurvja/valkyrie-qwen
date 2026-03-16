import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache, DynamicLayer

class SinkLayer(DynamicLayer):
    """
    A single layer for SinkCache that assembles the first `num_sink_tokens` 
    and the most recent `window_length - num_sink_tokens` tokens.
    """
    def __init__(self, window_length: int, num_sink_tokens: int):
        super().__init__()
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cumulative_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        seq_len = self.keys.shape[-2]
        
        # Evict middle tokens if we exceed window length
        if seq_len > self.window_length:
            sink_k = self.keys[..., :self.num_sink_tokens, :]
            sink_v = self.values[..., :self.num_sink_tokens, :]
            
            recent_k = self.keys[..., -self.window_length + self.num_sink_tokens:, :]
            recent_v = self.values[..., -self.window_length + self.num_sink_tokens:, :]
            
            self.keys = torch.cat([sink_k, recent_k], dim=-2)
            self.values = torch.cat([sink_v, recent_v], dim=-2)
            
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.cumulative_length


class SinkCache(Cache):
    """
    A cache that assembles the first `num_sink_tokens` (Attention Sinks)
    and the most recent `window_length - num_sink_tokens` tokens.
    """
    def __init__(self, window_length: int, num_sink_tokens: int):
        # We must provide layer_class_to_replicate in this transformers iteration
        super().__init__(
            layer_class_to_replicate=lambda: SinkLayer(window_length, num_sink_tokens)
        )
        num_hidden_layers = 100 # Large enough default
        self.conv_states = [None] * num_hidden_layers
        self.recurrent_states = [None] * num_hidden_layers
        self.layer_types = ["full_attention"] * num_hidden_layers
        self.transformer_layers = list(range(num_hidden_layers))

    @property
    def has_previous_state(self) -> bool:
        return self.get_seq_length() > 0
