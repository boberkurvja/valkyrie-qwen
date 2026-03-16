"""
Project Valkyrie — Phase 4: Routing Mechanism
Routes hidden states either directly to LM Head (standard tokens)
or through the HRM coprocessor (reasoning tokens).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# Special routing token
REASON_TOKEN = "<|reason|>"
REASON_TOKEN_ID = None  # Set during tokenizer modification


def add_reason_token(tokenizer) -> int:
    """
    Add the <|reason|> routing token to the tokenizer.
    Returns the token ID.
    """
    global REASON_TOKEN_ID

    # Check if already added
    if REASON_TOKEN in tokenizer.get_vocab():
        REASON_TOKEN_ID = tokenizer.convert_tokens_to_ids(REASON_TOKEN)
    else:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": [REASON_TOKEN]})
        REASON_TOKEN_ID = tokenizer.convert_tokens_to_ids(REASON_TOKEN)
        print(f"  Added {REASON_TOKEN} token (ID: {REASON_TOKEN_ID})")

    return REASON_TOKEN_ID


def get_reason_token_id() -> int:
    """Get the reason token ID (must be set by add_reason_token first)."""
    if REASON_TOKEN_ID is None:
        raise ValueError("Reason token not initialized. Call add_reason_token(tokenizer) first.")
    return REASON_TOKEN_ID


class ReasoningRouter(nn.Module):
    """
    Routes hidden states based on token content:
    - Standard tokens → direct path to LM Head
    - <|reason|> tokens → route through HRM coprocessor

    Works at the sequence level: if ANY token in the sequence is <|reason|>,
    the entire sequence is routed through HRM (for batch efficiency).
    """

    def __init__(self, reason_token_id: Optional[int] = None):
        super().__init__()
        self.reason_token_id = reason_token_id

    def set_reason_token_id(self, token_id: int):
        self.reason_token_id = token_id

    def compute_routing_mask(self, input_ids: Tensor) -> Tensor:
        """
        Determine which sequences should be routed through HRM.

        Args:
            input_ids: [B, L] token IDs
        Returns:
            route_mask: [B] boolean tensor (True = route through HRM)
        """
        if self.reason_token_id is None:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        # A sequence is routed to HRM if it contains the reason token
        has_reason = (input_ids == self.reason_token_id).any(dim=-1)  # [B]
        return has_reason

    def forward(
        self,
        input_ids: Tensor,
        hidden_states: Tensor,
        hrm_forward_fn=None,
        direct_head_fn=None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Route hidden states through either direct path or HRM path.

        Args:
            input_ids: [B, L] token IDs
            hidden_states: [B, L, H] backbone output
            hrm_forward_fn: callable that processes hidden states through HRM
                           (proj_in -> HRM -> proj_out) -> [B, L, H]
            direct_head_fn: callable that processes hidden states directly
                           (identity or norm) -> [B, L, H]
        Returns:
            output: [B, L, H] processed hidden states
            route_mask: [B] routing decisions
        """
        route_mask = self.compute_routing_mask(input_ids)

        if not route_mask.any():
            # All standard → direct path
            if direct_head_fn is not None:
                return direct_head_fn(hidden_states), route_mask
            return hidden_states, route_mask

        if route_mask.all():
            # All reasoning → HRM path
            if hrm_forward_fn is not None:
                return hrm_forward_fn(hidden_states), route_mask
            return hidden_states, route_mask

        # Mixed batch: split and process separately
        B = hidden_states.shape[0]
        output = hidden_states.clone()

        # Process HRM-routed sequences
        hrm_indices = route_mask.nonzero(as_tuple=True)[0]
        direct_indices = (~route_mask).nonzero(as_tuple=True)[0]

        if hrm_forward_fn is not None and len(hrm_indices) > 0:
            hrm_input = hidden_states[hrm_indices]
            hrm_output = hrm_forward_fn(hrm_input)
            output[hrm_indices] = hrm_output

        if direct_head_fn is not None and len(direct_indices) > 0:
            direct_input = hidden_states[direct_indices]
            direct_output = direct_head_fn(direct_input)
            output[direct_indices] = direct_output

        return output, route_mask
