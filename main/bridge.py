"""
Project Valkyrie — Phase 4: HRM Bridge Modules
Projections between Valkyrie hidden space (1024) and HRM space (512).
StableMax LM head replacement for numerical stability.
"""
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import BridgeConfig

# Import StableMax from HRM losses
sys.path.insert(0, "/root/model/HRM")
from models.losses import s as stablemax_s, log_stablemax, stablemax_cross_entropy


class ProjIn(nn.Module):
    """
    Inbound projection: Valkyrie hidden_size → HRM hidden_size.
    Maps Valkyrie's 1024-dim representations into HRM's 512-dim latent space.
    """

    def __init__(self, valkyrie_dim: int, hrm_dim: int):
        super().__init__()
        self.proj = nn.Linear(valkyrie_dim, hrm_dim, bias=False, dtype=torch.bfloat16)
        # Xavier init for stable projection
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class ProjOut(nn.Module):
    """
    Outbound projection: HRM hidden_size → Valkyrie hidden_size.
    Maps HRM's 512-dim output back to Valkyrie's 1024-dim space.
    """

    def __init__(self, hrm_dim: int, valkyrie_dim: int):
        super().__init__()
        self.proj = nn.Linear(hrm_dim, valkyrie_dim, bias=False, dtype=torch.bfloat16)
        # Zero init for residual-friendly behavior
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class StableMaxLMHead(nn.Module):
    """
    LM Head using StableMax instead of standard Softmax.
    Prevents softmax collapse and handles large-scale logit blowups,
    crucial for small-sample reasoning tasks.

    StableMax s(x) = { 1/(1-x+ε) if x < 0; x+1 if x >= 0 }
    log_stablemax(x) = log(s(x) / Σ s(x))
    """

    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=bias, dtype=torch.bfloat16)

    def forward(self, x: Tensor) -> Tensor:
        """Returns raw logits (apply stablemax at loss time)."""
        return self.linear(x)

    def compute_loss(self, logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
        """Compute StableMax cross-entropy loss."""
        return stablemax_cross_entropy(logits, labels, ignore_index=ignore_index).mean()

    @staticmethod
    def log_probs(logits: Tensor) -> Tensor:
        """Compute log probabilities using StableMax."""
        return log_stablemax(logits, dim=-1)


class HRMBridge(nn.Module):
    """
    Complete bridge between Valkyrie backbone and HRM coprocessor.
    Manages Proj_In, Proj_Out, and the StableMax LM head.
    """

    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config

        # Projections
        self.proj_in = ProjIn(config.valkyrie_hidden_size, config.hrm_hidden_size)
        self.proj_out = ProjOut(config.hrm_hidden_size, config.valkyrie_hidden_size)

    def to_hrm_space(self, x: Tensor) -> Tensor:
        """Project from Valkyrie space to HRM space."""
        return self.proj_in(x)

    def from_hrm_space(self, x: Tensor) -> Tensor:
        """Project from HRM space back to Valkyrie space."""
        return self.proj_out(x)

    def get_bridge_parameters(self):
        """Get all bridge-specific parameters for selective training."""
        params = list(self.proj_in.parameters()) + list(self.proj_out.parameters())
        return params
