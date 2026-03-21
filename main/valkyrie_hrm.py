"""
Project Valkyrie — Phase 4: Combined Valkyrie + HRM Model
Integrates the Valkyrie backbone, HRM coprocessor, bridge projections,
and routing mechanism into a unified model.
"""
import sys
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoModelForCausalLM

from config import ValkyrieConfig, BridgeConfig, get_config
from bridge import HRMBridge, StableMaxLMHead
from routing import ReasoningRouter, add_reason_token, REASON_TOKEN

# Import HRM from the existing codebase
sys.path.insert(0, "/root/model/HRM")
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1_Inner,
)
from models.losses import stablemax_cross_entropy


@dataclass
class ValkyrieHRMOutput:
    """Output from the combined Valkyrie+HRM model."""
    logits: Tensor
    loss: Optional[Tensor] = None
    route_mask: Optional[Tensor] = None  # Which sequences went through HRM
    avg_act_steps: Optional[Tensor] = None  # Average computation steps for the batch
    hrm_delta: Optional[Tensor] = None      # Magnitude of HRM residual change
    backbone_hidden: Optional[Tensor] = None  # Pre-routing hidden states
    new_carry: Optional[Tuple[Tensor, Tensor]] = None # Tuple of (z_H, z_L)
    hrm_states: Optional[Tuple[Tensor, Tensor]] = None # Full sequence (z_H, z_L)
    q_logits: Optional[Tensor] = None # Q-learning logits for ACT [B, 2]


class ValkyrieHRM(nn.Module):
    """
    Combined Valkyrie + HRM model.

    Forward pass:
    1. Valkyrie backbone (S5 blocks) processes input
    2. Router decides: standard → LM Head, or reasoning → HRM → LM Head
    3. HRM path: Proj_In → HRM (N×T latent cycles) → Proj_Out → StableMax LM Head
    4. Direct path: → Standard LM Head

    The StableMax LM head replaces standard softmax for numerical stability.
    """

    def __init__(self, config: ValkyrieConfig):
        super().__init__()
        self.config = config
        self.bridge_config = config.bridge

        # 1. Qwen backbone
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.teacher.model_name,
            torch_dtype=config.teacher.dtype,
            trust_remote_code=True,
            # Fall back to SDPA because flash-attn takes too long to install
        )
        
        # We need to access the text model inside (e.g. for Qwen3.5 it's .model)
        self.text_model = self.backbone.model

        # 2. HRM Bridge (projections between dimensions)
        self.bridge = HRMBridge(config.bridge)

        # 3. HRM Coprocessor (loaded from existing implementation)
        # We create a simplified inner model for integration
        self.hrm_inner = self._create_hrm_inner(config.bridge)

        # 4. Router
        self.router = ReasoningRouter()

        # 5. StableMax LM Head (replaces backbone's standard LM head)
        self.stablemax_head = StableMaxLMHead(
            hidden_size=config.teacher.hidden_size,
            vocab_size=config.teacher.vocab_size,
        )

        # Compile HRM forward for kernel fusion (reduces Python dispatch overhead)
        self._compiled_hrm_forward = torch.compile(
            self._hrm_forward, mode="reduce-overhead"
        )

        # Copy backbone's LM head weights to stablemax head
        if hasattr(self.backbone, "lm_head"):
            with torch.no_grad():
                self.stablemax_head.linear.weight.copy_(self.backbone.lm_head.weight)
        
        # Ensure backbone is frozen as per the objective
        self.freeze_backbone()

    def _create_hrm_inner(self, bridge_config: BridgeConfig) -> HierarchicalReasoningModel_ACTV1_Inner:
        """
        Create HRM inner model compatible with the existing implementation.
        We instantiate the inner module directly for tighter integration.
        """
        hrm_config = HierarchicalReasoningModel_ACTV1Config(
            batch_size=1,  # Dynamic batching handled externally
            seq_len=1,  # Sequence processed token-by-token through bridge
            puzzle_emb_ndim=0,  # No puzzle embeddings in Valkyrie integration
            num_puzzle_identifiers=1,  # Placeholder
            vocab_size=self.config.teacher.vocab_size,
            H_cycles=bridge_config.hrm_H_cycles,
            L_cycles=bridge_config.hrm_L_cycles,
            H_layers=bridge_config.hrm_H_layers,
            L_layers=bridge_config.hrm_L_layers,
            hidden_size=bridge_config.hrm_hidden_size,
            expansion=bridge_config.hrm_expansion,
            num_heads=bridge_config.hrm_num_heads,
            pos_encodings=bridge_config.hrm_pos_encodings,
            rms_norm_eps=bridge_config.hrm_rms_norm_eps,
            rope_theta=bridge_config.hrm_rope_theta,
            halt_max_steps=bridge_config.hrm_halt_max_steps,
            halt_exploration_prob=bridge_config.hrm_halt_exploration_prob,
            forward_dtype=bridge_config.hrm_forward_dtype,
        )

        return HierarchicalReasoningModel_ACTV1_Inner(hrm_config)

    def setup_tokenizer(self, tokenizer):
        """Add the <|reason|> token and configure the router."""
        reason_id = add_reason_token(tokenizer)
        self.router.set_reason_token_id(reason_id)

        if not hasattr(self.backbone, "lm_head"):
            return

        # ── Weight Surgery ────────────────────────────────────────────────────
        # Ensure backbone covers the new token ID
        if reason_id >= self.backbone.config.vocab_size:
            self.backbone.resize_token_embeddings(reason_id + 1)

        backbone_weight    = self.backbone.lm_head.weight.data   # [backbone_vocab, H]
        new_vocab_size     = backbone_weight.shape[0]
        current_head_vocab = self.stablemax_head.linear.weight.shape[0]

        if new_vocab_size != current_head_vocab:
            # Vocab grew: rebuild head, copy old rows, seed new rows with mean.
            hidden_size = self.config.teacher.hidden_size
            new_head = StableMaxLMHead(hidden_size, new_vocab_size).to(
                device=backbone_weight.device, dtype=backbone_weight.dtype
            )
            with torch.no_grad():
                copy_rows = min(current_head_vocab, new_vocab_size)
                new_head.linear.weight[:copy_rows].copy_(
                    self.stablemax_head.linear.weight[:copy_rows]
                )
                if new_vocab_size > current_head_vocab:
                    # Seed new token rows (e.g. <|reason|>) to mean of existing embed
                    mean_embed = self.stablemax_head.linear.weight.mean(dim=0)
                    new_head.linear.weight[current_head_vocab:] = mean_embed
            self.stablemax_head = new_head
        else:
            # Vocab unchanged: just re-sync from backbone (idempotent)
            with torch.no_grad():
                self.stablemax_head.linear.weight.copy_(backbone_weight)

    def _hrm_forward(self, hidden_states: Tensor, carry: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Process hidden states through the HRM coprocessor.
        hidden_states: [B, L, H_valkyrie]
        carry: Optional Tuple of (z_H, z_L) from previous step
        returns: (output, new_carry)
        """
        B, L, H = hidden_states.shape
        hrm_dim = self.bridge_config.hrm_hidden_size

        # Project to HRM space: [B, L, H_hrm]
        hrm_input = self.bridge.to_hrm_space(hidden_states)

        # Initialize or use provided carry states
        if carry is not None:
            if isinstance(carry, (list, tuple)) and len(carry) == 2:
                z_H_prev, z_L_prev = carry
                z_H = z_H_prev
                z_L = z_L_prev
            else:
                # Robustness: if carry is invalid, log and reset to initial states
                # This protects against KV-Cache pollution from other layers
                z_H = self.hrm_inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1).clone()
                z_L = self.hrm_inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1).clone()
        else:
            z_H = self.hrm_inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1).clone()
            z_L = self.hrm_inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1).clone()

            
        seq_info = dict(
            cos_sin=self.hrm_inner.rotary_emb() if hasattr(self.hrm_inner, "rotary_emb") else None,
        )


        # Run HRM dual-loop computation
        H_cycles = self.bridge_config.hrm_H_cycles
        L_cycles = self.bridge_config.hrm_L_cycles

        # No-grad iterations (all but last)
        with torch.no_grad():
            for h_step in range(H_cycles):
                for l_step in range(L_cycles):
                    if not ((h_step == H_cycles - 1) and (l_step == L_cycles - 1)):
                        z_L = self.hrm_inner.L_level(z_L, z_H + hrm_input, **seq_info)

                if not (h_step == H_cycles - 1):
                    z_H = self.hrm_inner.H_level(z_H, z_L, **seq_info)

        # 1-step gradient iteration (last step only)
        z_L = self.hrm_inner.L_level(z_L, z_H + hrm_input, **seq_info)
        z_H = self.hrm_inner.H_level(z_H, z_L, **seq_info)

        # Project back to Valkyrie space: [B, L, H_valkyrie]
        output = self.bridge.from_hrm_space(z_H)
        
        # New carry holds the complete temporal sequence state
        new_carry = (z_H, z_L)

        # Q-learning Head as shown in the original code, using first token
        q_logits = self.hrm_inner.q_head(z_H[:, 0]).to(torch.float32)

        # Residual connection: add original hidden states
        return hidden_states + output, new_carry, float(H_cycles * L_cycles), (z_H, z_L), q_logits

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
        force_hrm: bool = False,
        hrm_carry: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> ValkyrieHRMOutput:
        """
        Combined forward pass.

        Args:
            input_ids: [B, L] token IDs
            labels: [B, L] target labels (optional)
            past_key_values: standard HF cache state (e.g. SinkCache)
            use_cache: whether to use KV caching
            force_hrm: bypass routing, always use HRM
        """
        # 1. Single backbone forward — returns pre-head hidden states
        # Provide attention_mask to handle SinkCache properly if passed
        backbone_out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=False, # last_hidden_state is always returned
        )
        h = backbone_out.last_hidden_state

        # 2. Routing (use compiled HRM forward for CUDA graph fusion)
        # Note: torch.compile handles carry if we are careful, but for now we bypass it if carry is used
        hrm_fn = self._hrm_forward # Bypass compile for stateful calls to be safe
        
        # hrm_carry is passed explicitly to avoid conflicts with HF text_model past_key_values


        if force_hrm or self.router.compute_routing_mask(input_ids).any():
            # If HRM is triggered, we run it and update carry
            res, new_carry, act_steps, hrm_states_tuple, q_logits = hrm_fn(h, carry=hrm_carry)
            route_mask = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            
            # Carry is returned in outputs natively, no need to hack past_key_values
            pass

            
            # Metric: HRM Change Delta
            with torch.no_grad():
                # L2 norm of the difference, averaged over batch/seq
                hrm_delta = torch.norm(res - h, p=2, dim=-1).mean()
            
            routed_hidden = res
        else:
            # Standard path
            routed_hidden = h
            route_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            act_steps = 0.0
            hrm_delta = torch.tensor(0.0, device=h.device)
            hrm_states_tuple = None
            q_logits = None


        # 3. LM Head (StableMax)
        logits = self.stablemax_head(routed_hidden)

        # 4. Loss
        loss = None
        if labels is not None:
            if self.bridge_config.use_stablemax:
                # StableMax cross-entropy
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.stablemax_head.compute_loss(
                    shift_logits.view(-1, logits.shape[-1]),
                    shift_labels.view(-1),
                )
            else:
                # Standard cross-entropy
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        return ValkyrieHRMOutput(
            logits=logits,
            loss=loss,
            route_mask=route_mask,
            avg_act_steps=torch.tensor(act_steps, device=h.device),
            hrm_delta=hrm_delta,
            backbone_hidden=h.detach(),
            new_carry=new_carry if (force_hrm or self.router.compute_routing_mask(input_ids).any()) else None,
            hrm_states=hrm_states_tuple,
            q_logits=q_logits,
        )

    def get_hrm_parameters(self) -> List[nn.Parameter]:
        """Get HRM + bridge parameters for Phase 5 training."""
        params = list(self.bridge.get_bridge_parameters())
        params.extend(self.hrm_inner.parameters())
        return params

    def get_s5_parameters(self) -> List[nn.Parameter]:
        """Deprecated: S5 parameters no longer exist."""
        return []

    def freeze_backbone(self):
        """Freeze entire Valkyrie backbone (for HRM training)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.stablemax_head.parameters():
            p.requires_grad = False

    def unfreeze_for_orpo(self):
        """Unfreeze bridge parameters for ORPO."""
        # Unfreeze bridge
        for p in self.bridge.get_bridge_parameters():
            p.requires_grad = True

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["backbone_total"] = sum(p.numel() for p in self.backbone.parameters())
        counts["bridge_proj_in"] = sum(p.numel() for p in self.bridge.proj_in.parameters())
        counts["bridge_proj_out"] = sum(p.numel() for p in self.bridge.proj_out.parameters())
        counts["hrm"] = sum(p.numel() for p in self.hrm_inner.parameters())
        counts["stablemax_head"] = sum(p.numel() for p in self.stablemax_head.parameters())
        counts["grand_total"] = sum(p.numel() for p in self.parameters())
        return counts
