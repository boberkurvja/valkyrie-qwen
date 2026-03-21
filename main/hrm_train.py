"""
Project Valkyrie — Phase 5: Deep Supervision Training (The Logic Engine)

Trains the HRM coprocessor directly in a non-autoregressive seq2seq fashion,
following the HRM paper exactly:
  - Input grid tokens → HRM embed_tokens → H/L cycles → HRM lm_head → output grid
  - Non-causal full attention (causal=False in HRM blocks)
  - Loss computed ONLY on output grid positions (labels=-100 on input)
  - 1-step gradient with state detachment between segments → O(1) memory
  - Adam-atan2 optimizer (scale-invariant, bounds weights via L∞ constraint)

Auto-upload:
  After each checkpoint, the .pt file + a per-checkpoint training log are
  uploaded to HuggingFace Hub (leonidas123/valkyrie-v1), then the local .pt
  is deleted to free disk space.
  Pass --hf-upload to enable; set HF_TOKEN env var or --hf-token.
"""
import io
import os
import sys
import math
import argparse
import datetime
from typing import Optional, Dict
import torch._dynamo
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Multi-GPU imports
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from config import get_config, ValkyrieConfig
from valkyrie_hrm import ValkyrieHRM

# ── Adam-atan2 ────────────────────────────────────────────────────────────────
sys.path.insert(0, "/root/model/HRM")
try:
    from adam_atan2_pytorch import AdamAtan2 as AdamATan2
except ImportError:
    print("Warning: adam-atan2-pytorch not installed, falling back to AdamW")
    AdamATan2 = None

from models.losses import stablemax_cross_entropy, IGNORE_LABEL_ID

# ── HuggingFace Auto-Upload ───────────────────────────────────────────────────
try:
    from huggingface_hub import HfApi, CommitOperationAdd
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class LogBuffer:
    """
    Accumulates per-step log strings between checkpoints so they can be
    bundled as a .log file and uploaded alongside the .pt checkpoint.
    """
    def __init__(self):
        self._lines: list[str] = []
        self._start_step: int = 0

    def write(self, line: str):
        """Append a log line (called wherever progress.write / print is used)."""
        self._lines.append(line)

    def flush_to_str(self, step: int) -> str:
        """Return accumulated log as a string and reset the buffer."""
        header = (
            f"=== Valkyrie HRM Training Log ===\n"
            f"Checkpoint step : {step}\n"
            f"Generated at    : {datetime.datetime.utcnow().isoformat()} UTC\n"
            f"Steps in buffer : {self._start_step} → {step}\n"
            f"{'=' * 40}\n"
        )
        body = "\n".join(self._lines)
        self._lines = []
        self._start_step = step
        return header + body


class HFUploader:
    """
    Uploads checkpoint .pt files and companion .log files to a HuggingFace
    model repository, then optionally deletes the local .pt.

    Usage:
        uploader = HFUploader(repo_id="leonidas123/valkyrie-v1", token="hf_...")
        uploader.upload_checkpoint("/path/to/hrm_step_5000.pt", log_text, delete_local=True)
    """

    def __init__(self, repo_id: str, token: str = None):
        if not _HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub is not installed. "
                "Run: python -m pip install huggingface_hub"
            )
        self.repo_id = repo_id
        self.api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        # Ensure the repo exists (creates it as private model repo if not)
        try:
            self.api.repo_info(repo_id=repo_id, repo_type="model")
        except Exception:
            print(f"  [HF] Creating repo {repo_id} on HuggingFace...")
            self.api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

    def upload_checkpoint(
        self,
        local_path: str,
        log_text: str = "",
        delete_local: bool = True,
        subfolder: str = "checkpoints",
    ) -> bool:
        """
        Upload `local_path` (.pt) and a companion .log to HuggingFace.
        Deletes the local .pt on success if delete_local=True.
        Returns True on success, False on failure.
        """
        basename = os.path.basename(local_path)            # hrm_step_5000.pt
        stem     = os.path.splitext(basename)[0]           # hrm_step_5000
        pt_dest  = f"{subfolder}/{basename}"               # checkpoints/hrm_step_5000.pt
        log_dest = f"{subfolder}/{stem}.log"               # checkpoints/hrm_step_5000.log

        print(f"  [HF] Uploading {basename} → {self.repo_id}/{pt_dest} ...")
        try:
            operations = [
                CommitOperationAdd(
                    path_in_repo=pt_dest,
                    path_or_fileobj=local_path,
                ),
                CommitOperationAdd(
                    path_in_repo=log_dest,
                    path_or_fileobj=log_text.encode("utf-8"),
                ),
            ]
            self.api.create_commit(
                repo_id=self.repo_id,
                repo_type="model",
                operations=operations,
                commit_message=f"Add checkpoint {stem}",
            )
            print(f"  [HF] ✓ Upload complete: {self.repo_id}/{pt_dest}")

            if delete_local and os.path.exists(local_path):
                os.remove(local_path)
                print(f"  [HF] ✓ Deleted local checkpoint: {local_path}")

            return True

        except Exception as e:
            print(f"  [HF] ✗ Upload failed: {e}")
            print(f"  [HF]   Local checkpoint preserved at: {local_path}")
            return False
from models.layers import CastedLinear, CastedEmbedding

# ─────────────────────────────────────────────────────────────────────────────
#  Curated Autoregressive Dataset (Local Disk)
# ─────────────────────────────────────────────────────────────────────────────
import json, random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from routing import REASON_TOKEN

class CuratedReasoningDataset(Dataset):
    """
    Loads a curated reasoning dataset from local disk where the <|reason|>
    token has already been inserted perfectly before logical conclusions.
    """
    def __init__(self, data_dir: str, tokenizer: AutoTokenizer, seq_len: int = 2048, max_examples: Optional[int] = None):
        self.examples = []
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        # Ensure the reason token is in the tokenizer
        if REASON_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [REASON_TOKEN]})
            
        self.reason_token_id = self.tokenizer.convert_tokens_to_ids(REASON_TOKEN)
        self._load(data_dir, max_examples)

    def _load(self, data_dir: str, max_examples: Optional[int]):
        from datasets import load_from_disk
        
        dataset = load_from_disk(data_dir)
        
        for i, row in enumerate(dataset):
            if max_examples and len(self.examples) >= max_examples:
                break
                
            text = row["text"]
            
            # Tokenize text (The <|reason|> tokens are already in the string)
            tokens = self.tokenizer(text, truncation=True, max_length=self.seq_len, return_tensors="pt")["input_ids"].squeeze(0)

            # Pad sequence if necessary
            if len(tokens) < self.seq_len:
                pad_len = self.seq_len - len(tokens)
                tokens = torch.cat([tokens, torch.full((pad_len,), self.tokenizer.pad_token_id)])

            # For Causal LM, labels are exactly the input_ids. 
            labels = tokens.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            # --- 1. PROMPT MASKING (Valkyrie MoS Stability) ---
            # Mask all tokens before <|reason|> to focus HRM on reasoning logic only.
            reason_indices = (tokens == self.reason_token_id).nonzero(as_tuple=True)[0]
            if len(reason_indices) > 0:
                labels[:reason_indices[0] + 1] = -100

            self.examples.append({
                "inputs": tokens.to(torch.long),
                "labels": labels.to(torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  HRM Trainer
# ─────────────────────────────────────────────────────────────────────────────
class HRMTrainer:
    def __init__(
        self,
        config: ValkyrieConfig,
        model: ValkyrieHRM,
        device: str = "cuda",
        is_ddp: bool = False,
        local_rank: int = 0,
        hf_uploader: "HFUploader" = None,
    ):
        self.config      = config
        self.model       = model
        self.device      = device
        self.is_ddp      = is_ddp
        self.local_rank  = local_rank
        self.hf_uploader = hf_uploader
        self.log_buffer  = LogBuffer()
        self.hrm_config = config.hrm_train

        # 1. Freeze everything by default
        self.model.freeze_backbone()

        # =====================================================================
        # ULTIMATE NUMERIC STABILITY PATCH (FP16 OVERFLOW FIX)
        # =====================================================================
        self.model.bridge.to(torch.float32)
        self.model.hrm_inner.to(torch.float32)
        self.model.stablemax_head.to(torch.float32)

        # Patch HRM Forward to upcast inputs and downcast outputs
        orig_hrm_forward = self.model._hrm_forward
        def safe_hrm_forward(hidden_states, carry=None):
            orig_dtype = hidden_states.dtype
            out_fp32, new_carry, act_steps, hrm_states_tuple, q_logits = orig_hrm_forward(hidden_states.to(torch.float32), carry=carry)
            return out_fp32.to(orig_dtype), new_carry, act_steps, hrm_states_tuple, q_logits
            
        self.model._hrm_forward = safe_hrm_forward


        
        # Re-enable torch.compile on the safe FP32 wrapper
        torch._dynamo.config.suppress_errors = True
        self.model._compiled_hrm_forward = torch.compile(safe_hrm_forward, mode="default")

        # Patch StableMax Forward to upcast inputs
        orig_stablemax = self.model.stablemax_head.forward
        def safe_stablemax(hidden_states):
            return orig_stablemax(hidden_states.to(torch.float32))
        self.model.stablemax_head.forward = safe_stablemax
        # =====================================================================

        # 2. Extract strictly necessary parameters using Python id()
        self.trainable_params = []
        hrm_param_ids = {id(p) for p in self.model.get_hrm_parameters()}

        for name, p in self.model.named_parameters():
            if id(p) in hrm_param_ids:
                # Filter out the unused giant token layers
                # UNFREEZE hrm_inner.lm_head to allow trainable influence on vocabulary
                if "hrm_inner.embed_tokens" in name:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
                    self.trainable_params.append(p)
            else:
                p.requires_grad_(False)

        # stablemax_head stays FROZEN (Option A):
        # ProjOut is zero-init, so at step 0 the HRM contributes ~0 and the frozen
        # Qwen-initialized head is already correct. The HRM must learn to output
        # residuals that live in Qwen's existing token embedding space.
        # Unfreezing would add ~253M trainable params for marginal gain at this stage.

        trainable = sum(p.numel() for p in self.trainable_params)

        
        # 3. Optimizer
        if AdamATan2 is not None:
            self.optimizer = AdamATan2(
                self.trainable_params,
                lr=self.hrm_config.learning_rate,
                weight_decay=self.hrm_config.weight_decay,
                betas=(self.hrm_config.beta1, self.hrm_config.beta2),
            )
            optim_str = "Adam-atan2"
        else:
            self.optimizer = torch.optim.AdamW(
                self.trainable_params,
                lr=self.hrm_config.learning_rate,
                weight_decay=self.hrm_config.weight_decay,
                betas=(self.hrm_config.beta1, self.hrm_config.beta2),
            )
            optim_str = "AdamW (fallback)"

        if self.local_rank == 0:
            print(f"  Trainable parameters (HRM + Bridge): {trainable / 1e6:.2f}M")
            print(f"  Optimizer: {optim_str}")

        self.global_step = 0

    def _update_lr(self) -> float:
        warmup = self.hrm_config.warmup_steps
        step   = self.global_step
        lr     = self.hrm_config.learning_rate * min(1.0, step / max(1, warmup))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def train(self, tokenizer):
        cfg = self.hrm_config

        if self.local_rank == 0:
            print("=" * 60)
            print("Autoregressive Curated Training (Combined Model) - Stable FP32")
            print("=" * 60)
        
        self.model.setup_tokenizer(tokenizer)

        # --- INITIAL WEIGHT SYNC ---
        if self.is_ddp:
            with torch.no_grad():
                for p in self.trainable_params:
                    dist.broadcast(p.data, src=0)

        raw_model = self.model

        if self.local_rank == 0:
            print("Loading Curated Valkyrie dataset from local disk...")

        # Pointing to your local dataset path
        dataset = CuratedReasoningDataset(
            data_dir="/root/valkyrie_reasoning_data",
            tokenizer=tokenizer,
            seq_len=cfg.seq_len,
            max_examples=cfg.num_examples,
        )
        
        if self.local_rank == 0:
            print(f"  Loaded {len(dataset)} Causal LM examples from disk")

        # ── Distributed Sampler ────────────────────────────────────────────
        sampler = DistributedSampler(dataset) if self.is_ddp else None
        loader = DataLoader(
            dataset,
            batch_size =cfg.batch_size,
            shuffle    =(sampler is None),
            sampler    =sampler,
            num_workers=2,
            pin_memory =True,
            drop_last  =True,
        )

        if self.local_rank == 0:
            try:
                from tqdm import tqdm
                
                # Calculate total steps, accounting for max_steps if you set it
                total_steps = cfg.num_epochs * len(loader)
                if hasattr(cfg, 'max_steps'):
                    total_steps = min(total_steps, cfg.max_steps)
                
                # Set 'initial' so the bar visually starts at your resumed step
                progress = tqdm(
                    total=total_steps, 
                    initial=self.global_step, 
                    desc="Autoregressive Training"
                )
            except ImportError:
                progress = None
        else:
            progress = None

        raw_model.train()

        for epoch in range(cfg.num_epochs):
            if self.is_ddp:
                sampler.set_epoch(epoch)

            for batch in loader:
                # --- EXPLICIT STEP CAP ---
                if hasattr(cfg, 'max_steps') and self.global_step >= cfg.max_steps:
                    if self.local_rank == 0:
                        msg = f"\n[Step {self.global_step}] Reached max_steps ({cfg.max_steps}). Halting to prevent late-stage overfitting."
                        if progress: progress.write(msg)
                        else: print(msg)
                    break 

                inp_seq = batch["inputs"].to(self.device)   # [B, Seq_Len]
                labels  = batch["labels"].to(self.device)   # [B, Seq_Len]
                b_sz = inp_seq.shape[0]

                # --- CURRICULUM UNROLLING ---
                # Slowly scale from 2 thoughts to 16 thoughts over 4000 steps
                # This helps the Bridge mapping stabilize before deep iteration.
                base_M = getattr(cfg, 'M_min', 2)
                target_M = getattr(cfg, 'M_max', 16)
                curr_M_max = int(min(target_M, base_M + (self.global_step / 4000) * (target_M - base_M)))
                
                # Stochastic M_min (exploration)
                epsilon = getattr(cfg, 'halt_exploration_prob', 0.1)
                if torch.rand(1).item() < epsilon:
                    M_min = torch.randint(base_M, target_M + 1, (1,)).item()
                else:
                    M_min = 1

                hrm_carry = None
                halted = torch.zeros(b_sz, dtype=torch.bool, device=self.device)
                # Track effective steps for ACT metric
                seq_steps = torch.zeros(b_sz, dtype=torch.float32, device=self.device)

                for segment in range(1, curr_M_max + 1):
                    if halted.all():
                        break

                    pr_h, pr_l = 0.0, 0.0

                    self.optimizer.zero_grad()

                    # Full forward pass through ValkyrieHRM natively
                    outputs = raw_model(input_ids=inp_seq, labels=labels, hrm_carry=hrm_carry)
                    # Extract native Cross-Entropy loss from the forward pass
                    base_ce_loss = outputs.loss
                    
                    # 1. TIME-DISCOUNTED CURRICULUM LOSS (Quadratic Penalty)
                    # We penalize early "fast" thoughts less than final "deliberative" results.
                    segment_weight = (segment / curr_M_max) ** 2
                    discounted_loss = base_ce_loss * segment_weight
                    
                    loss = discounted_loss

                    # Compute Q-learning targets & loss
                    if outputs.q_logits is not None:
                        q_halt_logits = outputs.q_logits[..., 0]
                        q_continue_logits = outputs.q_logits[..., 1]
                        
                        with torch.no_grad():
                            shift_logits = outputs.logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            mask = shift_labels != -100
                            # Per-sequence correctness (all tokens match)
                            preds = shift_logits.argmax(dim=-1)
                            is_correct = ((preds == shift_labels) | ~mask).all(dim=-1).float()
                            
                            target_q_halt = is_correct
                            # target_q_continue should strictly reflect if continuing leads to success.
                            # Usually, if we are not correct yet and not at M_max, continue is valid.
                            target_q_continue = (1.0 - is_correct)

                        # Q-Loss computation ensures no NaN leaks
                        q_loss_halt = F.binary_cross_entropy_with_logits(q_halt_logits, target_q_halt)
                        q_loss_cont = F.binary_cross_entropy_with_logits(q_continue_logits, target_q_continue)
                        # We mask out continue target loss at terminal segment because continuing is impossible.
                        if segment == curr_M_max:
                            q_loss = q_loss_halt
                        else:
                            q_loss = q_loss_halt + q_loss_cont
                            
                        loss = loss + q_loss

                    # --- 1. RANK DIVERSITY & DECORRELATION LOSS (Prevent Neural Collapse) ---
                    rank_loss = torch.tensor(0.0, device=self.device)
                    if getattr(outputs, "hrm_states", None) is not None:
                        z_H, z_L = outputs.hrm_states
                        
                        def calc_pr_and_decorr(x, target_pr):
                            # Flatten batch and sequence dims
                            x_flat = x.view(-1, x.shape[-1]).float()
                            
                            # 1. Variance Loss (Hinge loss to ensure variance is at least 1.0 per dimension)
                            # This prevents the representation from shrinking to zero.
                            std = torch.sqrt(x_flat.var(dim=0) + 1e-4)
                            var_loss = torch.mean(F.relu(1.0 - std))
                            
                            # Mean center for Covariance
                            x_centered = x_flat - x_flat.mean(dim=0)
                            
                            # Covariance matrix
                            cov = torch.mm(x_centered.T, x_centered) / max(1, x_centered.shape[0] - 1)
                            
                            # 2. Decorrelation Loss (Minimize off-diagonal elements)
                            # This forces the features to become orthogonal, expanding the rank.
                            off_diag = cov - torch.diag(torch.diag(cov))
                            decorr_loss = (off_diag ** 2).mean()
                            
                            # 3. Participation Ratio (for telemetry only, or light gradients)
                            tr_cov = torch.trace(cov)
                            tr_cov_sq = torch.trace(torch.mm(cov, cov))
                            pr_val = (tr_cov**2) / (tr_cov_sq.detach() + 1e-6)
                            
                            # Combine losses: Heavily penalize low variance and high correlation
                            collapse_loss = var_loss + (10.0 * decorr_loss)
                            
                            # Add the target PR hinge
                            pr_loss = F.relu(target_pr - pr_val)
                            
                            return pr_val.item(), collapse_loss + pr_loss

                        # We target PR_H ~ 90 and PR_L ~ 30
                        pr_h_val, collapse_loss_h = calc_pr_and_decorr(z_H, 90.0)
                        pr_l_val, collapse_loss_l = calc_pr_and_decorr(z_L, 30.0)
                        
                        # Scale this up slightly now that the base CE loss is stable
                        rank_loss = 0.05 * (collapse_loss_h + collapse_loss_l)
                        loss = loss + rank_loss
                        
                        # Update local telemetry variables for logging
                        pr_h = pr_h_val
                        pr_l = pr_l_val

                    # --- 1. LOSS NAN GUARD ---
                    is_loss_nan = torch.tensor(float(loss.isnan().any()), device=self.device)
                    if self.is_ddp:
                        dist.all_reduce(is_loss_nan, op=dist.ReduceOp.MAX)
                    
                    if is_loss_nan.item() > 0:
                        if self.local_rank == 0:
                            msg = f"[Step {self.global_step} Seg {segment}] ⚠️ NaN loss detected in forward pass. Skipping segment."
                            if progress:
                                progress.write(msg)
                            else:
                                print(msg)
                        self.optimizer.zero_grad() 
                        continue

                    # Native Backward Pass (No Hooks to crash)
                    if loss.requires_grad:
                        loss.backward()
                    
                    # --- 2. GRADIENT SANITIZATION & MANUAL DDP SYNC ---
                    if self.is_ddp:
                        world_size = dist.get_world_size()
                    
                    local_nan_count = 0    
                    for p in self.trainable_params:
                        # 1. Handle completely skipped paths
                        if p.grad is None:
                            if self.is_ddp:
                                p.grad = torch.zeros_like(p.data)
                            else:
                                continue
                                
                        # Track pre-sanitization NaNs locally for telemetry
                        if not p.grad.isfinite().all():
                            local_nan_count += 1
                                
                        # 2. SANITIZE: Safely clamp Inf/NaN
                        p.grad.data.nan_to_num_()
                        
                        # 3. Safe FP32 Reduction
                        if self.is_ddp:
                            grad_fp32 = p.grad.data.to(torch.float32)
                            dist.all_reduce(grad_fp32, op=dist.ReduceOp.SUM)
                            p.grad.data.copy_(grad_fp32 / world_size)

                    # Collect the maximum absolute gradient value across all parameters (post-sync)
                    max_grad_val = 0.0
                    for p in self.trainable_params:
                        if p.grad is not None:
                            curr_max = p.grad.data.abs().max().item()
                            if curr_max > max_grad_val:
                                max_grad_val = curr_max

                    # clip_grad_norm_ returns the total L2 norm BEFORE clipping is applied.
                    pre_clip_norm = nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                    if isinstance(pre_clip_norm, torch.Tensor):
                        pre_clip_norm = pre_clip_norm.item()

                    lr = self._update_lr()
                    self.optimizer.step()
                    self.global_step += 1
                    
                    # --- DETACH FOR 1-STEP GRADIENT (Section 2.1) ---
                    # We MUST detach everything to prevent computation graph accumulation
                    if getattr(outputs, "new_carry", None) is not None:
                        new_carry = outputs.new_carry
                        if isinstance(new_carry, tuple):
                            new_carry = tuple(c.detach() for c in new_carry)
                        elif isinstance(new_carry, torch.Tensor):
                            new_carry = new_carry.detach()
                        hrm_carry = new_carry
                        
                    # Halting Logic
                    with torch.no_grad():
                        if outputs.q_logits is not None:
                            # seq_steps tracks how many segments each sequence has been active for
                            seq_steps[~halted] += 1
                            should_halt = (q_halt_logits > q_continue_logits) & (segment >= M_min)
                            halted = halted | should_halt
                        # Terminate if we reach the curriculum limit
                        halted = halted | (segment >= curr_M_max)

                    # --- 3. DETAILED PER-STEP TELEMETRY LOGGING ---
                    if self.local_rank == 0:
                        with torch.no_grad():
                            ppl = math.exp(min(loss.item(), 20.0))
                            
                            shift_logits = outputs.logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            mask = shift_labels != -100
                            if mask.sum() > 0:
                                preds = shift_logits.argmax(dim=-1)
                                acc = (preds == shift_labels)[mask].float().mean().item()
                            else:
                                acc = 0.0

                            # Using precision PR values from the rank_loss calculation above
                            pass
                                    
                            # Telemetry strings will use pr_h and pr_l variables

                        routed_count = outputs.route_mask.sum().item() if outputs.route_mask is not None else 0
                        avg_act_segments = seq_steps.mean().item()
                        hrm_delta = outputs.hrm_delta.item() if outputs.hrm_delta is not None else 0.0
                        logit_mag = torch.mean(torch.abs(outputs.logits)).item()
                        
                        log_str = (
                            f"[Step {self.global_step:04d} Seg {segment:02d}/{curr_M_max}] "
                            f"Loss: {loss.item():7.4f} | "
                            f"PPL: {ppl:6.1f} | "
                            f"Acc: {acc:.2f} | "
                            f"Norm: {pre_clip_norm:6.2f} | "
                            f"HRM_Δ: {hrm_delta:6.4f} | "
                            f"ACT_M: {avg_act_segments:5.2f} | "
                            f"PR_H: {pr_h:5.1f} | "
                            f"PR_L: {pr_l:5.1f} | "
                            f"Logit: {logit_mag:6.2f} | "
                            f"Routed: {routed_count}"
                        )

                        
                        # Buffer every log line for the upcoming checkpoint upload
                        self.log_buffer.write(log_str)

                        if progress:
                            progress.write(log_str)
                            progress.update(1)
                            progress.set_postfix_str(f"loss={loss.item():.4f} lr={lr:.2e}")
                        else:
                            print(log_str)

                        if self.global_step % cfg.save_steps == 0:
                            self._save_checkpoint(raw_model)

            # Break the outer loop if max_steps was triggered
            if hasattr(cfg, 'max_steps') and self.global_step >= cfg.max_steps:
                break

        if self.local_rank == 0:
            if progress:
                progress.close()
            self._save_checkpoint(raw_model, final=True)
            print("\nAutoregressive training complete!")

    def _save_checkpoint(self, raw_model, final: bool = False):
        if self.local_rank != 0:
            return

        cfg = self.hrm_config
        os.makedirs(cfg.output_dir, exist_ok=True)
        suffix = "final" if final else f"step_{self.global_step}"
        path   = os.path.join(cfg.output_dir, f"hrm_{suffix}.pt")

        torch.save({
            "model_state_dict":     raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step":          self.global_step,
        }, path)
        print(f"  Checkpoint saved: {path}")

        # ── HuggingFace Auto-Upload ───────────────────────────────────────
        if self.hf_uploader is not None:
            log_text = self.log_buffer.flush_to_str(step=self.global_step)
            # Also write the log to disk next to the checkpoint (small, keep it)
            log_path = path.replace(".pt", ".log")
            with open(log_path, "w") as f:
                f.write(log_text)
            self.hf_uploader.upload_checkpoint(
                local_path=path,
                log_text=log_text,
                delete_local=True,   # free disk space after upload
            )


# ─────────────────────────────────────────────────────────────────────────────
def main():
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Autoregressive Curated training (Combined Model)")
    parser.add_argument("--model-checkpoint", type=str, required=False, default="")
    parser.add_argument("--device", type=str, default="cuda")
    # ── HuggingFace upload args ──────────────────────────────────────────────
    parser.add_argument(
        "--hf-upload", action="store_true",
        help="Upload each checkpoint to HuggingFace and delete the local .pt file."
    )
    parser.add_argument(
        "--hf-repo", type=str, default="leonidas123/valkyrie-v1",
        help="HuggingFace repo ID to push checkpoints to (default: leonidas123/valkyrie-v1)."
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace API token. Falls back to HF_TOKEN environment variable."
    )
    args = parser.parse_args()

    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = args.device

    config = get_config()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.teacher.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ValkyrieHRM(config).to(device)
    
    ckpt = None
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        ckpt = torch.load(args.model_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if local_rank == 0:
            print(f"Loaded checkpoint weights: {args.model_checkpoint}")

    # ── HuggingFace uploader (rank-0 only) ──────────────────────────────────
    hf_uploader = None
    if args.hf_upload and local_rank == 0:
        if not _HF_AVAILABLE:
            print("[WARN] --hf-upload requested but huggingface_hub is not installed. Skipping.")
            print("       Install with: python -m pip install huggingface_hub")
        else:
            token = args.hf_token or os.environ.get("HF_TOKEN")
            if not token:
                print("[WARN] --hf-upload: no HF token found. Set --hf-token or HF_TOKEN env var.")
            else:
                try:
                    hf_uploader = HFUploader(repo_id=args.hf_repo, token=token)
                    print(f"  [HF] Auto-upload enabled → {args.hf_repo}")
                except Exception as e:
                    print(f"[WARN] Failed to initialize HF uploader: {e}")

    trainer = HRMTrainer(
        config, model,
        device=device,
        is_ddp=is_ddp,
        local_rank=local_rank,
        hf_uploader=hf_uploader,
    )
    
    # --- ADD THIS BLOCK TO FULLY RESUME TRAINING STATE ---
    if ckpt is not None and "optimizer_state_dict" in ckpt:
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.global_step = ckpt.get("global_step", 0)
        if local_rank == 0:
            print(f"Resumed optimizer state and global step ({trainer.global_step})")
    # -----------------------------------------------------

    trainer.train(tokenizer=tokenizer)

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()