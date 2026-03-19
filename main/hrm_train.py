"""
Project Valkyrie — Phase 5: Deep Supervision Training (The Logic Engine)

Trains the HRM coprocessor directly in a non-autoregressive seq2seq fashion,
following the HRM paper exactly:
  - Input grid tokens → HRM embed_tokens → H/L cycles → HRM lm_head → output grid
  - Non-causal full attention (causal=False in HRM blocks)
  - Loss computed ONLY on output grid positions (labels=-100 on input)
  - 1-step gradient with state detachment between segments → O(1) memory
  - Adam-atan2 optimizer (scale-invariant, bounds weights via L∞ constraint)
"""
import os
import sys
import math
import argparse
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
            labels[labels == self.tokenizer.pad_token_id] = -100 # Ignore padding in loss

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
    def __init__(self, config: ValkyrieConfig, model: ValkyrieHRM, device: str = "cuda", is_ddp: bool = False, local_rank: int = 0):
        self.config     = config
        self.model      = model
        self.device     = device
        self.is_ddp     = is_ddp
        self.local_rank = local_rank
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
        def safe_hrm_forward(hidden_states):
            orig_dtype = hidden_states.dtype
            out_fp32 = orig_hrm_forward(hidden_states.to(torch.float32))
            return out_fp32.to(orig_dtype)
            
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
                if "hrm_inner.lm_head" in name or "hrm_inner.embed_tokens" in name:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
                    self.trainable_params.append(p)
            else:
                p.requires_grad_(False)

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
                total = cfg.num_epochs * len(loader)
                progress = tqdm(total=total, desc="Autoregressive Training")
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

                self.optimizer.zero_grad()
                
                inp_seq = batch["inputs"].to(self.device)   # [B, Seq_Len]
                labels  = batch["labels"].to(self.device)   # [B, Seq_Len]

                # Full forward pass through ValkyrieHRM natively
                outputs = raw_model(input_ids=inp_seq, labels=labels)
                loss = outputs.loss

                # --- 1. LOSS NAN GUARD ---
                is_loss_nan = torch.tensor(float(loss.isnan().any()), device=self.device)
                if self.is_ddp:
                    dist.all_reduce(is_loss_nan, op=dist.ReduceOp.MAX)
                
                if is_loss_nan.item() > 0:
                    if self.local_rank == 0:
                        msg = f"[Step {self.global_step}] ⚠️ NaN loss detected in forward pass. Skipping batch."
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

                # --- 3. DETAILED PER-STEP TELEMETRY LOGGING ---
                if self.local_rank == 0:
                    routed_count = outputs.route_mask.sum().item() if outputs.route_mask is not None else 0
                    
                    log_str = (
                        f"[Step {self.global_step:04d}] "
                        f"Loss: {loss.item():.4f} | "
                        f"Pre-Clip Norm: {pre_clip_norm:8.2f} | "
                        f"Max Grad: {max_grad_val:8.2f} | "
                        f"NaN Tensors: {local_nan_count:2d} | "
                        f"Routed: {routed_count}"
                    )
                    
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


# ─────────────────────────────────────────────────────────────────────────────
def main():
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Autoregressive Curated training (Combined Model)")
    parser.add_argument("--model-checkpoint", type=str, required=False, default="")
    parser.add_argument("--device", type=str, default="cuda")
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

    trainer = HRMTrainer(config, model, device=device, is_ddp=is_ddp, local_rank=local_rank)
    
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