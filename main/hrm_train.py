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
#  FineWeb Autoregressive Dataset
# ─────────────────────────────────────────────────────────────────────────────
import json, random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from routing import REASON_TOKEN

class FineWebDataset(Dataset):
    """
    Loads general text data (e.g., FineWeb) and formats it for causal language modeling.
    Periodically injects the <|reason|> token to activate the HRM coprocessor.
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
        from datasets import load_dataset
        
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        
        for i, row in enumerate(dataset):
            if max_examples and len(self.examples) >= max_examples:
                break
                
            text = row["text"]
            
            # Tokenize text
            tokens = self.tokenizer(text, truncation=True, max_length=self.seq_len, return_tensors="pt")["input_ids"].squeeze(0)
            
            # Inject <|reason|> token randomly to train the router/HRM path
            if len(tokens) > 10 and random.random() > 0.5:
                insert_idx = random.randint(5, len(tokens) - 5)
                tokens = torch.cat([
                    tokens[:insert_idx], 
                    torch.tensor([self.reason_token_id]), 
                    tokens[insert_idx:]
                ])[:self.seq_len]

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

        # 2. Extract strictly necessary parameters using Python id() to prevent PyTorch broadcasting crashes
        self.trainable_params = []
        
        # Get the memory IDs of the parameters designated for training (Bridge + HRM)
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
            print("Autoregressive FineWeb Training (Combined Model) - Custom Multi-GPU")
            print("=" * 60)
        
        # Add tokenizer setup to initialize router token
        self.model.setup_tokenizer(tokenizer)

        # --- INITIAL WEIGHT SYNC ---
        # Ensure all GPUs start with the exact same initialized weights. 
        if self.is_ddp:
            with torch.no_grad():
                for p in self.trainable_params:
                    dist.broadcast(p.data, src=0)

        raw_model = self.model

        if self.local_rank == 0:
            print("Downloading/Loading FineWeb dataset...")

        dataset = FineWebDataset(
            data_dir=cfg.logic_dataset_path,
            tokenizer=tokenizer,
            seq_len=cfg.seq_len,
            max_examples=cfg.num_examples,
        )
        
        if self.local_rank == 0:
            print(f"  Loaded {len(dataset)} Causal LM examples from FineWeb")

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
                        print(f"\n  [NaN guard] step={self.global_step}: NaN loss detected in forward pass, skipping batch.")
                    self.optimizer.zero_grad() 
                    continue

                # Native Backward Pass (No Hooks to crash)
                if loss.requires_grad:
                    loss.backward()
                
                # --- 2. GRADIENT SANITIZATION & MANUAL DDP SYNC ---
                if self.is_ddp:
                    world_size = dist.get_world_size()
                    
                for p in self.trainable_params:
                    # 1. Handle completely skipped paths
                    if p.grad is None:
                        if self.is_ddp:
                            p.grad = torch.zeros_like(p.data)
                        else:
                            continue
                            
                    # 2. SANITIZE: Safely clamp Inf/NaN from float16 overflow to finite bounds
                    p.grad.data.nan_to_num_()
                    
                    # 3. Safe FP32 Reduction (Prevents float16 overflow when summing across GPUs)
                    if self.is_ddp:
                        grad_fp32 = p.grad.data.to(torch.float32)
                        dist.all_reduce(grad_fp32, op=dist.ReduceOp.SUM)
                        p.grad.data.copy_(grad_fp32 / world_size)

                # Now the gradients are guaranteed finite and synced. Safe to clip and step.
                nn.utils.clip_grad_norm_(self.trainable_params, 1.0)

                lr = self._update_lr()
                self.optimizer.step()
                self.global_step += 1

                if self.local_rank == 0:
                    if progress:
                        progress.update(1)
                        progress.set_postfix_str(
                            f"loss={outputs.loss.item():.4f} "
                            f"lr={lr:.2e} "
                            f"hrm_routed={outputs.route_mask.sum().item() if outputs.route_mask is not None else 0}"
                        )

                    # Periodic checkpoint
                    if self.global_step % cfg.save_steps == 0:
                        self._save_checkpoint(raw_model)

        if self.local_rank == 0:
            if progress:
                progress.close()
            self._save_checkpoint(raw_model, final=True)
            print("\nAutoregressive training complete!")

    def _save_checkpoint(self, raw_model, final: bool = False):
        if self.local_rank != 0: 
            return # Only rank 0 saves the model
            
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
    parser = argparse.ArgumentParser(description="Autoregressive FineWeb training (Combined Model)")
    parser.add_argument("--model-checkpoint", type=str, required=False, default="")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. Initialize DDP
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

    # 2. Initialize the Tokenizer from the teacher model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.teacher.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Initialize the Model
    model = ValkyrieHRM(config).to(device)
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        ckpt = torch.load(args.model_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if local_rank == 0:
            print(f"Loaded checkpoint: {args.model_checkpoint}")

    # 4. Initialize Trainer and pass the DDP variables
    trainer = HRMTrainer(config, model, device=device, is_ddp=is_ddp, local_rank=local_rank)
    trainer.train(tokenizer=tokenizer)

    # 5. Cleanup
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()