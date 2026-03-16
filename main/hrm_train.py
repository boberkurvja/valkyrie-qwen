"""
Project Valkyrie — Phase 5: Deep Supervision Training (The Logic Engine)

Trains the HRM coprocessor directly in a non-autoregressive seq2seq fashion,
following the HRM paper exactly:
  - Input grid tokens → HRM embed_tokens → H/L cycles → HRM lm_head → output grid
  - Non-causal full attention (causal=False in HRM blocks)
  - Loss computed ONLY on output grid positions (labels=-100 on input)
  - 1-step gradient with state detachment between segments → O(1) memory
  - Adam-atan2 optimizer (scale-invariant, bounds weights via L∞ constraint)

The paper mandates:
  z^m = HRM(z^(m-1), x; θ)      # forward pass
  loss = Loss(ŷ^m, y)           # supervise at EVERY segment
  θ ← optimizer(∇_θ loss)       # update after EACH segment
  z^m = detach(z^m)             # cut graph for O(1) memory
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

from config import get_config, ValkyrieConfig
from valkyrie_hrm import ValkyrieHRM

# ── Adam-atan2 ────────────────────────────────────────────────────────────────
sys.path.insert(0, "/root/model/HRM")
try:
    from adam_atan2_pytorch import AdamAtan2 as AdamATan2
except ImportError:
    print("Warning: adam-atan2-pytorch not installed, falling back to AdamW")
    AdamATan2 = None

# ── HRM losses / layers ──────────────────────────────────────────────────────
from models.losses import stablemax_cross_entropy, IGNORE_LABEL_ID
from models.layers import CastedLinear, CastedEmbedding

# ── ARC Grid constants (must match build_arc_dataset.py) ─────────────────────
#   token 0  = PAD
#   token 1  = EOS
#   tokens 2-11 = digits 0-9
ARC_VOCAB_SIZE = 12    # 12 tokens: PAD, EOS, 0-9
ARC_GRID_SIZE  = 30    # max 30×30 grids  → seq_len = 900
ARC_SEQ_LEN    = ARC_GRID_SIZE * ARC_GRID_SIZE  # 900


# ─────────────────────────────────────────────────────────────────────────────
#  Lightweight ARC Dataset  (raw .json files from fchollet/ARC-AGI)
# ─────────────────────────────────────────────────────────────────────────────
import json, random
from torch.utils.data import Dataset, DataLoader

class ARCDataset(Dataset):
    """
    Loads ARC-AGI raw JSON puzzles and converts them to flat grid token sequences.
    Input  tokens: input-grid  → [0-11] in row-major order, padded to 900
    Output tokens: output-grid → [0-11] in row-major order, padded to 900
    Labels: -100 everywhere EXCEPT output grid positions (seq2seq masking).
    """

    def __init__(self, data_dir: str, split: str = "train", max_examples: Optional[int] = None):
        self.examples = []
        self._load(data_dir, split, max_examples)

    @staticmethod
    def _grid_to_seq(grid) -> np.ndarray:
        """Flatten 2D grid to 1D token array, encoding digit d as (d+2)."""
        rows = []
        for row in grid:
            for v in row:
                rows.append(int(v) + 2)              # digit d→token d+2
        arr = np.array(rows, dtype=np.int32)
        # Pad to ARC_SEQ_LEN with 0 (PAD)
        if len(arr) < ARC_SEQ_LEN:
            arr = np.concatenate([arr, np.zeros(ARC_SEQ_LEN - len(arr), dtype=np.int32)])
        else:
            arr = arr[:ARC_SEQ_LEN]
        return arr

    def _load(self, data_dir: str, split: str, max_examples: Optional[int]):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"ARC data directory not found: {data_dir}")

        puzzle_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".json")
        )

        rng = random.Random(42)
        rng.shuffle(puzzle_files)

        for fpath in puzzle_files:
            if max_examples and len(self.examples) >= max_examples:
                break
            try:
                puzzle = json.load(open(fpath))
            except Exception:
                continue

            pairs = puzzle.get(split, puzzle.get("train", []))
            for pair in pairs:
                if max_examples and len(self.examples) >= max_examples:
                    break
                inp  = pair.get("input", [])
                out  = pair.get("output", [])
                if not inp or not out:
                    continue

                inp_seq = self._grid_to_seq(inp)
                out_seq = self._grid_to_seq(out)

                # labels: -100 on input tokens, real tokens on output tokens
                # HRM takes [inputs] and predicts [labels] at same positions.
                labels = out_seq.copy()
                # Replace PAD positions in output with ignore
                labels[labels == 0] = IGNORE_LABEL_ID

                self.examples.append({
                    "inputs": torch.tensor(inp_seq, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                })

        print(f"  Loaded {len(self.examples)} {split} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  HRM Seq2Seq Trainer
# ─────────────────────────────────────────────────────────────────────────────
class HRMTrainer:
    """
    Deep Supervision trainer for the HRM coprocessor — paper-accurate.

    Key principles (from Section 2 of the HRM paper):
    1. Freeze Valkyrie backbone; train only HRM inner + bridge.
    2. Each training step = one deep-supervision SEGMENT:
         z^m, ŷ^m  = HRM_inner(z^(m-1), x)
         loss       = Loss(ŷ^m, y)
         optimizer.step()
         z^m        = z^m.detach()     ← O(1) memory
    3. The HRM uses its OWN embedding + LM head (ARC vocab=12), NOT Qwen.
    4. Non-autoregressive (no token shift), loss averaged over output tokens.
    5. Adam-atan2 keeps weights bounded (L∞ constraint → Q-learning stability).
    """

    def __init__(self, config: ValkyrieConfig, model: ValkyrieHRM, device: str = "cuda"):
        self.config     = config
        self.model      = model
        self.device     = device
        self.hrm_config = config.hrm_train

        # Freeze backbone, only train HRM + bridge
        self.model.freeze_backbone()

        # ── Bug 1 fix: replace lm_head and embed_tokens with ARC-vocab sizes ──
        # ValkyrieHRM._create_hrm_inner() uses the Qwen vocab_size (248,320),
        # but we only need ARC_VOCAB_SIZE=12 here.  The large random lm_head
        # causes float32 overflow in stablemax (248K terms) → NaN immediately.
        hrm_inner = self.model.hrm_inner
        arc_dim = hrm_inner.config.hidden_size
        hrm_inner.embed_tokens = CastedEmbedding(
            ARC_VOCAB_SIZE, arc_dim,
            init_std=1.0 / hrm_inner.embed_scale,
            cast_to=hrm_inner.forward_dtype,
        ).to(device)
        hrm_inner.lm_head = CastedLinear(arc_dim, ARC_VOCAB_SIZE, bias=False).to(device)
        print(f"  HRM lm_head replaced: {arc_dim} → {ARC_VOCAB_SIZE} (ARC vocab)")

        # Rebuild param list AFTER replacing the modules so the new weights are included
        hrm_params = self.model.get_hrm_parameters()
        for p in hrm_params:
            p.requires_grad_(True)

        trainable = sum(p.numel() for p in hrm_params if p.requires_grad)
        print(f"  Trainable parameters (HRM + Bridge): {trainable / 1e6:.2f}M")

        # Optimizer
        if AdamATan2 is not None:
            self.optimizer = AdamATan2(
                hrm_params,
                lr=self.hrm_config.learning_rate,
                weight_decay=self.hrm_config.weight_decay,
                betas=(self.hrm_config.beta1, self.hrm_config.beta2),
            )
            print("  Optimizer: Adam-atan2")
        else:
            self.optimizer = torch.optim.AdamW(
                hrm_params,
                lr=self.hrm_config.learning_rate,
                weight_decay=self.hrm_config.weight_decay,
                betas=(self.hrm_config.beta1, self.hrm_config.beta2),
            )
            print("  Optimizer: AdamW (fallback)")

        self.global_step = 0

    # ── LR schedule: linear warmup then constant ──────────────────────────────
    def _update_lr(self) -> float:
        warmup = self.hrm_config.warmup_steps
        step   = self.global_step
        lr     = self.hrm_config.learning_rate * min(1.0, step / max(1, warmup))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    # ── One segment forward + backward ───────────────────────────────────────
    def _segment_step(
        self,
        hrm,
        inp_seq: torch.Tensor,    # [B, L]  ARC grid tokens (0-11)
        labels:  torch.Tensor,    # [B, L]  target tokens or -100
        z_H:     Optional[torch.Tensor],
        z_L:     Optional[torch.Tensor],
    ) -> Dict:
        """
        One deep-supervision segment following the paper pseudocode.

        Forward path (paper Figure 4):
          x̃ = embed_scale * embed_tokens(input)
          [no-grad] all H×L cycles except the very last step of each module
          [grad]    last L-step: z_L = L_level(z_L, z_H + x̃)
          [grad]    last H-step: z_H = H_level(z_H, z_L)
          ŷ  = lm_head(z_H)            ← full-sequence, non-causal
          loss = mean(stablemax_CE(ŷ, y))   only where labels != -100
        """
        B, L = inp_seq.shape
        hrm_inner = hrm  # model.hrm_inner

        # Rotary positional encoding (pre-computed, cos_sin)
        seq_info = dict(
            cos_sin=hrm_inner.rotary_emb() if hasattr(hrm_inner, "rotary_emb") else None,
        )

        # ── Bug 3 fix: compute embedding OUTSIDE no_grad so embed_tokens gets grads ──
        # The paper's 1-step gradient approximation only stops grads through the
        # recurrent carry (z_H / z_L), NOT through the input embedding.  We pass a
        # detached copy into the convergence loop and the live tensor to the 1-step.
        x_emb = hrm_inner.embed_scale * hrm_inner.embed_tokens(
            inp_seq.to(torch.int32)
        )  # [B, L, H]  — full gradient kept
        x_emb_sg = x_emb.detach()  # stop-gradient copy for convergence loop

        # Initialise states if first segment
        dtype = hrm_inner.forward_dtype
        if z_H is None:
            z_H = hrm_inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1).clone().to(dtype)
            z_L = hrm_inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1).clone().to(dtype)

        H_cycles = hrm_inner.config.H_cycles
        L_cycles = hrm_inner.config.L_cycles

        # ── No-grad convergence iterations (carry convergence only) ───────────
        with torch.no_grad():
            for h in range(H_cycles):
                for l in range(L_cycles):
                    if not (h == H_cycles - 1 and l == L_cycles - 1):
                        z_L = hrm_inner.L_level(z_L, z_H + x_emb_sg, **seq_info)
                if h < H_cycles - 1:
                    z_H = hrm_inner.H_level(z_H, z_L, **seq_info)

        # ── 1-step gradient (last iteration — full grad through x_emb) ────────
        z_L = hrm_inner.L_level(z_L, z_H + x_emb, **seq_info)
        z_H = hrm_inner.H_level(z_H, z_L, **seq_info)

        # ── Outputs ──────────────────────────────────────────────────────────
        # lm_head maps H → ARC_VOCAB_SIZE (12) — NOT Qwen's 150K vocab
        logits = hrm_inner.lm_head(z_H).to(torch.float32)   # [B, L, 12]
        q_logits = hrm_inner.q_head(z_H[:, 0]).to(torch.float32)  # [B, 2]

        # ── Loss: seq2seq, NO token shifting (non-autoregressive) ───────────
        lm_loss = stablemax_cross_entropy(
            logits.view(-1, logits.shape[-1]),   # [B*L, 12]
            labels.view(-1),                      # [B*L]
            ignore_index=IGNORE_LABEL_ID,
        )
        # average only over non-masked positions
        valid_mask = labels.view(-1) != IGNORE_LABEL_ID
        num_valid  = valid_mask.sum().clamp(min=1)
        lm_loss    = lm_loss.sum() / num_valid

        # ── Q-head loss (ACT) ──────────────────────────────────────────────
        with torch.no_grad():
            mask        = labels != IGNORE_LABEL_ID
            is_correct  = mask & (torch.argmax(logits, dim=-1) == labels)
            loss_counts = mask.sum(-1)
            seq_correct = is_correct.sum(-1) == loss_counts

        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_logits[:, 0],
            seq_correct.float(),
            reduction="mean",
        )

        total_loss = lm_loss + 0.5 * q_halt_loss

        # Detach states for next segment — this is the O(1) memory trick
        new_z_H = z_H.detach()
        new_z_L = z_L.detach()

        with torch.no_grad():
            accuracy    = is_correct.float().mean()
            exact_acc   = seq_correct.float().mean()

        return {
            "loss":           total_loss,
            "lm_loss":        lm_loss.detach(),
            "q_halt_loss":    q_halt_loss.detach(),
            "accuracy":       accuracy,
            "exact_accuracy": exact_acc,
            "z_H":            new_z_H,
            "z_L":            new_z_L,
            "q_halt":         q_logits[:, 0].detach(),
            "q_continue":     q_logits[:, 1].detach(),
        }

    # ── Training loop ─────────────────────────────────────────────────────────
    def train(self):
        cfg = self.hrm_config

        print("=" * 60)
        print("Phase 5: Deep Supervision Training (Logic Engine)")
        print("=" * 60)
        print(f"  Learning rate:  {cfg.learning_rate}")
        print(f"  Batch size:     {cfg.batch_size}")
        print(f"  M_max (ACT):    {cfg.M_max}")
        print(f"  Seq len:        {ARC_SEQ_LEN}")

        # ── Dataset / DataLoader ───────────────────────────────────────────
        dataset = ARCDataset(
            data_dir=cfg.logic_dataset_path,
            split   ="train",
            max_examples=cfg.num_examples,
        )
        loader = DataLoader(
            dataset,
            batch_size =cfg.batch_size,
            shuffle    =True,
            num_workers=2,
            pin_memory =True,
            drop_last  =True,
        )

        hrm = self.model.hrm_inner

        try:
            from tqdm import tqdm
            total = cfg.num_epochs * len(loader)
            progress = tqdm(total=total, desc="HRM Training")
        except ImportError:
            progress = None

        self.model.train()

        for epoch in range(cfg.num_epochs):
            for batch in loader:
                inp_seq = batch["inputs"].to(self.device)   # [B, 900]
                labels  = batch["labels"].to(self.device)   # [B, 900]

                # ── Deep supervision: M segments, each with its own backward ─
                z_H = z_L = None
                last_result = None

                for seg in range(cfg.M_max):
                    self.optimizer.zero_grad()

                    result = self._segment_step(hrm, inp_seq, labels, z_H, z_L)
                    result["loss"].backward()

                    # ── NaN guard: skip optimizer step on NaN loss ─────────────
                    # NaN in the loss or gradients means the states are already
                    # corrupted; applying the optimizer step would make it worse.
                    # Detach and move on — the next batch will recover.
                    if result["loss"].isnan().any():
                        print(f"  [NaN guard] step={self.global_step} seg={seg}: "
                              f"NaN loss detected, skipping optimizer step.")
                        self.optimizer.zero_grad()  # clear poisoned grads
                        z_H = z_L = None            # reset carry for next batch
                        break

                    nn.utils.clip_grad_norm_(self.model.get_hrm_parameters(), 1.0)

                    lr = self._update_lr()
                    self.optimizer.step()

                    z_H = result["z_H"]
                    z_L = result["z_L"]
                    last_result = result
                    self.global_step += 1

                    # Early halting via ACT
                    if seg >= cfg.M_min - 1:
                        with torch.no_grad():
                            if (result["q_halt"] > result["q_continue"]).all():
                                break

                if progress and last_result:
                    progress.update(1)
                    progress.set_postfix_str(
                        f"loss={last_result['lm_loss']:.4f} "
                        f"acc={last_result['exact_accuracy']:.2%} "
                        f"lr={lr:.2e}"
                    )

                # Periodic checkpoint
                if self.global_step % cfg.save_steps == 0:
                    self._save_checkpoint()

        if progress:
            progress.close()
        self._save_checkpoint(final=True)
        print("\nDeep supervision training complete!")

    def _save_checkpoint(self, final: bool = False):
        cfg = self.hrm_config
        os.makedirs(cfg.output_dir, exist_ok=True)
        suffix = "final" if final else f"step_{self.global_step}"
        path   = os.path.join(cfg.output_dir, f"hrm_{suffix}.pt")
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step":          self.global_step,
        }, path)
        print(f"  Checkpoint saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HRM deep supervision training (paper-accurate)")
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = get_config()

    model = ValkyrieHRM(config).to(args.device)
    if os.path.exists(args.model_checkpoint):
        ckpt = torch.load(args.model_checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint: {args.model_checkpoint}")

    trainer = HRMTrainer(config, model, device=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
