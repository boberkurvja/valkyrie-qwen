"""
Project Valkyrie — Phase 6: ORPO Alignment
Odds Ratio Preference Optimization using the trl library.
Unfreezes S5 layers and HRM bridges for alignment training.
"""
import os
import sys
import argparse
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from config import get_config, ValkyrieConfig


class ValkyrieORPOWrapper(nn.Module):
    """
    Wrapper to make ValkyrieHRM compatible with trl's ORPOTrainer.
    Adapts the interface to match HuggingFace's model API.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self._make_hf_config()

    def _make_hf_config(self):
        """Create a minimal HF-compatible config object."""
        from types import SimpleNamespace
        return SimpleNamespace(
            is_encoder_decoder=False,
            pad_token_id=0,
            eos_token_id=None,
            model_type="valkyrie",
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """HF-compatible forward pass."""
        from transformers.modeling_outputs import CausalLMOutputWithPast

        output = self.model(
            input_ids=input_ids,
            labels=labels,
        )

        return CausalLMOutputWithPast(
            loss=output.loss,
            logits=output.logits,
        )

    def generate(self, *args, **kwargs):
        """Placeholder for generation (not used during ORPO training)."""
        raise NotImplementedError("Use generate.py for generation")

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def get_input_embeddings(self):
        return self.model.backbone.embed_tokens


def run_orpo(config: ValkyrieConfig, model, tokenizer, device: str = "cuda"):
    """
    Run ORPO alignment training.

    ORPO maximizes the likelihood of chosen responses while penalizing
    rejected responses, without needing a separate reward model.
    """
    from datasets import load_dataset
    from trl import ORPOConfig as TRLORPOConfig, ORPOTrainer

    cfg = config.orpo
    print("=" * 60)
    print("Phase 6: ORPO Alignment")
    print("=" * 60)

    # 1. Prepare model: unfreeze S5 + bridge
    model.unfreeze_for_orpo()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M total")

    # 2. Load preference dataset
    print(f"  Loading dataset: {cfg.dataset_name}")
    dataset = load_dataset(cfg.dataset_name, split="train_prefs")

    def format_preference_pair(example):
        """Format UltraFeedback examples for ORPO."""
        prompt = example.get("prompt", example.get("instruction", ""))

        # Handle different dataset formats
        chosen = example.get("chosen", [])
        rejected = example.get("rejected", [])

        if isinstance(chosen, list) and len(chosen) > 0:
            if isinstance(chosen[0], dict):
                chosen_text = chosen[-1].get("content", str(chosen[-1]))
                rejected_text = rejected[-1].get("content", str(rejected[-1]))
            else:
                chosen_text = str(chosen[-1])
                rejected_text = str(rejected[-1])
        elif isinstance(chosen, str):
            chosen_text = chosen
            rejected_text = rejected if isinstance(rejected, str) else str(rejected)
        else:
            chosen_text = str(chosen)
            rejected_text = str(rejected)

        return {
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    dataset = dataset.map(format_preference_pair)

    # 3. Wrap model for trl compatibility
    wrapped_model = ValkyrieORPOWrapper(model)

    # 4. Configure ORPO
    training_args = TRLORPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        beta=cfg.orpo_alpha,
        logging_steps=10,
        save_steps=500,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    # 5. Run ORPO training
    trainer = ORPOTrainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("  Starting ORPO training...")
    trainer.train()

    # 6. Save final model
    final_path = os.path.join(cfg.output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
    }, final_path)
    print(f"\n  Saved aligned model to: {final_path}")
    print("ORPO alignment complete!")


def main():
    parser = argparse.ArgumentParser(description="ORPO alignment training")
    parser.add_argument("--model-checkpoint", type=str, required=True,
                        help="Path to Valkyrie+HRM checkpoint (post Phase 5)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = get_config()

    # Load model
    from valkyrie_hrm import ValkyrieHRM
    model = ValkyrieHRM(config).to(args.device)
    if os.path.exists(args.model_checkpoint):
        ckpt = torch.load(args.model_checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Get tokenizer
    from load_teacher import TeacherModel
    teacher = TeacherModel(config.teacher)
    teacher.load()
    tokenizer = teacher.tokenizer
    model.setup_tokenizer(tokenizer)

    # Run ORPO
    run_orpo(config, model, tokenizer, device=args.device)


if __name__ == "__main__":
    main()
