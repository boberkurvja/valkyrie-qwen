"""
Project Valkyrie — Build Student from Teacher
Instantiates the Valkyrie student model and copies weights from the teacher.
Only S5 blocks are randomly initialized; everything else comes from the teacher.
"""
import sys
import argparse
from typing import Dict

import torch
import torch.nn as nn

from config import get_config, ValkyrieConfig, StudentConfig
from load_teacher import TeacherModel, load_teacher
from valkyrie_model import ValkyrieModel


def _copy_embedding(teacher: TeacherModel, student: ValkyrieModel, config: StudentConfig):
    """Copy embedding weights from teacher to student."""
    teacher_embed = teacher.get_embedding()
    with torch.no_grad():
        # Handle vocabulary size differences
        t_vocab, t_dim = teacher_embed.weight.shape
        s_vocab, s_dim = student.embed_tokens.weight.shape

        copy_vocab = min(t_vocab, s_vocab)
        copy_dim = min(t_dim, s_dim)

        student.embed_tokens.weight[:copy_vocab, :copy_dim] = \
            teacher_embed.weight[:copy_vocab, :copy_dim].to(student.embed_tokens.weight.dtype)

    print(f"  ✓ Embedding: copied {copy_vocab}×{copy_dim} weights")


def _copy_lm_head(teacher: TeacherModel, student: ValkyrieModel, config: StudentConfig):
    """Copy LM head weights (may be tied with embeddings)."""
    if config.tie_word_embeddings:
        print("  ✓ LM Head: tied with embeddings (no separate copy needed)")
        return

    teacher_head = teacher.get_lm_head()
    with torch.no_grad():
        t_vocab, t_dim = teacher_head.weight.shape
        s_vocab, s_dim = student.lm_head.weight.shape

        copy_vocab = min(t_vocab, s_vocab)
        copy_dim = min(t_dim, s_dim)

        student.lm_head.weight[:copy_vocab, :copy_dim] = \
            teacher_head.weight[:copy_vocab, :copy_dim].to(student.lm_head.weight.dtype)

    print(f"  ✓ LM Head: copied {copy_vocab}×{copy_dim} weights")


def _copy_final_norm(teacher: TeacherModel, student: ValkyrieModel):
    """Copy final layer norm weights."""
    teacher_norm = teacher.get_final_norm()
    with torch.no_grad():
        if hasattr(teacher_norm, "weight") and hasattr(student.norm, "weight"):
            student.norm.weight.copy_(teacher_norm.weight.to(student.norm.weight.dtype))
    print(f"  ✓ Final norm: copied")


def _copy_layer_weights(
    teacher: TeacherModel,
    student: ValkyrieModel,
    teacher_layer_idx: int,
    student_layer_idx: int,
):
    """
    Copy MLP and RMSNorm weights from a specific teacher layer to student layer.
    S5 weights are NOT copied (they use HiPPO-N initialization).
    """
    teacher_layer = teacher.get_layer(teacher_layer_idx)
    student_block = student.layers[student_layer_idx]

    copied = []

    with torch.no_grad():
        # ── Copy RMSNorm weights ──
        # Try to find matching norm layers
        for t_name, s_name in [
            ("input_layernorm", "input_layernorm"),
            ("post_attention_layernorm", "post_attention_layernorm"),
        ]:
            t_norm = getattr(teacher_layer, t_name, None)
            s_norm = getattr(student_block, s_name, None)

            if t_norm is not None and s_norm is not None:
                if hasattr(t_norm, "weight") and hasattr(s_norm, "weight"):
                    s_norm.weight.copy_(t_norm.weight.to(s_norm.weight.dtype))
                    copied.append(t_name)

        # ── Copy MLP weights ──
        teacher_mlp = None
        for attr in ["mlp", "feed_forward", "ffn"]:
            if hasattr(teacher_layer, attr):
                teacher_mlp = getattr(teacher_layer, attr)
                break

        if teacher_mlp is not None:
            student_mlp = student_block.mlp

            # Map teacher MLP weight names to student
            mlp_mappings = [
                ("gate_proj", "gate_proj"),
                ("up_proj", "up_proj"),
                ("down_proj", "down_proj"),
                ("gate_up_proj", None),  # Some models fuse gate+up
            ]

            for t_attr, s_attr in mlp_mappings:
                t_mod = getattr(teacher_mlp, t_attr, None)
                if t_mod is None:
                    continue

                if s_attr is not None:
                    s_mod = getattr(student_mlp, s_attr, None)
                    if s_mod is not None and hasattr(t_mod, "weight"):
                        s_mod.weight.copy_(t_mod.weight.to(s_mod.weight.dtype))
                        if hasattr(t_mod, "bias") and t_mod.bias is not None and hasattr(s_mod, "bias") and s_mod.bias is not None:
                            s_mod.bias.copy_(t_mod.bias.to(s_mod.bias.dtype))
                        copied.append(f"mlp.{t_attr}")

    return copied


def build_student(config: ValkyrieConfig, teacher: TeacherModel, device: str = "cuda") -> ValkyrieModel:
    """
    Build the Valkyrie student model by:
    1. Instantiating ValkyrieModel with student config
    2. Copying embeddings, norms, MLPs from teacher
    3. Leaving S5 blocks with HiPPO-N initialization
    """
    print("=" * 60)
    print("Building Valkyrie Student Model")
    print("=" * 60)

    teacher_layers = teacher.get_num_layers()
    student_layers = config.student.num_layers
    layers_to_prune = config.student.layers_to_prune

    print(f"\n  Teacher layers: {teacher_layers}")
    print(f"  Pruning last {layers_to_prune} layers → {student_layers} student layers")

    if teacher_layers - layers_to_prune != student_layers:
        print(f"  WARNING: teacher({teacher_layers}) - pruned({layers_to_prune}) = "
              f"{teacher_layers - layers_to_prune} ≠ student({student_layers})")
        print(f"  Using min({teacher_layers - layers_to_prune}, {student_layers}) layers for copy")

    # 1. Instantiate student
    student = ValkyrieModel(config.student).to(device)

    # 2. Copy embeddings
    print("\nCopying weights from teacher:")
    _copy_embedding(teacher, student, config.student)

    # 3. Copy LM head
    _copy_lm_head(teacher, student, config.student)

    # 4. Copy final norm
    _copy_final_norm(teacher, student)

    # 5. Copy per-layer MLP and norm weights
    layers_to_copy = min(teacher_layers - layers_to_prune, student_layers)
    for i in range(layers_to_copy):
        copied = _copy_layer_weights(teacher, student, teacher_layer_idx=i, student_layer_idx=i)
        if i < 3 or i == layers_to_copy - 1:
            print(f"  ✓ Layer {i}: copied {', '.join(copied)}")
        elif i == 3:
            print(f"  ... (copying layers 3-{layers_to_copy - 2})")

    # 6. Report parameter counts
    counts = student.count_parameters()
    print(f"\n{'=' * 60}")
    print(f"Student Model Parameter Summary:")
    print(f"{'=' * 60}")
    for name, count in counts.items():
        print(f"  {name:>20s}: {count:>12,d}  ({count / 1e6:.2f}M)")
    print(f"{'=' * 60}")

    return student


def main():
    parser = argparse.ArgumentParser(description="Build Valkyrie student from teacher")
    parser.add_argument("--save-path", type=str, default="checkpoints/student_init.pt",
                        help="Path to save initialized student")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only count parameters, don't save")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = get_config()

    # Load teacher
    teacher = load_teacher(config, device=args.device)

    # Build student
    student = build_student(config, teacher, device=args.device)

    if not args.dry_run:
        import os
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({
            "model_state_dict": student.state_dict(),
            "config": config,
        }, args.save_path)
        print(f"\n  Saved initialized student to: {args.save_path}")
    else:
        print("\n  Dry run complete — no file saved.")

    # Quick forward pass test
    print("\n  Running forward pass test...")
    student.eval()
    with torch.no_grad():
        dummy = torch.tensor([[1, 2, 3, 4, 5]], device=args.device)
        output = student(dummy)
        print(f"  Input: {dummy.shape} → Logits: {output.logits.shape}")
        print(f"  ✓ Forward pass successful!")


if __name__ == "__main__":
    main()
