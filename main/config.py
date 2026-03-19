"""
Project Valkyrie — Central Configuration
All hyperparameters and architecture configs in one place.
"""
from dataclasses import dataclass, field
from typing import Optional
import torch


# ═══════════════════════════════════════════════════════════════════
# Teacher (Qwen3.5-0.8B) Configuration
# ═══════════════════════════════════════════════════════════════════
@dataclass
class TeacherConfig:
    model_name: str = "Qwen/Qwen3.5-0.8B"
    dtype: torch.dtype = torch.bfloat16
    # Qwen3.5-0.8B text model specs (from HF config.json)
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000
    max_position_embeddings: int = 262144


# S5Config and StudentConfig have been removed as we are using a frozen Qwen backbone


# ═══════════════════════════════════════════════════════════════════
# HRM Bridge Configuration
# ═══════════════════════════════════════════════════════════════════
@dataclass
class BridgeConfig:
    # Valkyrie hidden dim → HRM hidden dim
    valkyrie_hidden_size: int = 1024
    hrm_hidden_size: int = 512
    # HRM architecture config (from /root/model/HRM/config/arch/hrm_v1.yaml)
    hrm_H_cycles: int = 2
    hrm_L_cycles: int = 2
    hrm_H_layers: int = 4
    hrm_L_layers: int = 4
    hrm_num_heads: int = 8
    hrm_expansion: float = 4.0
    hrm_pos_encodings: str = "rope"
    hrm_rope_theta: float = 10000.0
    hrm_rms_norm_eps: float = 1e-5
    hrm_halt_max_steps: int = 16
    hrm_halt_exploration_prob: float = 0.1
    hrm_forward_dtype: str = "bfloat16"
    # Routing token
    reason_token: str = "<|reason|>"
    # Use StableMax in LM head
    use_stablemax: bool = True


# DistillConfig has been removed as we use zero-training inference


# ═══════════════════════════════════════════════════════════════════
# Phase 5: HRM Deep Supervision Training Configuration
# ═══════════════════════════════════════════════════════════════════
@dataclass
class HRMTrainConfig:
    # Dataset
    logic_dataset_path: str = "/root/model/HRM/dataset/raw-data/ARC-AGI/data/training"
    num_examples: int = 5000
    seq_len: int = 256
    # Training
    batch_size: int = 2
    learning_rate: float = 3e-5
    warmup_steps: int = 500
    num_epochs: int = 50
    # ACT config
    M_max: int = 16
    M_min: int = 2
    # Optimizer: Adam-atan2 (scale-invariant)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    # Checkpointing
    save_steps: int = 5000
    eval_steps: int = 100
    output_dir: str = "checkpoints/hrm_train"


# ═══════════════════════════════════════════════════════════════════
# Phase 6: ORPO Alignment Configuration
# ═══════════════════════════════════════════════════════════════════
@dataclass
class ORPOConfig:
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    learning_rate: float = 5e-6
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    max_prompt_length: int = 512
    orpo_alpha: float = 1.0  # ORPO loss weight
    output_dir: str = "checkpoints/orpo"


# ═══════════════════════════════════════════════════════════════════
# Phase 7: Inference Configuration
# ═══════════════════════════════════════════════════════════════════
@dataclass
class InferenceConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    # HRM ACT: can increase beyond training M_max for harder problems
    M_max_inference: int = 32
    # Bounded cache parameters
    cache_size: int = 256
    sink_size: int = 4
    # torch.compile
    compile_mode: str = "reduce-overhead"
    use_compile: bool = True


# ═══════════════════════════════════════════════════════════════════
# Master Configuration
# ═══════════════════════════════════════════════════════════════════
@dataclass
class ValkyrieConfig:
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    hrm_train: HRMTrainConfig = field(default_factory=HRMTrainConfig)
    orpo: ORPOConfig = field(default_factory=ORPOConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    # Global
    seed: int = 42
    device: str = "cuda"


def get_config(**overrides) -> ValkyrieConfig:
    """Create config with optional overrides."""
    config = ValkyrieConfig()
    for key, value in overrides.items():
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return config
