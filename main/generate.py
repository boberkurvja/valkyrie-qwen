"""
Project Valkyrie — Phase 7: Custom Recurrent Generation Loop
O(1) state-based generation: pass current token + previous state only.
Supports dynamic M_max scaling for inference-time computation.
"""
import os
import sys
import argparse
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoTokenizer
from sink_cache import SinkCache

from config import get_config, ValkyrieConfig, InferenceConfig
from valkyrie_hrm import ValkyrieHRM

# Import StableMax
sys.path.insert(0, "/root/model/HRM")
from models.losses import log_stablemax


class ValkyrieGenerator:
    """
    Custom generation loop for Valkyrie + HRM.

    Uses `SinkCache` (StreamingLLM) to maintain $O(1)$ memory representation
    (bounding the KV cache context) while the HRM handles complex logic offline.

    Supports:
    - Top-k / Top-p sampling
    - Temperature scaling
    - Dynamic M_max for HRM ACT (inference-time scaling)
    - Streaming output
    """

    def __init__(
        self,
        model: ValkyrieHRM,
        tokenizer,
        config: InferenceConfig = InferenceConfig(),
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Put model in eval mode
        self.model.eval()

    def _sample_token(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        use_stablemax: bool = True,
    ) -> Tensor:
        """
        Sample next token from logits using top-k/top-p.
        Uses StableMax for probability computation.
        """
        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)

            if use_stablemax:
                cumulative_probs = torch.exp(log_stablemax(sorted_logits, dim=-1)).cumsum(dim=-1)
            else:
                cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample
        if use_stablemax:
            probs = torch.exp(log_stablemax(logits.to(torch.float64), dim=-1)).to(torch.float32)
        else:
            probs = F.softmax(logits, dim=-1)

        # Clamp to avoid sampling errors
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        use_hrm: bool = False,
        hrm_M_max: Optional[int] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            stream: If True, print tokens as they're generated
            use_hrm: If True, route through HRM coprocessor
            hrm_M_max: Override M_max for HRM ACT (inference-time scaling)
        """
        cfg = self.config
        max_new_tokens = max_new_tokens or cfg.max_new_tokens
        temperature = temperature or cfg.temperature
        top_k = top_k if top_k is not None else cfg.top_k
        top_p = top_p if top_p is not None else cfg.top_p

        # Tokenize prompt
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.device)
        B, L = input_ids.shape

        # Initialize SinkCache for O(1) bounded Generation
        sink_cache = SinkCache(
            window_length=cfg.cache_size, 
            num_sink_tokens=cfg.sink_size
        )

        # Process prompt through the backbone to build initial state and cache
        # Single forward pass for context
        out = self.model(
            input_ids=input_ids[:, :-1], 
            past_key_values=sink_cache, 
            use_cache=True
        )

        # Generate new tokens
        generated_tokens = []
        current_token = input_ids[:, -1:]  # Start from last prompt token

        if stream:
            sys.stdout.write(prompt)
            sys.stdout.flush()

        for step in range(max_new_tokens):
            # Evaluate current token while rolling the cache
            out = self.model(
                input_ids=current_token,
                past_key_values=sink_cache,
                use_cache=True,
            )
            
            # The resulting backbone representation is evaluated by HRM or standard head
            h_t = out.backbone_hidden[:, -1:] # [B, 1, H]

            # Optional HRM routing
            if use_hrm:
                hrm_input = self.model.bridge.to_hrm_space(h_t)
                # Run HRM with potentially increased M_max
                M = hrm_M_max or cfg.M_max_inference
                hrm = self.model.hrm_inner

                z_H = hrm.H_init.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).clone()
                z_L = hrm.L_init.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).clone()

                seq_info = dict(
                    cos_sin=hrm.rotary_emb() if hasattr(hrm, "rotary_emb") else None,
                )

                H_cycles = self.model.bridge_config.hrm_H_cycles
                L_cycles = self.model.bridge_config.hrm_L_cycles

                for _ in range(M):
                    for h_step in range(H_cycles):
                        for l_step in range(L_cycles):
                            z_L = hrm.L_level(z_L, z_H + hrm_input, **seq_info)
                        z_H = hrm.H_level(z_H, z_L, **seq_info)

                hrm_output = self.model.bridge.from_hrm_space(z_H.squeeze(1))
                h_t = h_t + hrm_output

            # Logits
            logits = self.model.stablemax_head(h_t)  # [B, 1, V]
            
            # Sample (we take [:, 0, :] to strip the seq dimension for sampling)
            next_token = self._sample_token(
                logits[:, 0, :], temperature=temperature, top_k=top_k, top_p=top_p,
            )  # [B, 1]

            generated_tokens.append(next_token.squeeze(-1))
            current_token = next_token

            # Stream output
            if stream:
                token_text = self.tokenizer.decode(next_token.cpu().tolist())
                sys.stdout.write(token_text)
                sys.stdout.flush()

            # Stop at EOS
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                if (next_token == self.tokenizer.eos_token_id).all():
                    break

        if stream:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Decode
        generated = torch.stack(generated_tokens, dim=1)
        full_sequence = torch.cat([input_ids, generated], dim=1)
        text = self.tokenizer.decode(full_sequence[0].cpu().tolist(), skip_special_tokens=True)

        return text


def main():
    parser = argparse.ArgumentParser(description="Valkyrie text generation")
    parser.add_argument("--model-checkpoint", type=str, required=True,
                        help="Path to final model checkpoint")
    parser.add_argument("--prompt", type=str, default="Hello, I am Valkyrie, and I",
                        help="Generation prompt")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--use-hrm", action="store_true",
                        help="Route through HRM coprocessor")
    parser.add_argument("--hrm-m-max", type=int, default=None,
                        help="Override M_max for inference-time scaling")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = get_config()

    # Load model
    model = ValkyrieHRM(config).to(args.device)
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        ckpt = torch.load(args.model_checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.teacher.model_name, 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.setup_tokenizer(tokenizer)

    # Generate
    generator = ValkyrieGenerator(model, tokenizer, config.inference, device=args.device)

    text = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stream=args.stream,
        use_hrm=args.use_hrm,
        hrm_M_max=args.hrm_m_max,
    )

    if not args.stream:
        print(f"\n{'=' * 60}")
        print(text)
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
