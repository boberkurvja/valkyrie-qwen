"""
Project Valkyrie — Phase 7: Inference Compilation with torch.compile
Wraps the generation loop in torch.compile(mode="reduce-overhead") for
maximum throughput on the RTX 3090.
"""
import os
import sys
import time
import argparse
from typing import Optional

import torch
import torch.nn as nn

from config import get_config, ValkyrieConfig, InferenceConfig
from valkyrie_hrm import ValkyrieHRM
from generate import ValkyrieGenerator


class CompiledValkyrieGenerator(ValkyrieGenerator):
    """
    Compiled version of ValkyrieGenerator.
    Uses torch.compile(mode="reduce-overhead") to fuse CUDA kernels
    and reduce CPU overhead, pushing generation to the RTX 3090's limits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compiled = False

    def compile(self, mode: str = "reduce-overhead"):
        """
        Compile the model's forward passes for optimal inference.
        mode="reduce-overhead" focuses on reducing CPU↔GPU latency.
        """
        print(f"Compiling model with torch.compile(mode='{mode}')...")

        # Compile individual blocks for recurrent step
        for i, block in enumerate(self.model.backbone.layers):
            block.s5 = torch.compile(block.s5, mode=mode, dynamic=False)

        # Compile norms and projections
        self.model.backbone.norm = torch.compile(
            self.model.backbone.norm, mode=mode, dynamic=False
        )

        # Compile the LM head
        self.model.stablemax_head = torch.compile(
            self.model.stablemax_head, mode=mode, dynamic=False
        )

        # Compile HRM bridge
        self.model.bridge = torch.compile(
            self.model.bridge, mode=mode, dynamic=False
        )

        self._compiled = True
        print("  Compilation complete (will JIT on first run)")

    def warmup(self, num_warmup_tokens: int = 10):
        """
        Run a warmup pass to trigger JIT compilation.
        The first run with torch.compile is slow; subsequent runs are fast.
        """
        print(f"Warming up with {num_warmup_tokens} tokens...")
        dummy_prompt = "Hello"
        _ = self.generate(
            prompt=dummy_prompt,
            max_new_tokens=num_warmup_tokens,
            stream=False,
        )
        print("  Warmup complete")


def benchmark(
    generator: CompiledValkyrieGenerator,
    prompt: str = "Once upon a time",
    max_new_tokens: int = 100,
    num_runs: int = 3,
    warmup_runs: int = 1,
):
    """
    Benchmark generation throughput.
    Reports tokens/second with and without compilation.
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {max_new_tokens} tokens, {num_runs} runs")
    print(f"{'=' * 60}")

    # Warmup
    for _ in range(warmup_runs):
        generator.generate(prompt=prompt, max_new_tokens=10, stream=False)

    # Benchmark
    times = []
    for run in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        text = generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            stream=False,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        tokens_per_sec = max_new_tokens / elapsed
        print(f"  Run {run + 1}: {elapsed:.3f}s ({tokens_per_sec:.1f} tok/s)")

    avg_time = sum(times) / len(times)
    avg_tps = max_new_tokens / avg_time

    print(f"\n  Average: {avg_time:.3f}s ({avg_tps:.1f} tokens/sec)")
    print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    return avg_tps


def export_model(
    model: ValkyrieHRM,
    save_path: str,
    include_compile: bool = True,
):
    """
    Export the final model for edge deployment.
    Saves model state dict and optionally a torch.compile'd version.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save standard state dict
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "hidden_size": model.config.student.hidden_size,
            "num_layers": model.config.student.num_layers,
            "vocab_size": model.config.student.vocab_size,
            "s5_state_dim": model.config.student.s5.state_dim,
            "hrm_hidden_size": model.config.bridge.hrm_hidden_size,
        },
    }, save_path)
    print(f"  Model exported to: {save_path}")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total / 1e6:.2f}M")
    print(f"  Model size: {os.path.getsize(save_path) / 1024**2:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Compile and benchmark Valkyrie inference")
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--benchmark-tokens", type=int, default=100)
    parser.add_argument("--benchmark-runs", type=int, default=3)
    parser.add_argument("--export-path", type=str, default=None,
                        help="Export model to this path")
    parser.add_argument("--prompt", type=str, default="Hello, I am Valkyrie, and I")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = get_config()

    # Load model
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

    # Create generator
    generator = CompiledValkyrieGenerator(
        model, tokenizer, config.inference, device=args.device
    )

    # Compile
    if args.compile:
        generator.compile(mode=args.compile_mode)
        generator.warmup()

    # Benchmark
    if args.benchmark:
        benchmark(
            generator,
            prompt=args.prompt,
            max_new_tokens=args.benchmark_tokens,
            num_runs=args.benchmark_runs,
        )

    # Export
    if args.export_path:
        export_model(model, args.export_path)

    # Interactive generation
    if not args.benchmark and not args.export_path:
        text = generator.generate(
            prompt=args.prompt,
            max_new_tokens=100,
            stream=True,
        )


if __name__ == "__main__":
    main()
