import os
import sys
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from config import get_config
from valkyrie_hrm import ValkyrieHRM
from generate import ValkyrieGenerator

def grid_to_str(grid):
    """Converts a 2D integer list into a multi-line string matrix."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

def run_interactive_chat(generator: ValkyrieGenerator, use_hrm: bool, hrm_m_max: int):
    """Runs a continuous CLI chat loop to test logic and conversational skills."""
    print("\n" + "="*60)
    print(f"Valkyrie Interactive Inference (HRM Routing: {use_hrm}, M_max: {hrm_m_max})")
    print("Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            # Format strictly to how it was trained
            prompt = f"User: {user_input}\nAssistant: "
            
            # If we want to force reasoning, we append the reason token
            if use_hrm:
                prompt += "<|reason|>\n"

            print("Assistant: ", end="")
            
            # The generator handles the SinkCache natively
            _ = generator.generate(
                prompt=prompt,
                max_new_tokens=512,
                stream=True,  # Stream directly to console
                use_hrm=use_hrm,
                hrm_M_max=hrm_m_max
            )
            print() # Newline after generation

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break

def run_arc_benchmark(generator: ValkyrieGenerator, arc_dir: str, use_hrm: bool, hrm_m_max: int):
    """Automatically evaluates the model on the unseen ARC validation set."""
    if not os.path.exists(arc_dir):
        print(f"\n[!] Error: ARC directory {arc_dir} not found.")
        return

    print("\n" + "="*60)
    print(f"ARC-AGI Benchmark (HRM Routing: {use_hrm}, M_max: {hrm_m_max})")
    print("="*60)

    json_files = [f for f in os.listdir(arc_dir) if f.endswith('.json')]
    total_puzzles = 0
    exact_matches = 0

    progress = tqdm(json_files, desc="Evaluating ARC tasks")

    for file_name in progress:
        with open(os.path.join(arc_dir, file_name), 'r') as f:
            task = json.load(f)

        # 1. Build the context prompt exactly as it was during training
        prompt_context = "Solve the following abstract reasoning puzzle. Observe the spatial and color patterns in the training examples and apply them to the final test input.\n\n"
        for i, ex in enumerate(task.get("train", [])):
            prompt_context += f"--- Train Example {i+1} ---\nInput:\n{grid_to_str(ex['input'])}\nOutput:\n{grid_to_str(ex['output'])}\n\n"

        # 2. Evaluate all test pairs in the puzzle
        for i, ex in enumerate(task.get("test", [])):
            test_prompt = prompt_context + f"--- Test Example {i+1} ---\nInput:\n{grid_to_str(ex['input'])}\n"
            ground_truth_str = grid_to_str(ex['output'])

            # Format with the reason token trigger
            full_prompt = f"User: {test_prompt}\nAssistant: <|reason|>\n"

            # 3. Generate Prediction
            generated_text = generator.generate(
                prompt=full_prompt,
                max_new_tokens=256,
                stream=False,
                use_hrm=use_hrm,
                hrm_M_max=hrm_m_max,
                temperature=0.1, # Low temp for deterministic math tasks
                top_p=1.0
            )

            # Strip the prompt to extract just the model's generated grid
            prediction = generated_text[len(full_prompt):].strip()
            
            # Compare
            if prediction == ground_truth_str.strip():
                exact_matches += 1
            total_puzzles += 1

            # Update progress bar stats
            current_acc = (exact_matches / total_puzzles) * 100
            progress.set_postfix_str(f"Acc: {current_acc:.2f}% ({exact_matches}/{total_puzzles})")

    print("\n" + "="*60)
    print(f"FINAL ARC ACCURACY: {(exact_matches / total_puzzles) * 100:.2f}%")
    print(f"Total Test Grids Solved: {exact_matches} out of {total_puzzles}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Valkyrie Evaluation Suite")
    parser.add_argument("--model-checkpoint", type=str, required=True, help="Path to your hrm_step_X.pt file")
    parser.add_argument("--mode", type=str, choices=["chat", "arc"], default="chat", help="Which mode to run")
    parser.add_argument("--arc-dir", type=str, default="../dataset/raw-data/ARC-AGI/data/evaluation", help="Path to unseen ARC eval JSONs")
    parser.add_argument("--use-hrm", action="store_true", help="Force routing through the MoS Coprocessor")
    parser.add_argument("--hrm-m-max", type=int, default=1, help="Test-Time Compute: Number of recurrent reasoning cycles")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = get_config()

    print(f"Loading ValkyrieHRM from {args.model_checkpoint}...")
    model = ValkyrieHRM(config).to(args.device)
    
    ckpt = torch.load(args.model_checkpoint, map_location=args.device)
    load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if load_result.missing_keys:
        print(f"[WARN] Missing keys in checkpoint ({len(load_result.missing_keys)}): {load_result.missing_keys[:10]}")
    if load_result.unexpected_keys:
        print(f"[WARN] Unexpected keys in checkpoint ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys[:10]}")

    tokenizer = AutoTokenizer.from_pretrained(config.teacher.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.setup_tokenizer(tokenizer)

    # Initialize the customized O(1) inference generator
    generator = ValkyrieGenerator(model, tokenizer, config.inference, device=args.device)

    if args.mode == "chat":
        run_interactive_chat(generator, args.use_hrm, args.hrm_m_max)
    elif args.mode == "arc":
        run_arc_benchmark(generator, args.arc_dir, args.use_hrm, args.hrm_m_max)

if __name__ == "__main__":
    main()