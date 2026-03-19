import os
import json
from datasets import load_dataset, concatenate_datasets, Dataset

# =====================================================================
# CONFIGURATION & AUTHENTICATION
# =====================================================================
# Uncomment and add your token if you need to access gated datasets
# from huggingface_hub import login
# login(token="hf_YOUR_TOKEN_HERE")

OUTPUT_DIR = "./valkyrie_reasoning_data"
ARC_DATA_DIR = "/root/model/HRM/dataset/raw-data/ARC-AGI/data"

# =====================================================================
# DATASET 1: Opus-4.6-Reasoning-3300x
# =====================================================================
DATASET_1 = "Crownelius/Opus-4.6-Reasoning-3300x"
print(f"Downloading {DATASET_1}...")
ds1 = load_dataset(DATASET_1, split="train")

def format_ds1(example):
    problem = example.get("problem", "")
    solution = example.get("solution", "")
    # Inject the reasoning router token
    return {"text": f"User: {problem}\nAssistant: <|reason|>\n{solution}"}

print(f"Formatting {DATASET_1}...")
ds1 = ds1.map(format_ds1, num_proc=8, remove_columns=ds1.column_names)

# =====================================================================
# DATASET 2: claude-opus-4.6-10000x
# =====================================================================
DATASET_2 = "Roman1111111/claude-opus-4.6-10000x"
print(f"Downloading {DATASET_2} (Bypassing faulty metadata via Parquet)...")
ds2 = load_dataset("json", data_files=f"hf://datasets/{DATASET_2}/**/*.json*", split="train")

def format_ds2(example):
    messages = example.get("messages", [])
    prompt = ""
    response = ""
    
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            prompt = msg.get("content", "")
        elif role == "assistant":
            # Extract only the final content, ignoring intermediate dict keys
            response = msg.get("content", "")
            
    return {"text": f"User: {prompt}\nAssistant: <|reason|>\n{response}"}

print(f"Formatting {DATASET_2}...")
ds2 = ds2.map(format_ds2, num_proc=8, remove_columns=ds2.column_names)

# =====================================================================
# DATASET 3: ARC-AGI (Text-Stringified for Qwen Tokenizer)
# =====================================================================
def grid_to_str(grid):
    """Converts a 2D integer list into a multi-line string matrix."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

def load_arc_as_text(arc_data_dir):
    arc_examples = []
    
    if not os.path.exists(arc_data_dir):
        print(f"\n[!] WARNING: ARC directory '{arc_data_dir}' not found.")
        print("Please ensure you have the ARC-AGI JSON files in this path. Skipping ARC for now.\n")
        return Dataset.from_list([])

    print(f"Loading and formatting ARC dataset from {arc_data_dir}...")
    for file_name in os.listdir(arc_data_dir):
        if not file_name.endswith('.json'): continue
        
        with open(os.path.join(arc_data_dir, file_name), 'r') as f:
            task = json.load(f)

        # Build the context from the training pairs
        prompt = "Solve the following abstract reasoning puzzle. Observe the spatial and color patterns in the training examples and apply them to the final test input.\n\n"
        
        for i, ex in enumerate(task.get("train", [])):
            prompt += f"--- Train Example {i+1} ---\nInput:\n{grid_to_str(ex['input'])}\nOutput:\n{grid_to_str(ex['output'])}\n\n"

        # Apply the learned context to the test pairs
        for i, ex in enumerate(task.get("test", [])):
            test_prompt = prompt + f"--- Test Example {i+1} ---\nInput:\n{grid_to_str(ex['input'])}\n"
            test_solution = grid_to_str(ex['output'])

            # Inject the <|reason|> token so the HRM coprocessor engages
            arc_examples.append({
                "text": f"User: {test_prompt}\nAssistant: <|reason|>\n{test_solution}"
            })

    return Dataset.from_list(arc_examples)

ds_arc = load_arc_as_text(ARC_DATA_DIR)

# =====================================================================
# FUSION & SAVING
# =====================================================================
print("\nFusing datasets...")
datasets_to_fuse = [ds1, ds2]
if len(ds_arc) > 0:
    datasets_to_fuse.append(ds_arc)

final_dataset = concatenate_datasets(datasets_to_fuse)

# Shuffle heavily so spatial grids and textual logic are perfectly interleaved
print("Shuffling unified dataset...")
final_dataset = final_dataset.shuffle(seed=42)

final_dataset.save_to_disk(OUTPUT_DIR)

print(f"\n✅ Done! {len(final_dataset)} combined examples successfully saved to {OUTPUT_DIR}.")

# Print samples to verify formatting
print("\n" + "="*80)
print("SAMPLE 1: CLAUDE LOGIC")
print("="*80)
print(final_dataset[0]['text'][:500] + "...\n")

if len(ds_arc) > 0:
    # Find and print one ARC example just to verify
    for item in final_dataset:
        if "abstract reasoning puzzle" in item['text']:
            print("="*80)
            print("SAMPLE 2: ARC SPATIAL REASONING")
            print("="*80)
            print(item['text'][:800] + "...\n")
            break