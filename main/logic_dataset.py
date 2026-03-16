"""
Project Valkyrie — Phase 5: Logic Puzzle Dataset
Loads ARC-AGI / logic puzzle input-output pairs.
No chain-of-thought — pure input→output for deep supervision.
"""
import os
import json
from typing import Optional, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from config import HRMTrainConfig


class LogicPuzzleDataset(Dataset):
    """
    Dataset for pure logic puzzles (ARC-AGI style).
    Format: input grid → output grid, no chain-of-thought.

    Expected data structure (JSON files):
    {
        "train": [{"input": [[...]], "output": [[...]]}, ...],
        "test":  [{"input": [[...]], "output": [[...]]}]
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_len: int = 256,
        split: str = "train",
        max_examples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.examples = []

        self._load_data(data_path, max_examples)

    def _grid_to_tokens(self, grid: list) -> str:
        """Convert a 2D grid to a token string."""
        rows = []
        for row in grid:
            rows.append(" ".join(str(x) for x in row))
        return "\n".join(rows)

    def _load_arc_format(self, data_path: str, max_examples: Optional[int]):
        """Load ARC-AGI format puzzles from JSON files."""
        puzzle_files = []

        # Handle directory of JSON files
        if os.path.isdir(data_path):
            for fname in sorted(os.listdir(data_path)):
                if fname.endswith(".json"):
                    puzzle_files.append(os.path.join(data_path, fname))
        elif os.path.isfile(data_path):
            puzzle_files = [data_path]

        puzzle_id = 0
        for fpath in puzzle_files:
            if max_examples and len(self.examples) >= max_examples:
                break

            try:
                with open(fpath) as f:
                    puzzle = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Extract train/test pairs
            pairs = puzzle.get(self.split, puzzle.get("train", []))
            for pair in pairs:
                if max_examples and len(self.examples) >= max_examples:
                    break

                inp = pair.get("input", [])
                out = pair.get("output", [])

                if inp and out:
                    self.examples.append({
                        "input_grid": inp,
                        "output_grid": out,
                        "puzzle_id": puzzle_id,
                    })

            puzzle_id += 1

    def _load_data(self, data_path: str, max_examples: Optional[int]):
        """Load data from the specified path."""
        if not os.path.exists(data_path):
            print(f"  Warning: {data_path} not found, creating synthetic examples")
            self._create_synthetic_examples(max_examples or 1000)
            return

        self._load_arc_format(data_path, max_examples)

        if not self.examples:
            print(f"  Warning: No examples found in {data_path}, creating synthetic")
            self._create_synthetic_examples(max_examples or 1000)

        print(f"  Loaded {len(self.examples)} {self.split} examples")

    def _create_synthetic_examples(self, n: int):
        """Create synthetic logic puzzles for testing."""
        import random
        random.seed(42)

        for i in range(n):
            # Simple pattern: identity, rotation, color mapping
            size = random.randint(3, 8)
            grid = [[random.randint(0, 9) for _ in range(size)] for _ in range(size)]

            # Simple transformation: increment all values mod 10
            out_grid = [[(v + 1) % 10 for v in row] for row in grid]

            self.examples.append({
                "input_grid": grid,
                "output_grid": out_grid,
                "puzzle_id": i,
            })

    def _encode_example(self, example: dict) -> Dict[str, torch.Tensor]:
        """Encode a single example as token tensors."""
        input_str = self._grid_to_tokens(example["input_grid"])
        output_str = self._grid_to_tokens(example["output_grid"])

        # Format: "INPUT:\n{grid}\nOUTPUT:\n{grid}"
        full_text = f"INPUT:\n{input_str}\nOUTPUT:\n{output_str}"

        tokens = self.tokenizer(
            full_text,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)

        # Labels: mask the input portion, only compute loss on output
        separator = self.tokenizer("\nOUTPUT:\n", add_special_tokens=False)["input_ids"]
        sep_len = len(separator)

        # Find where "OUTPUT:" starts
        labels = input_ids.clone()
        input_tokens = self.tokenizer(
            f"INPUT:\n{input_str}\nOUTPUT:\n",
            add_special_tokens=False,
        )["input_ids"]
        input_len = len(input_tokens)

        # Mask input portion
        labels[:min(input_len, self.seq_len)] = -100

        # Mask padding
        if tokens.get("attention_mask") is not None:
            labels[tokens["attention_mask"].squeeze(0) == 0] = -100

        return {
            "inputs": input_ids,
            "labels": labels,
            "puzzle_identifiers": torch.tensor(example["puzzle_id"], dtype=torch.int32),
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self._encode_example(self.examples[idx])


def create_logic_dataloader(
    config: HRMTrainConfig,
    tokenizer,
    split: str = "train",
) -> DataLoader:
    """Create DataLoader for logic puzzle training."""
    dataset = LogicPuzzleDataset(
        data_path=config.logic_dataset_path,
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        split=split,
        max_examples=config.num_examples,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
