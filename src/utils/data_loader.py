"""
Dataset utilities for calibration and evaluation.

Provides:
- WikiText-2 (perplexity benchmark)
- HellaSwag (accuracy benchmark)
- Calibration data for quantization
"""

from datasets import load_dataset
from typing import Dict, List, Optional

from src.config.settings import (
    WIKITEXT_DATASET, WIKITEXT_CONFIG, HELLASWAG_DATASET, PILEVAL_DATASET,
    CALIBRATION_NUM_SAMPLES, CALIBRATION_SEQ_LEN, CALIBRATION_SOURCE, SEED,
)


def get_wikitext(split: str = "test") -> List[str]:
    """Load WikiText-2 dataset split."""
    dataset = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split=split)
    return dataset["text"]


def get_wikitext_raw_text(split: str = "train", max_length: Optional[int] = None) -> str:
    """Load WikiText-2 as a single concatenated string (for perplexity computation)."""
    texts = get_wikitext(split)
    full_text = "\n\n".join(texts)
    if max_length:
        # Truncate tokens (approximate by characters)
        full_text = full_text[:max_length * 4]
    return full_text


def get_hellaswag(split: str = "train") -> dict:
    """Load HellaSwag dataset."""
    return load_dataset(HELLASWAG_DATASET, split=split)


def get_calibration_data(
    tokenizer,
    num_samples: int = CALIBRATION_NUM_SAMPLES,
    seq_len: int = CALIBRATION_SEQ_LEN,
    source: str = CALIBRATION_SOURCE,
    seed: int = SEED,
) -> List[Dict[str, List[int]]]:
    """
    Build fixed-length calibration blocks shared by all quantization methods.

    Documents are shuffled with a fixed seed, tokenized, concatenated, and cut
    into `num_samples` blocks of exactly `seq_len` tokens, so AWQ, GPTQ and
    SmoothQuant all see identical calibration tokens.

    Args:
        tokenizer: Tokenizer of the model being quantized.
        num_samples: Number of blocks.
        seq_len: Tokens per block.
        source: "pile" (pile-val, default) or "wikitext" (train split; inflates WikiText PPL).
        seed: Shuffle seed.

    Returns:
        List of {"input_ids": [...], "attention_mask": [...]} dicts.
    """
    if source == "pile":
        dataset = load_dataset(PILEVAL_DATASET, split="validation")
    elif source == "wikitext":
        dataset = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split="train")
    else:
        raise ValueError(f"Unknown calibration source: {source}")
    dataset = dataset.shuffle(seed=seed)

    needed = num_samples * seq_len
    tokens: List[int] = []
    for doc in dataset:
        text = doc["text"].strip()
        if not text:
            continue
        tokens.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(tokens) >= needed:
            break
    if len(tokens) < needed:
        raise ValueError(f"Not enough calibration tokens in {source}: {len(tokens)} < {needed}")

    samples = []
    for i in range(num_samples):
        ids = tokens[i * seq_len:(i + 1) * seq_len]
        samples.append({"input_ids": ids, "attention_mask": [1] * seq_len})

    print(f"[DataLoader] Using {num_samples} x {seq_len}-token calibration blocks from {source} (seed={seed}).")
    return samples


def get_longbench():
    """Load LongBench dataset (for long-context evaluation)."""
    return load_dataset("THUDM/LongBench")


# ── Legacy compatibility ───────────────────────────────────────────────────
def get_hellaswag():
    """Legacy function for backward compatibility."""
    return load_dataset(HELLASWAG_DATASET)