"""
Perplexity computation utilities.

Protocol: WikiText-2 test split, joined with "\\n\\n", tokenized once, cut into
non-overlapping windows over the full split (trailing partial window dropped).
Each window is BOS + (seq_len - 1) text tokens; PPL = exp(mean NLL over the
seq_len - 1 text tokens of every window). Differs from the GPTQ/AWQ papers
only in the per-window BOS.

Provides:
- wikitext_windows: the shared token windows (used by HF and vLLM paths)
- wikitext_ppl: HF-model perplexity on those windows
- batch_ppl: Batched perplexity for any dataset
"""

import math

import torch
from tqdm import tqdm
from typing import List

from src.config.settings import PPL_SEQ_LEN
from src.utils.data_loader import get_wikitext


def wikitext_windows(tokenizer, seq_len: int = PPL_SEQ_LEN, split: str = "test") -> List[List[int]]:
    """Tokenize the full WikiText-2 split and cut it into non-overlapping windows.

    Each window is BOS + (seq_len - 1) text tokens, as in real prompts and
    lm_eval's rolling loglikelihood. Without BOS, Llama-3 must form its
    attention sink on an arbitrary mid-sentence token, which 4-bit models
    often cannot (GPTQ W4 PPL 54 vs 8.4 with BOS); FP16 is barely affected.
    """
    text = "\n\n".join(get_wikitext(split))
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    body = seq_len - 1
    n_windows = ids.numel() // body
    bos = tokenizer.bos_token_id
    return [[bos] + ids[i * body:(i + 1) * body].tolist() for i in range(n_windows)]


def wikitext_ppl(
    model,
    tokenizer,
    seq_len: int = PPL_SEQ_LEN,
    split: str = "test",
) -> float:
    """
    Compute perplexity on WikiText-2 with non-overlapping `seq_len` windows.

    Args:
        model: PyTorch model.
        tokenizer: Tokenizer instance.
        seq_len: Window length in tokens.
        split: Dataset split to evaluate on.

    Returns:
        Perplexity score.
    """
    device = next(model.parameters()).device
    windows = wikitext_windows(tokenizer, seq_len, split)

    total_nll = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for window in tqdm(windows, desc="WikiText PPL"):
            input_ids = torch.tensor([window], device=device)
            loss = model(input_ids=input_ids, labels=input_ids).loss
            total_nll += loss.float().item() * (seq_len - 1)
            total_tokens += seq_len - 1
            del input_ids, loss

    torch.cuda.empty_cache()
    return math.exp(total_nll / total_tokens)


def batch_ppl(
    model,
    tokenizer,
    dataset: List[str],
    max_length: int = 2048,
    batch_size: int = 4,
) -> float:
    """
    Compute perplexity on a list of text samples.

    Args:
        model: PyTorch model.
        tokenizer: Tokenizer instance.
        dataset: List of text strings.
        max_length: Maximum sequence length.
        batch_size: Batch size.

    Returns:
        Perplexity score.
    """
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Computing PPL"):
            batch = dataset[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()


if __name__ == "__main__":
    from src.utils.model_loader import ModelLoader

    loader = ModelLoader("fp16")
    model, tokenizer = loader.load("meta-llama/Meta-Llama-3-8B-Instruct")
    ppl = wikitext_ppl(model, tokenizer)
    print(f"Perplexity: {ppl:.4f}")
