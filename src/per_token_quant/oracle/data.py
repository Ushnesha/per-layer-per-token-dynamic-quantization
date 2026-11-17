from transformers import PreTrainedTokenizerBase
from datasets import load_dataset
import torch

def load_wikitext(tokenizer: PreTrainedTokenizerBase, chunk_size: int = 512, split: str = "test") -> torch.Tensor:
    """Loads WikiText-2 and splits it into fixed-length chunks.

    Downloads the specified split of raw WikiText-2 dataset, tokenizes the entire text corpus as a single joined
    sequence, and reshapes it into uniform blocks. The text is joined using two newline characters in order to be consistent with 
    GPTQ and AWQ. If the tokens can not be evenly divided by `chunk_size`, then any leftover tokens will be dropped.

    Args:
        tokenizer: The Hugging Face tokenizer instance used to encode the text. Should correspond to the model being evaluated.
        chunk_size: The target token length for each individual sequence chunk. Defaults to 512.
        split: The dataset split to download (e.g., 'train', 'test', or 'validation'). Defaults to 'test'.
    
    Returns:
        torch.Tensor: A 2D integer tensor of shape `(num_chunks, chunk_size)`.
    """
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    tokenized_data = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")["input_ids"]
    num_chunks = tokenized_data.shape[1] // chunk_size
    chunks = tokenized_data[:, :num_chunks*chunk_size].reshape(num_chunks, chunk_size)
    return chunks
