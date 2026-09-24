# src/utils/perplexity.py
import torch
from tqdm import tqdm

def perplexity(model, tokenizer, dataset, max_length=2048, batch_size=4):
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        # Assuming batch is a list of strings
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()