import torch
from tqdm import tqdm
from data_loader import get_wikitext
from model_loader import get_model_and_tokenizer

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

def wikitext_ppl(model, tokenizer, device, stride = 512, split="test"):
    wikitext = get_wikitext(split)
    text = "\n\n".join(wikitext)
    max_len = model.config.max_position_embeddings
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    ids = tokens["input_ids"]
    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    with torch.no_grad():
        for end in range(stride, ids.size(1) + stride, stride):
            end = min(end, ids.size(1))
            begin = max(0, end - max_len)
            target_len = end - prev_end

            input_ids = ids[:, begin:end]
            labels = input_ids.clone()
            labels[:,:-target_len] = -100 #mask previously computed tokens from loss

            loss = model(input_ids=input_ids, labels=labels)
            total_nll += loss.loss.item() * target_len #loss.loss.item() is for one token
            total_tokens += target_len
            prev_end = end

            if end == ids.size(1):
                break

    avg_nll = total_nll / total_tokens
    return torch.exp(torch.tensor(avg_nll)).item()

if __name__ == "__main__":
    model, tokenizer = get_model_and_tokenizer()
    ppl = wikitext_ppl(model, tokenizer, device = model.device)
    print(f"Perplexity: {ppl}")