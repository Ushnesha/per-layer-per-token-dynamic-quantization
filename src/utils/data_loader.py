from datasets import load_dataset

# WikiText-2 (perplexity benchmark)
wikitext = load_dataset("wikitext", "wikitext-2-v1")

# HellaSwag (downstream accuracy)
hellaswag = load_dataset("Rowan/hellaswag")

# LongBench (long-context evaluation)
longbench = load_dataset("THUDM/LongBench")