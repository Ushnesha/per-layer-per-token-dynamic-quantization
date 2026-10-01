from datasets import load_dataset

# WikiText-2 (perplexity benchmark)
def get_wikitext(split):
    wiki_test = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split=split)
    return wiki_test["text"]

def get_hellaswag():
    return load_dataset("Rowan/hellaswag")

def get_longbench():
    return load_dataset("THUDM/LongBench")