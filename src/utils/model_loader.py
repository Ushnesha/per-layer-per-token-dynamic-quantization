from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_model = None
_tokenizer = None

def get_model_and_tokenizer(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    global _model, _tokenizer
    if _model is None:
        print("Loading model...")
        _model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _model, _tokenizer