from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
import torch

def get_model_and_tokenizer(model_name: str, dtype: torch.dtype, device: torch.device) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads a Hugging Face model and its corresponding tokenizer. 
    
    Uses eager attention to get the full attention matrix to do analysis on. Using eager attention would be slower, so it would not
    be used for any latency measurements. 

    Args:
        model_name: The repository ID or path of the model to load from the Hugging Face Hub (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct').
        dtype: The target floating-point data type for the model weights (e.g., torch.bfloat16 or torch.float32).
        device: The target device where the model will be allocated and executed (e.g., torch.device('cuda')).
    
    Returns:
        A tuple that holds the model and tokenizer in that order.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", dtype=dtype, device_map=device)
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer