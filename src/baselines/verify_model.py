import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.model_loader import get_model_and_tokenizer

model, tokenizer = get_model_and_tokenizer()

# Quick inference test
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(outputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))