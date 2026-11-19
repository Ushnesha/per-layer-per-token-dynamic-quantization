"""Chat framing for generation tasks; retain explicit token-budget accounting."""

import hashlib


def chat_parts(tokenizer, before: str, after: str) -> tuple[list[int], list[int]]:
    """Reserve the full chat wrapper before allocating/truncating context tokens."""
    marker = '__ADITYA_CONTEXT_7dce8621__'
    if marker in before or marker in after:
        raise ValueError('Context marker occurs in prompt instructions.')
    if not tokenizer.chat_template:
        raise ValueError('Instruction evaluation requires the checkpoint chat template.')
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': before + marker + after}],
        tokenize=False, add_generation_prompt=True)
    if rendered.count(marker) != 1:
        raise ValueError('Chat template did not preserve the context boundary.')
    prefix, suffix = rendered.split(marker)
    # The template already includes BOS; adding special tokens again duplicates it.
    encode = lambda text: tokenizer(text, add_special_tokens=False)['input_ids']
    return encode(prefix), encode(suffix)


def stop_token_ids(tokenizer) -> list[int]:
    ids = []
    if type(tokenizer.eos_token_id) is int:
        ids.append(tokenizer.eos_token_id)
    # Llama-3 instruction turns end with eot_id, which may differ from EOS.
    eot = tokenizer.get_vocab().get('<|eot_id|>')
    if type(eot) is int:
        ids.append(eot)
    return sorted(set(ids))


def prompt_protocol(tokenizer) -> dict:
    return {'apply_chat_template': True,
            'prompting_version': 'checkpoint_chat_template_v1',
            'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            'stop_token_ids': stop_token_ids(tokenizer)}
