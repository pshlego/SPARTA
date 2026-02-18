from abc import ABC

class BaseLlm(ABC):
    def __init__(self, cfg, tokenizer):
        self.tokenizer = tokenizer

        self.max_tokens = cfg.llm.max_tokens
        self.max_new_tokens = cfg.llm.max_new_tokens
        self.temperature = cfg.llm.temperature

    def inference(self, system_prompt, user_prompt, max_new_tokens, temperature, **kwargs) -> str:
        return {}

    def is_overflow(self, system_prompt, user_prompt) -> bool:
        return {}

    def trim(self, raw_text, trim_length):
        tokenized_text = self.tokenizer.encode(raw_text)
        trimmed_tokenized_text = tokenized_text[ : trim_length]
        trimmed_text = self.tokenizer.decode(trimmed_tokenized_text)
        return trimmed_text


