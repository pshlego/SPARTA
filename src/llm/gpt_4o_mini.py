from .base_llm import BaseLlm
from .registry import register_llm
from openai import OpenAI
import os
import tiktoken


@register_llm("gpt-4o-mini")
class Gpt_4o_mini(BaseLlm):
    def __init__(self, cfg, acc=None):
        # gpt-4o-mini also uses the o200k_base or cl100k_base encoding.
        # cl100k_base is standard for gpt-4 series.
        tokenizer = tiktoken.get_encoding("cl100k_base")
        super().__init__(cfg, tokenizer)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.llm_name = cfg.llm.name

    def inference(
        self,
        messages=[],
        system_prompt="",
        user_prompt="",
        max_new_tokens=None,
        temperature=None,
        **kwargs
    ):
        
        # Prepare arguments
        params = {
            "model": self.llm_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ] if messages == [] else messages,
            "max_completion_tokens" : max_new_tokens if max_new_tokens else self.max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            **kwargs,
        }

        try:
            # gpt-4o-mini uses the standard Chat Completion API
            response = self.client.chat.completions.create(**params)
            if 'tools' in kwargs:
                output = response.choices[0].message
            else:
                output = response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

        print(output) 

        return output

    def is_overflow(self, system_prompt, user_prompt, max_tokens=None):
        prompt = system_prompt + user_prompt
        tokens = self.tokenizer.encode(prompt)

        limit = max_tokens if max_tokens is not None else self.max_tokens
        return len(tokens) > limit

    def trim(self, raw_text, trim_length):
        tokenized_text = self.tokenizer.encode(raw_text)
        trimmed_tokenized_text = tokenized_text[:trim_length]
        trimmed_text = self.tokenizer.decode(trimmed_tokenized_text)
        return trimmed_text