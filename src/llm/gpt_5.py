from .base_llm import BaseLlm
from .registry import register_llm
from openai import OpenAI
import os
import tiktoken


@register_llm("gpt-5")
class Gpt_5(BaseLlm):
    def __init__(self, cfg, acc=None):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        super().__init__(cfg, tokenizer)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.llm_name = cfg.llm.name

    def inference(
        self,
        system_prompt,
        user_prompt,
        max_new_tokens=None,
        temperature=None,
        is_reasoning=False,
        **kwargs,
    ) -> str:
        if max_new_tokens is None:
            if is_reasoning:
                response = self.client.responses.create(
                    model=self.llm_name,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=(
                        temperature if temperature is not None else self.temperature
                    ),
                    stop=stop,
                )
            else:
                response = self.client.responses.create(
                    model=self.llm_name,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=(
                        temperature if temperature is not None else self.temperature
                    ),
                    reasoning={"effort": "low"},
                    text={"verbosity": "low"},
                    stop=stop,
                )
        else:
            response = self.client.responses.create(
                model=self.llm_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max(max_new_tokens, self.max_new_tokens),
                temperature=(
                    temperature if temperature is not None else self.temperature
                ),
                stop=stop,
            )

        output = response.output_text  # .choices[0].message.content

        if output.strip() == "":
            print("Need more tokens!!!")
            output = self.inference(
                system_prompt,
                user_prompt,
                max(max_new_tokens, self.max_new_tokens) * 2,
                temperature,
                stop,
            )
        else:
            print(output)

        return output

    def is_overflow(self, system_prompt, user_prompt, max_tokens=None):
        prompt = system_prompt + user_prompt
        tokens = self.tokenizer.encode(prompt)

        if max_tokens is None:
            return len(tokens) > self.max_tokens
        else:
            return len(tokens) > max_tokens

    def trim(self, raw_text, trim_length):
        tokenized_text = self.tokenizer.encode(raw_text)
        trimmed_tokenized_text = tokenized_text[:trim_length]
        trimmed_text = self.tokenizer.decode(trimmed_tokenized_text)
        return trimmed_text
