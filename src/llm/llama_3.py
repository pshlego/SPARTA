

import torch
from transformers import AutoTokenizer, pipeline

from .base_llm import BaseLlm
from .registry import register_llm



@register_llm("llama-3.1-8b-instruct")
class Llama_3(BaseLlm):
    def __init__(self, cfg, acc=None):
        llm_path = cfg.llm.llm_path
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        super().__init__(cfg, tokenizer)

        self.generator = pipeline(
            'text-generation',
            model=llm_path,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device=0 if acc is None else acc.process_index,
            temperature=self.temperature,
        )

    def _connect_prompt(self, system_prompt, user_prompt):
        if system_prompt == '':
            return '<|begin_of_text|><|start_header_id|>user<|end_header_id|>' +  user_prompt + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        else:
            return '<|begin_of_text|><|start_header_id|>system<|end_header_id|>' + system_prompt + '<|eot_id|><|start_header_id|>user<|end_header_id|>' + user_prompt + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'


    def inference(self, system_prompt, user_prompt, max_new_tokens=None, temperature=None, **kwargs) -> str:
        prompt = self._connect_prompt(system_prompt, user_prompt)

        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = self.generator(
                prompt, 
                return_full_text=False, 
                max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                temperature=temperature if temperature is not None else self.temperature,
            )
        return outputs[0]['generated_text']
    
    
    def is_overflow(self, system_prompt, user_prompt, max_tokens=None):
        prompt = self._connect_prompt(system_prompt, user_prompt)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)

        if max_tokens is None:
            return len(tokens) > self.max_tokens
        else:
            return len(tokens) > max_tokens
        

