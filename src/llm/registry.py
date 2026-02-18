from typing import Dict, Type
from .base_llm import BaseLlm

LLM_REGISTRY: Dict[str, Type[BaseLlm]] = {}


def register_llm(name: str):
    def _wrap(cls: Type[BaseLlm]) -> Type[BaseLlm]:
        if name in LLM_REGISTRY:
            raise ValueError(f"Llm '{name}' already registered.")
        if not issubclass(cls, BaseLlm):
            raise TypeError(f"{cls.__name__} must subclass BaseLlm.")
        LLM_REGISTRY[name] = cls
        return cls

    return _wrap


def get_llm_cls(name: str) -> Type[BaseLlm]:
    try:
        return LLM_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(LLM_REGISTRY.keys()))
        raise KeyError(f"Unknown llm '{name}'. Available: {available}")
