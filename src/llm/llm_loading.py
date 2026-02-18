from .registry import get_llm_cls
import src.llm


def load_llm(cfg, acc=None):
    name = cfg.llm.name
    llm_cls = get_llm_cls(name)
    return llm_cls(cfg, acc)
