from typing import Dict, Type
from .base_dataset import BaseDataset

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}


def register_dataset(name: str):

    def _wrap(cls: Type[BaseDataset]) -> Type[BaseDataset]:
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' already registered.")
        if not issubclass(cls, BaseDataset):
            raise TypeError(f"{cls.__name__} must subclass BaseDataset.")
        DATASET_REGISTRY[name] = cls
        return cls

    return _wrap
