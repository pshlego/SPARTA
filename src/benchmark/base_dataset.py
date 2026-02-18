import os
import abc
import pickle
from datasets import Dataset
from omegaconf import DictConfig
from functools import cached_property
from typing import List, Optional, Dict
from src.benchmark.types import DataTuple
from src.benchmark.utils import id_to_idx
from src.benchmark.base_collator import BaseCollator


class BaseDataset(abc.ABC):
    def __init__(self, cfg: DictConfig, global_cfg: DictConfig) -> None:
        self.cfg: DictConfig = cfg
        self.global_cfg: DictConfig = global_cfg
        self.name: str = self.cfg.name
        self.data_instances: Optional[List[DataTuple]] = None

    def __len__(self) -> int:
        return len(self.data_instances)

    def __getitem__(self, idx: int) -> DataTuple:
        return self.data_instances[idx]

    def setup(self) -> None:
        """
        Setup the dataset for training. Loads preprocessed data instances from disk.
        """
        assert os.path.exists(
            self.data_instances_path
        ), f"Data instances not found at {self.data_instances_path}. Please run `prepare_data` first."

        with open(self.data_instances_path, "rb") as f:
            self.data_instances = pickle.load(f)

    @abc.abstractmethod
    def prepare_data(self) -> None:
        """
        Prepare data for the dataset. Must be implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def table_dataset(self) -> Dataset:
        """
        Get the table dataset containing all tables.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def passage_dataset(self) -> Dataset:
        """
        Get the passage dataset containing all passages.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abc.abstractmethod
    def table_id_column(self) -> str:
        """
        Return the column name for table IDs.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @cached_property
    def table_dataset_id_to_idx(self) -> Dict[str, int]:
        """
        Create a mapping from table IDs to their indices in the table dataset.
        """
        table_dataset_ids: List[str] = list(self.table_dataset[self.table_id_column])
        return id_to_idx(table_dataset_ids)

    @property
    @abc.abstractmethod
    def passage_id_column(self) -> str:
        """
        Return the column name for passage IDs.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @cached_property
    def passage_dataset_id_to_idx(self) -> Dict[str, int]:
        """
        Create a mapping from passage IDs to their indices in the passage dataset.
        """
        passage_dataset_ids: List[str] = list(
            self.passage_dataset[self.passage_id_column]
        )
        return id_to_idx(passage_dataset_ids)

    @property
    def data_instances_path(self) -> str:
        """
        Get the path where processed data instances are stored.
        """
        return os.path.join(self.cfg.data_dir_path, self.name, "data_instances.pkl")

    @cached_property
    def collator(self) -> BaseCollator:
        """
        Get the dataset collator for batching.
        """
        return BaseCollator(global_cfg=self.global_cfg)
