import logging
import lightning as L
from omegaconf import DictConfig
from functools import cached_property
from src.benchmark.registry import DATASET_REGISTRY
from torch.utils.data import DataLoader
from src.benchmark.base_dataset import BaseDataset

logger = logging.getLogger("DataModule")


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        if cfg.get("seed") is not None:
            L.seed_everything(cfg.seed, workers=True)

    @cached_property
    def test_dataset(self) -> BaseDataset:
        dataset_name = self.cfg.benchmark.name
        dataset: BaseDataset = DATASET_REGISTRY[dataset_name](
            cfg=self.cfg.benchmark, global_cfg=self.cfg
        )
        return dataset

    def prepare_data(self) -> None:
        """Downloads the dataset if not already present.
        This method is called only on a single process in distributed training.
        """
        dataset = self.test_dataset
        dataset.prepare_data()
        return None

    def setup(self, stage: str | None = None) -> None:
        """Loads and preprocesses the dataset for training.
        This method is called on every process in distributed training.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'. Currently unused.
        """
        dataset = self.test_dataset
        dataset.setup()
        return None

    def test_dataloader(self) -> DataLoader:
        dataset = self.test_dataset
        shuffle = self.cfg.benchmark.get("dataloader", {}).get("shuffle", False)
        dataloader = DataLoader(dataset, collate_fn=dataset.collator, shuffle=shuffle)
        return dataloader
