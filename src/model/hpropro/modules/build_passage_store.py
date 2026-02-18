"""Generate passage_store.json -- a flat dict of {passage_key: passage_text}.

Usage:
    python -m src.model.hpropro.modules.build_passage_store benchmark=sparta benchmark.domain=nba
    python -m src.model.hpropro.modules.build_passage_store benchmark=hybridqa
    python -m src.model.hpropro.modules.build_passage_store benchmark=ottqa
"""

import os
import json
import logging

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from src.benchmark import DataModule

logger = logging.getLogger("BuildPassageStore")


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)

    benchmark_name = cfg.benchmark.name
    output_path = cfg.model.passage_store_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        logger.info(f"Passage store already exists at {output_path}")
        return

    logger.info(f"Building passage store for benchmark: {benchmark_name}")

    # Load benchmark
    data_module = DataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.test_dataset

    passage_store = {}
    pid_col = dataset.passage_id_column

    if benchmark_name == "sparta":
        # SPARTA: text_data.json keyed by pid
        for row in dataset.passage_dataset:
            pid = row[pid_col]
            if pid == "dummy":
                continue
            passage_store[pid] = row["text"]

    elif benchmark_name == "hybridqa":
        # HybridQA: passage dataset keyed by /wiki/Entity_Name
        for row in dataset.passage_dataset:
            pid = row[pid_col]
            passage_store[pid] = row["text"]

    elif benchmark_name == "ottqa":
        # OTT-QA: gold passage dataset
        if hasattr(dataset, "gold_passage_dataset"):
            gold_pid_col = dataset.gold_passage_id_column
            for row in dataset.gold_passage_dataset:
                pid = row[gold_pid_col]
                passage_store[pid] = row["text"]
        else:
            for row in dataset.passage_dataset:
                pid = row[pid_col]
                passage_store[pid] = row["text"]

    else:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}")

    logger.info(f"Saving {len(passage_store)} passages to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(passage_store, f, ensure_ascii=False)

    logger.info("Done.")


if __name__ == "__main__":
    main()
