"""Generate preprocessed_tables.pkl from current benchmark's table data.

Usage:
    python -m src.model.hpropro.modules.preprocess_tables benchmark=sparta benchmark.domain=nba
    python -m src.model.hpropro.modules.preprocess_tables benchmark=hybridqa
    python -m src.model.hpropro.modules.preprocess_tables benchmark=ottqa
"""

import os
import json
import pickle
import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from src.benchmark import DataModule
from src.model.hpropro.process_table import linearize_table

logger = logging.getLogger("PreprocessTables")


def preprocess_from_raw_dir(tables_dir: str) -> dict:
    """Build preprocessed table repr from a directory of raw table JSON files."""
    all_tables = {}
    for filename in tqdm(sorted(os.listdir(tables_dir)), desc="Preprocessing tables"):
        if not filename.endswith(".json"):
            continue
        table_name = filename.replace(".json", "")
        table_path = os.path.join(tables_dir, filename)

        with open(table_path, "r", encoding="utf-8") as f:
            table_data = json.load(f)

        table_str, table_pd = linearize_table(table_data, table_name=table_name)
        all_tables[table_name] = {"linearized": table_str, "dataframe": table_pd}
    return all_tables


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)

    benchmark_name = cfg.benchmark.name
    output_path = cfg.model.preprocessed_tables_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        logger.info(f"Preprocessed tables already exist at {output_path}")
        return

    logger.info(f"Preprocessing tables for benchmark: {benchmark_name}")

    if benchmark_name == "sparta":
        from src.benchmark.benchmarks.sparta import SPARTA

        domain_dir = SPARTA.DOMAIN_TO_DIR[cfg.benchmark.domain]
        tables_dir = os.path.join(
            cfg.benchmark.dataset_dir_path, domain_dir, "source_tables"
        )
        all_tables = preprocess_from_raw_dir(tables_dir)

    elif benchmark_name == "hybridqa":
        tables_dir = cfg.benchmark.raw_table_dir_path
        all_tables = preprocess_from_raw_dir(tables_dir)

    elif benchmark_name == "ottqa":
        tables_dir = cfg.benchmark.raw_gold_table_dir_path
        all_tables = preprocess_from_raw_dir(tables_dir)

    else:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}")

    logger.info(f"Saving {len(all_tables)} preprocessed tables to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(all_tables, f)

    logger.info("Done.")


if __name__ == "__main__":
    main()
