import os
import ast
import json
import pickle
import logging
from typing import List
from pathlib import Path
from datasets import Dataset, load_dataset, load_from_disk, Features, Value, Sequence
from omegaconf import DictConfig
from functools import cached_property
from src.benchmark.types import DataTuple, SQLMetadata
from src.benchmark.base_dataset import BaseDataset
from src.benchmark.registry import register_dataset

logger = logging.getLogger("SPARTADataset")


@register_dataset("sparta")
class SPARTA(BaseDataset):
    """SPARTA benchmark dataset with domain-specific table and passage data.

    Each instance is initialized with a specific domain (nba, movie, medical).
    The table_dataset and passage_dataset properties return data only for the
    configured domain.
    """

    # Mapping from workload domain names to dataset directory names
    DOMAIN_TO_DIR = {
        "nba": "nba",
        "movie": "imdb",
        "medical": "medical",
    }

    # Tables that have associated text data (passages) for each domain
    DOMAIN_TEXT_TABLES = {
        "movie": ["role_mapping", "director_mapping", "names"],
        "medical": ["patients", "doctors"],
        "nba": [
            "nba_player_game_stats",
            "nba_team_game_stats",
            "nba_game_information",
        ],
    }

    def __init__(self, cfg: DictConfig, global_cfg: DictConfig) -> None:
        super().__init__(cfg, global_cfg)
        self.domain = cfg.domain  # Required: nba | movie | medical
        self.split = cfg.split  # dev | test
        self._validate_domain()

    def _validate_domain(self) -> None:
        if self.domain not in self.DOMAIN_TO_DIR:
            raise ValueError(
                f"Invalid domain: {self.domain}. "
                f"Must be one of {list(self.DOMAIN_TO_DIR.keys())}"
            )

    def setup(self) -> None:
        """Load data instances and optionally filter to modified QIDs only."""
        super().setup()

        qid_filter_path = self.cfg.get("qid_filter_path", None)
        if qid_filter_path is None:
            return

        if not os.path.exists(qid_filter_path):
            raise FileNotFoundError(f"QID filter file not found: {qid_filter_path}")

        with open(qid_filter_path, "r", encoding="utf-8") as f:
            qid_filter_data = json.load(f)

        if self.domain not in qid_filter_data:
            logger.warning(
                f"Domain '{self.domain}' not found in QID filter file. No filtering applied."
            )
            return

        modified_qids = set(qid_filter_data[self.domain].get("union_qids", []))
        if not modified_qids:
            logger.warning(
                f"No union_qids for domain '{self.domain}'. No filtering applied."
            )
            return

        original_count = len(self.data_instances)
        # DataTuple.qid = "{domain}:{question_id}", filter file uses bare question_id
        self.data_instances = [
            dt for dt in self.data_instances if dt.qid.split(":", 1)[1] in modified_qids
        ]
        logger.info(
            f"QID filter: {self.domain} {original_count} -> {len(self.data_instances)} instances "
            f"({len(modified_qids)} modified QIDs in filter)"
        )

    @property
    def domain_dir(self) -> str:
        """Get the dataset directory name for the current domain."""
        return self.DOMAIN_TO_DIR[self.domain]

    @property
    def data_instances_path(self) -> str:
        """Get domain-specific path for data instances."""
        return os.path.join(
            self.cfg.source_data_dir, self.name, self.domain, self.split, "data_instances.pkl"
        )

    def prepare_data(self) -> None:
        """Create and cache data instances for the configured domain."""
        if os.path.exists(self.data_instances_path):
            logger.info(f"Data instances already exist at {self.data_instances_path}")
            return None

        # Load raw workload data from HuggingFace dataset
        logger.info(
            f"Loading HuggingFace dataset '{self.cfg.hf_dataset_name}' "
            f"(domain={self.domain}, split={self.split})..."
        )
        hf_dataset = load_dataset(self.cfg.hf_dataset_name, name=self.domain, split=self.split)
        raw_dataset = list(hf_dataset)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_instances_path), exist_ok=True)

        # Create data instances
        logger.info(f"Creating data instances for domain '{self.domain}'...")
        data_instances = self._create_data_instances(raw_dataset)

        # Save data instances to disk
        logger.info(f"Saving data instances to {self.data_instances_path}...")
        with open(self.data_instances_path, "wb") as f:
            pickle.dump(data_instances, f)
        logger.info(
            f"Saved {len(data_instances)} data instances to {self.data_instances_path}"
        )

        del data_instances
        return None

    def _create_data_instances(self, raw_dataset: List[dict]) -> List[DataTuple]:
        """Create DataTuple instances from raw workload data."""
        instances: List[DataTuple] = []

        for raw_datum in raw_dataset:
            qid = raw_datum['question_id']
            question = raw_datum["question"]
            answers = raw_datum["answer"]

            if not isinstance(answers, list):
                answers = [str(answers)]
            else:
                answers = [str(a) for a in answers]

            # Get accessed tables from SQL metadata
            accessed_tables = raw_datum.get("table", [])
            if not isinstance(accessed_tables, list):
                accessed_tables = [accessed_tables]

            # Create SQLMetadata
            sql_metadata = SQLMetadata(
                sql_query=raw_datum["sql_query"],
                accessed_tables=accessed_tables,
                is_nested=raw_datum.get("is_nested", False),
                is_aggregated=raw_datum.get("is_aggregated", False),
                height=raw_datum.get("height", 0),
                breadth=raw_datum.get("breadth"),
                max_breadth=raw_datum.get("max_breadth", 0),
                query_type=raw_datum.get("type", ""),
                predicate_types=raw_datum.get("predicate_types"),
                clause_usage=raw_datum.get("clause_usage"),
                aggregate_usage=raw_datum.get("aggregate_usage"),
                source_file=raw_datum.get("source_file"),
            )

            instance = DataTuple(
                qid=qid,
                question=question,
                answers=answers,
                domain=self.domain,
                sql_metadata=sql_metadata,
            )
            instances.append(instance)

        return instances

    @cached_property
    def table_dataset(self) -> Dataset:
        """Get the table dataset containing all tables for the current domain.

        Returns:
            Dataset: A HuggingFace Dataset containing table data with features:
                - tid: Table ID (string) - format: "{title}"
                - title: Name of the table (string)
                - header: Column names (List[str])
                - data: Table rows (List[List[str]]) - cells serialized as "['text', [passage_ids]]"
                - text: Linearized text representation (string)
        """
        hf_table_data_path = os.path.join(
            self.cfg.source_data_dir, self.name, self.domain, "table_data"
        )

        # If dataset already exists, load and return it
        if os.path.exists(os.path.join(hf_table_data_path, "dataset_info.json")):
            logger.info(f"Loading cached table dataset from {hf_table_data_path}")
            return load_from_disk(hf_table_data_path)

        # Otherwise, create the dataset
        os.makedirs(hf_table_data_path, exist_ok=True)

        # Path to source tables for this domain
        source_tables_dir = os.path.join(
            self.cfg.dataset_dir_path, self.domain_dir, "source_tables"
        )

        if not os.path.exists(source_tables_dir):
            raise FileNotFoundError(
                f"Source tables directory not found: {source_tables_dir}"
            )

        # Collect all table data
        tables = {
            "tid": [],
            "title": [],
            "header": [],
            "data": [],
            "text": [],
        }

        table_files = list(Path(source_tables_dir).glob("*.json"))
        logger.info(f"Loading {len(table_files)} tables from {source_tables_dir}")

        # add additional dummy table to connect with isolated passage node
        tables["tid"].append("dummy")
        tables["title"].append("dummy")
        tables["header"].append([""])
        tables["data"].append([["['', []]"]])
        tables["text"].append("")

        for table_file in table_files:
            title = table_file.stem
            if title in [
                "nba_player_game_stats",
                "nba_team_game_stats",
                "nba_game_information",
                "doctors",
                "patients",
                "role_mapping",
                "director_mapping",
                "names",
            ]:
                continue

            with open(table_file, "r", encoding="utf-8") as f:
                table_data = json.load(f)

            # Extract header (first element of each header entry)
            header = [col[0] for col in table_data["header"]]

            # Extract data with passage IDs serialized as "['text', [passage_ids]]"
            # Strip '/id/' prefix from passage IDs to match passage_dataset keys
            data = []
            for row in table_data["data"]:
                row_values = []
                for cell in row:
                    if isinstance(cell, list) and len(cell) >= 2:
                        cell_text = str(cell[0])
                        raw_pids = cell[1] if isinstance(cell[1], list) else []
                        passage_ids = [pid.replace("/id/", "") for pid in raw_pids]
                        row_values.append(str([cell_text, passage_ids]))
                    elif isinstance(cell, list) and len(cell) == 1:
                        row_values.append(str([str(cell[0]), []]))
                    else:
                        row_values.append(str([str(cell), []]))
                data.append(row_values)

            # Create linearized text representation
            text = self._table_data_to_text(header, data)

            tables["tid"].append(title)
            tables["title"].append(title)
            tables["header"].append(header)
            tables["data"].append(data)
            tables["text"].append(text)

        # Create dataset with explicit features
        features = Features(
            {
                "tid": Value("string"),
                "title": Value("string"),
                "header": Sequence(Value("string")),
                "data": Sequence(Sequence(Value("string"))),
                "text": Value("string"),
            }
        )

        dataset = Dataset.from_dict(tables, features=features)
        dataset.save_to_disk(hf_table_data_path)
        logger.info(
            f"Saved table dataset with {len(dataset)} tables to {hf_table_data_path}"
        )

        return dataset

    def _table_data_to_text(self, header: List[str], data: List[List[str]]) -> str:
        """Convert table data into a textual representation.

        Args:
            header (List[str]): List of column headers.
            data (List[List[str]]): List of rows, where each row is a list of serialized cells.

        Returns:
            str: Textual representation of the table.
        """
        text = ", ".join(header) + "\n"
        for row in data:
            cell_values = []
            for cell in row:
                try:
                    parsed = ast.literal_eval(cell)
                    if isinstance(parsed, list) and len(parsed) >= 1:
                        cell_values.append(str(parsed[0]))
                    else:
                        cell_values.append(str(cell))
                except (ValueError, SyntaxError):
                    cell_values.append(str(cell))
            text += ", ".join(cell_values) + "\n"

        return text.strip()

    @cached_property
    def passage_dataset(self) -> Dataset:
        """Get the passage dataset containing text data for the current domain.

        Returns:
            Dataset: A HuggingFace Dataset containing passage data with features:
                - pid: Passage ID (string)
                - text: Text content (string)
        """
        hf_passage_data_path = os.path.join(
            self.cfg.source_data_dir, self.name, self.domain, "passage_data"
        )

        # If dataset already exists, load and return it
        if os.path.exists(os.path.join(hf_passage_data_path, "dataset_info.json")):
            logger.info(f"Loading cached passage dataset from {hf_passage_data_path}")
            return load_from_disk(hf_passage_data_path)

        # Otherwise, create the dataset
        os.makedirs(hf_passage_data_path, exist_ok=True)

        # Path to text data for this domain
        text_data_path = os.path.join(
            self.cfg.dataset_dir_path, self.domain_dir, "text_data.json"
        )

        passages = {"pid": [], "text": []}

        if os.path.exists(text_data_path):
            with open(text_data_path, "r", encoding="utf-8") as f:
                text_data = json.load(f)

            logger.info(f"Loading {len(text_data)} passages from {text_data_path}")

            # add dummy passage to connect with isolated table segment node
            passages["pid"].append("dummy")
            passages["text"].append("")

            for pid, text in text_data.items():
                passages["pid"].append(pid)
                passages["text"].append(text)

        else:
            logger.warning(
                f"Text data file not found: {text_data_path}. "
                f"Creating empty passage dataset."
            )

        # Create and save dataset
        dataset = Dataset.from_dict(passages)
        dataset.save_to_disk(hf_passage_data_path)
        logger.info(
            f"Saved passage dataset with {len(dataset)} passages to {hf_passage_data_path}"
        )

        return dataset

    @property
    def table_id_column(self) -> str:
        return "tid"

    @property
    def passage_id_column(self) -> str:
        return "pid"

    @property
    def text_tables(self) -> List[str]:
        """Get the list of tables that have associated text data for this domain."""
        return self.DOMAIN_TEXT_TABLES.get(self.domain, [])

    def get_tables_for_question(self, data_tuple: DataTuple) -> List[str]:
        """Get the table IDs accessed by a question's SQL query."""
        if data_tuple.sql_metadata and data_tuple.sql_metadata.accessed_tables:
            return data_tuple.sql_metadata.accessed_tables
        return []

    @cached_property
    def linked_passage_map(self) -> dict:
        """table_id -> row_list -> col_list (passage list)

        Passage IDs have '/id/' prefix stripped to match passage_dataset keys.
        (text_data.json uses keys without '/id/' prefix)
        """
        save_path = os.path.join(
            self.cfg.source_data_dir, self.name, self.domain, "linked_passage_map.json"
        )

        if os.path.exists(save_path):
            logger.info(f"Loading passage map from JSON: {save_path}")
            with open(save_path, "r", encoding="utf-8") as f:
                return json.load(f)

        source_tables_dir = os.path.join(
            self.cfg.dataset_dir_path, self.domain_dir, "source_tables"
        )

        passage_map = {}
        table_files = list(Path(source_tables_dir).glob("*.json"))

        for table_file in table_files:
            table_id = table_file.stem
            with open(table_file, "r", encoding="utf-8") as f:
                table_data = json.load(f)

            table_rows = []
            for row in table_data.get("data", []):
                current_row_passages = []
                for cell in row:
                    # Strip '/id/' prefix to match passage_dataset keys (text_data.json format)
                    if (
                        isinstance(cell, list)
                        and len(cell) > 1
                        and isinstance(cell[1], list)
                    ):
                        passages = [c.replace("/id/", "") for c in cell[1]]
                    else:
                        passages = []
                    current_row_passages += passages
                table_rows.append(current_row_passages)

            passage_map[table_id] = table_rows

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(passage_map, f, ensure_ascii=False, indent=4)

        logger.info(f"Saved linked_passage_map to {save_path}")
        return passage_map

    @cached_property
    def unlinked_passages(self) -> list:
        """Get passage IDs not linked to any table cell.

        Computed from linked_passage_map without file caching.
        """
        linked_ids = set()
        for table_rows in self.linked_passage_map.values():
            for row_passages in table_rows:
                linked_ids.update(row_passages)

        all_passage_ids = set(self.passage_dataset[self.passage_id_column])
        return [
            pid for pid in all_passage_ids if pid not in linked_ids and pid != "dummy"
        ]
