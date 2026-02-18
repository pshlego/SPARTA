import os
import ast
import json
import pickle
import logging
from typing import List, Dict
from datasets import Dataset, load_from_disk
from omegaconf import DictConfig
from functools import cached_property
from src.benchmark.utils import id_to_idx
from src.benchmark.types import DataTuple
from src.benchmark.base_dataset import BaseDataset
from src.benchmark.registry import register_dataset
from src.benchmark.utils import get_json_files_from_dir

logger = logging.getLogger("OTTQADataset")


@register_dataset("ottqa")
class OTTQA(BaseDataset):

    def __init__(self, cfg: DictConfig, global_cfg: DictConfig) -> None:
        super().__init__(cfg, global_cfg)

    def prepare_data(self) -> None:
        # Create data instances and save to disk
        if not os.path.exists(self.data_instances_path):
            raw_dataset = json.load(
                open(self.cfg.raw_dataset_path, "r", encoding="utf-8")
            )

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.data_instances_path), exist_ok=True)

            # Create training data instances from qrels
            logger.info(f"Creating data instances...")
            data_instances = self._create_data_instances(raw_dataset)

            # Save data instances to disk for efficient loading during training
            logger.info(f"Saving data instances to {self.data_instances_path}...")
            with open(self.data_instances_path, "wb") as f:
                pickle.dump(data_instances, f)
            logger.info(f"Data instances saved to {self.data_instances_path}")

            # Remove data instances from memory to free up space
            del data_instances

        return None

    def _create_data_instances(self, raw_dataset: List[dict]) -> List[DataTuple]:
        instances: List[DataTuple] = []
        for raw_datum in raw_dataset:
            qid = raw_datum["question_id"]
            question = raw_datum["question"]
            answers = raw_datum["answer-text"]
            pos_table_ids = raw_datum["table_id"]
            answer_node = raw_datum["answer-node"]

            if not isinstance(answers, list):
                answers = [answers]

            if not isinstance(pos_table_ids, list):
                pos_table_ids = [pos_table_ids]

            pos_passage_ids = []

            for single_node in answer_node:
                node_type = single_node[-1]
                if node_type == "passage":
                    passage_id = single_node[-2]
                    pos_passage_ids.append(passage_id)

            # Create instance as a DataTuple
            instance = DataTuple(
                qid=qid,
                question=question,
                answers=answers,
                pos_table_ids=pos_table_ids,
                pos_passage_ids=pos_passage_ids,
                answer_node=answer_node,
            )
            instances.append(instance)

        return instances

    @property
    def gold_table_id_column(self) -> str:
        return "tid"

    @property
    def gold_passage_id_column(self) -> str:
        return "pid"

    @cached_property
    def gold_table_dataset(self) -> Dataset:
        """Get the table dataset containing all tables for the HybridQA dataset.
        Returns:
            Dataset: A HuggingFace Dataset containing table data with features:
                - tid: Table ID (string)
                - title: Title of the table (string)
                - header: Header information of the table (List[str])
                - data: Data entries of the table (list of strings)
                - text: Text content of the table (string)
        """
        hf_table_data_path = os.path.join(
            self.cfg.source_data_dir, self.cfg.name, "gold_table_data"
        )

        # If dataset already exists, load and return it
        if os.path.exists(os.path.join(hf_table_data_path, "dataset_info.json")):
            return load_from_disk(hf_table_data_path)

        # Otherwise, create the dataset
        os.makedirs(hf_table_data_path, exist_ok=True)

        table_data_paths = get_json_files_from_dir(
            os.path.join(self.cfg.raw_gold_table_dir_path)
        )

        # Collect all table data
        tables = {"tid": [], "title": [], "header": [], "data": [], "text": []}

        for table_data_path in table_data_paths:
            with open(table_data_path, "r", encoding="utf-8") as f:
                table_data = json.load(f)

            tid = table_data["uid"]
            title = table_data["title"]
            header = [column_info[0] for column_info in table_data["header"]]
            data = table_data["data"]
            text = self._gold_table_data_to_text(header, data)

            tables["tid"].append(tid)
            tables["title"].append(title)
            tables["header"].append(header)
            tables["data"].append(data)
            tables["text"].append(text)

        # Create dataset with explicit features to handle nested lists
        from datasets import Features, Value, Sequence

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

    def _gold_table_data_to_text(self, header: List[str], data: List[List[str]]) -> str:
        """Convert table data into a textual representation.

        Args:
            header (List[str]): List of column headers.
            data (List[List[str]]): List of rows, where each row is a list of cell values.

        Returns:
            str: Textual representation of the table.
        """
        text = ", ".join(header) + "\n"
        for row in data:
            cell_values = [cell[0] for cell in row]
            text += ", ".join(cell_values) + "\n"

        return text.strip()

    @cached_property
    def gold_passage_dataset(self) -> Dataset:
        """Get the passage dataset containing all passages for the HybridQA dataset.
        Returns:
            Dataset: A HuggingFace Dataset containing passage data with features:
                - pid: Passage ID (string)
                - title: Title of the passage (string)
                - text: Text content of the passage (string)
        """
        hf_passage_data_path = os.path.join(
            self.cfg.source_data_dir, self.cfg.name, "gold_passage_data"
        )

        # If dataset already exists, load and return it
        if os.path.exists(os.path.join(hf_passage_data_path, "dataset_info.json")):
            return load_from_disk(hf_passage_data_path)

        # Otherwise, create the dataset
        os.makedirs(hf_passage_data_path, exist_ok=True)

        passage_data_paths = get_json_files_from_dir(
            os.path.join(self.cfg.raw_gold_passage_dir_path)
        )

        # Collect all passage data
        passages = {"pid": [], "title": [], "text": []}

        for passage_data_path in passage_data_paths:
            with open(passage_data_path, "r", encoding="utf-8") as f:
                passage_data = json.load(f)

            for pid, content in passage_data.items():
                passages["pid"].append(pid)
                title = pid[len("/wiki/") :].replace("_", " ")
                passages["title"].append(title)
                passages["text"].append(content)

        # Create and save dataset
        dataset = Dataset.from_dict(passages)
        dataset.save_to_disk(hf_passage_data_path)

        return dataset

    @property
    def table_id_column(self) -> str:
        return "tid"

    @property
    def passage_id_column(self) -> str:
        return "pid"

    @cached_property
    def table_dataset(self) -> Dataset:
        """Get the table dataset containing all tables for the OTT-QA dataset.
        Returns:
            Dataset: A HuggingFace Dataset containing table data with features:
                - tid: Table ID (string)
                - title: Title of the table (string)
                - header: Header information of the table (List[str])
                - data: Data entries of the table (List[List[str]]) - cells serialized as "['text', []]"
                - text: Text content of the table (string)
        """
        hf_table_data_path = os.path.join(
            self.cfg.source_data_dir, self.cfg.name, "table_data"
        )

        # If dataset already exists, load and return it
        if os.path.exists(os.path.join(hf_table_data_path, "dataset_info.json")):
            return load_from_disk(hf_table_data_path)

        # Otherwise, create the dataset
        os.makedirs(hf_table_data_path, exist_ok=True)

        table_data_path = self.cfg.raw_source_table_file_path

        with open(table_data_path, "r", encoding="utf-8") as f:
            table_data = json.load(f)

        # Collect all table data
        tables = {"tid": [], "title": [], "header": [], "data": [], "text": []}

        for table_datum in table_data.values():

            tid = table_datum["uid"]
            title = table_datum["title"]
            header = [column_info for column_info in table_datum["header"]]

            # Serialize cells with empty passage IDs: "['text', []]"
            data = []
            for row in table_datum["data"]:
                row_values = [str([str(cell), []]) for cell in row]
                data.append(row_values)

            text = self._table_data_to_text(header, data)

            tables["tid"].append(tid)
            tables["title"].append(title)
            tables["header"].append(header)
            tables["data"].append(data)
            tables["text"].append(text)

        # Create dataset with explicit features to handle nested lists
        from datasets import Features, Value, Sequence

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

        return dataset

    @cached_property
    def passage_dataset(self) -> Dataset:
        """Get the passage dataset containing all passages for the HybridQA dataset.
        Returns:
            Dataset: A HuggingFace Dataset containing passage data with features:
                - pid: Passage ID (string)
                - title: Title of the passage (string)
                - text: Text content of the passage (string)
        """
        hf_passage_data_path = os.path.join(
            self.cfg.source_data_dir, self.cfg.name, "passage_data"
        )

        # If dataset already exists, load and return it
        if os.path.exists(os.path.join(hf_passage_data_path, "dataset_info.json")):
            return load_from_disk(hf_passage_data_path)

        # Otherwise, create the dataset
        os.makedirs(hf_passage_data_path, exist_ok=True)

        passage_data_path = self.cfg.raw_source_passage_file_path

        with open(passage_data_path, "r", encoding="utf-8") as f:
            passage_data = json.load(f)

        # Collect all passage data
        passages = {"pid": [], "title": [], "text": []}

        for pid, content in passage_data.items():
            passages["pid"].append(pid)
            title = pid[len("/wiki/") :].replace("_", " ")
            passages["title"].append(title)
            passages["text"].append(content)

        # Create and save dataset
        dataset = Dataset.from_dict(passages)
        dataset.save_to_disk(hf_passage_data_path)

        return dataset

    @cached_property
    def gold_table_dataset_id_to_idx(self) -> Dict[str, int]:
        """
        Create a mapping from table IDs to their indices in the table dataset.
        """
        table_dataset_ids: List[str] = [
            _id for _id in self.gold_table_dataset[self.gold_table_id_column]
        ]
        return id_to_idx(table_dataset_ids)

    @cached_property
    def gold_passage_dataset_id_to_idx(self) -> Dict[str, int]:
        """
        Create a mapping from passage IDs to their indices in the passage dataset.
        """
        passage_dataset_ids: List[str] = [
            _id for _id in self.gold_passage_dataset[self.gold_passage_id_column]
        ]
        return id_to_idx(passage_dataset_ids)
