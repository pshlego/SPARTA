"""Adapts the current BaseDataset/DataTuple to HProPro's expected data format.

This is the critical bridging layer that translates the new benchmark system
(BaseDataset, DataTuple, HuggingFace Datasets) into the old-style dicts and
method calls that the HProPro model code expects.
"""

import os
import ast
import json
import pickle
import logging
from typing import Any, Dict, List, Optional
from functools import cached_property

from omegaconf import DictConfig
from src.benchmark.base_dataset import BaseDataset
from src.benchmark.types import DataTuple
from src.model.hpropro.process_table import linearize_table
from src.model.hpropro import embedding_cache

logger = logging.getLogger("DatasetAdapter")


class DatasetAdapter:
    """Adapts current BaseDataset/DataTuple to HProPro's expected data format."""

    def __init__(self, dataset: BaseDataset, cfg: DictConfig):
        self.dataset = dataset
        self.cfg = cfg
        self._raw_table_cache: Dict[str, dict] = {}
        self._passage_store_cache: Optional[Dict[str, str]] = None

        # Load HELIOS processed data for query-specific SPARTA mode
        self._helios_processed: Optional[Dict[str, dict]] = None
        path = cfg.model.get("helios_processed_path")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._helios_processed = {str(item["question_id"]): item for item in raw}
            logger.info(
                f"Loaded HELIOS processed data: {len(self._helios_processed)} entries from {path}"
            )

    @property
    def benchmark_name(self) -> str:
        return self.dataset.name

    @property
    def is_sparta(self) -> bool:
        return self.benchmark_name == "sparta"

    @property
    def is_hybridqa(self) -> bool:
        return self.benchmark_name == "hybridqa"

    @property
    def is_ottqa(self) -> bool:
        return self.benchmark_name == "ottqa"

    # ---- DataTuple -> old example dict ----

    def to_example(self, data_tuple: DataTuple) -> dict:
        """Convert a DataTuple into the old HProPro example dict format."""
        if self.is_sparta:
            return self._to_example_sparta(data_tuple)
        elif self.is_hybridqa:
            return self._to_example_hybridqa(data_tuple)
        elif self.is_ottqa:
            return self._to_example_ottqa(data_tuple)
        else:
            raise ValueError(f"Unsupported benchmark: {self.benchmark_name}")

    def _to_example_sparta(self, dt: DataTuple) -> dict:
        # Strip domain prefix from qid (e.g., "nba:123" -> "123")
        raw_qid = dt.qid
        if ":" in raw_qid:
            raw_qid = raw_qid.split(":", 1)[1]

        # If HELIOS processed data available for this qid, use per-query mode
        if self._helios_processed and raw_qid in self._helios_processed:
            processed = self._helios_processed[raw_qid]

            # Build table_data from processed tables
            table_data = {}
            table_list = []
            for table_info in processed.get("tables", []):
                tid = table_info["table_id"]
                table_list.append(tid)
                table_data[tid] = {
                    "header": table_info["header"],
                    "data": table_info["data"],
                }

            return {
                "question_id": raw_qid,
                "question": dt.question,
                "answer": dt.answers,
                "table": table_list,
                "table_data": table_data,
                "text": processed.get("text", {}),
            }

        # Fallback: original SQL metadata behavior
        table_list = []
        if dt.sql_metadata and dt.sql_metadata.accessed_tables:
            table_list = dt.sql_metadata.accessed_tables

        table_data = {}
        for tid in table_list:
            table_data[tid] = self.get_raw_table_data(tid)

        return {
            "question_id": raw_qid,
            "question": dt.question,
            "answer": dt.answers,
            "table": table_list,
            "table_data": table_data,
        }

    def _to_example_hybridqa(self, dt: DataTuple) -> dict:
        table_ids = dt.pos_table_ids or []
        table_ids = [t for t in table_ids if t is not None]

        table_data = {}
        for tid in table_ids:
            table_data[tid] = self.get_raw_table_data(tid)

        # Build text map (passage store) for this example
        text_map = {}
        for tid in table_ids:
            raw = table_data[tid]
            for row in raw.get("data", []):
                for cell in row:
                    if (
                        isinstance(cell, list)
                        and len(cell) >= 2
                        and isinstance(cell[1], list)
                    ):
                        for link in cell[1]:
                            passage_text = self._get_passage_text_hybridqa(link)
                            if passage_text:
                                text_map[link] = passage_text

        return {
            "question_id": dt.qid,
            "question": dt.question,
            "answer": dt.answers,
            "table": table_ids[0] if len(table_ids) == 1 else table_ids,
            "table_data": table_data,
            "text": text_map,
        }

    def _to_example_ottqa(self, dt: DataTuple) -> dict:
        table_ids = dt.pos_table_ids or []
        table_ids = [t for t in table_ids if t is not None]

        table_data = {}
        for tid in table_ids:
            table_data[tid] = self.get_raw_table_data(tid)

        # Build text map for OTT-QA
        text_map = {}
        for tid in table_ids:
            raw = table_data[tid]
            for row in raw.get("data", []):
                for cell in row:
                    if (
                        isinstance(cell, list)
                        and len(cell) >= 2
                        and isinstance(cell[1], list)
                    ):
                        for link in cell[1]:
                            passage_text = self._get_passage_text_ottqa(link)
                            if passage_text:
                                text_map[link] = passage_text

        return {
            "question_id": dt.qid,
            "question": dt.question,
            "answer": dt.answers,
            "table": table_ids[0] if len(table_ids) == 1 else table_ids,
            "table_data": table_data,
            "text": text_map,
        }

    # ---- Raw table data access (with hyperlinks preserved) ----

    def get_raw_table_data(self, table_id: str) -> dict:
        """Get raw table data dict {header: [...], data: [...]} for a table_id.

        Loads from raw JSON files to preserve hyperlink info (cell[1]).
        """
        if table_id in self._raw_table_cache:
            return self._raw_table_cache[table_id]

        if self.is_sparta:
            raw = self._load_raw_table_sparta(table_id)
        elif self.is_hybridqa:
            raw = self._load_raw_table_hybridqa(table_id)
        elif self.is_ottqa:
            raw = self._load_raw_table_ottqa(table_id)
        else:
            raise ValueError(f"Unsupported benchmark: {self.benchmark_name}")

        self._raw_table_cache[table_id] = raw
        return raw

    def _load_raw_table_sparta(self, table_id: str) -> dict:
        domain_dir = self.dataset.domain_dir
        path = os.path.join(
            self.cfg.benchmark.dataset_dir_path,
            domain_dir,
            "source_tables",
            f"{table_id}.json",
        )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_raw_table_hybridqa(self, table_id: str) -> dict:
        path = os.path.join(self.cfg.benchmark.raw_table_dir_path, f"{table_id}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_raw_table_ottqa(self, table_id: str) -> dict:
        path = os.path.join(
            self.cfg.benchmark.raw_gold_table_dir_path, f"{table_id}.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- Passage access ----

    def _get_passage_text_hybridqa(self, link: str) -> Optional[str]:
        """Look up passage text for HybridQA by link (e.g., /wiki/Entity_Name)."""
        store = self._get_passage_store_hybridqa()
        return store.get(link)

    def _get_passage_text_ottqa(self, link: str) -> Optional[str]:
        """Look up passage text for OTT-QA by link."""
        store = self._get_passage_store_ottqa()
        return store.get(link)

    @cached_property
    def _hybridqa_passage_store(self) -> Dict[str, str]:
        """Build a flat {pid: text} dict from raw passage dir for HybridQA."""
        store = {}
        passage_dir = self.cfg.benchmark.raw_passage_dir_path
        if not os.path.isdir(passage_dir):
            logger.warning(f"Passage dir not found: {passage_dir}")
            return store

        for fname in os.listdir(passage_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(passage_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                passage_data = json.load(f)
            for pid, text in passage_data.items():
                store[pid] = text
        return store

    def _get_passage_store_hybridqa(self) -> Dict[str, str]:
        return self._hybridqa_passage_store

    @cached_property
    def _ottqa_passage_store(self) -> Dict[str, str]:
        """Build a flat {pid: text} dict from raw passage dir for OTT-QA."""
        store = {}
        passage_dir = self.cfg.benchmark.raw_gold_passage_dir_path
        if not os.path.isdir(passage_dir):
            logger.warning(f"Passage dir not found: {passage_dir}")
            return store

        for fname in os.listdir(passage_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(passage_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                passage_data = json.load(f)
            for pid, text in passage_data.items():
                store[pid] = text
        return store

    def _get_passage_store_ottqa(self) -> Dict[str, str]:
        return self._ottqa_passage_store

    # ---- Preprocessed table repr ----

    @cached_property
    def preprocessed_table_repr(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Load preprocessed_tables.pkl: {table_name: {linearized: str, dataframe: pd.DataFrame}}"""
        # When HELIOS processed data is available, use per-query tables instead
        if self._helios_processed is not None:
            return None
        path = self.cfg.model.get("preprocessed_tables_path")
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    # ---- Methods matching old dataset interface ----

    def get_passage_store(self, example: dict) -> dict:
        """Get passage_store dict {key: passage_text} for code execution."""
        if self.is_hybridqa:
            return example.get("text", {})
        elif self.is_ottqa:
            return example.get("text", {})
        elif self.is_sparta:
            if self._helios_processed is not None:
                # Per-query mode: keys are "/id/123", strip prefix for code_template
                # extract_info strips "/id/" from cell links before lookup
                raw_text = example.get("text", {})
                return {
                    k.split("/id/")[1] if "/id/" in k else k: v
                    for k, v in raw_text.items()
                }
            return embedding_cache.SUMMARY_DATA
        return {}

    def get_text_from_link(self, k: str, data_entry: dict) -> str:
        """Get passage text from link key."""
        if self.is_hybridqa or self.is_ottqa:
            return data_entry.get("text", {}).get(k, "")
        elif self.is_sparta:
            if self._helios_processed is not None:
                # Per-query mode: retriever_hybridqa passes link as-is from cell[1]
                return data_entry.get("text", {}).get(k, "")
            summary_id = k.split("/")[-1]
            return embedding_cache.SUMMARY_DATA.get(summary_id, "")
        return ""

    def has_link_to_passage(self, table_name: str, example: dict) -> bool:
        """Check if table has hyperlinks to passages."""
        if self.is_hybridqa:
            return True
        elif self.is_ottqa:
            # Check if any cell in the table has hyperlinks
            table_data = example.get("table_data", {}).get(table_name)
            if table_data is None:
                return False
            for row in table_data.get("data", []):
                for cell in row:
                    if isinstance(cell, list) and len(cell) >= 2:
                        if isinstance(cell[1], list) and len(cell[1]) > 0:
                            return True
            return False
        elif self.is_sparta:
            if self._helios_processed is not None:
                # Per-query mode: check actual cell links (like OTT-QA mode)
                table_data = example.get("table_data", {}).get(table_name)
                if table_data is None:
                    return False
                for row in table_data.get("data", []):
                    for cell in row:
                        if isinstance(cell, list) and len(cell) >= 2:
                            if isinstance(cell[1], list) and len(cell[1]) > 0:
                                return True
                return False
            return table_name in self.gt_linkable_source_tables
        return False

    # Domain-specific grounding tables (passage-only tables) for SPARTA
    _SPARTA_GROUNDING_TABLES = {
        "nba": {
            "nba_game_information",
            "nba_team_game_stats",
            "nba_player_game_stats",
        },
        "movie": {
            "role_mapping",
            "director_mapping",
            "names",
        },
        "medical": {"patients", "doctors"},
    }

    # Domain-specific tables that have links to passages for SPARTA
    _SPARTA_LINKABLE_TABLES = {
        "nba": ["nba_player_information", "nba_team_information"],
        "movie": ["movie"],
        "medical": ["billing", "appointments"],
    }

    @cached_property
    def grounding_tables(self) -> set:
        """Tables that are passage-only (no actual table content)."""
        if self.is_sparta:
            if self._helios_processed is not None:
                # Per-query mode: HELIOS already selected tables, no grounding needed
                return set()
            domain = self.dataset.domain
            return self._SPARTA_GROUNDING_TABLES.get(domain, set())
        return set()

    @cached_property
    def gt_linkable_source_tables(self) -> list:
        """Tables that have links to passages (for SPARTA)."""
        if self.is_sparta:
            domain = self.dataset.domain
            return self._SPARTA_LINKABLE_TABLES.get(domain, [])
        return []

    @property
    def has_individual_text(self) -> bool:
        """Whether each example has its own text map (vs shared global)."""
        if self.is_hybridqa or self.is_ottqa:
            return True
        if self.is_sparta and self._helios_processed is not None:
            return True
        return False

    @property
    def tables_json_output_path(self) -> Optional[str]:
        """Path to directory containing raw table JSONs.
        Used when loading table data for passage retrieval.
        """
        if self.is_sparta:
            domain_dir = self.dataset.domain_dir
            return os.path.join(
                self.cfg.benchmark.dataset_dir_path, domain_dir, "source_tables"
            )
        elif self.is_hybridqa:
            return self.cfg.benchmark.raw_table_dir_path
        elif self.is_ottqa:
            return self.cfg.benchmark.raw_gold_table_dir_path
        return None
