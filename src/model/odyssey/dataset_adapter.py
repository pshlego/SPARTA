"""Adapts the current BaseDataset/DataTuple to Odyssey's expected data format.

This is the critical bridging layer that translates the new benchmark system
(BaseDataset, DataTuple, DataModule) into the old-style dicts and method calls
that the Odyssey model code expects.

Pattern follows src/model/hpropro/dataset_adapter.py.
"""

import os
import json
import logging
import orjson
import numpy as np
from typing import Any, Dict, List, Optional
from functools import cached_property
from pathlib import Path
from collections import defaultdict

from omegaconf import DictConfig
from src.benchmark.base_dataset import BaseDataset
from src.benchmark.types import DataTuple
from src.model.odyssey.utils import GTLabelled

logger = logging.getLogger("OdysseyDatasetAdapter")


# ═══════════════════════════════════════════════════════════════════════════════
# SPARTA hardcoded schema constants (moved from old dataset classes)
# ═══════════════════════════════════════════════════════════════════════════════

_ALL_TABLES = {
    "nba": {
        "nba_player_award": ["season", "award", "player_name"],
        "nba_draft_combine_stats": [
            "season",
            "wingspan",
            "standing_reach",
            "percentage_of_body_fat",
            "hand_length",
            "hand_width",
            "standing_vertical_leap",
            "max_vertical_leap",
            "lane_agility_time",
            "three_quarter_sprint",
            "number_of_bench_press",
            "fifteen_corner_left",
            "fifteen_break_left",
            "fifteen_top_key",
            "fifteen_break_right",
            "fifteen_corner_right",
            "college_corner_left",
            "college_break_left",
            "college_top_key",
            "college_break_right",
            "college_corner_right",
            "nba_corner_left",
            "nba_break_left",
            "nba_top_key",
            "nba_break_right",
            "nba_corner_right",
            "off_dribble_fifteen_break_left",
            "off_dribble_fifteen_top_key",
            "off_dribble_fifteen_break_right",
            "off_dribble_college_break_left",
            "off_dribble_college_top_key",
            "off_dribble_college_break_right",
            "on_move_fifteen",
            "on_move_college",
            "player_name",
        ],
        "nba_player_information": [
            "birthdate",
            "birthplace",
            "college",
            "draft_pick",
            "draft_round",
            "draft_team",
            "draft_year",
            "highschool",
            "player_name",
            "position",
            "shoots",
            "weight",
            "birthyear",
            "height",
        ],
        "nba_champion_history": [
            "year",
            "final_sweep",
            "nationality_of_mvp_player",
            "mvp_player_name",
            "western_champion_name",
            "eastern_champion_name",
            "nba_champion_name",
            "nba_vice_champion_name",
        ],
        "nba_team_information": [
            "disbandment_year",
            "team_name",
            "founded_year",
            "arena",
            "arena_capacity",
            "owner",
            "generalmanager",
            "headcoach",
            "dleagueaffiliation",
        ],
        "nba_player_affiliation": [
            "salary",
            "season",
            "team_name",
            "player_name",
        ],
        "nba_game_information": [
            "summary_id",
            "game_place",
            "game_weekday",
            "game_stadium",
            "game_date",
            "home_team_name",
            "visitor_team_name",
        ],
        "nba_team_game_stats": [
            "team_name",
            "team_assist",
            "team_percentage_of_three_point_field_goal_made",
            "team_percentage_of_field_goal_made",
            "team_points",
            "team_points_in_quarter1",
            "team_points_in_quarter2",
            "team_points_in_quarter3",
            "team_points_in_quarter4",
            "team_rebound",
            "team_turnover",
            "summary_id",
        ],
        "nba_player_game_stats": [
            "player_name",
            "number_of_assist",
            "number_of_block",
            "number_of_defensive_rebounds",
            "number_of_three_point_field_goals_attempted",
            "number_of_three_point_field_goals_made",
            "percentage_of_three_point_field_goal_made",
            "number_of_field_goals_attempted",
            "number_of_field_goals_made",
            "percentage_of_field_goal_made",
            "number_of_free_throws_attempted",
            "number_of_free_throws_made",
            "percentage_of_free_throw_made",
            "minutes_played",
            "number_of_offensive_rebounds",
            "number_of_personal_fouls",
            "number_of_points",
            "number_of_rebound",
            "number_of_steal",
            "number_of_turnover",
            "summary_id",
        ],
    },
    "movie": {
        "genre": ["genre", "movie_title"],
        "ratings": ["avg_rating", "total_votes", "median_rating", "movie_title"],
        "movie": [
            "title",
            "year",
            "date_published",
            "duration",
            "country",
            "worlwide_gross_income",
            "languages",
            "production_company",
        ],
        "role_mapping": ["name_id", "category", "name", "movie_title"],
        "director_mapping": ["name_id", "name", "movie_title"],
        "names": [
            "name_id",
            "name",
            "height",
            "date_of_birth",
            "known_for_movie",
        ],
    },
    "medical": {
        "appointments": [
            "appointment_id",
            "appointment_date",
            "appointment_time",
            "reason_for_visit",
            "status",
            "patient_name",
            "doctor_name",
        ],
        "billing": [
            "bill_id",
            "treatment_id",
            "bill_date",
            "amount",
            "payment_method",
            "payment_status",
            "patient_name",
        ],
        "doctors": [
            "doctor_id",
            "specialization",
            "years_experience",
            "hospital_branch",
            "name",
        ],
        "patients": [
            "patient_id",
            "gender",
            "date_of_birth",
            "address",
            "registration_date",
            "insurance_provider",
            "name",
        ],
        "treatments": [
            "treatment_id",
            "appointment_id",
            "treatment_type",
            "description",
            "cost",
            "treatment_date",
        ],
    },
}

_GROUNDING_TABLES = {
    "nba": {"nba_game_information", "nba_team_game_stats", "nba_player_game_stats"},
    "movie": {"role_mapping", "director_mapping", "names"},
    "medical": {"patients", "doctors"},
}

_LINKABLE_HEADER_MAP = {
    "nba": {
        "nba_player_information": "player_name",
        "nba_team_information": "team_name",
    },
    "movie": {"movie": "title"},
    "medical": {
        "billing": "patient_name",
        "appointments": ["patient_name", "doctor_name"],
    },
}

_GT_LINKABLE_SOURCE_TABLES = {
    "nba": ["nba_player_information", "nba_team_information"],
    "movie": ["movie"],
    "medical": ["billing", "appointments"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Adapter class
# ═══════════════════════════════════════════════════════════════════════════════


class OdysseyDatasetAdapter:
    """Adapts current BaseDataset/DataTuple to Odyssey's expected data format."""

    def __init__(self, dataset: BaseDataset, cfg: DictConfig):
        self.dataset = dataset
        self.cfg = cfg
        self._raw_table_cache: Dict[str, dict] = {}

        self.context_mode = getattr(cfg.benchmark, "context_mode", "global")

        # SPARTA: load global text map
        self._text_map_global: Optional[Dict[str, str]] = None
        if self.is_sparta:
            text_data_path = os.path.join(
                cfg.benchmark.dataset_dir_path,
                self.dataset.domain_dir,
                "text_data.json",
            )
            if os.path.exists(text_data_path):
                with open(text_data_path, "r") as f:
                    self._text_map_global = json.load(f)

        # HELIOS retrieval results (query_specific mode)
        self._helios_text_map: Optional[Dict[str, Dict[str, str]]] = None
        if self.is_sparta and self.context_mode == "query_specific":
            helios_path = getattr(cfg.benchmark, "helios_text_path", None)
            if helios_path and os.path.exists(helios_path):
                with open(helios_path, "r") as f:
                    helios_data = json.load(f)
                # {question_id: {passage_id: text}}
                self._helios_text_map = {}
                for item in helios_data:
                    qid = str(item["question_id"])
                    # Strip /id/ prefix to match text_data.json keys
                    self._helios_text_map[qid] = {
                        k.replace("/id/", ""): v
                        for k, v in item.get("text", {}).items()
                    }

        # HybridQA: build _headers_map from raw table JSONs
        self._headers_map: Dict[str, List[str]] = {}
        if self.is_hybridqa:
            self._build_hybridqa_headers_map()

    # ── benchmark detection ──────────────────────────────────────────────

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

    # ── DataTuple -> old example dict ────────────────────────────────────

    def to_example(self, data_tuple: DataTuple) -> dict:
        """Convert a DataTuple into the old Odyssey example dict format."""
        if self.is_sparta:
            return self._to_example_sparta(data_tuple)
        elif self.is_hybridqa:
            return self._to_example_hybridqa(data_tuple)
        elif self.is_ottqa:
            return self._to_example_ottqa(data_tuple)
        else:
            raise ValueError(f"Unsupported benchmark: {self.benchmark_name}")

    def _to_example_sparta(self, dt: DataTuple) -> dict:
        raw_qid = dt.qid
        if ":" in raw_qid:
            raw_qid = raw_qid.split(":", 1)[1]

        table_list = []
        if dt.sql_metadata and dt.sql_metadata.accessed_tables:
            table_list = dt.sql_metadata.accessed_tables

        table_data = {}
        for tid in table_list:
            table_data[tid] = self._load_raw_table(tid)

        # Determine text based on context_mode
        if self.context_mode == "query_specific":
            text = self._get_query_specific_text(raw_qid, table_list)
        else:
            text = self._text_map_global or {}

        return {
            "question_id": raw_qid,
            "question": dt.question,
            "answer": dt.answers,
            "table": table_list,
            "table_data": table_data,
            "text": text,
        }

    def _to_example_hybridqa(self, dt: DataTuple) -> dict:
        table_ids = dt.pos_table_ids or []
        table_ids = [t for t in table_ids if t is not None]

        table_data = {}
        for tid in table_ids:
            table_data[tid] = self._load_raw_table(tid)

        # Build text map from hyperlinks
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
        table_headers = {}
        for tid in table_ids:
            raw = self._load_raw_table(tid)
            table_data[tid] = raw
            table_headers[tid] = [h[0] for h in raw.get("header", [])]

        # Build text map from hyperlinks
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
            "table_headers": table_headers,
            "text": text_map,
        }

    def _get_query_specific_text(self, qid: str, table_list: list) -> dict:
        """Return only passages relevant to the question.

        Priority: HELIOS retrieval results > table-linked passages fallback.
        If the question accesses grounding tables, all passages are needed.
        """
        # 1) HELIOS results take priority
        if self._helios_text_map and qid in self._helios_text_map:
            return self._helios_text_map[qid]

        if not self._text_map_global:
            return {}

        # 2) Grounding tables require all passages
        if self.grounding_tables and self.grounding_tables.intersection(table_list):
            return self._text_map_global

        # 3) Fallback: filter by table's linked_passage_map
        linked_pids: set = set()
        passage_map = self.dataset.linked_passage_map  # SPARTA cached_property
        for tid in table_list:
            for row_passages in passage_map.get(tid, []):
                linked_pids.update(row_passages)
        return {
            pid: self._text_map_global[pid]
            for pid in linked_pids
            if pid in self._text_map_global
        }

    # ── raw table loading ────────────────────────────────────────────────

    def _load_raw_table(self, table_id: str) -> dict:
        if table_id in self._raw_table_cache:
            return self._raw_table_cache[table_id]

        if self.is_sparta:
            domain_dir = self.dataset.domain_dir
            path = os.path.join(
                self.cfg.benchmark.dataset_dir_path,
                domain_dir,
                "source_tables",
                f"{table_id}.json",
            )
        elif self.is_hybridqa:
            path = os.path.join(
                self.cfg.benchmark.raw_table_dir_path, f"{table_id}.json"
            )
        elif self.is_ottqa:
            path = os.path.join(
                self.cfg.benchmark.raw_gold_table_dir_path, f"{table_id}.json"
            )
        else:
            raise ValueError(f"Unsupported benchmark: {self.benchmark_name}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._raw_table_cache[table_id] = raw
        return raw

    # ── passage access ───────────────────────────────────────────────────

    def _get_passage_text_hybridqa(self, link: str) -> Optional[str]:
        return self._hybridqa_passage_store.get(link)

    def _get_passage_text_ottqa(self, link: str) -> Optional[str]:
        return self._ottqa_passage_store.get(link)

    @cached_property
    def _hybridqa_passage_store(self) -> Dict[str, str]:
        store = {}
        passage_dir = self.cfg.benchmark.raw_passage_dir_path
        if not os.path.isdir(passage_dir):
            return store
        for fname in os.listdir(passage_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(passage_dir, fname), "r", encoding="utf-8") as f:
                passage_data = json.load(f)
            for pid, text in passage_data.items():
                store[pid] = text
        return store

    @cached_property
    def _ottqa_passage_store(self) -> Dict[str, str]:
        store = {}
        passage_dir = self.cfg.benchmark.raw_gold_passage_dir_path
        if not os.path.isdir(passage_dir):
            return store
        for fname in os.listdir(passage_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(passage_dir, fname), "r", encoding="utf-8") as f:
                passage_data = json.load(f)
            for pid, text in passage_data.items():
                store[pid] = text
        return store

    # ── HybridQA headers map ─────────────────────────────────────────────

    def _build_hybridqa_headers_map(self):
        """Build _headers_map from raw table JSONs (same as old HybridQA dataset)."""
        wiki_dir = Path(self.cfg.benchmark.raw_table_dir_path)
        for dt in self.dataset.data_instances:
            table_ids = dt.pos_table_ids or []
            for tid in table_ids:
                if tid is None or tid in self._headers_map:
                    continue
                table_file = wiki_dir / f"{tid}.json"
                if not table_file.exists():
                    continue
                with table_file.open("r", encoding="utf-8") as f:
                    table = json.load(f)
                headers = []
                for h in table.get("header", []):
                    if isinstance(h, list) and len(h) >= 1:
                        h0 = str(h[0]).strip()
                        if h0 in ("", "-"):
                            h0 = "-"
                        headers.append(h0)
                self._headers_map[tid] = headers

    # ── Odyssey-specific interface methods ────────────────────────────────

    def get_table_header_mapping(self, data) -> Dict[str, List[str]]:
        """Return {table_name: [header1, header2, ...]} for a given datum."""
        if self.is_sparta:
            domain = self.dataset.domain
            return _ALL_TABLES.get(domain, {})
        elif self.is_hybridqa:
            return self._headers_map
        elif self.is_ottqa:
            return data.get("table_headers", {})
        return {}

    def get_linkable_header(self, data, table_name):
        """Return header name(s) that link table cells to passages."""
        if self.is_sparta:
            domain = self.dataset.domain
            return _LINKABLE_HEADER_MAP.get(domain, {}).get(table_name)
        elif self.is_hybridqa:
            return data.get("headers", {}).get(table_name)
        elif self.is_ottqa:
            return data.get("headers", {}).get(table_name, [])
        return None

    @cached_property
    def grounding_tables(self) -> set:
        """Tables that are passage-only (no actual table content)."""
        if self.is_sparta:
            domain = self.dataset.domain
            return _GROUNDING_TABLES.get(domain, set())
        return set()

    @cached_property
    def source_tables_path(self) -> Optional[str]:
        if self.is_sparta:
            domain_dir = self.dataset.domain_dir
            return os.path.join(
                self.cfg.benchmark.dataset_dir_path, domain_dir, "source_tables"
            )
        elif self.is_hybridqa:
            return self.cfg.benchmark.raw_table_dir_path
        return None

    @property
    def data_has_table_cells(self) -> bool:
        """Whether table data is embedded in the data dict (OTT-QA) vs file-based."""
        if self.is_ottqa:
            return True
        return False

    @property
    def has_individual_ent_doc_graph(self) -> bool:
        """Whether EDGs are per-question (OTT-QA, SPARTA query_specific) vs per-table/global."""
        if self.is_ottqa:
            return True
        if self.is_sparta and self.context_mode == "query_specific":
            return True
        return False

    @property
    def has_individual_text(self) -> bool:
        if self.is_hybridqa or self.is_ottqa:
            return True
        return False

    @property
    def remove_text_before_save(self) -> bool:
        if self.is_sparta:
            return True
        return False

    def get_cell_passage_links(self, table_name, r, c):
        """Get passage links for a specific cell (used in subtableRetrieval join logic)."""
        tbl = self._load_raw_table(table_name)
        cell = tbl["data"][r][c]
        links = []
        if isinstance(cell, list):
            if len(cell) >= 2 and cell[1]:
                links = cell[1]
        return tuple(links) if links else "None"

    # ── EDG / graph caches ───────────────────────────────────────────────

    def _load_node_link(self, d):
        """Convert legacy NetworkX node-link JSON -> GTLabelled."""
        g = GTLabelled()
        id2lbl = {}
        for n in d["nodes"]:
            raw = n.get("id", n)
            try:
                lbl = json.loads(raw)
            except Exception:
                lbl = raw
            id2lbl[n["id"] if "id" in n else n] = lbl
            g._get_v(lbl)
        for l in d["links"]:
            g.add_edge(id2lbl[l["source"]], id2lbl[l["target"]])
        return g

    @cached_property
    def edg_cache(self) -> Dict[str, Any]:
        """Load entity-document graphs from disk."""
        edg_path = self.cfg.model.entityDocumentGraphPath
        edg_dir = Path(edg_path)

        if self.is_sparta:
            if self.context_mode == "query_specific":
                # Per-question files (same pattern as OTT-QA)
                cache = {}
                for f in edg_dir.glob("*.json"):
                    question_id = f.stem
                    with f.open("rb") as jf:
                        ed_json = orjson.loads(jf.read())
                    cache[question_id] = {
                        k: self._load_node_link(v) for k, v in ed_json.items()
                    }
                return cache
            else:
                # Global: single file
                file_path = edg_dir / "entity_doc_graph.json"
                with open(file_path, "rb") as f:
                    ed_json = orjson.loads(f.read())
                return {k: self._load_node_link(v) for k, v in ed_json.items()}

        elif self.is_hybridqa:
            # HybridQA: per-table files
            cache = {}
            for f in edg_dir.glob("*.json"):
                table_id = f.stem
                with f.open("rb") as jf:
                    ed_json = orjson.loads(jf.read())
                cache[table_id] = {
                    k: self._load_node_link(v) for k, v in ed_json.items()
                }
            return cache

        elif self.is_ottqa:
            # OTT-QA: per-question files
            cache = {}
            for f in edg_dir.glob("*.json"):
                question_id = f.stem
                with f.open("rb") as jf:
                    ed_json = orjson.loads(jf.read())
                cache[question_id] = {
                    k: self._load_node_link(v) for k, v in ed_json.items()
                }
            return cache

        return {}

    @cached_property
    def g_all_cache(self):
        """Unified graph(s) from EDGs."""
        if self.is_sparta:
            if self.context_mode == "query_specific":
                # Per-question unified graphs (same pattern as OTT-QA)
                g_all = {}
                entity_nbrs_all = {}
                for question_id, graphs in self.edg_cache.items():
                    G_all = GTLabelled()
                    all_edges = set()
                    entity_neighbors = defaultdict(set)
                    for g in graphs.values():
                        vprop = g._vprop
                        for e in g.g.edges():
                            u_lbl = vprop[e.source()]
                            v_lbl = vprop[e.target()]
                            all_edges.add((u_lbl, v_lbl))
                            entity_neighbors[u_lbl].add(v_lbl)
                            entity_neighbors[v_lbl].add(u_lbl)
                    G_all.add_edges(all_edges)
                    g_all[question_id] = G_all
                    entity_nbrs_all[question_id] = entity_neighbors
                self.__dict__["_entity_nbrs_cache_built"] = entity_nbrs_all
                return g_all
            else:
                # Single unified graph
                G_all = GTLabelled()
                all_edges = set()
                for g in self.edg_cache.values():
                    vprop = g._vprop
                    for e in g.g.edges():
                        u_lbl = vprop[e.source()]
                        v_lbl = vprop[e.target()]
                        all_edges.add((u_lbl, v_lbl))
                G_all.add_edges(all_edges)
                return G_all

        elif self.is_hybridqa:
            # Per-table unified graphs
            g_all = {}
            entity_nbrs_all = {}
            for table_id, graphs in self.edg_cache.items():
                G_all = GTLabelled()
                all_edges = set()
                entity_neighbors = defaultdict(set)
                for g in graphs.values():
                    vprop = g._vprop
                    for e in g.g.edges():
                        u_lbl = vprop[e.source()]
                        v_lbl = vprop[e.target()]
                        all_edges.add((u_lbl, v_lbl))
                        entity_neighbors[u_lbl].add(v_lbl)
                        entity_neighbors[v_lbl].add(u_lbl)
                G_all.add_edges(all_edges)
                g_all[table_id] = G_all
                entity_nbrs_all[table_id] = entity_neighbors
            self.__dict__["_entity_nbrs_cache_built"] = entity_nbrs_all
            return g_all

        elif self.is_ottqa:
            # Per-question unified graphs
            g_all = {}
            entity_nbrs_all = {}
            for question_id, graphs in self.edg_cache.items():
                G_all = GTLabelled()
                all_edges = set()
                entity_neighbors = defaultdict(set)
                for g in graphs.values():
                    vprop = g._vprop
                    for e in g.g.edges():
                        u_lbl = vprop[e.source()]
                        v_lbl = vprop[e.target()]
                        all_edges.add((u_lbl, v_lbl))
                        entity_neighbors[u_lbl].add(v_lbl)
                        entity_neighbors[v_lbl].add(u_lbl)
                G_all.add_edges(all_edges)
                g_all[question_id] = G_all
                entity_nbrs_all[question_id] = entity_neighbors
            self.__dict__["_entity_nbrs_cache_built"] = entity_nbrs_all
            return g_all

        return GTLabelled()

    def get_g_all_cache_for(self, data) -> GTLabelled:
        if self.is_sparta:
            if self.context_mode == "query_specific":
                question_id = data["question_id"]
                return self.g_all_cache.get(question_id, GTLabelled())
            return self.g_all_cache
        elif self.is_hybridqa:
            table_id = data["table"]
            if isinstance(table_id, list):
                table_id = table_id[0]
            return self.g_all_cache.get(table_id, GTLabelled())
        elif self.is_ottqa:
            question_id = data["question_id"]
            return self.g_all_cache.get(question_id, GTLabelled())
        return GTLabelled()

    @cached_property
    def entity_nbrs_cache(self):
        """Entity -> neighbors mapping."""
        reused = self.__dict__.get("_entity_nbrs_cache_built")
        if reused is not None:
            return reused

        if self.is_sparta:
            if self.context_mode == "query_specific":
                # Per-question entity neighbors (same pattern as OTT-QA)
                entity_nbrs_all = {}
                for question_id, graphs in self.edg_cache.items():
                    entity_neighbors = defaultdict(set)
                    for g in graphs.values():
                        vprop = g._vprop
                        for e in g.g.edges():
                            u_lbl = vprop[e.source()]
                            v_lbl = vprop[e.target()]
                            entity_neighbors[u_lbl].add(v_lbl)
                            entity_neighbors[v_lbl].add(u_lbl)
                    entity_nbrs_all[question_id] = entity_neighbors
                return entity_nbrs_all
            else:
                entity_neighbors = defaultdict(set)
                for g in self.edg_cache.values():
                    vprop = g._vprop
                    for e in g.g.edges():
                        u_lbl = vprop[e.source()]
                        v_lbl = vprop[e.target()]
                        entity_neighbors[u_lbl].add(v_lbl)
                        entity_neighbors[v_lbl].add(u_lbl)
                return entity_neighbors

        elif self.is_hybridqa:
            entity_nbrs_all = {}
            for table_id, graphs in self.edg_cache.items():
                entity_neighbors = defaultdict(set)
                for g in graphs.values():
                    vprop = g._vprop
                    for e in g.g.edges():
                        u_lbl = vprop[e.source()]
                        v_lbl = vprop[e.target()]
                        entity_neighbors[u_lbl].add(v_lbl)
                        entity_neighbors[v_lbl].add(u_lbl)
                entity_nbrs_all[table_id] = entity_neighbors
            return entity_nbrs_all

        elif self.is_ottqa:
            entity_nbrs_all = {}
            for question_id, graphs in self.edg_cache.items():
                entity_neighbors = defaultdict(set)
                for g in graphs.values():
                    vprop = g._vprop
                    for e in g.g.edges():
                        u_lbl = vprop[e.source()]
                        v_lbl = vprop[e.target()]
                        entity_neighbors[u_lbl].add(v_lbl)
                        entity_neighbors[v_lbl].add(u_lbl)
                entity_nbrs_all[question_id] = entity_neighbors
            return entity_nbrs_all

        return {}

    def get_entity_nbrs_cache_for(self, data):
        if self.is_sparta:
            if self.context_mode == "query_specific":
                question_id = data["question_id"]
                return self.entity_nbrs_cache.get(question_id, {})
            return self.entity_nbrs_cache
        elif self.is_hybridqa:
            table_id = data["table"]
            if isinstance(table_id, list):
                table_id = table_id[0]
            return self.entity_nbrs_cache.get(table_id, {})
        elif self.is_ottqa:
            question_id = data["question_id"]
            return self.entity_nbrs_cache.get(question_id, {})
        return {}

    # ── table encoding helpers ───────────────────────────────────────────

    @cached_property
    def table_files_for_table_encoding(self) -> Optional[List[Path]]:
        """Return list of table JSON file paths for table encoding, or None for in-memory."""
        if self.is_ottqa:
            return None
        source_path = Path(self.source_tables_path)
        skip = self.grounding_tables
        return sorted(p for p in source_path.glob("*.json") if p.stem not in skip)

    # ── ent-doc graph save ───────────────────────────────────────────────

    def save_ent_doc_graph(self, combined_graph, datum):
        edg_path = self.cfg.model.outputPath_make_ent_doc_graphs
        if self.is_sparta:
            if self.context_mode == "query_specific":
                file_path = os.path.join(edg_path, f"{datum['question_id']}.json")
            else:
                file_path = os.path.join(edg_path, "entity_doc_graph.json")
        elif self.is_hybridqa:
            table_id = datum["table"]
            if isinstance(table_id, list):
                table_id = table_id[0]
            file_path = os.path.join(edg_path, f"{table_id}.json")
        elif self.is_ottqa:
            file_path = os.path.join(edg_path, f"{datum['question_id']}.json")
        else:
            raise ValueError(f"Unsupported benchmark: {self.benchmark_name}")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(combined_graph, f, indent=2, ensure_ascii=False)

    # ── embedding cache helpers (for init_start_points) ──────────────────

    @cached_property
    def _entity_emb_cache(self):
        """Load pre-computed entity embeddings from disk."""
        root = Path(self.cfg.model.rootEmb)

        if self.is_sparta:
            node_path = root / "entity_nodes.json"
            emb_path = root / "entity_embeddings.npy"
            if not node_path.exists() or not emb_path.exists():
                return {}
            with open(node_path) as f:
                nodes = json.load(f)
            embs = np.load(emb_path)
            return {n: e for n, e in zip(nodes, embs)}

        elif self.is_hybridqa:
            nodes_dir = root / "entity_nodes"
            emb_dir = root / "entity_embeddings"
            cache = defaultdict(dict)
            if not nodes_dir.exists():
                return cache
            for node_path in nodes_dir.glob("*.json"):
                tid = node_path.stem
                emb_path = emb_dir / f"{tid}.npy"
                with node_path.open() as f:
                    labels = json.load(f)
                embs_arr = np.load(emb_path)
                cache[tid] = {lbl: vec for lbl, vec in zip(labels, embs_arr)}
            return cache

        return {}

    @cached_property
    def _new_entity_cache(self):
        return defaultdict(dict)

    @cached_property
    def _emb_dim(self):
        root = Path(self.cfg.model.rootEmb)
        if self.is_sparta:
            emb_path = root / "entity_embeddings.npy"
            if emb_path.exists():
                return np.load(emb_path, mmap_mode="r").shape[1]
        elif self.is_hybridqa:
            emb_dir = root / "entity_embeddings"
            first = next(emb_dir.glob("*.npy"), None) if emb_dir.exists() else None
            if first is not None:
                return int(np.load(first, mmap_mode="r").shape[1])
        return 768  # default INSTRUCTOR dim

    def _ensure_entity_embs(self, data, labels, ins, bs=1024):
        if not labels:
            return np.empty((0, self._emb_dim), dtype=np.float32)

        ins_prompt = self.cfg.model.insPrompt

        if self.is_sparta:
            miss = [
                l
                for l in labels
                if l not in self._entity_emb_cache and l not in self._new_entity_cache
            ]
            if miss:
                vecs = ins.encode([[ins_prompt, l] for l in miss], batch_size=bs)
                self._new_entity_cache.update(dict(zip(miss, vecs)))
            look = {**self._entity_emb_cache, **self._new_entity_cache}
            return np.stack([look[l] for l in labels])

        elif self.is_hybridqa:
            table_id = data["table"]
            if isinstance(table_id, list):
                table_id = table_id[0]
            base = self._entity_emb_cache.get(table_id, {})
            miss = [
                l
                for l in labels
                if l not in base and l not in self._new_entity_cache.get(table_id, {})
            ]
            if miss:
                vecs = ins.encode([[ins_prompt, l] for l in miss], batch_size=bs)
                if table_id not in self._new_entity_cache:
                    self._new_entity_cache[table_id] = {}
                self._new_entity_cache[table_id].update(dict(zip(miss, vecs)))
                base = {**base, **self._new_entity_cache[table_id]}
            return np.stack(
                [
                    (base[l] if l in base else self._new_entity_cache[table_id][l])
                    for l in labels
                ]
            )

        elif self.is_ottqa:
            # OTT-QA: compute on-demand, no pre-computed cache
            vec = ins.encode([[ins_prompt, "dummy"]], batch_size=1)
            emb_dim = vec.shape[1]
            if not labels:
                return np.empty((0, emb_dim), dtype=np.float32)
            uniq = list(dict.fromkeys(labels))
            vecs = ins.encode([[ins_prompt, l] for l in uniq], batch_size=bs)
            emb_map = dict(zip(uniq, vecs))
            return np.stack([emb_map[l] for l in labels])

        return np.empty((0, self._emb_dim), dtype=np.float32)

    def _ensure_table_embs(self, tbl, nodes, ins, bs=1024):
        ins_prompt = self.cfg.model.insPrompt
        root = Path(self.cfg.model.rootEmb)

        if self.is_ottqa:
            # On-demand for OTT-QA
            nodes = [n for n in nodes if n[-1] == tbl]
            if not nodes:
                vec = ins.encode([[ins_prompt, "dummy"]], batch_size=1)
                return np.empty((0, vec.shape[1]), dtype=np.float32)
            uniq = list(dict.fromkeys(nodes))
            vecs = ins.encode([[ins_prompt, n[0]] for n in uniq], batch_size=bs)
            emb_map = dict(zip(uniq, vecs))
            return np.stack([emb_map[n] for n in nodes])

        # File-based for HybridQA/SPARTA
        tbl_dir = Path(self.cfg.model.outputPath_table_encoding)
        emb_path = tbl_dir / f"{tbl}.npy"
        node_path = tbl_dir / f"{tbl}_nodes.json"

        if emb_path.exists() and node_path.exists():
            embs = np.load(emb_path)
            with open(node_path) as f:
                stored_nodes = json.load(f)

            # Filter to match requested nodes when a subset is passed
            if nodes is not None and len(nodes) < len(stored_nodes):

                def _nkey(n):
                    return tuple(tuple(x) if isinstance(x, list) else x for x in n)

                stored_idx = {_nkey(sn): i for i, sn in enumerate(stored_nodes)}
                indices = [
                    stored_idx[_nkey(n)] for n in nodes if _nkey(n) in stored_idx
                ]
                if indices:
                    return embs[indices]
                return np.empty((0, embs.shape[1]), dtype=np.float32)

            return embs
        return np.empty((0, self._emb_dim), dtype=np.float32)

    def _ensure_header_embs(self, tbl, hdrs, ins, bs=1024):
        ins_prompt = self.cfg.model.insPrompt
        root = Path(self.cfg.model.rootEmb)

        if self.is_ottqa:
            # On-demand for OTT-QA
            if not hdrs:
                vec = ins.encode([[ins_prompt, "dummy"]], batch_size=1)
                return np.empty((0, vec.shape[1]), dtype=np.float32)
            uniq = list(dict.fromkeys(hdrs))
            vecs = ins.encode([[ins_prompt, h] for h in uniq], batch_size=bs)
            emb_map = dict(zip(uniq, vecs))
            return np.stack([emb_map[h] for h in hdrs])

        # File-based for HybridQA/SPARTA
        tbl_dir = Path(self.cfg.model.outputPath_table_encoding)
        emb_path = tbl_dir / f"{tbl}_header.npy"
        node_path = tbl_dir / f"{tbl}_header_nodes.json"

        if emb_path.exists() and node_path.exists():
            embs = np.load(emb_path)
            with open(node_path) as f:
                stored_hdrs = json.load(f)
            # Filter to requested headers
            if set(hdrs) == set(stored_hdrs):
                return embs
            if not hdrs:
                return np.empty((0, embs.shape[1]), dtype=np.float32)
            # Compute specific headers on-demand
            uniq = list(dict.fromkeys(hdrs))
            vecs = ins.encode([[ins_prompt, h] for h in uniq], batch_size=bs)
            emb_map = dict(zip(uniq, vecs))
            return np.stack([emb_map[h] for h in hdrs])

        # Fallback: compute on demand
        if not hdrs:
            return np.empty((0, self._emb_dim), dtype=np.float32)
        uniq = list(dict.fromkeys(hdrs))
        vecs = ins.encode([[ins_prompt, h] for h in uniq], batch_size=bs)
        emb_map = dict(zip(uniq, vecs))
        return np.stack([emb_map[h] for h in hdrs])
