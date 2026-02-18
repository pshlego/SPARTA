from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SQLMetadata:
    """SPARTA-specific SQL query metadata for analysis and filtering."""

    sql_query: str
    accessed_tables: list[str]
    is_nested: bool = False
    is_aggregated: bool = False
    height: int = 0
    breadth: dict[str, int] | None = None
    max_breadth: int = 0
    query_type: str = ""  # "type" field renamed to avoid Python keyword
    predicate_types: list[str] | None = None
    clause_usage: dict[str, bool] | None = None  # WHERE, GROUP BY, etc.
    aggregate_usage: dict[str, bool] | None = None  # SUM, COUNT, etc.
    source_file: str | None = None


@dataclass
class DataTuple:
    qid: str
    question: str
    answers: list[str]
    pos_table_ids: Optional[list[str]] = field(default_factory=lambda: [None])
    pos_passage_ids: Optional[list[str]] = field(default_factory=lambda: [None])
    answer_node: Optional[list[list[str | list[int] | str | None]]] = field(default_factory=lambda: [None])
    domain: Optional[str] = None
    sql_metadata: Optional[SQLMetadata] = None  # SPARTA-specific


# @dataclass
# class TableTuple:
#     tid: str  # table_id
#     cid: str  # chunk_id
#     title: str
#     text: str
#     header: str
#     data: list[str]


# @dataclass
# class PassageTuple:
#     pid: str
#     title: str
#     text: str
