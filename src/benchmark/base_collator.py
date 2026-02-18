import abc
from typing import Any, Dict, List, Union

from omegaconf import DictConfig
from src.benchmark.types import DataTuple


class BaseCollator(abc.ABC):
    """
    Abstract base class for dataset collators.
    Only collates already-prepared fields from dataset items into batch tensors/lists.
    """

    def __init__(self, global_cfg: DictConfig) -> None:
        self.global_cfg: DictConfig = global_cfg

    def __call__(self, data: List[DataTuple]) -> Dict[str, Any]:
        """
        Collate a list of dataset items into a batch. Only collates already-prepared fields.
        Returns a dict with stacked tensors and batched lists.
        """
        batch: Dict[str, Any] = {}
        batch.update(self._collate_qid(data))
        batch.update(self._collate_question(data))
        batch.update(self._collate_answers(data))
        batch.update(self._collate_pos_table_ids(data))
        batch.update(self._collate_pos_passage_ids(data))
        batch.update(self._collate_answer_node(data))
        batch.update(self._collate_domain(data))
        batch.update(self._collate_sql_metadata(data))
        return batch

    def _collate_qid(self, batch: List[DataTuple]) -> Dict[str, Any]:
        """
        Stack query IDs into a list.
        """
        qids = [b.qid for b in batch]
        return {"qid": qids}

    def _collate_question(self, batch: List[DataTuple]) -> Dict[str, Any]:
        """
        Stack questions into a list.
        """
        questions = [b.question for b in batch]
        return {"question": questions}

    def _collate_answers(self, batch: List[DataTuple]) -> Dict[str, Any]:
        """
        Stack answers into a list of lists.
        """
        answers = [b.answers for b in batch]
        return {"answers": answers}

    def _collate_pos_table_ids(self, batch: List[DataTuple]) -> Dict[str, Any]:
        """
        Stack positive table IDs into a list of lists.
        """
        pos_table_ids = [b.pos_table_ids for b in batch]
        return {"pos_table_ids": pos_table_ids}

    def _collate_pos_passage_ids(self, batch: List[DataTuple]) -> Dict[str, Any]:
        """
        Stack positive passage IDs into a list of lists.
        """
        pos_passage_ids = [b.pos_passage_ids for b in batch]
        return {"pos_passage_ids": pos_passage_ids}

    def _collate_answer_node(self, batch: List[DataTuple]) -> Dict[str, Any]:
        """
        Stack answer nodes into a list of lists.
        """
        answer_node = [b.answer_node for b in batch]
        return {"answer_node": answer_node}

    def _collate_domain(self, batch: List[DataTuple]) -> Dict[str, Any]:
        """
        Stack domains into a list.
        """
        domains = [b.domain for b in batch]
        return {"domain": domains}

    def _collate_sql_metadata(self, batch: List[DataTuple]) -> Dict[str, Any]:
        """
        Stack SQL metadata into a list of dictionaries.
        """
        sql_metadata = [b.sql_metadata for b in batch]
        return {"sql_metadata": sql_metadata}
