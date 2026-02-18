import json
from typing import Dict, Any
from threading import Lock
from typing import Optional

# Global cache for summary data; initialized via load_summary_data
SUMMARY_DATA: Dict[str, Any] = {}

_SUMMARY_PATH: Optional[str] = None
_LOCK = Lock()


def load_summary_data(path: str) -> None:
    global SUMMARY_DATA, _SUMMARY_PATH
    with open(path, "r") as f:
        SUMMARY_DATA = json.load(f)
    _SUMMARY_PATH = path


def ensure_summary_loaded(path: str) -> None:
    global _SUMMARY_PATH
    if _SUMMARY_PATH == path and SUMMARY_DATA:
        return
    with _LOCK:
        if _SUMMARY_PATH == path and SUMMARY_DATA:
            return
        load_summary_data(path)
