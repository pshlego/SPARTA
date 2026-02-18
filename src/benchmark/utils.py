from pathlib import Path
from typing import List, Dict


def id_to_idx(ids: List[str]) -> Dict[str, int]:
    """
    Create a mapping from IDs to their indices.
    """
    return {_id: idx for idx, _id in enumerate(ids)}


def get_json_files_from_dir(dir_path: str) -> List[str]:
    """
    Get a list of all JSON files in the specified directory.

    Args:
        dir_path (str): The directory path to search for JSON files.

    Returns:
        List[str]: A list of JSON file paths.
    """
    source_tables_path = Path(dir_path)
    return sorted(source_tables_path.rglob("*.json"))
