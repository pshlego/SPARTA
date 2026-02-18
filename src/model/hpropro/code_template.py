from time import sleep
import dateparser
import re
import pandas as pd

from src.prompt.model.hpropro.convert_datetime import CONVERT_DATETIME
from src.prompt.model.hpropro.check_prompt import CHECK_PROMPT
from src.prompt.model.hpropro.extract_info import EXTRACT_INFO
from src.model.hpropro.query_api import query_API


def url2text(url):
    if url.startswith("https://en.wikipedia.org"):
        url = url.replace("https://en.wikipedia.org", "")
    return url.replace("/wiki/", "").replace("_", " ")


def check_same(obj1, obj2, model="gpt-3.5-turbo"):
    """Query the model to compare the two object if they are the same.

    Args:
        obj1 (str): Object 1
        obj2 (str): Object 2
        model (str, optional): The uesd api. Defaults to 'gpt-3.5-turbo'.

    Returns:
        Bool: True if same else False.
    """

    prompt = CHECK_PROMPT
    prompt = prompt.replace("[STRING1]", str(obj1))
    prompt = prompt.replace("[STRING2]", str(obj2))
    result = query_API(prompt, model=model)
    if result == "True":
        return True
    elif result == "False":
        return False


def check(obj1, obj2, op, model="gpt-3.5-turbo"):
    """Query the model to compare the two object

    Args:
        obj1 (str): Object 1
        obj2 (str): Object 2
        op (str): The relation. (in ['=', '<', '>'])
        model (str, optional): The used model. Defaults to 'gpt-3.5-turbo'.

    Returns:
        Bool: True if the relation is tenable else False.
    """
    prompt = ""
    if op not in ["==", ">", "<"]:
        return False
    try:
        val1 = float(obj1)
        val2 = float(obj2)
        if op == "==":
            return val1 == val2
        elif op == ">":
            return val1 > val2
        elif op == "<":
            return val1 < val2
    except (ValueError, TypeError):
        pass

    if op == "==" and isinstance(obj1, str) and isinstance(obj2, str):
        return obj1.strip().lower() == obj2.strip().lower()

    prompt = CHECK_PROMPT
    prompt = prompt.replace("[STRING1]", str(obj1))
    prompt = prompt.replace("[STRING2]", str(obj2))
    prompt = prompt.replace("[REL]", str(op))
    result = query_API(prompt, model=model)
    if result == "True":
        return True
    else:
        return False


def convert_time(time_str):
    """Use the model to convert a unregular time string to a datetime object.

    Args:
        time_str (str): The string containing a time.

    Returns:
        str: A datetime
    """
    prompt = CONVERT_DATETIME
    prompt = prompt.replace("[TIME]", time_str)
    result = query_API(prompt, model="gpt-3.5-turbo")
    result = dateparser.parse(result)
    return result


def extract_info(cell, query):
    """Get the answer from the text from the hyperlink according to the given query.

    Args:
        cell (str): target cell in the table
        query (str): the target information we want the model to get
    """
    source = globals().get("tables", None)
    if source is None:
        source = globals().get("table", None)

    passages_source = globals().get("passage_store", None)

    if cell == "" or cell == None or source is None:
        print(cell)
        return "NOT_AVAILABLE"

    cell_links = []
    if "/wiki/" in cell or "/id/" in cell:
        if "###" in cell:
            cell_links = [x.strip() for x in cell.split("###") if x.strip()]
        else:
            cell_links = [cell]
    else:
        if isinstance(source, dict):
            dataframes = source.values()
        else:
            dataframes = [source]
        for df in dataframes:
            for _, row in df.iterrows():
                for column in df.columns:
                    cell_value = row[column]
                    if isinstance(cell_value, list) and len(cell_value) == 2:
                        value, hyper_str = cell_value
                        if str(value).strip() == str(cell).strip():
                            if isinstance(hyper_str, list):
                                links = [str(x) for x in hyper_str]
                            elif isinstance(hyper_str, str):
                                if "###" in hyper_str:
                                    links = [
                                        x.strip()
                                        for x in hyper_str.split("###")
                                        if x.strip()
                                    ]
                                else:
                                    links = [hyper_str.strip()]
                            else:
                                links = []
                            cell_links.extend(links)

    cell_links = list(
        set(item.split("/id/")[1] if "/id/" in item else item for item in cell_links)
    )
    passages = "\n".join(
        [passages_source[k] for k in cell_links if k in passages_source]
    )

    if not passages.strip():
        print(f"No passages found for cell: {cell}")
        return "NOT_AVAILABLE"

    prompt = EXTRACT_INFO
    prompt = prompt.replace("[CELL_CONTENT]", cell)
    prompt = prompt.replace("[PASSAGES]", passages)
    prompt = prompt.replace("[QUERY]", query)

    result = query_API(prompt, model="gpt-5", extract_info_flag=True)
    print(f"result: {result}")
    pattern = r"So my answer is (.*?)\."
    match = re.search(pattern, result, re.DOTALL)
    if match:
        result = match.group(1)
    else:
        print("wrong format: ", result)
        result = "NOT_AVAILABLE"

    return result
