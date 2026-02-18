# -*- coding: utf-8 -*-

# The built-in lib
import os
import json
import logging
from typing import Dict, Any

# The third party lib
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from src.benchmark import DataModule

import nltk

nltk.data.path.insert(0, "/root/nltk_data")
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
# Lib in Project
from src.model.hpropro.utils import *
from src.model.hpropro.query_api import query_API
from src.prompt.model.hpropro.system_prompt import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NOT_TABLE,
)
from src.prompt.model.hpropro.simplify_query import SIMPLIFY_QUERY

from json import JSONDecoder

from src.model.hpropro.process_table import linearize_table

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME

from src.model.hpropro.retriever import Question_Passage_Retriever
from src.model.hpropro import embedding_cache
from src.model.hpropro.dataset_adapter import DatasetAdapter

import tiktoken

tokenizer = tiktoken.encoding_for_model(
    "gpt-3.5-turbo",
)


def estimate_reserved_token_count(
    system_prompt: str, question: str, footer: str
) -> int:
    return (
        len(tokenizer.encode(system_prompt))
        + len(tokenizer.encode(question))
        + len(tokenizer.encode(footer))
    )


def resolveOverflow(tablesTextList=None, tableNameList=None):
    remaining_budget = 13000
    tok = tokenizer
    tablesTextList = tablesTextList or []
    tableNameList = tableNameList or []

    num_tables = len(tablesTextList)
    if num_tables > 0:
        table_budget = remaining_budget // num_tables
    else:
        table_budget = remaining_budget

    total_tokens = []
    truncated_text = ""

    for idx, text in enumerate(tablesTextList):
        tokens = tok.encode(text)
        length = len(tokens)

        if length > table_budget:
            tokens = tokens[:table_budget]
            length = table_budget

        table_name = tableNameList[idx] if idx < len(tableNameList) else f"table{idx+1}"

        truncated_table_text = tok.decode(tokens)
        truncated_text += "[Table]\n"
        truncated_text += f"Table Name: {table_name}\n"
        truncated_text += f"{truncated_table_text}\n"

        total_tokens.extend(tokens)

    return truncated_text


def extract_json_object(text, decoder=JSONDecoder()):
    """Find and return the first valid JSON object (dict) from a text string"""
    pos = 0
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            return result
        except ValueError:
            pos = match + 1
    return None


def create_test_prompt(
    example: Dict[str, Any],
    simplify,
    adapter,
    qs_pas_retriever,
    model,
):
    """Generate the prompt for the test case.

    Args:
        example (Dict[str, Any]): an example from the hybridqa dataset (json file)

    Returns:
        str: The prompt of the test case
        pd.dataframe: The table in the pandas dataframe format.
    """
    table_name_list = (
        example["table"] if isinstance(example["table"], list) else [example["table"]]
    )
    table_name_list = [t for t in table_name_list if t not in adapter.grounding_tables]

    if len(table_name_list) == 1:
        table_name = table_name_list[0]
        if not adapter.preprocessed_table_repr:  # OTTQA or no preprocessed
            table_str, table_pd = linearize_table(example["table_data"][table_name])
            table_pd_list = table_pd
            truncated_table_text = resolveOverflow([table_str], [table_name])
        else:
            table_str = adapter.preprocessed_table_repr[table_name]["linearized"]
            table_pd = adapter.preprocessed_table_repr[table_name]["dataframe"]
            table_pd_list = table_pd
            truncated_table_text = resolveOverflow([table_str], [table_name])
        table_text = truncated_table_text

        if adapter.has_link_to_passage(table_name, example):
            if not adapter.preprocessed_table_repr:  # OTTQA
                table_data = example["table_data"][table_name]
            else:
                table_file = os.path.join(
                    adapter.tables_json_output_path, f"{table_name}.json"
                )
                table_data = json.load(open(table_file, "r"))
            try:
                retrieved_knowledge = qs_pas_retriever.retriever_hybridqa(
                    example, table_data, adapter, table_name_list
                )
            except Exception as e:
                retrieved_knowledge = []
        else:
            retrieved_knowledge = []
    else:
        table_pd_list = {}
        table_str_list = []
        table_data_list = []
        for i, table_name in enumerate(table_name_list):
            if not adapter.preprocessed_table_repr:  # OTTQA
                table_str, table_pd = linearize_table(example["table_data"][table_name])
            else:
                table_str = adapter.preprocessed_table_repr[table_name]["linearized"]
                table_pd = adapter.preprocessed_table_repr[table_name]["dataframe"]
            table_str_list.append(table_str)
            table_pd_list[table_name] = table_pd
            if adapter.has_link_to_passage(table_name, example):
                if not adapter.preprocessed_table_repr:  # OTTQA
                    table_data = example["table_data"][table_name]
                else:
                    table_file = os.path.join(
                        adapter.tables_json_output_path, f"{table_name}.json"
                    )
                    table_data = json.load(open(table_file, "r"))
                table_data_list.append(table_data)
        truncated_table_text = resolveOverflow(table_str_list, table_name_list)
        table_text = truncated_table_text
        try:
            retrieved_knowledge = qs_pas_retriever.retriever_hybridqa(
                example, table_data_list, adapter, table_name_list
            )
        except Exception as e:
            retrieved_knowledge = []

    """
    Simplify the query.
    """
    question = example["question"]
    if len(retrieved_knowledge) != 0 and simplify:
        simplify_prompt = SIMPLIFY_QUERY
        simplify_prompt = simplify_prompt.replace("[QUERY]", question)
        simplify_prompt = simplify_prompt.replace(
            "[KNOWLEDGE]", "\n".join(retrieved_knowledge)
        )
        question = query_API(simplify_prompt, model=model)

    question_text = question
    if len(table_name_list) == 1:
        code_text = "\nFirst describe your solution to clarify your thinking, and write python code to solve this problem like:\n```python\ndef solve(table) -> str:\n    xxx\n```\n"
    else:
        code_text = """
First describe your solution to clarify your thinking, and write python code to solve this problem like:
```python
def solve(tables) -> str:
# tables is a dictionary where each key is a table name (str), and each value is a pandas DataFrame.
xxx
```
"""
    prompt = f"""
    Now deal with this:
    {table_text}
    [Question]
    {question_text}
    [Code]
    {code_text}
    """

    return prompt, table_pd_list


def create_test_prompt_for_passages_only(
    example: Dict[str, Any],
    qs_pas_retriever,
    adapter,
):
    prompt = "\nNow deal with this:\n"
    prompt += "[Question]\n"
    question = example["question"]

    footer_text = """
Output Format:
Your output must be in JSON format with the key:
- Final Answers: ["<your_answer1>", "<your_answer2>", ......]

Return the results in a FLAT json.
*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""
    passage_budget = 13000

    all_passages = qs_pas_retriever.retriever_from_all_passages(example, adapter)
    selected_passages = []
    token_acc = 0
    for passage in all_passages:
        passage_tokens = len(tokenizer.encode(passage))
        if token_acc + passage_tokens > passage_budget:
            break
        selected_passages.append(passage)
        token_acc += passage_tokens

    prompt += question
    prompt += "\n[Retrieved Relevant Passages]\n"
    prompt += "\n".join(selected_passages)
    prompt += '\nPlease answer the question directly based on the above passages. If the answer cannot be found, respond with "None"'
    prompt += footer_text

    return prompt, None


def run_single_case(example, args, adapter) -> dict:

    # Initialize summary data cache and retriever based on benchmark config
    if not adapter.has_individual_text:
        if args.get("passage_store_path") and os.path.exists(args.passage_store_path):
            embedding_cache.ensure_summary_loaded(args.passage_store_path)
    qs_pas_retriever = Question_Passage_Retriever()

    table_name_list = (
        example["table"] if isinstance(example["table"], list) else [example["table"]]
    )
    multi_table = len(table_name_list) > 1
    passages_only_question = (
        all(table_name in adapter.grounding_tables for table_name in table_name_list)
        or len(table_name_list) == 0
    )

    if passages_only_question:
        system_prompt = SYSTEM_PROMPT_NOT_TABLE
        test_prompt, table_pd_list = create_test_prompt_for_passages_only(
            example,
            qs_pas_retriever=qs_pas_retriever,
            adapter=adapter,
        )
    else:
        system_prompt = SYSTEM_PROMPT
        # Add few-shot prompt
        few_shot_prompt = ""
        if args.shot_num > 0:  # In the few-shot setting
            few_shot_prompt = create_few_shot_code_prompt(
                args.shot_num, args.few_shot_case_path
            )
        system_prompt += "\n\n" + few_shot_prompt
        test_prompt, table_pd_list = create_test_prompt(
            example,
            simplify=args.simplify,
            adapter=adapter,
            qs_pas_retriever=qs_pas_retriever,
            model=args.model,
        )

    """Run the model to generate the code."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_prompt},
    ]
    response = query_API(messages, model=args.model, temperature=args.temperature)
    logging.info(
        "\n===================\nGenerated Content:\n-------------------\n"
        f"{response}\n===================\n"
    )

    if not passages_only_question:  ## Table Access Question
        response_code = parser_code_from_response(response)

        logging.info(
            "\n===================\nGenerated Code:\n-------------------\n"
            f"{response_code}\n===================\n"
        )
        ans = execute_generated_code(
            response_code, table_pd_list, multi_table, adapter, example
        )
        traceback_record = ""
        if ans != "" and ans != None:
            if ans.startswith("Traceback"):
                traceback_record = ans
                ans = None
        golden_answer = example["answer"] if "answer" in example else None

        logging.info(
            "\n=========================\n"
            f"answer: {ans}\n"
            f"golden_answer = {golden_answer}\n"
            "=========================\n"
        )

        result = {
            "question_id": example["question_id"],
            "query": example["question"],
            "full_prompt": system_prompt + test_prompt,
            "code": response_code,
            "pred": ans,
            "golden_answer": golden_answer,
        }

        refined_code = refined_answer = ""

        if args.reflection and (
            ans
            in [
                "",
                "None",
                "NOT_AVAILABLE",
                "NOT_FOUND",
                "null",
                "Execute_Failed",
            ]
            or ans == None
        ):
            logging.info(
                "\n=========================\n"
                "Reflection\n"
                "---------------------\n"
                f"trackback_record: {traceback_record}\n"
            )
            refined_code = refine(
                system_prompt + test_prompt, response_code, traceback_record, args.model
            )
            logging.info(
                "\n=========================\n"
                f"Refined Code:\n{refined_code}\n"
                "=========================\n"
            )

            if multi_table:
                refined_answer = execute_generated_code(
                    refined_code + "\n\nans=solve(tables)",
                    table_pd_list,
                    multi_table,
                    adapter,
                    example,
                )
            else:
                refined_answer = execute_generated_code(
                    refined_code + "\n\nans=solve(table)",
                    table_pd_list,
                    multi_table,
                    adapter,
                    example,
                )

            if refined_answer is not None and str(refined_answer).startswith(
                "Traceback"
            ):
                refined_answer = None
            logging.info(
                f"\nRefined answer: {refined_answer}\n"
                f"Golden Answer: {golden_answer}\n"
                "========================="
            )
            result["refined_code"] = refined_code
            result["old_pred"] = result["pred"]
            result["pred"] = refined_answer if refined_answer is not None else ""
            result["traceback"] = traceback_record
    else:
        response_obj = extract_json_object(response)
        if response_obj is None:
            response_obj = {}
        result = {
            "question_id": example["question_id"],
            "query": example["question"],
            "full_prompt": system_prompt + test_prompt,
            "pred": response_obj.get("Final Answers", ""),
            "golden_answer": example["answer"],
        }
    return result


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    model_conf = cfg.model

    output_dir = model_conf.output_path
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "qa_results_retrieval.json")
    error_output_path = os.path.join(output_dir, "errors_retrieval.json")

    # Load benchmark via DataModule
    data_module = DataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.test_dataset

    # Create adapter
    adapter = DatasetAdapter(dataset, cfg)

    log_level = logging.INFO if model_conf.logging else logging.WARNING
    logging.basicConfig(level=log_level)

    generated_result = []
    errors = []

    for idx in tqdm(range(len(dataset)), desc="Running cases"):
        data_tuple = dataset[idx]
        example = adapter.to_example(data_tuple)

        try:
            result = run_single_case(example, model_conf, adapter)
            generated_result.append(result)
        except Exception as e:
            logging.error(
                f"Error processing question {example.get('question_id', 'unknown')}: {e}"
            )
            errors.append(
                {
                    "question_id": example.get("question_id", "unknown"),
                    "error": str(e),
                }
            )
            continue

    # Save results
    if generated_result:
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(generated_result, fout, indent=2, ensure_ascii=False)
        logging.info(f"Saved {len(generated_result)} results to {output_path}")

    if errors:
        with open(error_output_path, "w", encoding="utf-8") as fout:
            json.dump(errors, fout, indent=2, ensure_ascii=False)
        logging.info(f"Saved {len(errors)} errors to {error_output_path}")

    logging.info("All finished.")


if __name__ == "__main__":
    main()
