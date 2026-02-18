import json
import hydra
import datetime
import requests
import pandas as pd
from tqdm import tqdm
from decimal import Decimal
from omegaconf import DictConfig
from src.benchmark.sparta.QueryGeneration.methods.utils import PostgreSQLDatabase

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME


def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj


def convert_dates_to_isoformat(lst):
    return [
        elem.isoformat() if isinstance(elem, datetime.date) else elem for elem in lst
    ]


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    cfg = cfg.sparta.question_verbalisation

    # with open(cfg.dataset_test, "r") as fp:
    #     single_query_blocks = fp.readlines()
    sql_query_data = json.load(open(cfg.dataset_test, "r", encoding="utf-8"))
    
    single_query_blocks = [datum["sql_query"] for datum in sql_query_data if datum["question"] == "None"]
    data_manager = PostgreSQLDatabase(
        "postgres", "postgres", "localhost", cfg.port, cfg.db_name
    )
    df = pd.read_csv(cfg.examples_path)
    prompts_all = df["prompt"].tolist()
    llama_sampling_params = {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_new_tokens": cfg.max_new_tokens,
    }
    translation_data = []
    for prompt in tqdm(prompts_all):
        prompt_with_inst = f"""
        ### System: You are an intelligent SQL Code assistant who effectively translates the intent and logic of the SQL queries into natural language that is easy to understand.
        ### User: Convert the given SQL query into a clear and concise natural language query.  Ensure that the request accurately represents the actions specified in the SQL query and is easy to understand for someone without technical knowledge of SQL. 
        ### Examples: 
        {prompt} """
        response_data = requests.post(
            cfg.llm_addr,
            json={"text": [prompt_with_inst], "sampling_params": llama_sampling_params},
        ).json()
        completion = response_data[0]["text"] if response_data else ""
        completion = completion.split("###")[0].replace('"', "").strip()
        translation_data.append(completion)

    final_data = []
    count = 0
    for translation_datum, single_query_block in zip(
        translation_data, single_query_blocks
    ):
        if "order by" in single_query_block.lower():
            modified_single_query_block = single_query_block
        else:
            modified_single_query_block = (
                "SELECT DISTINCT " + single_query_block[len("SELECT ") :]
            )
        # data_manager.execute(modified_single_query_block)
        # execution_results = data_manager.fetchall()
        # answer = convert_dates_to_isoformat(
        #     [
        #         decimal_default(execution_result[0])
        #         for execution_result in execution_results
        #     ]
        # )
        answer = []
        final_data.append(
            {
                "question_id": count,
                "question": translation_datum,
                "sql_query": single_query_block,
                "answer": answer,
            }
        )
        count += 1

    with open(cfg.output_path, "w") as fp:
        json.dump(final_data, fp, indent=4)


if __name__ == "__main__":
    main()
