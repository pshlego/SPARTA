import os
import sys
import json
import time
import hydra
import logging
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from requests.structures import CaseInsensitiveDict
from methods.utils import *
from methods.nested_query_generator import *
from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME


def run_generator(
    data_manager, schema, column_info, args, rng, table_info, logger, log_step=1
):
    all_table_set = set(schema["join_tables"])
    join_clause_list = schema["join_clauses"]
    join_keys = schema["join_keys"]

    join_key_list = list()
    for table, cols in join_keys.items():
        for col in cols:
            join_key_list.append(f"{table}.{col}")

    t1 = time.time()
    t2 = time.time()

    lines = list()

    dtype_dict = CaseInsensitiveDict(column_info["dtype_dict"])

    (
        args.IDS,
        args.HASH_CODES,
        args.NOTES,
        args.CATEGORIES,
        args.FOREIGN_KEYS,
        args.INVALID_IDS,
    ) = set_col_info(column_info)
    
    if args.hyperparams.is_agg:
        # remove '*' and deduplicate
        args.INVALID_IDS = [x for x in args.INVALID_IDS if x != "*"]
        args.INVALID_IDS = list(set(args.INVALID_IDS))

    pbar = tqdm(total=args.num_queries)

    num_success = 0
    num_iter = 0

    generated_queries = list()
    if os.path.exists(args.previous_generated_queries_path):
        with open(args.previous_generated_queries_path, "r") as fp:
            generated_queries = fp.readlines()
            generated_queries = [
                block.replace("\n", "").strip() for block in generated_queries
            ]

    with open(args.inner_query_block_wo_agg_path, "r") as fp:
        inner_query_blocks_wo_agg = fp.readlines()
        inner_query_blocks_wo_agg = [
            block.replace("\n", "").strip() for block in inner_query_blocks_wo_agg
        ]

    with open(args.inner_query_block_w_agg_path, "r") as fp:
        inner_query_blocks_w_agg = fp.readlines()
        inner_query_blocks_w_agg = [
            block.replace("\n", "").strip() for block in inner_query_blocks_w_agg
        ]

    if args.approach_name == "oneshotk":
        llm_query_generator = OneShotKNestedQueryGenerator(
            args,
            data_manager,
            rng,
            all_table_set,
            table_info,
            dtype_dict,
            join_key_list,
            join_clause_list,
            inner_query_blocks_wo_agg,
            inner_query_blocks_w_agg,
            logger,
        )
    elif args.approach_name == "postorder":
        llm_query_generator = PostOrderNestedQueryGenerator(
            args,
            data_manager,
            rng,
            all_table_set,
            table_info,
            dtype_dict,
            join_key_list,
            join_clause_list,
            inner_query_blocks_wo_agg,
            inner_query_blocks_w_agg,
            logger,
        )

    llm_query_generator.generated_sql_query_set = set(generated_queries)

    while num_success < args.num_queries:
        num_iter += 1
        if num_success == 0 and num_iter > 10000:
            logger.warning("This type might be cannot be generated..")
            break

        if args.debug:
            line = llm_query_generator.generate()
        else:
            try:
                line = llm_query_generator.generate()
                if line is None:
                    continue
            except Exception as e:
                logger.error(e)
                continue

            if line[0] == "expired\n":
                logger.warning("Exceeded the maximum number of calls to the LLM.")
                break

        lines = lines + line
        pbar.update(len(line))
        num_success += len(line)

        if (num_success + 1) % log_step == 0:
            with open(args.output_path, "at") as writer:
                writer.writelines(lines)
                lines = list()

            cur_time = time.time()
            time_diff = cur_time - t1
            time_diff_per_log_step = cur_time - t2
            txt = f"{num_success} queries are done. time: takes {time_diff:.2f}s\t{time_diff_per_log_step:.2f}s per {log_step}\tideal_calls: {llm_query_generator.ideal_calls}\ttotal_calls: {llm_query_generator.total_calls}\ntotal_tokens: {llm_query_generator.total_tokens}\nerror_dict: {llm_query_generator.error_dict}"
            logger.info(txt)
            t2 = time.time()

    pbar.close()

    with open(args.output_path, "at") as writer:
        writer.writelines(lines)

    cur_time = time.time()

    txt = f"Done. takes {cur_time-t1:.2f}s\t{cur_time-t2:.2f}s\n"
    logger.info(txt)


def setup_logger(args):
    formatter = logging.Formatter(
        "[[ %(levelname)s ]]::%(asctime)s::%(funcName)s::%(lineno)d - %(message)s"
    )

    consol_handler = logging.StreamHandler(sys.stdout)
    consol_handler.setLevel(logging.DEBUG)
    consol_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(args.log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(consol_handler)
    logger.addHandler(file_handler)
    return logger


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    # get arguments
    args = cfg.sparta.query_generation
    # setup logger
    logger = setup_logger(args)
    # set random seed
    rng = np.random.RandomState(args.seed)
    # load schema
    SCHEMA = json.load(open(args.schema_path))
    schema = CaseInsensitiveDict(lower_case_schema_data(SCHEMA[args.db]))
    # connect to database
    data_manager, table_info = connect_data_manager(
        "localhost", args.port, "postgres", "postgres", schema
    )
    # load column info
    COL_INFO = json.load(open(args.dtype_path))
    column_info = COL_INFO[schema["dataset"]]
    # run query generator
    run_generator(data_manager, schema, column_info, args, rng, table_info, logger)


if __name__ == "__main__":
    main()
