import hydra
import json
import os
import glob
import torch
import datetime
from tqdm import tqdm
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from src.benchmark import DataModule
from src.model.odyssey.dataset_adapter import OdysseyDatasetAdapter
from src.model.odyssey.odyssey import Odyssey
from src.model.odyssey.prompter import Prompter
from src.model.odyssey.utils import (
    writeJson,
    flatten,
    shard_for_accelerate,
    strip_heavy_keys,
    addEntityExtraction,
    addHeaderExtraction,
    make_accelerator,
)


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    model_conf = cfg.model
    acc = make_accelerator(getattr(model_conf, "use_accelerate", True))

    if getattr(model_conf, "use_accelerate", True):
        outputPath = (
            model_conf.paths.data_dir + f"/e2h_predictions_{acc.process_index}.json"
        )
    else:
        outputPath = model_conf.outputPath_entity_to_header_mapping
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    if os.path.exists(outputPath):
        os.remove(outputPath)

    # Load benchmark
    data_module = DataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.test_dataset
    adapter = OdysseyDatasetAdapter(dataset, cfg)

    all_examples = [adapter.to_example(dataset[i]) for i in range(len(dataset))]
    my_examples = shard_for_accelerate(all_examples, acc)

    model = Odyssey(model_conf, adapter)
    prompter = Prompter(adapter)

    addEntityExtraction(my_examples, model_conf)
    addHeaderExtraction(my_examples, model_conf, adapter)

    print("Entity to Header mapping ... ")
    results = []
    for datum in tqdm(my_examples):
        if isinstance(datum["table"], list):
            table_list = datum["table"]
        else:
            table_list = [datum["table"]]
        passage_only_question = all(
            table_name in adapter.grounding_tables for table_name in table_list
        )
        if passage_only_question:
            output = {}
            prompt = []
        else:
            prompt = prompter.entityToHeaderMappingPrompt_GPT(datum, adapter)
            cnt = 0
            while cnt < 20:
                output = model.inference_GPT(prompt)
                if isinstance(output, dict):
                    break
                cnt += 1
                print(f"Retry {cnt} times")

            if not isinstance(output, dict):
                output = {}

        post_processed_datum = prompter.processEntityToHeaderMapping_GPT(
            prompt, datum, output, adapter
        )
        results.append(post_processed_datum)

        with open(outputPath, "a+") as f:
            save_copy = {k: v for k, v in post_processed_datum.items()}
            strip_heavy_keys(save_copy)
            json.dump(save_copy, f)
            f.write("\n")

    acc.wait_for_everyone()
    if acc.is_main_process:
        if acc.num_processes > 1:
            shard_glob_pattern = os.path.join(
                model_conf.paths.data_dir, "e2h_predictions_*.json"
            )
            shards = sorted(glob.glob(shard_glob_pattern))
            merged = []
            for sp in shards:
                with open(sp, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            if adapter.remove_text_before_save:
                                strip_heavy_keys(obj)
                            merged.append(obj)
        else:
            merged = results
            if adapter.remove_text_before_save:
                for item in merged:
                    strip_heavy_keys(item)

        writeJson(model_conf.outputPath_entity_to_header_mapping, flatten(merged))


if __name__ == "__main__":
    main()
