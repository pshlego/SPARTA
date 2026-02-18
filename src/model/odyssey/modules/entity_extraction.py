import hydra
import json
import os
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
    make_accelerator,
    gather_if_distributed,
)


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    model_conf = cfg.model
    acc = make_accelerator(getattr(model_conf, "use_accelerate", True))

    if getattr(model_conf, "use_accelerate", True):
        outputPath = (
            model_conf.paths.data_dir + f"/ee_predictions_{acc.process_index}.json"
        )
    else:
        outputPath = model_conf.outputPath_entity_extraction
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    if os.path.exists(outputPath):
        os.remove(outputPath)

    # Load benchmark
    data_module = DataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.test_dataset
    adapter = OdysseyDatasetAdapter(dataset, cfg)

    # Convert all DataTuples to old-style dicts
    all_examples = [adapter.to_example(dataset[i]) for i in range(len(dataset))]
    my_examples = shard_for_accelerate(all_examples, acc)

    model = Odyssey(model_conf, adapter)
    prompter = Prompter(adapter)

    print("Entity extracting ... ")
    results = []
    for datum in tqdm(my_examples):
        prompt = prompter.entityExtractionPrompt_GPT(datum, adapter)
        output = model.inference_GPT(prompt)
        post_processed_datum = prompter.processEntityExtraction_GPT(
            prompt, datum, output
        )
        results.append(post_processed_datum)

        with open(outputPath, "a+") as f:
            save_copy = {k: v for k, v in post_processed_datum.items()}
            strip_heavy_keys(save_copy)
            json.dump(save_copy, f)
            f.write("\n")

    gathered_results = gather_if_distributed(results, acc)
    if acc.is_main_process:
        if adapter.remove_text_before_save:
            for item in gathered_results:
                strip_heavy_keys(item)
        writeJson(model_conf.outputPath_entity_extraction, flatten(gathered_results))


if __name__ == "__main__":
    main()
