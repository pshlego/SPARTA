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
from src.model.odyssey.utils import (
    GTLabelled,
    writeJson,
    flatten,
    shard_for_accelerate,
    strip_heavy_keys,
    addEntityExtraction,
    addHeaderExtraction,
    addEntityToHeaderMapping,
    addSubtableGraph,
    addHybridGraph,
    addStartNodes,
    make_accelerator,
    gather_if_distributed,
)


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    model_conf = cfg.model
    acc = make_accelerator(getattr(model_conf, "use_accelerate", True))

    outputPath = model_conf.paths.results_dir + f"/qa_results_{acc.process_index}.json"
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

    print("Inferencing ... ")
    addEntityExtraction(my_examples, model_conf)
    addHeaderExtraction(my_examples, model_conf, adapter)
    addEntityToHeaderMapping(my_examples, model_conf)
    addSubtableGraph(my_examples, model, adapter)
    addHybridGraph(my_examples, model, adapter)
    addStartNodes(my_examples, model_conf)

    results = []
    for input in tqdm(my_examples):
        input["ban_list"] = []

        if input["start_nodes"] == []:
            model.addFinalHop(input, adapter)
        else:
            maxIdx = input["start_scores"].index(max(input["start_scores"]))
            startNodes = [input["start_nodes"][maxIdx]]

            if startNodes[0] == "NoStartPoint":
                model.addFinalHop(input, adapter)

            else:
                G = GTLabelled()
                for n in startNodes:
                    G._get_v(n)
                input["subgraph"] = G

                if model.addOneHop(
                    1,
                    input,
                    adapter.get_g_all_cache_for(input),
                    adapter,
                ):
                    if model.addOneHop(
                        2,
                        input,
                        adapter.get_g_all_cache_for(input),
                        adapter,
                    ):
                        if model.addOneHop(
                            3,
                            input,
                            adapter.get_g_all_cache_for(input),
                            adapter,
                        ):
                            model.addFinalHop(input, adapter)

        input.pop("subgraph", None)
        input.pop("hybrid_graph", None)
        input.pop("ed_graph", None)
        input.pop("st_graph", None)

        result = input
        with open(outputPath, "a+") as file:
            save_copy = dict(result)
            strip_heavy_keys(save_copy)
            json.dump(save_copy, file)
            file.write("\n")

        results.append(result)

    gathered_results = gather_if_distributed(results, acc)
    if acc.is_main_process:
        if adapter.remove_text_before_save:
            for item in gathered_results:
                strip_heavy_keys(item)
        os.makedirs(os.path.dirname(model_conf.outputPath_inference), exist_ok=True)
        writeJson(model_conf.outputPath_inference, gathered_results)


if __name__ == "__main__":
    main()
