import hydra
import json
import os
import glob
import torch
import datetime
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from collections import defaultdict
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from src.benchmark import DataModule
from src.model.odyssey.dataset_adapter import OdysseyDatasetAdapter
from src.model.odyssey.odyssey import Odyssey
from src.model.odyssey.utils import (
    writeJson,
    flatten,
    shard_for_accelerate,
    strip_heavy_keys,
    addEntityExtraction,
    addHeaderExtraction,
    addEntityToHeaderMapping,
    addSubtableGraph,
    addHybridGraph,
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
            model_conf.paths.data_dir + f"/start_points_{acc.process_index}.json"
        )
    else:
        outputPath = model_conf.outputPath_start_point_initiation
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

    addEntityExtraction(my_examples, model_conf)
    addHeaderExtraction(my_examples, model_conf, adapter)
    addEntityToHeaderMapping(my_examples, model_conf)
    addSubtableGraph(my_examples, model, adapter)
    addHybridGraph(my_examples, model, adapter)

    ins = INSTRUCTOR("hkunlp/instructor-xl")

    def _memo_once(thunk):
        cached = None

        def wrapped():
            nonlocal cached
            if cached is None:
                cached = thunk()
            return cached

        return wrapped

    print("Making traversal start point ...")
    infos = []

    for data in tqdm(my_examples):
        tbl_names = (
            data["table"] if isinstance(data["table"], list) else [data["table"]]
        )
        tbl_names = [
            t
            for t in tbl_names
            if t not in adapter.grounding_tables and t in data["hIdxs"]
        ]
        table_nodes = [n for n in data["hybrid_graph"].nodes() if isinstance(n, tuple)]
        entity_nodes = [
            n
            for n in data["hybrid_graph"].nodes()
            if isinstance(n, str) and "/header/" not in n
        ]
        header_nodes = [
            n.split("/header/")[-1]
            for n in data["hybrid_graph"].nodes()
            if isinstance(n, str) and "/header/" in n
        ]

        table_nodes_by_tbl = defaultdict(list)
        header_nodes_by_tbl = defaultdict(list)

        for n in table_nodes:
            table_nodes_by_tbl[n[-1]].append(n)

        for n in data["hybrid_graph"].nodes():
            if isinstance(n, str) and "/header/" in n:
                tbl = n.split("/header/")[0].lstrip("/")
                h = n.split("/header/")[-1]
                if tbl in tbl_names:
                    header_nodes_by_tbl[tbl].append(h)

        get_ent_emb = _memo_once(
            lambda: adapter._ensure_entity_embs(data, entity_nodes, ins)
        )
        tbl_emb_get = {
            t: _memo_once(
                lambda t=t: adapter._ensure_table_embs(t, table_nodes_by_tbl[t], ins)
            )
            for t in tbl_names
        }

        hdr_emb_get = {
            t: _memo_once(
                lambda t=t: adapter._ensure_header_embs(t, header_nodes_by_tbl[t], ins)
            )
            for t in tbl_names
        }

        startNodes, startScores = [], []
        for qEnt in data["entities"]:
            if qEnt not in data["mappings"]:
                mappings = [{"Others": "Others"}]
            else:
                mappings = data["mappings"][qEnt]
            qEmb = ins.encode([[model_conf.insPrompt, qEnt]])
            maxScore, maxNode = model_conf.simThreshold, "NoStartPoint"

            if len(table_nodes) == 0:
                mappings = [{"Others": "Others"}]

            for mapping in mappings:
                for table_name, hdrs in mapping.items():
                    if not isinstance(hdrs, list):
                        hdrs = [hdrs]
                    if (
                        table_name not in data["hIdxs"]
                        or table_name in adapter.grounding_tables
                    ):
                        hdrs = ["Others"]
                    for h in hdrs:
                        if h == "Others":
                            nodes, nEmbs = entity_nodes, get_ent_emb()
                        else:
                            table_header_mapping = adapter.get_table_header_mapping(
                                data
                            )
                            cols = table_header_mapping.get(table_name, [])
                            post_processed_table_nodes = [
                                n for n in table_nodes if n[-1] == table_name
                            ]
                            nodes = [
                                n
                                for n in post_processed_table_nodes
                                if cols and cols[n[1][1]] == h
                            ]

                            hEmb = adapter._ensure_header_embs(table_name, [h], ins)
                            hSc = cosine_similarity(qEmb, hEmb)[0, 0]
                            if hSc > maxScore:
                                maxScore, maxNode = hSc, f"/{table_name}/header/{h}"

                            nEmbs = adapter._ensure_table_embs(table_name, nodes, ins)

                        if not nodes:
                            continue

                        scs = cosine_similarity(qEmb, nEmbs)[0]
                        if scs.max() > maxScore:
                            maxScore = scs.max()
                            maxNode = nodes[scs.argmax()]

            if maxNode == "NoStartPoint" and any(h != "Others" for h in hdrs):
                for t in tbl_names:
                    if t not in data["hIdxs"] or t in adapter.grounding_tables:
                        continue
                    t_embs = tbl_emb_get[t]()
                    if t_embs.shape[0]:
                        scs = cosine_similarity(qEmb, t_embs)[0]
                        if scs.max() > maxScore:
                            maxScore = scs.max()
                            maxNode = table_nodes_by_tbl[t][scs.argmax()]

                    h_embs = hdr_emb_get[t]()
                    if h_embs.shape[0]:
                        hScs = cosine_similarity(qEmb, h_embs)[0]
                        if hScs.max() > maxScore:
                            maxScore = hScs.max()
                            maxNode = (
                                f"/{t}/header/{header_nodes_by_tbl[t][hScs.argmax()]}"
                            )

            startNodes.append(maxNode)
            startScores.append(f"{maxScore:.6f}")

        info = {k: v for k, v in data.items() if k not in ("st_graph", "hybrid_graph")}
        info.update(start_nodes=startNodes, start_scores=startScores)

        infos.append(info)
        with open(outputPath, "a+") as f:
            save_copy = dict(info)
            strip_heavy_keys(save_copy)
            json.dump(save_copy, f)
            f.write("\n")

    acc.wait_for_everyone()
    if acc.is_main_process:
        if acc.num_processes > 1:
            shard_glob_pattern = os.path.join(
                model_conf.paths.data_dir, "start_points_*.json"
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
            merged = infos
            if adapter.remove_text_before_save:
                for item in merged:
                    strip_heavy_keys(item)
        writeJson(model_conf.outputPath_start_point_initiation, flatten(merged))


if __name__ == "__main__":
    main()
