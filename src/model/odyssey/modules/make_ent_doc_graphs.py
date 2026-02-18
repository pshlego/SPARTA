import hydra
import json
import spacy
import os
from tqdm import tqdm
import networkx as nx
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from src.benchmark import DataModule
from src.model.odyssey.dataset_adapter import OdysseyDatasetAdapter
from src.model.odyssey.utils import (
    shard_for_accelerate,
    make_accelerator,
    gather_if_distributed,
)


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    model_conf = cfg.model
    context_mode = getattr(cfg.benchmark, "context_mode", "global")
    acc = make_accelerator(getattr(model_conf, "use_accelerate", True))

    os.makedirs(model_conf.outputPath_make_ent_doc_graphs, exist_ok=True)

    # Load benchmark
    data_module = DataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.test_dataset
    adapter = OdysseyDatasetAdapter(dataset, cfg)

    nlp = spacy.load("en_core_web_trf")

    if adapter.is_sparta and context_mode == "global":
        # ── SPARTA global: process entire text_data once ──
        text_data = adapter._text_map_global or {}
        text_items = list(text_data.items())
        my_items = shard_for_accelerate(text_items, acc)

        combined_graph = {}
        for summary_id, summary in tqdm(my_items):
            if not (summary_id.startswith("/id/") or summary_id.startswith("/wiki/")):
                summary_node = f"/id/{summary_id}"
            else:
                summary_node = summary_id
            G = nx.Graph()
            G.add_node(summary_node)
            entityList = [e.text for e in nlp(summary).ents]
            G.add_nodes_from(entityList)
            G.add_edges_from((summary_node, e) for e in entityList)
            combined_graph[summary_node] = nx.node_link_data(G)

        # Gather & merge on main process
        all_partials = gather_if_distributed([combined_graph], acc)
        if acc.is_main_process:
            merged = {}
            for partial in all_partials:
                merged.update(partial)
            out_path = os.path.join(
                model_conf.outputPath_make_ent_doc_graphs, "entity_doc_graph.json"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
    else:
        # ── query_specific or HybridQA/OTT-QA: per-question processing ──
        all_examples = [adapter.to_example(dataset[i]) for i in range(len(dataset))]
        my_examples = shard_for_accelerate(all_examples, acc)

        # NER cache: avoid reprocessing the same passage across questions
        ner_cache = {}

        for datum in tqdm(my_examples):
            text_data = datum["text"]
            if not isinstance(text_data, dict):
                raise ValueError(
                    "Text data should be a dictionary with summary IDs as keys."
                )
            combined_graph = {}
            for summary_id, summary in tqdm(text_data.items()):
                if not (
                    summary_id.startswith("/id/") or summary_id.startswith("/wiki/")
                ):
                    summary_node = f"/id/{summary_id}"
                else:
                    summary_node = summary_id
                if summary_id in ner_cache:
                    combined_graph[summary_node] = ner_cache[summary_id]
                    continue
                G = nx.Graph()
                G.add_node(summary_node)
                entityList = [e.text for e in nlp(summary).ents]
                G.add_nodes_from(entityList)
                G.add_edges_from((summary_node, e) for e in entityList)
                graph_data = nx.node_link_data(G)
                combined_graph[summary_node] = graph_data
                ner_cache[summary_id] = graph_data

            adapter.save_ent_doc_graph(combined_graph, datum)


if __name__ == "__main__":
    main()
