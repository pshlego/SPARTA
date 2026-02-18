import os

os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libgomp.so.1"
import torch
import json
import numpy as np
import orjson
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from pathlib import Path

from InstructorEmbedding import INSTRUCTOR
from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from src.benchmark import DataModule
from src.model.odyssey.dataset_adapter import OdysseyDatasetAdapter
from src.model.odyssey.utils import shard_for_accelerate, make_accelerator

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _collect_sparta_entity_names(edg_dir: Path) -> list:
    """Extract entity names directly from EDG JSON (no GTLabelled construction)."""
    all_entities: set[str] = set()

    # Prefer global EDG (single file, fastest)
    global_file = edg_dir / "entity_doc_graph.json"
    if global_file.exists():
        json_files = [global_file]
    else:
        # Fallback: scan per-query files
        json_files = sorted(edg_dir.glob("*.json"))

    for jf in tqdm(json_files, desc="Scanning EDG files for entities"):
        with open(jf, "rb") as f:
            ed_json = orjson.loads(f.read())
        for graph_data in ed_json.values():
            for node in graph_data.get("nodes", []):
                raw = node.get("id", node)
                try:
                    lbl = json.loads(raw)
                except Exception:
                    lbl = raw
                if isinstance(lbl, str) and not lbl.startswith(
                    ("/id/", "/header/", "/wiki/")
                ):
                    all_entities.add(lbl)

    return sorted(all_entities)


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    args = cfg.model
    acc = make_accelerator(getattr(args, "use_accelerate", True))

    # Load benchmark
    data_module = DataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.test_dataset
    adapter = OdysseyDatasetAdapter(dataset, cfg)

    torch.set_float32_matmul_precision("high")
    device = acc.device
    ins = INSTRUCTOR("hkunlp/instructor-xl", device=device)

    batch_size = 1024

    if adapter.is_sparta:
        # SPARTA: entity embeddings are context_mode-independent.
        # Extract entity names directly from EDG JSON files — avoids
        # expensive GTLabelled graph construction via adapter.g_all_cache.
        edg_dir = Path(args.entityDocumentGraphPath)
        all_entity_nodes = _collect_sparta_entity_names(edg_dir)

        root = Path(args.rootEmb)
        os.makedirs(root, exist_ok=True)
        emb_path = root / "entity_embeddings.npy"
        node_path = root / "entity_nodes.json"

        if emb_path.exists() and node_path.exists():
            print("[skip] SPARTA global entity embeddings already exist.")
        else:
            my_nodes = shard_for_accelerate(all_entity_nodes, acc)
            all_inputs = [[args.insPrompt, n] for n in my_nodes]
            all_embeddings = []
            for i in tqdm(
                range(0, len(all_inputs), batch_size),
                disable=not acc.is_local_main_process,
                desc="Encoding entities (SPARTA global)",
            ):
                batch_inputs = all_inputs[i : i + batch_size]
                batch_embs = ins.encode(batch_inputs, batch_size=len(batch_inputs))
                all_embeddings.extend(batch_embs)

            entityEmbs = (
                np.array(all_embeddings) if all_embeddings else np.empty((0, 768))
            )
            np.save(emb_path, entityEmbs)
            with open(node_path, "w") as f:
                json.dump(my_nodes, f)
    else:
        # HybridQA / OTT-QA: per-table or per-question encoding
        g_all = adapter.g_all_cache
        g_all_caches = g_all if isinstance(g_all, dict) else {"all": g_all}

        embed_dir = Path(args.embed_save_path)
        node_dir = Path(args.node_save_path)
        os.makedirs(embed_dir, exist_ok=True)
        os.makedirs(node_dir, exist_ok=True)

        all_ids = sorted(g_all_caches.keys())
        my_ids = shard_for_accelerate(all_ids, acc)

        pbar_ids = tqdm(
            my_ids, disable=not acc.is_local_main_process, desc="Sharded graphs"
        )
        for id_ in pbar_ids:
            entity_graph = g_all_caches[id_]
            embed_path = embed_dir / f"{id_}.npy"
            node_path = node_dir / f"{id_}.json"
            if embed_path.exists() and node_path.exists():
                print(f"[skip] {id_}: embeddings & nodes already exist.")
                continue

            # get entity nodes
            entityNodes = [
                n
                for n in entity_graph.nodes()
                if isinstance(n, str)
                and not n.startswith(("/id/", "/header/", "/wiki/"))
            ]

            # encode entity nodes
            all_inputs = [
                [args.insPrompt, n[0] if isinstance(n, tuple) else n]
                for n in entityNodes
            ]

            all_embeddings = []
            for i in tqdm(
                range(0, len(all_inputs), batch_size),
                disable=not acc.is_local_main_process,
                desc="Encoding entities",
            ):
                batch_inputs = all_inputs[i : i + batch_size]
                batch_embs = ins.encode(batch_inputs, batch_size=len(batch_inputs))
                all_embeddings.extend(batch_embs)

            entitynEmbs = np.array(all_embeddings)
            np.save(embed_path, entitynEmbs)
            with open(node_path, "w") as f:
                json.dump(entityNodes, f)


if __name__ == "__main__":
    main()
