import os

os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libgomp.so.1"
import torch
import json
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from pathlib import Path

from InstructorEmbedding import INSTRUCTOR
from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from src.benchmark import DataModule
from src.model.odyssey.dataset_adapter import OdysseyDatasetAdapter
from src.model.odyssey.odyssey import Odyssey
from src.model.odyssey.utils import shard_for_accelerate, make_accelerator


def already_done(save_dir: Path, table_name: str) -> bool:
    need = [
        save_dir / f"{table_name}.npy",
        save_dir / f"{table_name}_nodes.json",
        save_dir / f"{table_name}_header.npy",
        save_dir / f"{table_name}_header_nodes.json",
    ]
    return all(p.exists() for p in need)


def save_table_embeddings(
    save_dir: Path, file_name: str, table_nodes, table_embs, header_nodes, header_embs
):
    np.save(save_dir / f"{file_name}.npy", table_embs)
    with open(save_dir / f"{file_name}_nodes.json", "w", encoding="utf-8") as f:
        json.dump(table_nodes, f, ensure_ascii=False)

    np.save(save_dir / f"{file_name}_header.npy", header_embs)
    with open(save_dir / f"{file_name}_header_nodes.json", "w", encoding="utf-8") as f:
        json.dump(header_nodes, f, ensure_ascii=False)


def encode_text_batches(ins, inputs, show_tqdm=True):
    all_embs = []
    batch_size = 1024
    for i in tqdm(range(0, len(inputs), batch_size), disable=(not show_tqdm)):
        batch = inputs[i : i + batch_size]
        embs = ins.encode(batch, batch_size=len(batch))
        all_embs.extend(embs)
    return np.array(all_embs)


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    model_conf = cfg.model
    acc = make_accelerator(getattr(model_conf, "use_accelerate", True))

    # Load benchmark
    data_module = DataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.test_dataset
    adapter = OdysseyDatasetAdapter(dataset, cfg)

    model = Odyssey(model_conf, adapter)

    torch.set_float32_matmul_precision("high")
    device = acc.device
    ins = INSTRUCTOR("hkunlp/instructor-xl", device=device)

    save_dir = Path(model_conf.outputPath_table_encoding)
    os.makedirs(save_dir, exist_ok=True)

    table_files = adapter.table_files_for_table_encoding
    if table_files:
        my_table_files = shard_for_accelerate(list(table_files), acc)
        pbar = tqdm(
            my_table_files,
            desc=f"Processing tables (rank {acc.process_index})",
            disable=not acc.is_local_main_process,
        )
        for fpath in pbar:
            data = {}
            table_name = fpath.stem

            if already_done(Path(save_dir), table_name):
                continue

            try:
                with fpath.open("r", encoding="utf-8") as f:
                    tbl = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to read {fpath}: {e}")
                continue

            data["table"] = table_name
            data["hIdxs"] = {table_name: list(range(len(tbl["header"])))}
            table_graph = model.subtableRetrieval(data, adapter)

            # Encode table nodes
            tableNodes = [list(n) for n in table_graph.nodes() if isinstance(n, tuple)]
            inputs = [[model_conf.insPrompt, n[0]] for n in tableNodes]
            tablenodeEmbs = encode_text_batches(ins, inputs)

            # Encode header nodes
            headerNodes = [
                n.split("/header/")[-1]
                for n in table_graph.nodes()
                if isinstance(n, str) and "/header/" in n
            ]
            inputs = [[model_conf.insPrompt, n] for n in headerNodes]
            headernodeEmbs = encode_text_batches(ins, inputs)

            save_table_embeddings(
                save_dir,
                table_name,
                tableNodes,
                tablenodeEmbs,
                headerNodes,
                headernodeEmbs,
            )

    else:  # in memory for ottqa
        all_examples = [adapter.to_example(dataset[i]) for i in range(len(dataset))]
        idxs = list(range(len(all_examples)))
        my_idxs = shard_for_accelerate(idxs, acc)
        pbar = tqdm(
            my_idxs, disable=not acc.is_local_main_process, desc="Sharded graphs"
        )
        for i in pbar:
            data = all_examples[i]
            table_name_list = data["table"]
            if not isinstance(table_name_list, list):
                table_name_list = [table_name_list]
            tableNodes = []
            headerNodes = []
            for table_name in table_name_list:
                data["table"] = table_name
                data["hIdxs"] = {
                    table_name: list(range(len(data["table_headers"][table_name])))
                }
                table_graph = model.subtableRetrieval(data, adapter)
                if table_graph is None:
                    continue
                tableNodes.extend(
                    [list(n) for n in table_graph.nodes() if isinstance(n, tuple)]
                )
                headerNodes.extend(
                    [
                        n.split("/header/")[-1]
                        for n in table_graph.nodes()
                        if isinstance(n, str) and "/header/" in n
                    ]
                )

            # de-duplication
            seen_tbl = set()
            dedup_tableNodes = []
            for n in tableNodes:
                k = tuple(tuple(x) if isinstance(x, list) else x for x in n)
                if k not in seen_tbl:
                    seen_tbl.add(k)
                    dedup_tableNodes.append(n)
            tableNodes = dedup_tableNodes

            seen_hdr = set()
            dedup_headerNodes = []
            for h in headerNodes:
                if h not in seen_hdr:
                    seen_hdr.add(h)
                    dedup_headerNodes.append(h)
            headerNodes = dedup_headerNodes

            # encoding
            table_inputs = [[model_conf.insPrompt, n[0]] for n in tableNodes]
            tablenodeEmbs = encode_text_batches(ins, table_inputs)

            header_inputs = [[model_conf.insPrompt, n] for n in headerNodes]
            headernodeEmbs = encode_text_batches(ins, header_inputs)

            save_table_embeddings(
                save_dir,
                data["question_id"],
                tableNodes,
                tablenodeEmbs,
                headerNodes,
                headernodeEmbs,
            )


if __name__ == "__main__":
    main()
