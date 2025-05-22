import hydra
from tqdm import tqdm
import json, numpy as np
from omegaconf import DictConfig
from model.odyssey import Odyssey
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from common.utils import *

_SKIP_TABLES = {
    "nba_game_information",
    "nba_team_game_stats",
    "nba_player_game_stats",
}


@hydra.main(config_path = "conf", config_name = "start_point_initiation")
def main(cfg: DictConfig):
    with open(cfg.datasetPath, "r") as f:
        dataset = json.load(f)

    model = Odyssey(cfg)
    addEntityExtraction(dataset, cfg)
    addHeaderExtraction(dataset, cfg)
    addEntityToHeaderMapping(dataset, cfg)
    addSubtableGraph(dataset, model)
    addHybridGraph(dataset, model)

    ins = INSTRUCTOR("hkunlp/instructor-xl")
    ROOT_EMB = cfg.rootEmb
    with open(f"{ROOT_EMB}/entity_nodes.json") as f:
        ENT_NODES = json.load(f)
    ENT_EMBS = np.load(f"{ROOT_EMB}/entity_embeddings.npy")
    EMB_DIM = ENT_EMBS.shape[1] 
    entity_cache       = {n: e for n, e in zip(ENT_NODES, ENT_EMBS)}
    new_entity_cache   = {}

    table_cache, header_cache = {}, {}
    new_table_cache, new_header_cache = {}, {}

    def _maybe_load_table(tbl):
        if tbl in table_cache:
            return
        try:
            with open(f"{ROOT_EMB}/{tbl}_nodes.json") as f:
                nodes = json.load(f)
            embs = np.load(f"{ROOT_EMB}/{tbl}.npy")
            table_cache[tbl] = {tuple([tuple(n_element) if isinstance(n_element, list) else n_element for n_element in n]): e for n, e in zip(nodes, embs)}
        except FileNotFoundError:
            table_cache[tbl] = {}

        try:
            with open(f"{ROOT_EMB}/{tbl}_header_nodes.json") as f:
                hdrs = json.load(f)
            hem = np.load(f"{ROOT_EMB}/{tbl}_header.npy")
            header_cache[tbl] = {h: e for h, e in zip(hdrs, hem)}
        except FileNotFoundError:
            header_cache[tbl] = {}

    def _ensure_entity_embs(labels, bs=1024):
        if not labels:
            return np.empty((0, EMB_DIM), dtype=np.float32)

        miss = [l for l in labels if l not in entity_cache and l not in new_entity_cache]
        if miss:
            vecs = ins.encode([[cfg.insPrompt, l] for l in miss], batch_size=bs)
            new_entity_cache.update(dict(zip(miss, vecs)))
        look = {**entity_cache, **new_entity_cache}
        return np.stack([look[l] for l in labels])

    def _ensure_table_embs(tbl, nodes, bs=1024):
        nodes = [n for n in nodes if n[-1] == tbl]
        if not nodes: 
            return np.empty((0, EMB_DIM), dtype=np.float32)
        _maybe_load_table(tbl)
        base = {**table_cache[tbl], **new_table_cache.get(tbl, {})}
        miss = [n for n in nodes if (n[0], n[1], n[2]) not in base]
        if miss:
            vecs = ins.encode([[cfg.insPrompt, n[0]] for n in miss], batch_size=bs)
            new_table_cache.setdefault(tbl, {}).update(dict(zip(miss, vecs)))
            base.update(new_table_cache[tbl])
        return np.stack([base[(n[0], n[1], n[2])] for n in nodes])

    def _ensure_header_embs(tbl, hdrs, bs=1024):
        if not hdrs:
            return np.empty((0, EMB_DIM), dtype=np.float32)
        _maybe_load_table(tbl)
        base = {**header_cache[tbl], **new_header_cache.get(tbl, {})}
        miss = [h for h in hdrs if h.replace(f'/{tbl}', '') not in base]
        if miss:
            vecs = ins.encode([[cfg.insPrompt, h] for h in miss], batch_size=bs)
            new_header_cache.setdefault(tbl, {}).update(dict(zip(miss, vecs)))
            base.update(new_header_cache[tbl])
        return np.stack([base[h] for h in hdrs])

    def _memo_once(thunk):
        """Return a parameter-less function that runs `thunk()` just once per sample."""
        cached = None
        def wrapped():
            nonlocal cached
            if cached is None:
                cached = thunk()
            return cached
        return wrapped

    print("Making traversal start point …")
    infos = []

    for data in tqdm(dataset):
        tbl_names = data["table"] if isinstance(data["table"], list) else [data["table"]]
        tbl_names = [t for t in tbl_names if t not in _SKIP_TABLES and t in data["hIdxs"]]
        table_nodes = [n for n in data["hybrid_graph"].nodes() if isinstance(n, tuple)]
        entity_nodes = [n for n in data["hybrid_graph"].nodes()
                        if isinstance(n, str) and '/header/' not in n]
        header_nodes = [n.split('/header/')[-1]
                        for n in data["hybrid_graph"].nodes()
                        if isinstance(n, str) and '/header/' in n]
        
        table_nodes_by_tbl  = defaultdict(list)
        header_nodes_by_tbl = defaultdict(list)
        
        for n in table_nodes:  
            table_nodes_by_tbl[n[-1]].append(n)

        for h in header_nodes: 
            for t in tbl_names:
                if h in all_tables.get(t, []): 
                    header_nodes_by_tbl[t].append(h)
                    break

        get_ent_emb  = _memo_once(lambda: _ensure_entity_embs(entity_nodes))
        tbl_emb_get = {t: _memo_once(
                        lambda t=t: _ensure_table_embs(t, table_nodes_by_tbl[t]))
                    for t in tbl_names}

        hdr_emb_get = {t: _memo_once(
                        lambda t=t: _ensure_header_embs(t, header_nodes_by_tbl[t]))
                    for t in tbl_names}

        startNodes, startScores = [], []
        for qEnt in data["entities"]:
            if qEnt not in data["mappings"]:
                mappings = [{"Others": "Others"}]
            else:
                mappings = data["mappings"][qEnt]
            qEmb = ins.encode([[cfg.insPrompt, qEnt]])
            maxScore, maxNode = cfg.simThreshold, 'NoStartPoint'

            if len(table_nodes) == 0:
                mappings = [{"Others": "Others"}]
            
            for mapping in mappings:
                for table_name, hdrs in mapping.items():
                    if not isinstance(hdrs, list):
                        hdrs = [hdrs]
                    if  table_name not in data["hIdxs"] or table_name in _SKIP_TABLES:
                        hdrs = ["Others"]
                    for h in hdrs:
                        if h == "Others":
                            nodes, nEmbs = entity_nodes, get_ent_emb()
                        else:
                            cols = all_tables.get(table_name, [])
                            post_processed_table_nodes = [n for n in table_nodes if n[-1] == table_name]
                            nodes = [n for n in post_processed_table_nodes if cols and cols[n[1][1]] == h]

                            hEmb = _ensure_header_embs(table_name, [h])
                            hSc  = cosine_similarity(qEmb, hEmb)[0,0]
                            if hSc > maxScore:
                                maxScore, maxNode = hSc, f"/{table_name}/header/{h}"

                            nEmbs = _ensure_table_embs(table_name, nodes)
                        
                        if not nodes:
                            continue
                        
                        scs = cosine_similarity(qEmb, nEmbs)[0]
                        if scs.max() > maxScore:
                            maxScore = scs.max()
                            maxNode  = nodes[scs.argmax()]
        
            if maxNode == "NoStartPoint" and any(h != "Others" for h in hdrs):
                for t in tbl_names:
                    if t not in data["hIdxs"] or t in _SKIP_TABLES:
                        continue
                    t_embs = tbl_emb_get[t]()
                    if t_embs.shape[0]:
                        scs = cosine_similarity(qEmb, t_embs)[0]
                        if scs.max() > maxScore:
                            maxScore = scs.max()
                            maxNode  = table_nodes_by_tbl[t][scs.argmax()]

                    h_embs = hdr_emb_get[t]()
                    if h_embs.shape[0]:
                        hScs = cosine_similarity(qEmb, h_embs)[0]
                        if hScs.max() > maxScore:
                            maxScore = hScs.max()
                            maxNode  = f"/{t}/header/{header_nodes_by_tbl[t][hScs.argmax()]}"

            startNodes.append(maxNode)
            startScores.append(f"{maxScore:.6f}")

        info = {k: v for k, v in data.items()
                if k not in ("st_graph", "hybrid_graph")}
        info.update(start_nodes=startNodes, start_scores=startScores)
        infos.append(info)

    writeJson(cfg.outputPath, infos)

if __name__ == "__main__":
    main()