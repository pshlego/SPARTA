import hydra
import json, spacy
from tqdm import tqdm
import networkx as nx
from common.utils import *
from omegaconf import DictConfig

@hydra.main(config_path = "conf", config_name = "make_ent_doc_graphs")
def main(cfg: DictConfig):
    with open(cfg.text_data_path, 'r') as f:
        text_data = json.load(f)

    nlp = spacy.load('en_core_web_trf')
    combined_graph = {}
    for summary_id, summary in tqdm(text_data.items()):
        summary_node = f"/id/{summary_id}"
        G = nx.Graph()
        G.add_node(summary_node)
        
        entityList = [e.text for e in nlp(summary).ents]
        G.add_nodes_from(entityList)
        G.add_edges_from((summary_node, e) for e in entityList)

        combined_graph[summary_node] = nx.node_link_data(G)
        print(combined_graph[summary_node])

    with open(cfg.outputPath, "w") as f:
        json.dump(combined_graph, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()