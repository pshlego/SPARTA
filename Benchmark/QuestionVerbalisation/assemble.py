import os
import json
import hydra
import torch
import numpy as np
import pandas as pd
from itertools import chain
from sqlglot import parse_one
from omegaconf import DictConfig
from util.ast_icl_gcn import GCN
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from util.prompt import create_incontext_prompt2
from sklearn.metrics.pairwise import cosine_similarity


def parse_query(query):
    ast = parse_one(query)
    idx = 0
    features = []
    edge_index = []
    mapping = {}
    for node in ast.walk():
        features.append(type(node).__name__)
        mapping[id(node)] = idx
        if node.depth > 0:
            parent = mapping[id(node.parent)]
            edge_index.append([parent, idx])
        idx = idx + 1
    return features, edge_index


def get_graphs(cfg):
    # Train
    with open(cfg.dataset_train, "r") as file1:
        train = json.load(file1)
    df_train = pd.DataFrame(train, columns=["question", "query"])
    df_train["utterance"] = df_train["question"]
    df_train[["features", "edge_index"]] = (
        df_train["query"].apply(parse_query).apply(pd.Series)
    )
    all_features = df_train["features"]
    unique_words = all_features.explode().unique().tolist()
    unique_words.append("<UNK>")
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    embeddings = torch.nn.Embedding(len(word_to_index), cfg.word_embedding_dim)
    train_graphs = df_train.apply(
        create_graph, axis=1, args=(word_to_index, embeddings)
    )

    # Test
    with open(cfg.dataset_test, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    data = [
        {"query": query, "utterances": [], "idx": str(idx)}
        for idx, query in enumerate(lines)
    ]

    refs = [x["utterances"] for x in data]
    queries = [x["query"] for x in data]
    df_test = pd.DataFrame({"utterances": refs, "query": queries})
    df_test[["features", "edge_index"]] = (
        df_test["query"].apply(parse_query).apply(pd.Series)
    )
    test_graphs = df_test.apply(create_graph, axis=1, args=(word_to_index, embeddings))
    return train_graphs, test_graphs, df_train, df_test


def create_graph(row, word_to_index, embeddings):
    UNK_IDX = word_to_index["<UNK>"]
    token_features = [word_to_index.get(word, UNK_IDX) for word in row["features"]]
    embed_features = embeddings(torch.tensor(token_features))
    graph = Data(
        edge_index=torch.tensor(row["edge_index"], dtype=torch.long).t(),
        x=embed_features,
        idx=row.name,
    )
    return graph


def train(model, loader):
    all_pooled = []
    with torch.no_grad():
        for batch in loader:
            embeddings = model(batch)
            pooled = global_mean_pool(embeddings, batch.batch)
            all_pooled.append(pooled)
    all_pooled_tensor = torch.cat(all_pooled, dim=0)
    all_pooled_numpy = all_pooled_tensor.cpu().numpy()
    return all_pooled_numpy


def get_samples_top(df, num, train_pool, test):
    differences = cosine_similarity(train_pool, [test])
    x = np.argpartition(np.squeeze(differences), -num)[-num:]
    sampled_data = df.iloc[x]
    return list(
        chain.from_iterable(zip(sampled_data["query"], sampled_data["utterance"]))
    )


def create_input_file(cfg, prompt, num_examples, method):
    data_dict = {"prompt": prompt}
    df = pd.DataFrame(data_dict)
    df.to_csv(f"{cfg.output_dir}/{method}-{num_examples}.csv", index=False)


@hydra.main(config_path="conf", config_name="assemble")
def main(cfg: DictConfig):
    train_graphs, test_graphs, df_train, df_test = get_graphs(cfg)
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    model = GCN(cfg.word_embedding_dim, cfg.hidden_dim, cfg.output_dim)
    train_pool = train(model, train_loader)
    test_pool = train(model, test_loader)

    kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=0).fit(train_pool)
    df_train["label"] = kmeans.labels_
    prompts = [[] for i in range(0, cfg.LEN)]

    for i in range(0, len(df_test)):
        prog = df_test.iloc[i]["query"]
        for n in range(0, cfg.LEN):
            samples = get_samples_top(df_train, n + 1, train_pool, test_pool[i]) + [
                prog
            ]
            prompt1 = create_incontext_prompt2(*samples)
            prompts[n].append(prompt1)

    create_input_file(cfg, prompts[i], 4, f"{cfg.method}-{cfg.dataset}-exp")


if __name__ == "__main__":
    main()
