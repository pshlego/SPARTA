import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import pairwise_distances
from thefuzz import fuzz
from collections import defaultdict
from tqdm import tqdm
import bm25s

from src.model.hpropro import embedding_cache

from _lcs_ext import longest_match_distance


class Question_Passage_Retriever(object):
    def __init__(self, threshold=0.95, best_threshold=0.80):
        self.resource_path = "./data/WikiTables-WithLinks"
        self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        stopWords = list(set(stopwords.words("english")))
        self.tfidf = TfidfVectorizer(
            strip_accents="unicode", ngram_range=(2, 3), stop_words=stopWords
        )
        self.best_threshold = best_threshold
        self.threshold = threshold

        if embedding_cache.SUMMARY_DATA:
            self.summary_items = list(embedding_cache.SUMMARY_DATA.items())
            self.titles = [k for k, _ in self.summary_items]
            self.corpus = [f"[{k}]: {v}" for k, v in self.summary_items]

            self.bm25_retriever = bm25s.BM25(corpus=self.corpus)
            self.bm25_retriever.index(bm25s.tokenize(self.corpus))
        else:
            self.summary_items = []
            self.titles = []
            self.corpus = []
            self.bm25_retriever = None

    def url2text(self, url):
        if url.startswith("https://en.wikipedia.org"):
            url = url.replace("https://en.wikipedia.org", "")
        return url.replace("/wiki/", "").replace("_", " ")

    def longestSubstringFinder(self, S, T):
        S = S.lower()
        T = T.lower()
        m = len(S)
        n = len(T)

        dp = [0] * (n + 1)
        longest = 0
        lcs_set = set()

        for i in range(m):
            prev = 0
            for j in range(n):
                temp = dp[j + 1]
                if S[i] == T[j]:
                    dp[j + 1] = prev + 1
                    if dp[j + 1] > longest:
                        longest = dp[j + 1]
                        lcs_set = {S[i - dp[j + 1] + 1 : i + 1]}
                    elif dp[j + 1] == longest:
                        lcs_set.add(S[i - dp[j + 1] + 1 : i + 1])
                else:
                    dp[j + 1] = 0
                prev = temp

        return longest, lcs_set

    def retriever_from_all_passages(self, data_entry, adapter):
        question = data_entry["question"]
        if adapter.has_individual_text:
            self.summary_items = list(data_entry["text"].items())
            self.titles = [k for k, _ in self.summary_items]
            self.corpus = [f"[{k}]: {v}" for k, v in self.summary_items]
            self.bm25_retriever = bm25s.BM25(corpus=self.corpus)
            self.bm25_retriever.index(bm25s.tokenize(self.corpus))

        corpus_size = len(self.bm25_retriever.corpus)
        bm25_results, _ = self.bm25_retriever.retrieve(
            bm25s.tokenize(question), k=corpus_size
        )
        passages = bm25_results[0] if len(bm25_results) > 0 else []
        summary_datas_dict = (
            data_entry["text"]
            if adapter.has_individual_text
            else embedding_cache.SUMMARY_DATA
        )
        ret_list = []
        seen_keys = set()
        for passage in passages:
            if passage.startswith("[") and "]" in passage:
                key = passage[1:].split("]")[0]
                if key in summary_datas_dict and key not in seen_keys:
                    seen_keys.add(key)
                    summary_text = summary_datas_dict[key]
                    ret_list.append(f"{summary_text}")

        return ret_list

    def retriever_hybridqa(self, data_entry, table_data_list, adapter, table_name_list):

        ret_list = []

        if not isinstance(table_data_list, list):
            table_data_list = [table_data_list]

        for idx, table_data in enumerate(
            table_data_list
        ):  # Loop through each table_data in the list
            # Mapping entity link to cell, entity link to surface word
            mapping_entity = {}
            for row_idx, row in enumerate(table_data["data"]):
                for col_idx, cell in enumerate(row):
                    for i, ent in enumerate(cell[1]):
                        mapping_entity[ent] = mapping_entity.get(ent, []) + [
                            (row_idx, col_idx)
                        ]

            # Convert the paragraph and question into TF-IDF features
            keys = []
            paras = []
            for k in mapping_entity:
                v = adapter.get_text_from_link(k, data_entry)
                for _ in self.tokenizer.tokenize(v):
                    keys.append(k)
                    paras.append(_)

            # Try to compute TF-IDF features for the paragraph and question
            qs = [data_entry["question"]]
            para_feature = self.tfidf.fit_transform(paras)
            q_feature = self.tfidf.transform(qs)
            dist_match, dist_pattern = longest_match_distance(qs, paras)
            dist_match, dist_pattern = dist_match[0], dist_pattern[0]
            dist = pairwise_distances(q_feature, para_feature, "cosine")[0]

            # Find out the best matched passages based on distance
            tfidf_nodes = []
            min_dist = {}
            tfidf_best_match = ("N/A", None, 1.0)
            for k, para, d in zip(keys, paras, dist):
                if d < min_dist.get(k, self.threshold):
                    min_dist[k] = d
                    if d < tfidf_best_match[-1]:
                        tfidf_best_match = (k, para, d)
                    if d <= self.best_threshold:
                        for loc in mapping_entity[k]:
                            tfidf_nodes.append((self.url2text(k), loc, k, para, d))

            if tfidf_best_match[0] != "N/A":
                if tfidf_best_match[-1] > self.best_threshold:
                    k = tfidf_best_match[0]
                    for loc in mapping_entity[k]:
                        tfidf_nodes.append(
                            (
                                self.url2text(k),
                                loc,
                                k,
                                tfidf_best_match[1],
                                tfidf_best_match[2],
                            )
                        )

            # Find the best matched paragraph string
            string_nodes = []
            min_dist = {}
            string_best_match = ("N/A", None, 1.0)

            for k, para, d, pat in zip(keys, paras, dist_match, dist_pattern):
                if d < min_dist.get(k, self.threshold):
                    min_dist[k] = d
                    if d < string_best_match[-1]:
                        string_best_match = (k, para, pat, d)
                    if d <= self.best_threshold:
                        for loc in mapping_entity[k]:
                            string_nodes.append(
                                (self.url2text(k), loc, k, para, pat, d)
                            )

            if string_best_match[0] != "N/A":
                if string_best_match[-1] > self.best_threshold:
                    k = string_best_match[0]
                    for loc in mapping_entity[k]:
                        string_nodes.append(
                            (
                                self.url2text(k),
                                loc,
                                k,
                                string_best_match[1],
                                string_best_match[2],
                                string_best_match[3],
                            )
                        )

            # Consider both tf-idf and longest substring
            union_nodes = [
                node[:-1]
                for node in string_nodes
                if node[:-2] in [item[:-1] for item in tfidf_nodes]
            ]

            table_contents = [cell[0] for cell in table_data["header"]]
            for i, row in enumerate(table_data["data"]):
                content = [cell[0] for cell in row]
                table_contents.extend(content)
            table_contents.append(table_name_list[idx])
            table_contents = [item.lower() for item in list(set(table_contents))]

            for node in union_nodes:
                if all(
                    [
                        fuzz.ratio(node[-1].strip().lower(), item) < 60
                        for item in table_contents
                    ]
                ):
                    # Use cell content instead of Wikipedia passage title.
                    row_index, column_index = node[1]
                    assert row_index < len(table_data["data"]) and column_index < len(
                        table_data["data"][0]
                    )
                    cell_content = table_data["data"][row_index][column_index][0]

                    additional_knowledge = "{}:\t{}".format(
                        cell_content, node[-2]
                    )  # Contain the entire sentence.

                    if additional_knowledge not in ret_list:
                        ret_list.append(additional_knowledge)

        return ret_list
