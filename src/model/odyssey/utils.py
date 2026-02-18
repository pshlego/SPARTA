import json
import numpy as np
import torch
from tqdm import tqdm
import graph_tool.all as gt
from json import JSONDecoder


class SingleProcess:
    """Drop-in replacement for Accelerator when running without multi-processing."""
    process_index = 0
    num_processes = 1
    is_main_process = True
    is_local_main_process = True

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def wait_for_everyone(self):
        pass


def make_accelerator(use_accelerate: bool = True):
    """Return Accelerator() or a single-process stub."""
    if use_accelerate:
        from accelerate import Accelerator
        return Accelerator()
    return SingleProcess()


def gather_if_distributed(results, acc):
    """gather_object when distributed, identity otherwise."""
    if isinstance(acc, SingleProcess):
        return results
    from accelerate.utils import gather_object
    return gather_object(results)


class GTLabelled:
    @staticmethod
    def _canon(label):
        try:
            hash(label)
            return label
        except TypeError:
            return tuple(label)

    def __init__(self, directed=False):
        self.g = gt.Graph(directed=directed)
        self._vprop = self.g.new_vertex_property("object")
        self.g.vertex_properties["label"] = self._vprop
        self._idx, self._v2lbl = {}, {}

    def _get_v(self, label):
        k = self._canon(label)
        v = self._idx.get(k)
        if v is None:
            v = self.g.add_vertex()
            self._vprop[v] = k
            self._idx[k] = v
            self._v2lbl[int(v)] = k
        return v

    def add_edge(self, u, v):
        self.g.add_edge(self._get_v(u), self._get_v(v))

    def add_complete(self, L):
        vids = [self._get_v(x) for x in L]
        self.g.add_edge_list(
            [
                (vids[i], vids[j])
                for i in range(len(vids))
                for j in range(i + 1, len(vids))
            ]
        )

    def nodes(self):
        return list(self._idx.keys())

    def neighbors(self, lbl):
        v = self._idx.get(self._canon(lbl))
        return [] if v is None else [self._v2lbl[int(w)] for w in v.all_neighbors()]

    def subgraph(self, keep):
        vfilt = self.g.new_vertex_property("bool")
        for lbl in keep:
            v = self._idx.get(self._canon(lbl))
            if v is not None:
                vfilt[v] = True

        view = gt.GraphView(self.g, vfilt=vfilt)

        sg_graph = gt.Graph(view, prune=True)

        sg = GTLabelled(self.g.is_directed())
        sg.g = sg_graph
        sg._vprop = sg_graph.vp["label"]

        sg._idx.clear()
        sg._v2lbl.clear()
        for v in sg_graph.vertices():
            lbl = sg._vprop[v]
            canon = sg._canon(lbl)
            sg._idx[canon] = v
            sg._v2lbl[int(v)] = lbl

        return sg

    def add_edges(self, pairs):
        """
        `pairs` is an iterable of (u_lbl, v_lbl) where labels can be
        any hashable object accepted by _get_v().
        Internally maps labels -> vertex handles only once and then
        emits a single add_edge_list().
        """
        mapped = []
        for u_lbl, v_lbl in pairs:
            u = self._get_v(u_lbl)
            v = self._get_v(v_lbl)
            mapped.append((u, v))

        # C++ bulk insert
        self.g.add_edge_list(mapped)

    def merge(self, other: "GTLabelled") -> None:
        """
        Fast union that    (i) re-uses existing vertices if their canonical label
        already exists in self,   (ii) creates missing vertices on-the-fly, and
        (iii) bulk-adds all edges from *other* at once.
        """
        assert self.g.is_directed() == other.g.is_directed()

        o2s: dict[int, int] = {}
        for ov in other.g.vertices():
            lbl = other._vprop[ov]
            canon = self._canon(lbl)

            sv = self._idx.get(canon)
            if sv is None:
                sv = self.g.add_vertex()
                self._vprop[sv] = lbl
                self._idx[canon] = sv
                self._v2lbl[int(sv)] = lbl

            o2s[int(ov)] = int(sv)

        mapped_edges = [
            (o2s[int(e.source())], o2s[int(e.target())]) for e in other.g.edges()
        ]

        self.g.add_edge_list(mapped_edges)

    def to_json(self, path):
        gt.write_graph(self.g, path, fmt="json")

    def debug_print(self):
        print("Nodes:")
        for v in self.g.vertices():
            print(f"  {int(v)}: {self._vprop[v]}")
        print("Edges:")
        for e in self.g.edges():
            u = self._v2lbl[int(e.source())]
            v = self._v2lbl[int(e.target())]
            print(f"  {u} -> {v}")


def gt_to_dict(g: GTLabelled):
    return {
        "nodes": [json.dumps(lbl, default=str) for lbl in g._idx.keys()],
        "edges": [
            (
                json.loads(g._vprop_label[e.source()]),
                json.loads(g._vprop_label[e.target()]),
            )
            for e in g.g.edges()
        ],
    }


def readJson(path):
    with open(path, "r") as f:
        return json.load(f)


def writeJson(path, datas):
    with open(path, "w") as f:
        json.dump(datas, f, indent=2)


def flatten(origin: list) -> list:
    flatted = []
    for o in origin:
        if isinstance(o, list):
            flatted += o
        else:
            flatted += [o]
    return flatted


def makeBatchs(dataList: list, batchSize: int) -> list:
    return [dataList[i : i + batchSize] for i in range(0, len(dataList), batchSize)]


def ordered(a, b):
    return (a, b) if repr(a) < repr(b) else (b, a)


## add 'entities' key in data_list
def addEntityExtraction(data_list, args):
    path = args.entityExtractionPath
    mapping = readJson(path)
    for data in tqdm(data_list):
        data["entities"] = [
            m["entities"] for m in mapping if m["question_id"] == data["question_id"]
        ][0]


## add 'headers' & 'hIdxs' key in data_list
def addHeaderExtraction(data_list, args, adapter):
    path = args.headerExtractionPath
    mapping = readJson(path)
    for data in tqdm(data_list):
        hIdxs = [
            m["hIdxs"] for m in mapping if m["question_id"] == data["question_id"]
        ][0]
        headers_dict = {}
        if isinstance(data["table"], str):
            data["table"] = [data["table"]]
        table_list = data["table"]
        for table_name in table_list:
            headers_for_table = adapter.get_table_header_mapping(data).get(
                table_name, []
            )
            relevant_indices = hIdxs.get(table_name, [])
            headers_dict[table_name] = [
                headers_for_table[i]
                for i in relevant_indices
                if 0 <= i < len(headers_for_table)
            ]
        data["headers"] = headers_dict
        data["hIdxs"] = hIdxs


## add 'mappings' key in data_list
def addEntityToHeaderMapping(data_list, args):
    path = args.entityToHeaderMappingPath
    mapping = readJson(path)
    for data in tqdm(data_list):
        data["mappings"] = [
            m["mappings"] for m in mapping if m["question_id"] == data["question_id"]
        ][0]


## add 'st_graph' key in data_list
def addSubtableGraph(data_list, model, adapter):
    for data in tqdm(data_list):
        data["st_graph"] = model.subtableRetrieval(data, adapter)


## add 'hybrid_graph' key in data_list
def addHybridGraph(data_list, model, adapter):
    for data in tqdm(data_list):
        data["hybrid_graph"] = model.makeHybridGraph(data, adapter)


## add 'start_nodes' & 'start_scores' keys in data_list
def addStartNodes(data_list, args):
    path = args.startPointsPath
    infos = readJson(path)
    for data in tqdm(data_list):
        tmp = [
            info["start_nodes"]
            for info in infos
            if data["question_id"] == info["question_id"]
        ][0]
        start_nodes = []
        for n in tmp:
            if isinstance(n, list):
                if isinstance(n[2], list):
                    n = tuple([n[0], tuple(n[1]), tuple(n[2]), n[3]])
                else:
                    n = tuple([n[0], tuple(n[1]), n[2], n[3]])
            start_nodes.append(n)
        data["start_nodes"] = start_nodes
        data["start_scores"] = [
            info["start_scores"]
            for info in infos
            if data["question_id"] == info["question_id"]
        ][0]


def extract_json_objects(text, decoder=JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data"""
    pos = 0
    results = []
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results


def shard_for_accelerate(items, acc):
    """Interleave-shard a list across Accelerate processes."""
    if not items:
        return []
    return [it for i, it in enumerate(items) if i % acc.num_processes == acc.process_index]


def strip_heavy_keys(d):
    """Remove bulky keys before JSON serialization."""
    for k in ("text", "tables", "table_data", "table_headers"):
        d.pop(k, None)
