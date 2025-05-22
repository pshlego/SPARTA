import json
import numpy as np
from tqdm import tqdm
import graph_tool.all as gt
from json import JSONDecoder

all_tables = {
            "nba_player_award": ['season', 'award', 'player_name'],
            "nba_draft_combine_stats": ['season', 'wingspan', 'standing_reach', 'percentage_of_body_fat', 'hand_length', 'hand_width', 'standing_vertical_leap', 'max_vertical_leap', 'lane_agility_time', 'three_quarter_sprint', 'number_of_bench_press', 'fifteen_corner_left', 'fifteen_break_left', 'fifteen_top_key', 'fifteen_break_right', 'fifteen_corner_right', 'college_corner_left', 'college_break_left', 'college_top_key', 'college_break_right', 'college_corner_right', 'nba_corner_left', 'nba_break_left', 'nba_top_key', 'nba_break_right', 'nba_corner_right', 'off_dribble_fifteen_break_left', 'off_dribble_fifteen_top_key', 'off_dribble_fifteen_break_right', 'off_dribble_college_break_left', 'off_dribble_college_top_key', 'off_dribble_college_break_right', 'on_move_fifteen', 'on_move_college', 'player_name'],
            "nba_player_information": ['birthdate', 'birthplace', 'college', 'draft_pick', 'draft_round', 'draft_team', 'draft_year', 'highschool', 'player_name', 'position', 'shoots', 'weight', 'birthyear', 'height'],
            "nba_champion_history": ['year', 'final_sweep', 'nationality_of_mvp_player', 'mvp_player_name', 'western_champion_name', 'eastern_champion_name', 'nba_champion_name', 'nba_vice_champion_name'],
            "nba_team_information": ['disbandment_year', 'team_name', 'founded_year', 'arena', 'arena_capacity', 'owner', 'generalmanager', 'headcoach', 'dleagueaffiliation'],
            "nba_player_affiliation": ['salary', 'season', 'team_name', 'player_name'],
            "nba_game_information": ['summary_id', 'game_place', 'game_weekday', 'game_stadium', 'game_date', 'home_team_name', 'visitor_team_name'],
            "nba_team_game_stats": ['team_name', 'team_assist', 'team_percentage_of_three_point_field_goal_made', 'team_percentage_of_field_goal_made', 'team_points', 'team_points_in_quarter1', 'team_points_in_quarter2', 'team_points_in_quarter3', 'team_points_in_quarter4', 'team_rebound', 'team_turnover', 'summary_id'],
            "nba_player_game_stats": ['player_name', 'number_of_assist', 'number_of_block', 'number_of_defensive_rebounds', 'number_of_three_point_field_goals_attempted', 'number_of_three_point_field_goals_made', 'percentage_of_three_point_field_goal_made', 'number_of_field_goals_attempted', 'number_of_field_goals_made', 'percentage_of_field_goal_made', 'number_of_free_throws_attempted', 'number_of_free_throws_made', 'percentage_of_free_throw_made', 'minutes_played', 'number_of_offensive_rebounds', 'number_of_personal_fouls', 'number_of_points', 'number_of_rebound', 'number_of_steal', 'number_of_turnover', 'summary_id']
        }

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
            [(vids[i], vids[j]) for i in range(len(vids))
                               for j in range(i + 1, len(vids))]
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
        Internally maps labels → vertex handles only once and then
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

       
        mapped_edges = [(o2s[int(e.source())], o2s[int(e.target())])
                        for e in other.g.edges()]

        self.g.add_edge_list(mapped_edges)

    def to_json(self, path):
        gt.write_graph(self.g, path, fmt="json")

def gt_to_dict(g: GTLabelled):
    return {
        "nodes": [json.dumps(lbl, default=str) for lbl in g._idx.keys()],
        "edges": [
            (
                json.loads(g._vprop_label[e.source()]),
                json.loads(g._vprop_label[e.target()])
            )
            for e in g.g.edges()
        ],
    }

def readJson(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def writeJson(path, datas):
    with open(path, 'w') as f:
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
    return [dataList[i:i+batchSize] for i in range(0, len(dataList), batchSize)]

def splitDataset(dataset, workerNum: int) -> list[list]:
    idxs = range(len(dataset))
    dataList = [dataset[i] for i in tqdm(idxs)]
    return [a.tolist() for a in np.array_split(dataList, workerNum)]

## add 'entities' key in dataset
def addEntityExtraction(dataset, args):
    path = args.entityExtractionPath
    mapping = readJson(path)
    for data in tqdm(dataset):
        data['entities'] = [m['entities'] for m in mapping if m['question_id'] == data['question_id']][0]

## add 'headers' & 'hIdxs' key in dataset
def addHeaderExtraction(dataset, args):
    path = args.headerExtractionPath
    mapping = readJson(path)
    for data in tqdm(dataset):
        hIdxs = [m['hIdxs'] for m in mapping if m['question_id'] == data['question_id']][0]
        headers_dict = {}
        if isinstance(data['table'], str):
            data['table'] = [data['table']]
        table_list = data['table']
        for table_name in table_list:
            headers_for_table = all_tables.get(table_name, [])
            relevant_indices = hIdxs.get(table_name, [])
            headers_dict[table_name] = [headers_for_table[i] for i in relevant_indices if 0 <= i < len(headers_for_table)]
        data['headers'] = headers_dict
        data['hIdxs'] = hIdxs

## add 'mappings' key in dataset
def addEntityToHeaderMapping(dataset, args):
    path = args.entityToHeaderMappingPath
    mapping = readJson(path)
    for data in tqdm(dataset):
        data['mappings'] = [m['mappings'] for m in mapping if m['question_id'] == data['question_id']][0]

def addEntityDocumentGraph(dataset, args):
    for data in tqdm(dataset):
        gt_g = gt.load_graph(args.entityDocumentGraphPath, fmt="json")
        lbl_prop = gt_g.vp["label"]
        wrapper  = GTLabelled()
        for e in gt_g.edges():
            u_lbl = json.loads(lbl_prop[e.source()])
            v_lbl = json.loads(lbl_prop[e.target()])
            wrapper.add_edge(u_lbl, v_lbl)

        data['ed_graph'] = wrapper

## add 'st_graph' key in dataset
def addSubtableGraph(dataset, model):
    for data in tqdm(dataset):
        data['st_graph'] = model.subtableRetrieval(data)

def ordered(a, b):
    return (a, b) if repr(a) < repr(b) else (b, a)

## add 'hybrid_graph' key in dataset
def addHybridGraph(dataset, model):
    for data in tqdm(dataset):
        data['hybrid_graph'] = model.makeHybridGraph(data)

## add 'start_nodes' & 'start_scores' keys in dataset
def addStartNodes(dataset, args):
    path = args.startPointsPath
    infos = readJson(path)
    for data in tqdm(dataset):
        tmp = [info['start_nodes'] for info in infos if data['question_id'] == info['question_id']][0]
        start_nodes = []
        for n in tmp:
            if isinstance(n, list):
                if isinstance(n[2], list):
                    n = tuple([n[0], tuple(n[1]), tuple(n[2]), n[3]])
                else:
                    n = tuple([n[0], tuple(n[1]), n[2], n[3]])
            start_nodes.append(n)
        data['start_nodes'] = start_nodes
        data['start_scores'] = [info['start_scores'] for info in infos if data['question_id'] == info['question_id']][0]   
        
        
def extract_json_objects(text, decoder=JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data
    """
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