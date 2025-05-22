import orjson
import os, json
from tqdm import tqdm
from openai import OpenAI
from itertools import combinations
from collections import defaultdict
from common.prompter import Prompter
from common.utils import GTLabelled, extract_json_objects, ordered
skip_tbl  = {
            "nba_game_information",
            "nba_team_game_stats",
            "nba_player_game_stats",
        }
# ─────────────────────────────────────────────────────────────────────────────
class Odyssey():
    def __init__(self, args):
        self.args = args
        self.llm_hyperparams = {"temperature": self.args.temperature,"top_p": self.args.top_p,"max_tokens": self.args.max_tokens}
        openai_api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=openai_api_key)
        self.llm = client
        self.prompter = Prompter(self.args)
        self.nba_player_information  = json.load(open(f"{self.args.source_tables_path}/nba_player_information.json", encoding="utf-8"))
        self.nba_team_information  = json.load(open(f"{self.args.source_tables_path}/nba_team_information.json", encoding="utf-8"))

    def inference_GPT(self, prompt):
        completion = self.llm.chat.completions.create(
            model=self.args.modelName,
            messages=prompt,
            **self.llm_hyperparams,
        ).choices[0].message.content
        
        try:
            response = json.loads(completion)
        except json.JSONDecodeError:
            response = extract_json_objects(completion)
    
        return response

   
    def subtableRetrieval(self, data):

        table_list = data["table"] if isinstance(data["table"], list) else [data["table"]]

        G               = GTLabelled()
        cell2lbl        = {}                
        bulk_edges      = set()               
        header_lbl_buf  = {}                
        cnt             = 0

        for table_name in table_list:
            if table_name in skip_tbl or table_name not in data["hIdxs"]:
                continue

            cnt += 1
            path = f"{self.args.source_tables_path}/{table_name}.json"
            tbl  = json.load(open(path, encoding="utf-8"))

            rows        = tbl["data"]
            headers     = [h[0] for h in tbl["header"]]
            hIdxs       = data["hIdxs"][table_name]

            for c in hIdxs:
                header_lbl_buf[(table_name, c)] = f"/{table_name}/header/{headers[c]}"

            for r, row in enumerate(rows):
                row_nodes = []

                for c in hIdxs:
                    val, links, join_links = row[c]

                    links = tuple(links) if table_name in (
                        "nba_player_information",
                        "nba_team_information") and links else "None"

                    lbl = (val, (r, c), links, table_name)
                    cell2lbl[(table_name, r, c)] = lbl
                    row_nodes.append(lbl)

                    bulk_edges.add((header_lbl_buf[(table_name, c)], lbl))

                    if join_links:
                        for jl in join_links:         
                            for tgt_tbl, coord_list in jl.items():
                                if tgt_tbl not in table_list or tgt_tbl in skip_tbl or tgt_tbl not in data["hIdxs"]:
                                    continue
                                for tr, tc in coord_list:
                                    bulk_edges.add(((table_name, r, c),
                                                    (tgt_tbl,   tr, tc)))
                                    if (tgt_tbl, tr, tc) not in cell2lbl:
                                        if tgt_tbl == "nba_player_information":
                                            passage_links = tuple(self.nba_player_information["data"][tr][tc][1]) if self.nba_player_information["data"][tr][tc][1] else "None"
                                            cell2lbl[(tgt_tbl, tr, tc)] = (val, (tr, tc), passage_links, tgt_tbl)
                                        elif tgt_tbl == "nba_team_information":
                                            passage_links = tuple(self.nba_team_information["data"][tr][tc][1]) if self.nba_team_information["data"][tr][tc][1] else "None"
                                            cell2lbl[(tgt_tbl, tr, tc)] = (val, (tr, tc), passage_links, tgt_tbl)
                                        else:
                                            cell2lbl[(tgt_tbl, tr, tc)] = (val, (tr, tc), "None", tgt_tbl)

                if len(row_nodes) > 1:
                    bulk_edges.update(combinations(row_nodes, 2))

        
        dedup = {ordered(src, dst) for src, dst in bulk_edges}

        final_pairs = set()
        for src_key, dst_key in dedup:
            src_lbl = cell2lbl.get(src_key, src_key)  
            dst_lbl = cell2lbl.get(dst_key, dst_key)
            final_pairs.add((src_lbl, dst_lbl))

        G.add_edges(final_pairs)

        return None if cnt == 0 else G

   
    def _load_node_link(self, d):
        """Convert legacy NetworkX node-link JSON → GTLabelled."""
        g = GTLabelled()
        id2lbl = {}
        for n in d["nodes"]:
            raw = n.get("id", n)
            try:
                lbl = json.loads(raw)
            except Exception:
                lbl = raw
            id2lbl[n["id"] if "id" in n else n] = lbl
            g._get_v(lbl)
        for l in d["links"]:
            g.add_edge(id2lbl[l["source"]], id2lbl[l["target"]])
        return g


    def makeHybridGraph(self, data):
        
        if not hasattr(self, "_edg_cache"):
            with open(self.args.entityDocumentGraphPath, "rb",) as f:
                ed_json = orjson.loads(f.read())
            self._edg_cache = {k: self._load_node_link(v) for k, v in ed_json.items()}

        if not hasattr(self, "_g_all_cache"):
            G_all             = GTLabelled()
            all_edges         = set()                
            entity_neighbors  = defaultdict(set)     

            for g in tqdm(self._edg_cache.values(), desc="Unifying EDGs"):
                vprop = g._vprop                  
                for e in g.g.edges():
                    u_lbl = vprop[e.source()]
                    v_lbl = vprop[e.target()]
                    all_edges.add((u_lbl, v_lbl))
                    entity_neighbors[u_lbl].add(v_lbl)
                    entity_neighbors[v_lbl].add(u_lbl)

            G_all.add_edges(all_edges)         
            self._g_all_cache        = G_all
            self._entity_nbrs_cache  = entity_neighbors

      
        STG = data["st_graph"]
        if STG is None:
            return self._g_all_cache
        
        if not isinstance(data['table'], list):
            table_list = [data['table']]
        else:
            table_list = list(set(data['table']))
        
        std_nodes = []
        for table_name in table_list:
            if table_name in skip_tbl or table_name not in data["hIdxs"]:
                continue
            
            if table_name in ["nba_player_information"]:
                std_nodes.extend(STG.neighbors('/nba_player_information/header/player_name'))
            elif table_name in ["nba_team_information"]:
                std_nodes.extend(STG.neighbors('/nba_team_information/header/team_name'))

        # connect STG tuple-nodes to linked entities in EDG
        nbr_cache  = self._entity_nbrs_cache
        bulk_edges = set()

        for n in tqdm(std_nodes, desc="Linking STG ⇄ EDG"):
            if not (isinstance(n, tuple) and n[2] != "None"):
                continue
            for link in n[2]:                       
                for nb in nbr_cache.get(link, ()):
                    edge = (n, nb)
                    bulk_edges.add(edge)
            
        STG.add_edges(bulk_edges)

        return STG


    def getNextHopGraph(self, data):
        HG: GTLabelled = data["hybrid_graph"]
        SG: GTLabelled = data["subgraph"]          

       
        new_nodes = set(SG._idx.keys())          

        idx     = HG._idx
        canon   = HG._canon
        vprop   = HG._vprop
        addN    = new_nodes.add

      
        for lbl in tuple(new_nodes):             
            v = idx.get(canon(lbl))              
            if v is None:
                continue
            for nb in v.all_neighbors():           
                addN(vprop[nb])                    

       
        return HG.subgraph(new_nodes)

    def extractGraphInfo(self, data):
        if "ban_list" not in data:
            data["ban_list"] = []
        
        ban          = set(data["ban_list"])
        table_nodes  = []
        passage_set  = set()

        append_tn    = table_nodes.append
        add_passage  = passage_set.add

        for n in data["subgraph"].nodes():        
            if isinstance(n, tuple):              
                append_tn(n)
                links = n[2]
                if links != "None":               
                    for title in links:
                        if title not in ban:
                            add_passage(title)
            elif isinstance(n, str) and n.startswith("/id/") and n not in ban:
                add_passage(n)

        return table_nodes, list(passage_set)
 
    def addFinalHop(self, inp):
        if "gpt" in self.args.modelName:
            prompt = self.prompter.llmTTPrompt_GPT(inp)
            out = self.inference_GPT(prompt)
            if isinstance(out, list):
                if len(out) == 0:
                    out = {}
                else:
                    out = out[0]
            
            if "Final Answers" in out:
                final_ans = out["Final Answers"] if out["Final Answers"] != [] else "None"
            else:
                final_ans = "None"
        else:
            prompt = self.prompter.llmTTPrompt(inp)
            final_ans = self.inference(prompt)
        inp.update(final_pred=final_ans, final_prompt=prompt, pred=final_ans)

    def addOneHop(self, n, inp):
        inp["subgraph"] = self.getNextHopGraph(inp)
        t_nodes, p_titles = self.extractGraphInfo(inp)
        inp[f"{n}-hop_table"], inp[f"{n}-hop_passages"] = t_nodes, p_titles

        if "gpt" in self.args.modelName:
            prompt = self.prompter.odysseyPrompt_GPT(self._g_all_cache, inp)
            out = self.inference_GPT(prompt)
            if isinstance(out, list):
                if len(out) == 0:
                    out = {}
                else:
                    out = out[0]
            
            if not isinstance(out, dict):
                out = {}
            
            if "Relevant Passages" in out:
                for p in out["Relevant Passages"]:
                    try:
                        p = int(p)
                        summary_id = f"/id/{p}"
                    except:
                        if '/id/' in p: summary_id = p
                        else: summary_id = f"/id/{p}"
                    
                    if summary_id in inp['ban_list']:
                        inp['ban_list'].remove(summary_id)

            if "Final Answers" in out:
                if len(out["Final Answers"]) == 0:
                    final_ans = "None"
                elif isinstance(out["Final Answers"][0], str) and out["Final Answers"][0].lower() == "none":
                    final_ans = "None"
                else:
                    final_ans = out["Final Answers"]
            else:
                final_ans = "None"
        else:
            prompt = self.prompter.odysseyPrompt(self._edg_cache, inp)
            out = self.inference(prompt)
            self.prompter.processRelevantPassages(inp, out)
            final_ans = self.prompter.processOdyssey(out)

        inp[f"{n}-hop_pred"] = out
        inp[f"{n}-hop_prompt"] = prompt
        inp["pred"] = final_ans
        return inp["pred"] == "None"