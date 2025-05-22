import json
import orjson
import tiktoken
from pathlib import Path
from collections import defaultdict
from common.prompt import traversal, ee, he, e2h

_TABLE_CACHE = {}

_SKIP_TABLES = {
    "nba_game_information",
    "nba_team_game_stats",
    "nba_player_game_stats",
}

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

def flatten(origin: list) -> list:
    flatted = []
    for o in origin:
        if isinstance(o, list):
            flatted += o
        else:
            flatted += [o]
    return flatted

class Prompter():
    def __init__(self, args):
        self.args = args
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.all_tables = all_tables
        text_data_path = args.text_data_path
        with open(text_data_path, 'r') as f:
            self.text_data = json.load(f)

    def table2text(self, title, headers, table):
        return '[TLE] ' + title + ' [TAB] col: | ' + ' | '.join(headers) + ' | [SEP] | ' + ' | [SEP] | '.join(' | '.join(r) for r in table) + ' | '
    
    def llmTTPrompt_GPT(self, data: dict):
        ban_set = set(data.setdefault("ban_list", []))

 
        passages = set()        
        for node in data["hybrid_graph"].nodes():
            if isinstance(node, tuple) and node[2] != "None":
                passages.update(node[2])      
        
        passages -= ban_set

        sd          = self.text_data
        mkpassage   = lambda k: f"Title: {k}\n\nSummary: {sd[k]}"
        passagesText = ' | \n\n'.join(
            mkpassage(key) for p in passages
            if (key := p.replace('/id/', '')) in sd
        )


        tbl_names = data["table"] if isinstance(data["table"], list) else [data["table"]]
        tablesTextList = []
        for tbl in tbl_names:
            if tbl in _SKIP_TABLES or tbl not in data["hIdxs"]:
                continue                        


            if tbl not in _TABLE_CACHE:
                p = Path(f"{self.args.source_tables_path}/{tbl}.json")
                _TABLE_CACHE[tbl] = orjson.loads(p.read_bytes())

            js      = _TABLE_CACHE[tbl]
            headers = self.all_tables.get(tbl, [])
            rows    = [[cell[0] for cell in r] for r in js["data"]]
            tablesTextList.append(self.table2text(tbl, headers, rows))

        system_prompt = traversal.system_final_answer_prompt
        user_prompt   = traversal.user_final_answer_prompt.format(
            tables   = "\n\n".join(tablesTextList),
            passages = passagesText,
            question = data["question"],
        )

        if self.isOverflow(user_prompt):
            user_prompt = self.resolveOverflow(
                prompt       = traversal.user_final_answer_prompt,
                tablesTextList   = tablesTextList,
                passagesText = passagesText,
                question     = data["question"],
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
    
    def entityExtractionPrompt_GPT(self, data:dict):
        if isinstance(data['table'], list):
            table_list = data['table']
        else:
            table_list = [data['table']]

        table_and_headers_text = []
        for i, table_name in enumerate(table_list):
            headers = all_tables.get(table_name, [])
            table_block = f"Table Name {i+1}: {table_name}\nTable Headers {i+1}: {headers}"
            table_and_headers_text.append(table_block)
        prompt = [
            {"role": "system", "content": ee.system_prompt},
            {"role": "user", "content": ee.user_prompt.format(question=data['question'], table_and_headers="\n\n".join(table_and_headers_text))}
        ]
        return prompt
    
    def processEntityExtraction_GPT(self, prompt, input:dict, output):
        if "Entities" in output:
            entities = output["Entities"]
        else:
            entities = []

        post_processed_entities = [str(entity).strip() for entity in entities if str(entity).lower().strip() in input['question'].lower()]

        result = input.copy()
        result['prompt'] = prompt
        result['pred'] = output
        result['entities'] = post_processed_entities

        return result
    
    def headerExtractionPrompt_GPT(self, data:dict):
        if isinstance(data['table'], list):
            table_list = data['table']
        else:
            table_list = [data['table']]

        table_and_headers_text = []
        for i, table_name in enumerate(table_list):
            headers = all_tables.get(table_name, [])
            
            table_block = f"Table Name {i+1}: {table_name}\nTable Headers {i+1}: {headers}"
            table_and_headers_text.append(table_block)

        prompt = [{"role": "system", "content": he.system_prompt}, {"role": "user", "content": he.user_prompt.format(table_and_headers="\n\n".join(table_and_headers_text), question=data['question'], entities=data['entities'])}]
        return prompt
    
    def processHeaderExtraction_GPT(self, prompt, input:dict, output):
        header_indices_dict = output
        processed = {}
        for table, header_names in header_indices_dict.items():
            full_header_names = self.all_tables.get(table, [])
            valid_header_names = [name for name in header_names if name in full_header_names]
            valid_indices = [full_header_names.index(valid_header) for valid_header in valid_header_names]
            processed[table] = sorted(set(valid_indices))
        
        result = input.copy()
        result['prompt'] = prompt
        result['pred'] = output
        result['hIdxs'] = processed
        return result
    
    def entityToHeaderMappingPrompt_GPT(self, data:dict):
        if isinstance(data['table'], list):
            table_list = data['table']
        else:
            table_list = [data['table']]
        
        hIdxs_dict = data['hIdxs']
        
        table_blocks = []
        for i, table_name in enumerate(table_list):
            headers_for_table = all_tables.get(table_name, [])
            relevant_indices = hIdxs_dict.get(table_name, [])
            if table_name not in data["hIdxs"]:
                continue
            headerTexts = []
            for j, hIdx in enumerate(relevant_indices):
                if hIdx in list(range(len(headers_for_table))):
                    headerTexts.append(headers_for_table[hIdx])

            block = f"Relevant Table Name {i+1}: {table_name}\nRelevant Headers {i+1}: {headerTexts}"
            table_blocks.append(block)
        
        prompt = [{"role": "system", "content": e2h.system_prompt}, {"role": "user", "content": e2h.user_prompt.format(table_and_headers="\n\n".join(table_blocks), question=data['question'], entities=data['entities'])}]
        return prompt

    def processEntityToHeaderMapping_GPT(self, prompt, input:dict, output):
        post_processed_maps = {}
        if output != {}:
            for entity, mappings in output.items():
                if entity in input['entities']:
                    if mappings == ["None"] or mappings == "None":
                        post_processed_maps[entity] = [{"Others": "Others"}]
                        continue
                    post_processed_mappings = []
                    if not isinstance(mappings, list):
                        mappings = [mappings]
                    for mapping in mappings:
                        if isinstance(mapping, str):
                            continue
                        post_processed_table_to_header_mapping = {}
                        for table_name, header_names in mapping.items():
                            if table_name in all_tables:
                                if not isinstance(header_names, list):
                                    header_names = [header_names]
                                if header_names == ["None"]:
                                    continue
                                post_processed_header_names = []
                                for header_name in header_names:
                                    if header_name in all_tables[table_name]:
                                        post_processed_header_names.append(header_name)
                                post_processed_table_to_header_mapping[table_name] = post_processed_header_names
                        if post_processed_table_to_header_mapping != {}:
                            post_processed_mappings.append(post_processed_table_to_header_mapping)
                    if post_processed_mappings == []:
                        post_processed_maps[entity] = [{"Others": "Others"}]
                    else:
                        post_processed_maps[entity] = post_processed_mappings
        else:
            for entity in input['entities']:
                post_processed_maps[entity] = [{"Others": "Others"}]
        
        input['prompt'] = prompt
        input['pred'] = output
        input['mappings'] = post_processed_maps
        return input

    def odysseyPrompt_GPT(self, g_all_cache, data: dict):
        SG        = data['subgraph']
        ban_set   = set(data['ban_list'])
        tables    = defaultdict(list)          
        passages  = set()                      

       
        for n in SG.nodes():
            if isinstance(n, tuple):           
                tables[n[3]].append(n)
                if n[2] != "None":             
                    passages.update(l for l in n[2] if l not in ban_set)
            elif isinstance(n, str):
                if n.startswith('/id/') and n not in ban_set:
                    passages.add(n)

       
        tablesTextList = []
        for tbl, nodes in tables.items():
            hIdxs = sorted({n[1][1] for n in nodes})
            rIdxs = sorted({n[1][0] for n in nodes})
            if not rIdxs:
                continue

            header_names  = [self.all_tables.get(tbl, [])[i] for i in hIdxs]
            col_pos       = {c: i for i, c in enumerate(hIdxs)}
            row_map       = {r: ['-'] * len(hIdxs) for r in rIdxs}

            for val, (r, c), *_ in nodes:      # n = (val, (r,c), links, tbl)
                row_map[r][col_pos[c]] = val
            subtable = [row_map[r] for r in rIdxs]

            tablesTextList.append(self.table2text(tbl, header_names, subtable))

       
        id_nodes = {
            nb for node in passages            
            for nb in g_all_cache.neighbors(node)
            if isinstance(nb, str) and nb.startswith('/id/')
        }
        passages.update(id_nodes - ban_set)    

        
        make_passage = lambda k: f"Title: {k}\n\nSummary: {self.text_data[k]}"
        passagesText = ' | \n\n'.join(
            make_passage(clean)
            for t in passages
            for clean in (t.replace('/id/', ''),)   
            if clean in self.text_data
        )

        
        system_prompt = traversal.system_one_hop_prompt
        user_prompt   = traversal.user_one_hop_prompt.format(
            tables="\n\n".join(tablesTextList),
            passages=passagesText,
            question=data['question']
        )

        if self.isOverflow(user_prompt):
            user_prompt = self.resolveOverflow(
                prompt      = traversal.user_one_hop_prompt,
                tablesTextList  = tablesTextList,
                passagesText= passagesText,
                question    = data['question']
            )

        data['ban_list'].extend(passages)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]

    def isOverflow(self, prompt):
        tokens = self.tokenizer.encode(prompt)
        return len(tokens) > 15000
    
    def resolveOverflow(self,
                        prompt: str = "",
                        tablesTextList=None,
                        passagesText: str = "",
                        question: str = ""):
        
        remaining_budget = 13000
        tok = self.tokenizer
        tablesTextList = tablesTextList or []


        tbl_info = []                          # [(len, tokens, text), …]
        for text in tablesTextList:
            tokens = tok.encode(text)
            tbl_info.append((len(tokens), tokens, text))

        tbl_info.sort(key=lambda x: x[0])

     
        NEWLINE_TOK = tok.encode("\n\n")     
        NL_LEN      = len(NEWLINE_TOK)

        tables_tokens    = []

        for length, tokens, _ in tbl_info:
            needed = length + (NL_LEN if tables_tokens else 0)
            if needed > remaining_budget:
                if remaining_budget <= 0:
                    break
                if tables_tokens:
                    tables_tokens.extend(NEWLINE_TOK)
                    remaining_budget -= NL_LEN
                tables_tokens.extend(tokens[:remaining_budget])
                remaining_budget = 0
                break

            if tables_tokens:
                tables_tokens.extend(NEWLINE_TOK)
            tables_tokens.extend(tokens)
            remaining_budget -= needed

        truncated_tables_text = tok.decode(tables_tokens)


        passage_budget = 14500 - len(tables_tokens)
        passages_tokens = tok.encode(passagesText)
        truncated_passages_text = tok.decode(passages_tokens[:passage_budget])


        return prompt.format(
            tables   = truncated_tables_text,
            passages = truncated_passages_text,
            question = question
        )