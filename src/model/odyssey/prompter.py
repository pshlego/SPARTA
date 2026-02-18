import json
import orjson
import tiktoken
from pathlib import Path
from collections import defaultdict
from src.prompt.model.odyssey import traversal, ee, he, e2h

_TABLE_CACHE = {}


def flatten(origin: list) -> list:
    flatted = []
    for o in origin:
        if isinstance(o, list):
            flatted += o
        else:
            flatted += [o]
    return flatted


class Prompter:
    def __init__(self, adapter):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        source_path = getattr(adapter, "source_tables_path", None)
        self.source_tables_path = Path(source_path) if source_path is not None else ""
        self.grounding_tables = getattr(adapter, "grounding_tables", {})

    def table2text(self, title, headers, table):
        return (
            "[TLE] "
            + title
            + " [TAB] col: | "
            + " | ".join(headers)
            + " | [SEP] | "
            + " | [SEP] | ".join(" | ".join(r) for r in table)
            + " | "
        )

    def llmTTPrompt_GPT(self, data: dict, adapter):
        ban_set = set(data.setdefault("ban_list", []))

        passages = set()
        for node in data["hybrid_graph"].nodes():
            if isinstance(node, tuple) and node[2] != "None":
                passages.update(node[2])

        passages -= ban_set

        sd = data["text"]
        mkpassage = lambda k: f"Title: {k}\n\nSummary: {sd[k]}"
        passagesText = " | \n\n".join(
            mkpassage(key if key in sd else f"/id/{key}")
            for p in passages
            if (key := p.replace("/id/", "")) in sd or f"/id/{key}" in sd
        )

        tbl_names = (
            data["table"] if isinstance(data["table"], list) else [data["table"]]
        )
        tablesTextList = []
        for tbl in tbl_names:
            if tbl in self.grounding_tables or tbl not in data["hIdxs"]:
                continue

            if adapter.data_has_table_cells:  # OttQA
                js = data["table_data"][tbl]
            else:
                if tbl not in _TABLE_CACHE:
                    p = Path(f"{self.source_tables_path}/{tbl}.json")
                    _TABLE_CACHE[tbl] = orjson.loads(p.read_bytes())

                js = _TABLE_CACHE[tbl]
            table_header_mapping = adapter.get_table_header_mapping(data)
            headers = table_header_mapping.get(tbl, [])
            rows = [[cell[0] for cell in r] for r in js["data"]]
            tablesTextList.append(self.table2text(tbl, headers, rows))

        system_prompt = traversal.system_final_answer_prompt
        user_prompt = traversal.user_final_answer_prompt.format(
            tables="\n\n".join(tablesTextList),
            passages=passagesText,
            question=data["question"],
        )

        if self.isOverflow(user_prompt):
            user_prompt = self.resolveOverflow(
                prompt=traversal.user_final_answer_prompt,
                tablesTextList=tablesTextList,
                passagesText=passagesText,
                question=data["question"],
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def entityExtractionPrompt_GPT(self, data: dict, adapter):
        if isinstance(data["table"], list):
            table_list = data["table"]
        else:
            table_list = [data["table"]]

        passage_only_question = all(
            table_name in adapter.grounding_tables for table_name in table_list
        )
        if not passage_only_question:
            table_and_headers_text = []
            source_tables_list = [
                table_name
                for table_name in table_list
                if table_name not in adapter.grounding_tables
            ]
            for i, table_name in enumerate(source_tables_list):
                headers = adapter.get_table_header_mapping(data).get(table_name, [])
                table_block = (
                    f"Table Name {i+1}: {table_name}\nTable Headers {i+1}: {headers}"
                )
                table_and_headers_text.append(table_block)

            prompt = [
                {"role": "system", "content": ee.system_prompt},
                {
                    "role": "user",
                    "content": ee.user_prompt.format(
                        question=data["question"],
                        table_and_headers="\n\n".join(table_and_headers_text),
                    ),
                },
            ]
        else:
            prompt = [
                {"role": "system", "content": ee.system_prompt_passage_only},
                {
                    "role": "user",
                    "content": ee.user_prompt_passage_only.format(
                        question=data["question"]
                    ),
                },
            ]
        return prompt

    def processEntityExtraction_GPT(self, prompt, input: dict, output):
        if "Entities" in output:
            entities = output["Entities"]
        else:
            entities = []

        post_processed_entities = [
            str(entity).strip()
            for entity in entities
            if str(entity).lower().strip() in input["question"].lower()
        ]

        result = input.copy()
        result["prompt"] = prompt
        result["pred"] = output
        result["entities"] = post_processed_entities

        return result

    def headerExtractionPrompt_GPT(self, data: dict, adapter):
        if isinstance(data["table"], list):
            table_list = data["table"]
        else:
            table_list = [data["table"]]

        source_tables_list = [
            table_name
            for table_name in table_list
            if table_name not in adapter.grounding_tables
        ]
        table_and_headers_text = []
        for i, table_name in enumerate(source_tables_list):
            headers = adapter.get_table_header_mapping(data).get(table_name, [])

            table_block = (
                f"Table Name {i+1}: {table_name}\nTable Headers {i+1}: {headers}"
            )
            table_and_headers_text.append(table_block)

        prompt = [
            {"role": "system", "content": he.system_prompt},
            {
                "role": "user",
                "content": he.user_prompt.format(
                    table_and_headers="\n\n".join(table_and_headers_text),
                    question=data["question"],
                    entities=data["entities"],
                ),
            },
        ]
        return prompt

    def processHeaderExtraction_GPT(self, prompt, input: dict, output, adapter):
        header_indices_dict = output
        processed = {}
        for table, header_names in header_indices_dict.items():
            full_header_names = adapter.get_table_header_mapping(input).get(table, [])
            valid_header_names = [
                name for name in header_names if name in full_header_names
            ]
            valid_indices = [
                full_header_names.index(valid_header)
                for valid_header in valid_header_names
            ]
            processed[table] = sorted(set(valid_indices))

        result = input.copy()
        result["prompt"] = prompt
        result["pred"] = output
        result["hIdxs"] = processed
        return result

    def entityToHeaderMappingPrompt_GPT(self, data: dict, adapter):
        if isinstance(data["table"], list):
            table_list = data["table"]
        else:
            table_list = [data["table"]]
        source_tables_list = [
            table_name
            for table_name in table_list
            if table_name not in adapter.grounding_tables
        ]

        hIdxs_dict = data["hIdxs"]

        table_blocks = []
        for i, table_name in enumerate(source_tables_list):
            headers_for_table = adapter.get_table_header_mapping(data).get(
                table_name, []
            )
            relevant_indices = hIdxs_dict.get(table_name, [])
            if table_name not in data["hIdxs"]:
                continue
            headerTexts = []
            for j, hIdx in enumerate(relevant_indices):
                if hIdx in list(range(len(headers_for_table))):
                    headerTexts.append(headers_for_table[hIdx])

            block = f"Relevant Table Name {i+1}: {table_name}\nRelevant Headers {i+1}: {headerTexts}"
            table_blocks.append(block)

        prompt = [
            {"role": "system", "content": e2h.system_prompt},
            {
                "role": "user",
                "content": e2h.user_prompt.format(
                    table_and_headers="\n\n".join(table_blocks),
                    question=data["question"],
                    entities=data["entities"],
                ),
            },
        ]
        return prompt

    def processEntityToHeaderMapping_GPT(self, prompt, input: dict, output, adapter):
        post_processed_maps = {}
        if output != {}:
            for entity, mappings in output.items():
                if entity in input["entities"]:
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
                            if table_name in adapter.get_table_header_mapping(input):
                                if not isinstance(header_names, list):
                                    header_names = [header_names]
                                if header_names == ["None"]:
                                    continue
                                post_processed_header_names = []
                                for header_name in header_names:
                                    if (
                                        header_name
                                        in adapter.get_table_header_mapping(input)[
                                            table_name
                                        ]
                                    ):
                                        post_processed_header_names.append(header_name)
                                post_processed_table_to_header_mapping[table_name] = (
                                    post_processed_header_names
                                )
                        if post_processed_table_to_header_mapping != {}:
                            post_processed_mappings.append(
                                post_processed_table_to_header_mapping
                            )
                    if post_processed_mappings == []:
                        post_processed_maps[entity] = [{"Others": "Others"}]
                    else:
                        post_processed_maps[entity] = post_processed_mappings
        else:
            for entity in input["entities"]:
                post_processed_maps[entity] = [{"Others": "Others"}]

        input["prompt"] = prompt
        input["pred"] = output
        input["mappings"] = post_processed_maps
        return input

    def odysseyPrompt_GPT(self, g_all_cache, data: dict, adapter):
        SG = data["subgraph"]
        ban_set = set(data["ban_list"])
        tables = defaultdict(list)
        passages = set()

        for n in SG.nodes():
            if isinstance(n, tuple):
                tables[n[3]].append(n)
                if n[2] != "None":
                    passages.update(l for l in n[2] if l not in ban_set)
            elif isinstance(n, str):
                if (
                    n.startswith("/id/") or n.startswith("/wiki/")
                ) and n not in ban_set:
                    passages.add(n)

        tablesTextList = []
        for tbl, nodes in tables.items():
            hIdxs = sorted({n[1][1] for n in nodes})
            rIdxs = sorted({n[1][0] for n in nodes})
            if not rIdxs:
                continue
            table_header_mapping = adapter.get_table_header_mapping(data)
            header_names = [table_header_mapping.get(tbl, [])[i] for i in hIdxs]
            col_pos = {c: i for i, c in enumerate(hIdxs)}
            row_map = {r: ["-"] * len(hIdxs) for r in rIdxs}

            for val, (r, c), *_ in nodes:  # n = (val, (r,c), links, tbl)
                row_map[r][col_pos[c]] = val
            subtable = [row_map[r] for r in rIdxs]

            tablesTextList.append(self.table2text(tbl, header_names, subtable))

        id_nodes = {
            nb
            for node in passages
            for nb in g_all_cache.neighbors(node)
            if isinstance(nb, str) and (nb.startswith("/id/") or nb.startswith("/wiki/"))
        }
        passages.update(id_nodes - ban_set)

        make_passage = lambda k: f"Title: {k}\n\nSummary: {data['text'][k]}"
        passagesText = " | \n\n".join(
            make_passage(key)
            for t in passages
            for clean in (t.replace("/id/", ""),)
            if (
                key := (
                    clean
                    if clean in data["text"]
                    else f"/id/{clean}" if f"/id/{clean}" in data["text"] else None
                )
            )
        )
        system_prompt = traversal.system_one_hop_prompt
        user_prompt = traversal.user_one_hop_prompt.format(
            tables="\n\n".join(tablesTextList),
            passages=passagesText,
            question=data["question"],
        )

        if self.isOverflow(user_prompt):
            user_prompt = self.resolveOverflow(
                prompt=traversal.user_one_hop_prompt,
                tablesTextList=tablesTextList,
                passagesText=passagesText,
                question=data["question"],
            )

        data["ban_list"].extend(passages)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def isOverflow(self, prompt):
        tokens = self.tokenizer.encode(prompt)
        return len(tokens) > 15000

    def resolveOverflow(
        self,
        prompt: str = "",
        tablesTextList=None,
        passagesText: str = "",
        question: str = "",
    ):

        remaining_budget = 13000
        tok = self.tokenizer
        tablesTextList = tablesTextList or []

        tbl_info = []
        for text in tablesTextList:
            tokens = tok.encode(text)
            tbl_info.append((len(tokens), tokens, text))

        tbl_info.sort(key=lambda x: x[0])

        NEWLINE_TOK = tok.encode("\n\n")
        NL_LEN = len(NEWLINE_TOK)

        tables_tokens = []

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
            tables=truncated_tables_text,
            passages=truncated_passages_text,
            question=question,
        )
