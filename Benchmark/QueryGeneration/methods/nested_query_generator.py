from utils import *
from prompts.nested import oneshotk
from non_nested_query_generator import EGQueryGenerator

class OneShotKNestedQueryGenerator(EGQueryGenerator):
    def __init__(self, args, data_manager, rng, all_table_set, table_info, dtype_dict, join_key_list, join_clause_list, single_query_blocks, single_query_blocks_w_agg, logger):
        super().__init__(args, data_manager, rng, all_table_set, table_info, dtype_dict, join_key_list, logger)
        self.join_clause_list = join_clause_list
        self.single_query_blocks = single_query_blocks
        self.single_query_blocks_w_agg = single_query_blocks_w_agg
        self.nested_predicate_prompt_temp = oneshotk
        self.table_to_single_query_blocks = map_tables_to_queries(single_query_blocks)
        self.table_to_single_query_blocks_w_agg = map_tables_to_queries(single_query_blocks_w_agg)
        self.column_to_tables = map_columns_to_tables(dtype_dict)

    def generate(self):
        self.sql_type_dict = select_clauses(self.args, self.rng)
        sql_clause_list = ["nested_predicate", "from"] + [clause_type for clause_type in ["where", "group", "having", "order", "limit"] if self.sql_type_dict[clause_type]] + ["select"]
        if self.sql_type_dict["group"]:
            self.ideal_calls += (len(sql_clause_list) - 3)
        else:
            self.ideal_calls += (len(sql_clause_list) - 2)
        self.ideal_calls += 2
        while True:
            history = []
            generated_clause_info_dict = {}
            database_representation = ""
            for clause_type in sql_clause_list:
                clause_gen_count = 0
                while clause_gen_count < self.args.hyperparams["max_clause_gen_count"]:
                    clause_gen_count += 1
                    # Generate the clause
                    try:
                        revised_clause, _ = self.generate_clause(database_representation, history, clause_type, generated_clause_info_dict)
                    except Exception as e:
                        print(e)
                        self.logger.error(f"Error generating clause {clause_type}")
                        continue

                    # Verify the clause with the database
                    error_type, instances = self.verify_with_database(clause_type, generated_clause_info_dict)
                    if error_type == "execution_error":
                        self.logger.error(f"Error verifying clause with the database {revised_clause}")
                        if clause_type == "from":
                            if self.sql_type_dict["group"]:
                                self.ideal_calls -= (len(sql_clause_list) - 3)
                            else:
                                self.ideal_calls -= (len(sql_clause_list) - 2)
                            break
                        continue
                    elif error_type == "empty_result_error":
                        self.logger.error(f"Empty result error for clause {revised_clause}")
                        continue

                    # Update the history
                    history = self.update_history(history, clause_type, revised_clause, generated_clause_info_dict)

                    if clause_type not in ["nested_predicate", "select"]:
                        filtered_rows, filtered_column_names = self.remove_all_none_columns(instances[0], instances[1])
                        if len(filtered_column_names) > 2:
                            database_representation = self.get_database_representation(filtered_rows, filtered_column_names)
                        else:
                            database_representation = self.get_database_representation(instances[0], instances[1])
                    break
                else:
                    # Delete the clause type which failed to generate
                    if clause_type == "from":
                        generated_clause_info_dict = {}
                        break
                    elif clause_type in generated_clause_info_dict:
                        del generated_clause_info_dict[clause_type]
                    else:
                        self.logger.error(f"Clause type {clause_type} not found in generated_clause_info_dict")
                        return None
            
            sql_query_list = self.get_sql_query(generated_clause_info_dict)
            if sql_query_list:
                break
            
        return sql_query_list
    
    def generate_clause(self, database_representation="", history=[], clause_type="", generated_clause_info_dict={}, relations=[], selected_inner_query_blocks=[]):
        if clause_type == "nested_predicate":
            generated_nested_predicate, accessed_tables, from_clause = self.nest_inner_query_block()
            self.update_generated_clause_info_dict(generated_clause_info_dict, clause_type, generated_nested_predicate, accessed_tables)
            self.update_generated_clause_info_dict(generated_clause_info_dict, "from", from_clause, accessed_tables)
            return generated_nested_predicate, accessed_tables
        elif clause_type == "from":
            if "relations" not in generated_clause_info_dict:
                accessed_table_list = []
                for selected_inner_query_block in selected_inner_query_blocks:
                    accessed_tables = get_accessed_tables(selected_inner_query_block)
                    accessed_table_list.extend(accessed_tables)
                if len(accessed_table_list) == 1:
                    joinable_tables = get_joinable_tables(accessed_table_list[0], self.join_clause_list)
                    relations = list(set(joinable_tables + accessed_table_list))
                else:
                    post_processed_outer_tables = find_join_path(self.join_clause_list, [] , accessed_table_list)[0]
                    relations = post_processed_outer_tables
                schema = self.get_schema(relations)
                system_prompt, user_prompt = self.get_prompt(clause_type, schema, [], selected_inner_query_block=selected_inner_query_blocks)
                is_error, response = self.call_llm(system_prompt, user_prompt, clause_type=clause_type, max_tokens=256)
                if is_error:
                    return None, None
                from_clause = self.revise_clause(response, "from")
                relations = extract_tables(from_clause)
                table_to_alias = from_clause_to_dict(from_clause)
                alias_to_table = {v: k for k, v in table_to_alias.items()}
                for alias, table in alias_to_table.items():
                    from_clause = from_clause.replace(alias, table).replace(f"AS {table}", "")
            else:
                relations = generated_clause_info_dict["relations"]
                if "from" not in generated_clause_info_dict:
                    from_clause = "FROM " + (
                    generate_from_clause_for_full_outer_view(
                        self.args, relations=relations, join_clause_list=self.join_clause_list, is_inner_join=True
                        ) if len(relations) > 1 else relations[0]
                    )
                    self.update_generated_clause_info_dict(generated_clause_info_dict, clause_type, from_clause, relations)
                else:
                    from_clause = generated_clause_info_dict["from"]
            return from_clause, relations
        else:
            return super().generate_clause(database_representation, history, clause_type, generated_clause_info_dict)

    def nest_inner_query_block(self,):
        history = []
        generated_outer_tables = []
        nested_predicate_types = self.args.hyperparams["nested_predicate_types"]
        nested_predicate_gen_count = 0
        generated_nested_predicates = ""
        generated_where_clause = ""
        while nested_predicate_gen_count < self.args.hyperparams["max_nested_predicate_gen_count"]:
            print(nested_predicate_types)
            selected_inner_query_blocks = self.get_candidate_inner_query_blocks(candidate_num=self.args.hyperparams["candidate_num"], nested_predicate_types=nested_predicate_types)
            from_clause, outer_tables = self.generate_clause(clause_type="from", selected_inner_query_blocks=selected_inner_query_blocks, relations=self.all_table_set)
            history = [from_clause]
            generated_nested_predicates = self.generate_nested_predicates(outer_tables, selected_inner_query_blocks, history, nested_predicate_types)
            error_type, sql_with_nested_predicate, __ = self.verify_nested_predicate_with_database(from_clause, outer_tables, generated_nested_predicates)
            
            if error_type == "execution_error":
                nested_predicate_gen_count += 1
                self.logger.error(f"Error verifying the nested predicate with the database {generated_nested_predicates}")
                continue
            elif error_type == "empty_result_error":
                nested_predicate_gen_count += 1
                self.logger.error(f"Empty result error for the nested predicate {generated_nested_predicates}")
                continue
    
            stype_list = detect_type_of_nested_predicate(sql_with_nested_predicate)
            gt_type_set = set(self.args.hyperparams["nested_predicate_types"])
            
            if self.args.hyperparams["height"] > 2:
                if set(stype_list[:1]) != gt_type_set or len(stype_list) != len(self.args.hyperparams["nested_predicate_types"])+(self.args.hyperparams["height"]-1):
                    nested_predicate_gen_count += 1
                    self.logger.error(f"Type mismatch for the nested predicate {generated_nested_predicates}")
                    continue
            else:
                if set(stype_list) != gt_type_set or len(stype_list) != len(self.args.hyperparams["nested_predicate_types"]):
                    nested_predicate_gen_count += 1
                    self.logger.error(f"Type mismatch for the nested predicate {generated_nested_predicates}")
                    continue

            break

        if self.total_calls >= self.max_calls:
            return {"sql": ["expired"]}, None
        
        generated_where_clause = f"WHERE {generated_nested_predicates}"
        generated_outer_tables = list(set(outer_tables))
        return generated_where_clause[len("WHERE"):].strip(), list(set(generated_outer_tables)), from_clause
    
    def get_candidate_inner_query_blocks(self, tables=None, candidate_num=1, nested_predicate_types=[]):
        candidate_inner_query_blocks = []
        total_candidate_inner_query_blocks_wo_agg = []
        total_candidate_inner_query_blocks_w_agg = []
        if tables is None:
            tables = self.all_table_set
        
        is_a = False
        is_n = False
        a_count = 0
        n_count = 0
        for nested_predicate_type in nested_predicate_types:
            if nested_predicate_type in ["type-a", "type-ja"]:
                is_a = True
                a_count += 1
            elif nested_predicate_type in ["type-n", "type-j"]:
                is_n = True
                n_count += 1
        
        if is_n:
            for table in tables:
                if table in self.table_to_single_query_blocks:
                    total_candidate_inner_query_blocks_wo_agg.extend(self.table_to_single_query_blocks[table])
        
            candidate_num = min(candidate_num, len(total_candidate_inner_query_blocks_wo_agg))
            candidate_inner_query_blocks.extend(self.rng.choice(total_candidate_inner_query_blocks_wo_agg, size=candidate_num, replace=False)[:n_count])
        
        if is_a:
            for table in tables:
                if table in self.table_to_single_query_blocks_w_agg:
                    total_candidate_inner_query_blocks_w_agg.extend(self.table_to_single_query_blocks_w_agg[table])
        
            if "type-a" in nested_predicate_types:
                total_candidate_inner_query_blocks_w_agg = [block for block in total_candidate_inner_query_blocks_w_agg if "count" not in block.lower()]
            
            candidate_num = min(candidate_num, len(total_candidate_inner_query_blocks_w_agg))
            candidate_inner_query_blocks.extend(self.rng.choice(total_candidate_inner_query_blocks_w_agg, size=candidate_num, replace=False)[:a_count])
        
        return [str(block) for block in candidate_inner_query_blocks]
    
    def generate_nested_predicates(self, relations, selected_inner_query_blocks, history, nested_predicate_types):
        schema = self.get_schema(relations)
        system_prompt, user_prompt = self.get_prompt("nested_predicate", schema, history, selected_inner_query_block=selected_inner_query_blocks, nested_predicate_types=nested_predicate_types)
        is_error, response = self.call_llm(system_prompt, user_prompt, clause_type="nested_predicate", max_tokens=256)
        generated_nested_predicates = post_process_nested_predicate(response['nested_predicate'][0].strip())            
        if is_error:
            return None
        
        if generated_nested_predicates == "":
            return None, None
        
        return generated_nested_predicates

    def get_prompt(self, clause_type, schema, history, selected_inner_query_block=None, subquery_execution_result=None, candidate_inner_query_blocks=None, nested_predicate_types=[]):
        if clause_type == "nested_predicate":
            nested_predicate_requirements = ""
            for pred_id, nested_predicate_type in enumerate(list(dict.fromkeys(nested_predicate_types))):
                if nested_predicate_type == "type-ja":
                    nested_predicate_requirements += f"{pred_id+4}) {self.nested_predicate_prompt_temp.type_ja_predicate_requirement}\n"
                elif nested_predicate_type == "type-j":
                    nested_predicate_requirements += f"{pred_id+4}) {self.nested_predicate_prompt_temp.type_j_predicate_requirement}\n"
                elif nested_predicate_type == "type-a":
                    nested_predicate_requirements += f"{pred_id+4}) {self.nested_predicate_prompt_temp.type_a_predicate_requirement}\n"
                elif nested_predicate_type == "type-n":
                    nested_predicate_requirements += f"{pred_id+4}) {self.nested_predicate_prompt_temp.type_n_predicate_requirement}\n"

            nested_predicate_formats = ""
            for pred_id, nested_predicate_type in enumerate(list(dict.fromkeys(nested_predicate_types))):
                if nested_predicate_type == "type-ja":
                    nested_predicate_formats += f"{self.nested_predicate_prompt_temp.type_ja_predicate_format}\n"
                elif nested_predicate_type == "type-j":
                    nested_predicate_formats += f"{self.nested_predicate_prompt_temp.type_j_predicate_format}\n"
                elif nested_predicate_type == "type-a":
                    nested_predicate_formats += f"{self.nested_predicate_prompt_temp.type_a_predicate_format}\n"
                elif nested_predicate_type == "type-n":
                    nested_predicate_formats += f"{self.nested_predicate_prompt_temp.type_n_predicate_format}\n"
            
            system_prompt = self.nested_predicate_prompt_temp.system_generate_nested_predicates_prompt.format(nested_predicate_types=nested_predicate_types, nested_predicate_requirements=nested_predicate_requirements, nested_predicate_formats=nested_predicate_formats, num_child_queries=len(nested_predicate_types), height=self.args.hyperparams["height"])
            user_prompt = self.nested_predicate_prompt_temp.user_generate_nested_predicates_prompt.format(schema=schema, history=history, candidate_inner_query_blocks=selected_inner_query_block, nested_predicate_requirements=nested_predicate_requirements)
        elif clause_type == "from":
            system_prompt = self.nested_predicate_prompt_temp.system_from_multiple_tables_prompt
            user_prompt = self.nested_predicate_prompt_temp.user_from_multiple_tables_prompt.format(schema=schema, subquery=selected_inner_query_block)
        else:
            system_prompt, user_prompt = super().get_prompt(clause_type, schema, history)
        
        return system_prompt, user_prompt

    def verify_nested_predicate_with_database(self, from_clause, outer_tables, generated_nested_predicates):
        error_type = None
        select_clause = "SELECT " +  generate_select_clause_for_full_outer_view(self.table_info, relations=outer_tables, use_alias=True)
        where_clause = f"WHERE {generated_nested_predicates}"
        table_to_alias = from_clause_to_dict(from_clause)
        for table in outer_tables:
            if table in table_to_alias:
                alias = table_to_alias[table]
                select_clause = select_clause.replace(table, alias)
        
        sql_with_nested_predicate = f"{select_clause} {from_clause} {where_clause}"
        # Verify the sql with the database
        try:  
            execution_result_rows, _ = self.data_manager.sample_rows_with_sql(sql_with_nested_predicate, self.args.hyperparams["sample_size"], self.rng)
            if len(execution_result_rows) == 0:
                error_type = "empty_result_error"
                return error_type, sql_with_nested_predicate, ""
        except Exception as e:
            print(e)
            error_type = "execution_error"
            return error_type, sql_with_nested_predicate, e
        
        return error_type, sql_with_nested_predicate, ""

    def update_history(self, history, clause_type, revised_clause, query_info):
        if clause_type == "where" and not str(history[0]).startswith("FROM"):
            where_clause_with_nested_predicate = f"{revised_clause} AND {history[0][len('WHERE'):].strip()}"
            history.append(where_clause_with_nested_predicate)
            history = history[1:]
        elif clause_type == "nested_predicate":
            if len(query_info['relations']) > 1:
                revised_clause = revised_clause.replace(".", "__")
            else:
                revised_clause = revised_clause.replace(f"{query_info['relations'][0]}.", "")
            history.append(f"WHERE {revised_clause}")
        else:
            history.append(f"{revised_clause}")
        return history

    def get_sql_query(self, generated_clause_info_dict):
        if "nested_predicate" in generated_clause_info_dict and "where" in generated_clause_info_dict:
            generated_clause_info_dict["where"] = "WHERE " + generated_clause_info_dict["nested_predicate"] + " AND " + generated_clause_info_dict["where"][len("WHERE"):]
        elif "nested_predicate" in generated_clause_info_dict:
            generated_clause_info_dict["where"] =  f'WHERE {generated_clause_info_dict["nested_predicate"]}'
        return super().get_sql_query(generated_clause_info_dict)
    