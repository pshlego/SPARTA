from src.utils.sparta import *
from src.benchmark.sparta.QueryGeneration.methods.M_Schema.utils import (
    examples_to_str,
    read_json,
    write_json,
)
from src.benchmark.sparta.QueryGeneration.methods.prov_query_fixer import (
    ProvQueryFixer,
)
from src.benchmark.sparta.QueryGeneration.methods.non_nested_query_generator import (
    EGQueryGenerator,
)
from src.prompt.sparta.QueryGeneration.nested import (
    oneshotk,
    oneshotk_movie,
    oneshotk_medical,
    postorder,
    postorder_movie,
    postorder_medical,
    correction,
    correction_movie,
    correction_medical,
)


class OneShotKNestedQueryGenerator(EGQueryGenerator):
    def __init__(
        self,
        args,
        data_manager,
        rng,
        all_table_set,
        table_info,
        dtype_dict,
        join_key_list,
        join_clause_list,
        single_query_blocks,
        single_query_blocks_w_agg,
        logger,
    ):
        super().__init__(
            args,
            data_manager,
            rng,
            all_table_set,
            table_info,
            dtype_dict,
            join_key_list,
            logger,
        )
        self.join_clause_list = join_clause_list
        self.single_query_blocks = single_query_blocks
        self.single_query_blocks_w_agg = single_query_blocks_w_agg
        if self.args.domain == "movie":
            self.nested_predicate_prompt_temp = oneshotk_movie
        elif self.args.domain == "medical":
            self.nested_predicate_prompt_temp = oneshotk_medical
        else:
            self.logger.warning(
                f"Unknown domain '{self.args.domain}' encountered. Falling back to 'oneshotk' prompt template."
            )
            self.nested_predicate_prompt_temp = oneshotk
        self.table_to_single_query_blocks = map_tables_to_queries(single_query_blocks)
        self.table_to_single_query_blocks_w_agg = map_tables_to_queries(
            single_query_blocks_w_agg
        )
        self.column_to_tables = map_columns_to_tables(dtype_dict)

    def generate(self):
        self.sql_type_dict = select_clauses(self.args, self.rng)
        sql_clause_list = (
            ["nested_predicate", "from"]
            + [
                clause_type
                for clause_type in ["where", "group", "having", "order", "limit"]
                if self.sql_type_dict[clause_type]
            ]
            + ["select"]
        )
        if self.sql_type_dict["group"]:
            self.ideal_calls += len(sql_clause_list) - 3
        else:
            self.ideal_calls += len(sql_clause_list) - 2
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
                        revised_clause, _ = self.generate_clause(
                            database_representation,
                            history,
                            clause_type,
                            generated_clause_info_dict,
                        )
                    except Exception as e:
                        print(e)
                        self.logger.error(f"Error generating clause {clause_type}")
                        continue

                    if "NULL" in revised_clause:
                        self.logger.error(
                            f"NULL value found in clause {revised_clause}"
                        )
                        continue
                    if clause_type == "select" or clause_type == "group":
                        sql_query_list = self.get_sql_query(generated_clause_info_dict, consider_duplicate=False)[0]
                        equality_columns = extract_equality_predicate_columns(sql_query_list)
                        invalid_columns = equality_columns.union(self.args.INVALID_IDS)
                        wrong_count = sum([True if invalid_column in revised_clause else False for invalid_column in invalid_columns])
                        if wrong_count > 0:
                            self.logger.error(
                                f"Equality column found in {clause_type} clause {revised_clause}, invalid_columns: {invalid_columns}"
                            )
                            continue
                        
                    if clause_type == "where":
                        if generated_clause_info_dict["nested_predicate"] in revised_clause:
                            revised_clause = revised_clause.replace(f"{generated_clause_info_dict['nested_predicate']} AND", "").replace(f"AND {generated_clause_info_dict['nested_predicate']}", "").replace(f"{generated_clause_info_dict['nested_predicate']}", "").strip()
                            if generated_clause_info_dict["nested_predicate"] in revised_clause or revised_clause == "WHERE":
                                self.logger.error(
                                    f"Duplicate nested predicate found in where clause {revised_clause}"
                                )
                                continue
                            generated_clause_info_dict[clause_type] = revised_clause
                    

                    # Verify the clause with the database
                    error_type, instances = self.verify_with_database(
                        clause_type, generated_clause_info_dict
                    )
                    if error_type == "execution_error":
                        self.logger.error(
                            f"Error verifying clause with the database {revised_clause}"
                        )
                        if clause_type == "from":
                            if self.sql_type_dict["group"]:
                                self.ideal_calls -= len(sql_clause_list) - 3
                            else:
                                self.ideal_calls -= len(sql_clause_list) - 2
                            break
                        continue
                    elif error_type == "empty_result_error":
                        self.logger.error(
                            f"Empty result error for clause {revised_clause}"
                        )
                        continue

                    # Update the history
                    history = self.update_history(
                        history, clause_type, revised_clause, generated_clause_info_dict
                    )

                    if clause_type not in ["nested_predicate", "select"]:
                        filtered_rows, filtered_column_names = (
                            self.remove_all_none_columns(instances[0], instances[1])
                        )
                        if len(filtered_column_names) > 2:
                            database_representation = self.get_database_representation(
                                filtered_rows, filtered_column_names
                            )
                        else:
                            database_representation = self.get_database_representation(
                                instances[0], instances[1]
                            )
                    break
                else:
                    # Delete the clause type which failed to generate
                    if clause_type == "from":
                        generated_clause_info_dict = {}
                        break
                    elif clause_type in generated_clause_info_dict:
                        del generated_clause_info_dict[clause_type]
                    else:
                        self.logger.error(
                            f"Clause type {clause_type} not found in generated_clause_info_dict"
                        )
                        return None

            sql_query_list = self.get_sql_query(generated_clause_info_dict)
            if sql_query_list:
                break

        return sql_query_list

    def generate_clause(
        self,
        database_representation="",
        history=[],
        clause_type="",
        generated_clause_info_dict={},
        relations=[],
        selected_inner_query_blocks=[],
    ):
        if clause_type == "nested_predicate":
            generated_nested_predicate, accessed_tables, from_clause = (
                self.nest_inner_query_block()
            )
            self.update_generated_clause_info_dict(
                generated_clause_info_dict,
                clause_type,
                generated_nested_predicate,
                accessed_tables,
            )
            self.update_generated_clause_info_dict(
                generated_clause_info_dict, "from", from_clause, accessed_tables
            )
            return generated_nested_predicate, accessed_tables
        elif clause_type == "from":
            if "relations" not in generated_clause_info_dict:
                accessed_table_list = []
                for selected_inner_query_block in selected_inner_query_blocks:
                    accessed_tables = get_accessed_tables(selected_inner_query_block)
                    accessed_table_list.extend(accessed_tables)
                if len(accessed_table_list) == 1:
                    joinable_tables = get_joinable_tables(
                        accessed_table_list[0], self.join_clause_list
                    )
                    relations = list(set(joinable_tables + accessed_table_list))
                else:
                    post_processed_outer_tables = find_join_path(
                        self.join_clause_list, [], accessed_table_list
                    )[0]
                    relations = post_processed_outer_tables
                schema = self.get_schema(relations)
                system_prompt, user_prompt = self.get_prompt(
                    clause_type,
                    schema,
                    [],
                    selected_inner_query_block=selected_inner_query_blocks,
                )
                is_error, response = self.call_llm(
                    system_prompt, user_prompt, clause_type=clause_type, max_tokens=256
                )
                if is_error:
                    return None, None
                from_clause = self.revise_clause(response, "from")
                relations = extract_tables(from_clause)
                table_to_alias = from_clause_to_dict(from_clause)
                alias_to_table = {v: k for k, v in table_to_alias.items()}
                for alias, table in alias_to_table.items():
                    from_clause = from_clause.replace(alias, table).replace(
                        f"AS {table}", ""
                    )
            else:
                relations = generated_clause_info_dict["relations"]
                if "from" not in generated_clause_info_dict:
                    from_clause = "FROM " + (
                        generate_from_clause_for_full_outer_view(
                            self.args,
                            relations=relations,
                            join_clause_list=self.join_clause_list,
                            is_inner_join=True,
                        )
                        if len(relations) > 1
                        else relations[0]
                    )
                    self.update_generated_clause_info_dict(
                        generated_clause_info_dict, clause_type, from_clause, relations
                    )
                else:
                    from_clause = generated_clause_info_dict["from"]
            return from_clause, relations
        else:
            return super().generate_clause(
                database_representation,
                history,
                clause_type,
                generated_clause_info_dict,
            )

    def nest_inner_query_block(
        self,
    ):
        history = []
        generated_outer_tables = []
        nested_predicate_types = self.args.hyperparams["nested_predicate_types"]
        nested_predicate_gen_count = 0
        generated_nested_predicates = ""
        generated_where_clause = ""
        while (
            nested_predicate_gen_count
            < self.args.hyperparams["max_nested_predicate_gen_count"]
        ):
            print(nested_predicate_types)
            selected_inner_query_blocks = self.get_candidate_inner_query_blocks(
                candidate_num=self.args.hyperparams["candidate_num"],
                nested_predicate_types=nested_predicate_types,
            )
            from_clause, outer_tables = self.generate_clause(
                clause_type="from",
                selected_inner_query_blocks=selected_inner_query_blocks,
                relations=self.all_table_set,
            )
            history = [from_clause]
            generated_nested_predicates = self.generate_nested_predicates(
                outer_tables,
                selected_inner_query_blocks,
                history,
                nested_predicate_types,
            )
            error_type, sql_with_nested_predicate, __ = (
                self.verify_nested_predicate_with_database(
                    from_clause, outer_tables, generated_nested_predicates
                )
            )

            if error_type == "execution_error":
                nested_predicate_gen_count += 1
                self.logger.error(
                    f"Error verifying the nested predicate with the database {generated_nested_predicates}"
                )
                continue
            elif error_type == "empty_result_error":
                nested_predicate_gen_count += 1
                self.logger.error(
                    f"Empty result error for the nested predicate {generated_nested_predicates}"
                )
                continue

            stype_list = detect_type_of_nested_predicate(sql_with_nested_predicate)
            gt_type_set = set(self.args.hyperparams["nested_predicate_types"])

            if self.args.hyperparams["height"] > 1:
                is_strange_type_1 = (
                    len(stype_list[:len(self.args.hyperparams["nested_predicate_types"])])
                    != len(self.args.hyperparams["nested_predicate_types"])
                )
                gt_type_set = set(
                    self.args.hyperparams["nested_predicate_types"][
                        : 1 + 1
                    ]
                )
                is_strange_type_2 = set(stype_list[:len(self.args.hyperparams["nested_predicate_types"])]) != gt_type_set


                if is_strange_type_2 or is_strange_type_1:
                    nested_predicate_gen_count += 1
                    self.logger.error(
                        f"Type mismatch for the nested predicate {generated_nested_predicates}"
                    )
                    continue
            else:
                if set(stype_list) != gt_type_set or len(stype_list) != len(
                    self.args.hyperparams["nested_predicate_types"]
                ):
                    nested_predicate_gen_count += 1
                    self.logger.error(
                        f"Type mismatch for the nested predicate {generated_nested_predicates}"
                    )
                    continue

            break

        if self.total_calls >= self.max_calls:
            return {"sql": ["expired"]}, None

        generated_where_clause = f"WHERE {generated_nested_predicates}"
        generated_outer_tables = list(set(outer_tables))
        return (
            generated_where_clause[len("WHERE") :].strip(),
            list(set(generated_outer_tables)),
            from_clause,
        )

    def get_candidate_inner_query_blocks(
        self, tables=None, candidate_num=1, nested_predicate_types=[]
    ):
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
                    total_candidate_inner_query_blocks_wo_agg.extend(
                        self.table_to_single_query_blocks[table]
                    )

            candidate_num = min(
                candidate_num, len(total_candidate_inner_query_blocks_wo_agg)
            )
            candidate_inner_query_blocks.extend(
                self.rng.choice(
                    total_candidate_inner_query_blocks_wo_agg,
                    size=candidate_num,
                    replace=False,
                )[:n_count]
            )

        if is_a:
            for table in tables:
                if table in self.table_to_single_query_blocks_w_agg:
                    total_candidate_inner_query_blocks_w_agg.extend(
                        self.table_to_single_query_blocks_w_agg[table]
                    )

            if "type-a" in nested_predicate_types:
                total_candidate_inner_query_blocks_w_agg = [
                    block
                    for block in total_candidate_inner_query_blocks_w_agg
                    if "count" not in block.lower()
                ]

            candidate_num = min(
                candidate_num, len(total_candidate_inner_query_blocks_w_agg)
            )
            candidate_inner_query_blocks.extend(
                self.rng.choice(
                    total_candidate_inner_query_blocks_w_agg,
                    size=candidate_num,
                    replace=False,
                )[:a_count]
            )

        return [str(block) for block in candidate_inner_query_blocks]

    def generate_nested_predicates(
        self, relations, selected_inner_query_blocks, history, nested_predicate_types
    ):
        schema = self.get_schema(relations)
        system_prompt, user_prompt = self.get_prompt(
            "nested_predicate",
            schema,
            history,
            selected_inner_query_block=selected_inner_query_blocks,
            nested_predicate_types=nested_predicate_types,
        )
        is_error, response = self.call_llm(
            system_prompt, user_prompt, clause_type="nested_predicate", max_tokens=256
        )
        generated_nested_predicates = post_process_nested_predicate(
            response["nested_predicate"][0].strip()
        )
        if is_error:
            return None

        if generated_nested_predicates == "":
            return None, None

        return generated_nested_predicates

    def get_prompt(
        self,
        clause_type,
        schema,
        history,
        selected_inner_query_block=None,
        subquery_execution_result=None,
        candidate_inner_query_blocks=None,
        nested_predicate_types=[],
    ):
        if clause_type == "nested_predicate":
            nested_predicate_requirements = ""
            for pred_id, nested_predicate_type in enumerate(
                list(dict.fromkeys(nested_predicate_types))
            ):
                if nested_predicate_type == "type-ja":
                    nested_predicate_requirements += f"{pred_id+4}) {self.nested_predicate_prompt_temp.type_ja_predicate_requirement}\n"
                elif nested_predicate_type == "type-j":
                    nested_predicate_requirements += f"{pred_id+4}) {self.nested_predicate_prompt_temp.type_j_predicate_requirement}\n"
                elif nested_predicate_type == "type-a":
                    nested_predicate_requirements += f"{pred_id+4}) {self.nested_predicate_prompt_temp.type_a_predicate_requirement}\n"
                elif nested_predicate_type == "type-n":
                    nested_predicate_requirements += f"{pred_id+4}) {self.nested_predicate_prompt_temp.type_n_predicate_requirement}\n"

            nested_predicate_formats = ""
            for pred_id, nested_predicate_type in enumerate(
                list(dict.fromkeys(nested_predicate_types))
            ):
                if nested_predicate_type == "type-ja":
                    nested_predicate_formats += f"{self.nested_predicate_prompt_temp.type_ja_predicate_format}\n"
                elif nested_predicate_type == "type-j":
                    nested_predicate_formats += (
                        f"{self.nested_predicate_prompt_temp.type_j_predicate_format}\n"
                    )
                elif nested_predicate_type == "type-a":
                    nested_predicate_formats += (
                        f"{self.nested_predicate_prompt_temp.type_a_predicate_format}\n"
                    )
                elif nested_predicate_type == "type-n":
                    nested_predicate_formats += (
                        f"{self.nested_predicate_prompt_temp.type_n_predicate_format}\n"
                    )

            system_prompt = self.nested_predicate_prompt_temp.system_generate_nested_predicates_prompt.format(
                nested_predicate_types=nested_predicate_types,
                nested_predicate_requirements=nested_predicate_requirements,
                nested_predicate_formats=nested_predicate_formats,
                num_child_queries=len(nested_predicate_types),
                height=self.args.hyperparams["height"],
            )
            user_prompt = self.nested_predicate_prompt_temp.user_generate_nested_predicates_prompt.format(
                schema=schema,
                history=history,
                candidate_inner_query_blocks=selected_inner_query_block,
                nested_predicate_requirements=nested_predicate_requirements,
            )
        elif clause_type == "from":
            system_prompt = (
                self.nested_predicate_prompt_temp.system_from_multiple_tables_prompt
            )
            user_prompt = self.nested_predicate_prompt_temp.user_from_multiple_tables_prompt.format(
                schema=schema, subquery=selected_inner_query_block
            )
        else:
            system_prompt, user_prompt = super().get_prompt(
                clause_type, schema, history
            )

        return system_prompt, user_prompt

    def verify_nested_predicate_with_database(
        self, from_clause, outer_tables, generated_nested_predicates
    ):
        error_type = None
        select_clause = "SELECT " + generate_select_clause_for_full_outer_view(
            self.table_info, relations=outer_tables, use_alias=True
        )
        where_clause = f"WHERE {generated_nested_predicates}"
        table_to_alias = from_clause_to_dict(from_clause)
        for table in outer_tables:
            if table in table_to_alias:
                alias = table_to_alias[table]
                select_clause = select_clause.replace(table, alias)

        sql_with_nested_predicate = f"{select_clause} {from_clause} {where_clause}"
        # Verify the sql with the database
        try:
            execution_result_rows, _ = self.data_manager.sample_rows_with_sql(
                sql_with_nested_predicate,
                self.args.hyperparams["sample_size"],
                self.rng,
            )
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
            where_clause_with_nested_predicate = (
                f"{revised_clause} AND {history[0][len('WHERE'):].strip()}"
            )
            history.append(where_clause_with_nested_predicate)
            history = history[1:]
        elif clause_type == "nested_predicate":
            if len(query_info["relations"]) > 1:
                revised_clause = revised_clause.replace(".", "__")
            else:
                revised_clause = revised_clause.replace(
                    f"{query_info['relations'][0]}.", ""
                )
            history.append(f"WHERE {revised_clause}")
        else:
            history.append(f"{revised_clause}")
        return history

    def get_sql_query(self, generated_clause_info_dict, consider_duplicate=True):
        if consider_duplicate:
            if (
                "nested_predicate" in generated_clause_info_dict
                and "where" in generated_clause_info_dict
            ):
                generated_clause_info_dict["where"] = (
                    "WHERE "
                    + generated_clause_info_dict["nested_predicate"]
                    + " AND "
                    + generated_clause_info_dict["where"][len("WHERE") :]
                )
            elif "nested_predicate" in generated_clause_info_dict:
                generated_clause_info_dict["where"] = (
                    f'WHERE {generated_clause_info_dict["nested_predicate"]}'
                )
        return super().get_sql_query(generated_clause_info_dict, consider_duplicate)


class PostOrderNestedQueryGenerator(OneShotKNestedQueryGenerator):
    def __init__(
        self,
        args,
        data_manager,
        rng,
        all_table_set,
        table_info,
        dtype_dict,
        join_key_list,
        join_clause_list,
        single_query_blocks,
        single_query_blocks_w_agg,
        logger,
    ):
        super().__init__(
            args,
            data_manager,
            rng,
            all_table_set,
            table_info,
            dtype_dict,
            join_key_list,
            join_clause_list,
            single_query_blocks,
            single_query_blocks_w_agg,
            logger,
        )
        if self.args.domain == "movie":
            self.nested_predicate_prompt_temp = postorder_movie
        elif self.args.domain == "medical":
            self.nested_predicate_prompt_temp = postorder_medical
        else:
            self.nested_predicate_prompt_temp = postorder
        if self.args.is_prov:
            if self.args.domain == "movie":
                self.nested_predicate_type_prompt = oneshotk_movie
            elif self.args.domain == "medical":
                self.nested_predicate_type_prompt = oneshotk_medical
            else:
                self.nested_predicate_type_prompt = oneshotk
            self.prov_query_fixer = ProvQueryFixer(data_manager, rng)
            if self.args.domain == "movie":
                self.correction_prompt_temp = correction_movie
            elif self.args.domain == "medical":
                self.correction_prompt_temp = correction_medical
            else:
                self.correction_prompt_temp = correction
        self.ideal_calls += (
            (len(self.args.hyperparams["nested_predicate_types"]) - 1)
            * self.args.num_queries
            * 2
        )

    def generate_clause(
        self,
        database_representation="",
        history=[],
        clause_type="",
        generated_clause_info_dict={},
        relations=[],
        selected_inner_query_block="",
    ):
        if clause_type == "nested_predicate":
            # Generate nested predicates
            generated_nested_predicate, accessed_tables, from_clause = (
                self.nest_inner_query_block()
            )
            self.update_generated_clause_info_dict(
                generated_clause_info_dict,
                clause_type,
                generated_nested_predicate,
                accessed_tables,
            )
            self.update_generated_clause_info_dict(
                generated_clause_info_dict, "from", from_clause, accessed_tables
            )
            return generated_nested_predicate, accessed_tables
        elif clause_type == "from":
            # Generate from clause
            if "relations" not in generated_clause_info_dict:
                schema = self.get_schema(relations)
                system_prompt, user_prompt = self.get_prompt(
                    clause_type,
                    schema,
                    selected_inner_query_block=[selected_inner_query_block],
                )
                is_error, response = self.call_llm(
                    system_prompt, user_prompt, clause_type=clause_type, max_tokens=256
                )
                if is_error:
                    return None, None
                # relations = [response['from'][0].replace("FROM", "").split("AS")[0].strip()]
                from_clause = self.revise_clause(response, "from")
                for relation in relations:
                    if relation in from_clause:
                        relations = [relation]
                        from_clause = f"FROM {relation}"
                        break
            else:
                relations = generated_clause_info_dict["relations"]
                if "from" not in generated_clause_info_dict:
                    from_clause = "FROM " + (
                        generate_from_clause_for_full_outer_view(
                            self.args,
                            relations=relations,
                            join_clause_list=self.join_clause_list,
                            is_inner_join=True,
                        )
                        if len(relations) > 1
                        else relations[0]
                    )
                    self.update_generated_clause_info_dict(
                        generated_clause_info_dict, clause_type, from_clause, relations
                    )
                else:
                    from_clause = generated_clause_info_dict["from"]
            return from_clause, relations
        else:
            # Generate other clauses
            return super().generate_clause(
                database_representation,
                history,
                clause_type,
                generated_clause_info_dict,
            )

    def nest_inner_query_block(
        self,
    ):
        if self.args.is_prov:
            self.prov_query_fixer.highlighter.missing_tables = []
        generated_nested_predicates = []
        generated_outer_tables = []
        generated_where_clause = ""
        generated_from_clause = ""
        for child_query_id, nested_predicate_type in enumerate(
            self.args.hyperparams["nested_predicate_types"]
        ):
            nested_predicate_gen_count = 0
            while (
                nested_predicate_gen_count
                < self.args.hyperparams["max_nested_predicate_gen_count"]
            ):
                print(
                    f"Total predicate types: {self.args.hyperparams['nested_predicate_types']}, Current predicate type: {nested_predicate_type}"
                )
                nested_predicate_gen_count += 1
                # Select inner query block
                selected_inner_query_block, inner_table = self.select_inner_query_block(
                    nested_predicate_type,
                    generated_where_clause,
                    generated_from_clause,
                    generated_outer_tables,
                )
                if selected_inner_query_block is None:
                    self.logger.error(f"Error selecting inner query block")
                    continue

                if child_query_id == 0:
                    # Generate from clause
                    candidate_outer_tables = list(
                        set(
                            get_joinable_tables(inner_table, self.join_clause_list)
                            + [inner_table]
                        )
                    )
                    from_clause, outer_tables = self.generate_clause(
                        clause_type="from",
                        selected_inner_query_block=selected_inner_query_block,
                        relations=candidate_outer_tables,
                    )
                    outer_tables = [
                        table for table in outer_tables if table in self.all_table_set
                    ]
                    if from_clause is None:
                        self.logger.error(f"Error generating from clause")
                        continue

                # Generate nested predicate
                accessed_tables = list(set([inner_table] + outer_tables))
                generated_nested_predicate, logical_operator = (
                    self.generate_nested_predicate(
                        nested_predicate_type,
                        accessed_tables,
                        selected_inner_query_block,
                        generated_where_clause,
                        from_clause,
                    )
                )
                if generated_nested_predicate is None:
                    self.logger.error(f"Error generating nested predicate")
                    continue
                elif generated_nested_predicate in generated_nested_predicates:
                    self.logger.error(
                        f"Duplicate nested predicate {generated_nested_predicate}"
                    )
                    continue

                error_type, sql_with_nested_predicate, error_log = (
                    self.verify_nested_predicate_with_database(
                        outer_tables,
                        generated_nested_predicate,
                        logical_operator,
                        generated_where_clause,
                    )
                )
                if self.args.is_prov:
                    stype_list = detect_type_of_nested_predicate(
                        sql_with_nested_predicate
                    )
                    if self.args.hyperparams["height"] > 1:
                        is_strange_type_1 = (
                            len(stype_list[:len(self.args.hyperparams["nested_predicate_types"])])
                            != len(self.args.hyperparams["nested_predicate_types"])
                        )
                        gt_type_set = set(
                            self.args.hyperparams["nested_predicate_types"][
                                : child_query_id + 1
                            ]
                        )
                        is_strange_type_2 = set(stype_list[:len(self.args.hyperparams["nested_predicate_types"])]) != gt_type_set
                    else:
                        is_strange_type_1 = len(stype_list) != len(
                            self.args.hyperparams["nested_predicate_types"][
                                : child_query_id + 1
                            ]
                        )
                        gt_type_set = set(
                            self.args.hyperparams["nested_predicate_types"][
                                : child_query_id + 1
                            ]
                        )
                        is_strange_type_2 = set(stype_list) != gt_type_set

                    if is_strange_type_1:
                        self.logger.error(
                            f"Type mismatch for the nested predicate {generated_nested_predicate}"
                        )
                        continue

                    if is_strange_type_2 or is_strange_type_1:
                        stype_set = set(stype_list)
                        self.logger.warning(
                            f"Type mismatch for the nested predicate {generated_nested_predicate}"
                        )
                        if error_type is None:
                            error_type = "nested_predicate_type_mismatch"
                            error_log = self.get_error_log_for_nested_predicate(
                                generated_nested_predicate, stype_set, gt_type_set
                            )
                            generated_nested_predicate = self.correct_query(
                                error_type,
                                sql_with_nested_predicate,
                                error_log,
                                nested_predicate_type,
                                accessed_tables,
                                outer_tables[0],
                            )
                            if generated_nested_predicate is None:
                                self.logger.error(
                                    f"Type mismatch for the nested predicate {generated_nested_predicate}"
                                )
                                continue
                            error_type, sql_with_nested_predicate, error_log = (
                                self.verify_nested_predicate_with_database(
                                    outer_tables,
                                    generated_nested_predicate,
                                    logical_operator,
                                    generated_where_clause,
                                )
                            )
                            stype_list = detect_type_of_nested_predicate(
                                sql_with_nested_predicate
                            )
                            gt_type_set = set(
                                self.args.hyperparams["nested_predicate_types"][
                                    : child_query_id + 1
                                ]
                            )

                            if self.args.hyperparams["height"] > 1:
                                is_strange_type_1 = (
                                    len(stype_list[:len(self.args.hyperparams["nested_predicate_types"])])
                                    != len(self.args.hyperparams["nested_predicate_types"])
                                )
                                gt_type_set = set(
                                    self.args.hyperparams["nested_predicate_types"][
                                        : child_query_id + 1
                                    ]
                                )
                                is_strange_type_2 = set(stype_list[:len(self.args.hyperparams["nested_predicate_types"])]) != gt_type_set
                            else:
                                is_strange_type_1 = len(stype_list) != len(
                                    self.args.hyperparams["nested_predicate_types"][
                                        : child_query_id + 1
                                    ]
                                )
                                gt_type_set = set(
                                    self.args.hyperparams["nested_predicate_types"][
                                        : child_query_id + 1
                                    ]
                                )
                                is_strange_type_2 = set(stype_list) != gt_type_set

                            if is_strange_type_1 or is_strange_type_2:
                                self.logger.error(
                                    f"Type mismatch for the nested predicate {generated_nested_predicate}; Generated: {stype_list}; GT: {gt_type_set}"
                                )
                                continue
                                    
                        else:
                            continue

                if self.args.is_prov:
                    if error_type == "execution_error":
                        self.logger.warning(
                            f"Execution error for the nested predicate {generated_nested_predicate}"
                        )
                        generated_nested_predicate = self.correct_query(
                            error_type,
                            sql_with_nested_predicate,
                            error_log,
                            nested_predicate_type,
                            accessed_tables,
                        )
                    elif error_type == "empty_result_error":
                        self.logger.warning(
                            f"Empty result error for the nested predicate {generated_nested_predicate}"
                        )
                        generated_nested_predicate = self.correct_query(
                            error_type,
                            sql_with_nested_predicate,
                            error_log,
                            nested_predicate_type,
                        )

                    if generated_nested_predicate is None:
                        self.logger.error(
                            f"Error correcting the nested predicate {generated_nested_predicate}"
                        )
                        continue

                    if error_type is not None:
                        error_type, sql_with_nested_predicate, error_log = (
                            self.verify_nested_predicate_with_database(
                                outer_tables,
                                generated_nested_predicate,
                                logical_operator,
                                generated_where_clause,
                            )
                        )

                if error_type == "execution_error":
                    self.logger.error(
                        f"Error verifying the nested predicate with the database {generated_nested_predicate}"
                    )
                    continue
                elif error_type == "empty_result_error":
                    self.logger.error(
                        f"Empty result error for the nested predicate {generated_nested_predicate}"
                    )
                    continue

                stype_list = detect_type_of_nested_predicate(sql_with_nested_predicate)
                if self.args.hyperparams["height"] > 1:
                    is_strange_type_1 = (
                        len(stype_list[:len(self.args.hyperparams["nested_predicate_types"])])
                        != len(self.args.hyperparams["nested_predicate_types"])
                    )
                    gt_type_set = set(
                        self.args.hyperparams["nested_predicate_types"][
                            : child_query_id + 1
                        ]
                    )
                    is_strange_type_2 = set(stype_list[:len(self.args.hyperparams["nested_predicate_types"])]) != gt_type_set
                else:
                    is_strange_type_1 = len(stype_list) != len(
                        self.args.hyperparams["nested_predicate_types"][
                            : child_query_id + 1
                        ]
                    )
                    gt_type_set = set(
                        self.args.hyperparams["nested_predicate_types"][
                            : child_query_id + 1
                        ]
                    )
                    is_strange_type_2 = set(stype_list) != gt_type_set

                if is_strange_type_1 or is_strange_type_2:
                    self.logger.error(
                        f"Type mismatch for the nested predicate {generated_nested_predicate}; Generated: {stype_list}; GT: {gt_type_set}"
                    )
                    continue

                generated_nested_predicates.append(generated_nested_predicate)
                if child_query_id == 0:
                    generated_where_clause = f"WHERE {generated_nested_predicate}"
                else:
                    generated_where_clause = f"{generated_where_clause} {logical_operator} {generated_nested_predicate}"

                generated_outer_tables.extend(outer_tables)
                if len(outer_tables) > 1:
                    generated_from_clause = "FROM " + (
                        generate_from_clause_for_full_outer_view(
                            self.args,
                            relations=outer_tables,
                            join_clause_list=self.join_clause_list,
                            is_inner_join=True,
                        )
                    )
                else:
                    generated_from_clause = from_clause

                break
            else:
                if child_query_id == 0:
                    return None, None
                elif len(generated_nested_predicates) == 0:
                    return None, None
                else:
                    break

        return (
            generated_where_clause[len("WHERE") :].strip(),
            list(set(generated_outer_tables)),
            generated_from_clause,
        )

    def select_inner_query_block(
        self,
        nested_predicate_type,
        generated_where_clause,
        generated_from_clause,
        generated_outer_tables,
    ):
        if generated_where_clause == "":
            selected_inner_query_block = str(
                self.get_candidate_inner_query_blocks(
                    nested_predicate_type=nested_predicate_type
                )[0]
            )
            accessed_table = get_accessed_tables(selected_inner_query_block)[0]
        else:
            candidate_inner_query_tables = []
            if nested_predicate_type in ["type-a", "type-ja"]:
                candidate_inner_query_tables = generated_outer_tables
            else:
                for outer_table in generated_outer_tables:
                    joinable_tables = get_joinable_tables(
                        outer_table, self.join_clause_list
                    )
                    candidate_inner_query_tables.extend(joinable_tables)

            schema = self.get_schema(generated_outer_tables)
            candidate_inner_query_blocks = self.get_candidate_inner_query_blocks(
                nested_predicate_type=nested_predicate_type,
                tables=candidate_inner_query_tables,
                candidate_num=self.args.hyperparams["candidate_num"],
            )
            system_prompt, user_prompt = self.get_prompt(
                "inner_query_block",
                schema,
                generated_from_clause=[generated_from_clause],
                generated_where_clause=[generated_where_clause],
                candidate_inner_query_blocks=candidate_inner_query_blocks,
            )
            is_error, response = self.call_llm(
                system_prompt,
                user_prompt,
                clause_type="inner_query_block",
                max_tokens=256,
            )
            if is_error:
                return None, None
            selected_inner_query_block = response["inner_query_block"][0]
            accessed_table = get_accessed_tables(selected_inner_query_block)[0]

        return selected_inner_query_block, accessed_table

    def get_candidate_inner_query_blocks(
        self, nested_predicate_type="type-n", tables=None, candidate_num=1
    ):
        candidate_inner_query_blocks = []
        if tables is None:
            tables = self.all_table_set

        if self.args.is_prov:
            tables = [
                table
                for table in tables
                if table not in self.prov_query_fixer.highlighter.missing_tables
            ]
            if len(tables) == 0:
                tables = self.all_table_set

        table_to_single_query_blocks = (
            self.table_to_single_query_blocks
            if nested_predicate_type in ["type-n", "type-j"]
            else self.table_to_single_query_blocks_w_agg
        )
        for table in tables:
            if table in table_to_single_query_blocks:
                candidate_inner_query_blocks.extend(table_to_single_query_blocks[table])

        if nested_predicate_type in ["type-a", "type-ja"]:
            candidate_inner_query_blocks = [
                block
                for block in candidate_inner_query_blocks
                if "count" not in block.lower()
            ]

        candidate_num = min(candidate_num, len(candidate_inner_query_blocks))
        candidate_inner_query_blocks = self.rng.choice(
            candidate_inner_query_blocks, size=candidate_num, replace=False
        )
        return candidate_inner_query_blocks

    def generate_nested_predicate(
        self,
        nested_predicate_type,
        accessed_tables,
        selected_inner_query_block,
        generated_where_clause,
        generated_from_clause,
    ):
        schema = self.get_schema(accessed_tables)
        system_prompt, user_prompt = self.get_prompt(
            "nested_predicate",
            schema,
            nested_predicate_type=nested_predicate_type,
            selected_inner_query_block=[selected_inner_query_block],
            generated_where_clause=[generated_where_clause],
            generated_from_clause=[generated_from_clause],
        )
        is_error, response = self.call_llm(
            system_prompt, user_prompt, clause_type="nested_predicate", max_tokens=512
        )
        if is_error:
            return None, None

        generated_nested_predicate = post_process_nested_predicate(
            response["nested_predicate"][0]
        )
        logical_operator = response["logical_operator"]
        if generated_nested_predicate == "":
            return None, None

        return generated_nested_predicate, logical_operator

    def get_prompt(
        self,
        clause_type,
        schema="",
        history=[],
        selected_inner_query_block=[],
        candidate_inner_query_blocks=None,
        nested_predicate_type="",
        generated_where_clause="",
        generated_from_clause="",
        query="",
        provenance_analysis_results="",
        problematic_subquery="",
        problematic_condition="",
        error_log="",
        problematic_subquery_execution_result="",
    ):
        if clause_type == "nested_predicate":
            if nested_predicate_type == "type-n":
                system_prompt = self.nested_predicate_prompt_temp.system_generate_nested_predicate_type_n_prompt.format(
                    height=self.args.hyperparams["height"] + 1
                )
            elif nested_predicate_type == "type-j":
                system_prompt = self.nested_predicate_prompt_temp.system_generate_nested_predicate_type_j_prompt.format(
                    height=self.args.hyperparams["height"] + 1
                )
            elif nested_predicate_type == "type-a":
                system_prompt = self.nested_predicate_prompt_temp.system_generate_nested_predicate_type_a_prompt.format(
                    height=self.args.hyperparams["height"] + 1
                )
            elif nested_predicate_type == "type-ja":
                system_prompt = self.nested_predicate_prompt_temp.system_generate_nested_predicate_type_ja_prompt.format(
                    height=self.args.hyperparams["height"] + 1
                )
            user_prompt = self.nested_predicate_prompt_temp.user_generate_nested_predicate_prompt.format(
                schema=schema,
                generated_from_clause=generated_from_clause,
                generated_where_clause=generated_where_clause,
                selected_inner_query_block=selected_inner_query_block,
            )

        elif clause_type == "inner_query_block":
            system_prompt = (
                self.nested_predicate_prompt_temp.system_select_inner_query_block_prompt
            )
            user_prompt = self.nested_predicate_prompt_temp.user_select_inner_query_block_prompt.format(
                schema=schema,
                generated_from_clause=generated_from_clause,
                generated_where_clause=generated_where_clause,
                candidate_inner_query_blocks=candidate_inner_query_blocks,
            )

        elif clause_type == "from":
            system_prompt = self.nested_predicate_prompt_temp.system_from_prompt
            user_prompt = self.nested_predicate_prompt_temp.user_from_prompt.format(
                schema=schema, subquery=selected_inner_query_block
            )
        elif clause_type == "execution_error":
            system_prompt = (
                self.correction_prompt_temp.system_execution_error_correction_prompt
            )
            user_prompt = self.correction_prompt_temp.user_execution_error_correction_prompt.format(
                query=query, error_log=error_log, schema=schema
            )
        elif clause_type == "empty_result_error":
            system_prompt = (
                self.correction_prompt_temp.system_empty_result_correction_prompt
            )
            user_prompt = self.correction_prompt_temp.user_empty_result_nested_predicate_correction_prompt.format(
                query=query,
                provenance_analysis_results=provenance_analysis_results,
                problematic_subquery=problematic_subquery,
                problematic_condition=problematic_condition,
                problematic_subquery_execution_result=problematic_subquery_execution_result,
            )
        else:
            system_prompt, user_prompt = super().get_prompt(
                clause_type, schema, history
            )

        return system_prompt, user_prompt

    def verify_nested_predicate_with_database(
        self,
        outer_tables,
        generated_nested_predicate,
        logical_operator,
        generated_where_clause,
    ):
        error_type = None
        # Generate the where clause
        if generated_where_clause == "":
            where_clause = f"WHERE {generated_nested_predicate}"
        else:
            where_clause = f"{generated_where_clause} {logical_operator} {generated_nested_predicate}"

        # Generate the from clause
        outer_columns, _ = get_columns_of_nested_predicate(generated_nested_predicate)
        if len(outer_columns) == 0:
            error_type = "execution_error"
            return error_type, "", ""

        outer_column = outer_columns[0]
        if "." in outer_column:
            outer_tables = list(
                set([outer_column.split(".")[0]]).union(set(outer_tables))
            )

        outer_tables = [table for table in outer_tables if table in self.all_table_set]

        if len(outer_tables) > 1:
            from_clause = "FROM " + (
                generate_from_clause_for_full_outer_view(
                    self.args,
                    relations=outer_tables,
                    join_clause_list=self.join_clause_list,
                    is_inner_join=True,
                )
            )
        else:
            from_clause = "FROM " + outer_tables[0]

        # Generate the select clause
        select_clause = "SELECT " + generate_select_clause_for_full_outer_view(
            self.table_info, relations=outer_tables, use_alias=True
        )

        # Generate the sql with nested predicate
        sql_with_nested_predicate = f"{select_clause} {from_clause} {where_clause}"

        # Verify the sql with the database
        try:
            execution_result_rows, _ = self.data_manager.sample_rows_with_sql(
                sql_with_nested_predicate,
                self.args.hyperparams["sample_size"],
                self.rng,
            )
            if len(execution_result_rows) == 0:
                error_type = "empty_result_error"
                return error_type, sql_with_nested_predicate, ""
        except Exception as e:
            print(e)
            error_type = "execution_error"
            return error_type, sql_with_nested_predicate, e

        return error_type, sql_with_nested_predicate, ""

    def correct_query(
        self,
        error_type,
        query,
        error_log,
        nested_predicate_type,
        accessed_tables=None,
        outer_table=None,
    ):
        if error_type in ["execution_error", "nested_predicate_type_mismatch"]:
            ast = parse_one(query)
            ast.set("expressions", [exp.Star()])
            query = str(ast)
            if (
                error_type == "nested_predicate_type_mismatch"
                and "type-j" in nested_predicate_type
            ):
                query = query.replace(
                    f"SELECT * FROM {outer_table}",
                    f"SELECT * FROM {outer_table} AS T1 ",
                )
            schema = self.get_schema(accessed_tables)
            system_prompt, user_prompt = self.get_prompt(
                "execution_error", query=query, error_log=error_log, schema=schema
            )
        elif error_type == "empty_result_error":
            try:
                (
                    provenance_analysis_results,
                    problematic_condition,
                    problematic_subquery,
                    problematic_subquery_execution_result,
                ) = self.prov_query_fixer.fix_query(query, nested_predicate_type)
            except Exception as e:
                return None

            if provenance_analysis_results is None:
                return None

            ast = parse_one(query)
            ast.set("expressions", [exp.Star()])
            query = str(ast)
            system_prompt, user_prompt = self.get_prompt(
                error_type,
                query=query,
                provenance_analysis_results=provenance_analysis_results,
                problematic_subquery=problematic_subquery,
                problematic_condition=problematic_condition,
                problematic_subquery_execution_result=problematic_subquery_execution_result,
            )

        is_error, response = self.call_llm(
            system_prompt, user_prompt, clause_type="corrected_query", max_tokens=256
        )

        if is_error:
            return None
        corrected_query = response["corrected_query"][0]
        print(f"error query: {query}")
        print(f"corrected query: {response['corrected_query'][0]}")
        diff = self.prov_query_fixer.diff_sql_parts(corrected_query, query)
        nested_predicate = diff["WHERE"][0]

        if nested_predicate == "":
            return None

        if error_type in ["execution_error", "nested_predicate_type_mismatch"]:
            table_to_alias = from_clause_to_dict(
                parse_one(corrected_query).args.get("from")
            )
            alias_to_table = {v: k for k, v in table_to_alias.items()}
            for alias, table in alias_to_table.items():
                nested_predicate = nested_predicate.replace(alias, table).replace(
                    f"AS {table}", ""
                )

        return nested_predicate

    def get_error_log_for_nested_predicate(
        self, generated_nested_predicate, stype_set, gt_type_set
    ):
        try:
            incorrect_type = list(stype_set - gt_type_set.intersection(stype_set))[0]
        except:
            incorrect_type = list(stype_set)[0]

        correct_type = list(gt_type_set - stype_set.intersection(gt_type_set))[0]
        error_log = f"""This is the type mismatch error for the nested predicate "{generated_nested_predicate}". The nested predicate is of {incorrect_type}, but it should be of {correct_type}.\n"""

        if correct_type == "type-ja":
            error_log += (
                f"{self.nested_predicate_type_prompt.type_ja_predicate_requirement}\n"
            )
        elif correct_type == "type-j":
            error_log += (
                f"{self.nested_predicate_type_prompt.type_j_predicate_requirement}\n"
            )
        elif correct_type == "type-a":
            error_log += (
                f"{self.nested_predicate_type_prompt.type_a_predicate_requirement}\n"
            )
        elif correct_type == "type-n":
            error_log += (
                f"{self.nested_predicate_type_prompt.type_n_predicate_requirement}\n"
            )

        if correct_type == "type-ja":
            error_log += f"{self.nested_predicate_type_prompt.type_ja_predicate_format}\n\nMANDATORY WHERE-Clause Join Predicate Placement \n–When correcting a type-ja nested predicate, ensure that the inner query Q contains a join predicate referencing the outer query’s relation inside Q’s WHERE clause. \n\n-It is strictly forbidden to place any join predicate, explicit JOIN, or additional table specification in Q’s FROM clause. \n\n-If you must reference the outer table, do so only through a correlation predicate in Q’s WHERE clause (e.g., WHERE outer_table.column = inner_table.column).\nABSOLUTE RULE: If any join predicate is placed in a FROM or JOIN clause, the answer is invalid. The join predicate must live in the WHERE clause of the inner query.\n"
        elif correct_type == "type-j":
            error_log += f"{self.nested_predicate_type_prompt.type_j_predicate_format}\n\nMANDATORY WHERE-Clause Join Predicate Placement \n–When correcting a type-ja nested predicate, ensure that the inner query Q contains a join predicate referencing the outer query’s relation inside Q’s WHERE clause. \n\n-It is strictly forbidden to place any join predicate, explicit JOIN, or additional table specification in Q’s FROM clause. \n\n-If you must reference the outer table, do so only through a correlation predicate in Q’s WHERE clause (e.g., WHERE outer_table.column = inner_table.column).\nABSOLUTE RULE: If any join predicate is placed in a FROM or JOIN clause, the answer is invalid. The join predicate must live in the WHERE clause of the inner query.\n"
        elif correct_type == "type-a":
            error_log += (
                f"{self.nested_predicate_type_prompt.type_a_predicate_format}\n\n"
            )
        elif correct_type == "type-n":
            error_log += (
                f"{self.nested_predicate_type_prompt.type_n_predicate_format}\n\n"
            )

        return error_log
