import json
import requests
import tiktoken
from utils import *
from prompts.non_nested import oneshot, cbc, eg
from sqlalchemy import create_engine
from M_Schema.schema_engine import SchemaEngine

class OneShotQueryGenerator:
    def __init__(self, args, data_manager, rng, all_table_set):
        self.args = args
        self.data_manager = data_manager
        self.rng = rng
        self.prompt_temp = oneshot
        self.all_table_set = all_table_set
        self.sampling_params = {"temperature": self.args.llm_hyperparams["temperature"], "top_p": self.args.llm_hyperparams["top_p"], "max_new_tokens": self.args.llm_hyperparams["max_tokens"]}
        self.mschema = SchemaEngine(engine=create_engine(f"postgresql+psycopg2://postgres:postgres@localhost:{self.args.port}/{self.args.db}"), db_name=self.args.db, rng=self.rng).mschema
        self.generated_sql_query_set = set()
        self.error_dict = {"duplicate": 0, "invalid": 0, "empty": 0, "parsing": 0}
        self.total_calls = 0
        self.max_calls = 10000
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.total_tokens = 0
        self.ideal_calls = 0

    def generate(self,):
        self.ideal_calls += 1
        initial_relation = self.choose_initial_relation()
        schema = self.get_schema([initial_relation])
        while True:
            sql_query_list = self.generate_sql_query(schema)["sql"]
            if len(sql_query_list) > 0:
                break
        return [sql_query + "\n" for sql_query in sql_query_list]
    
    def choose_initial_relation(self,):
        avaliable_tables = sorted(list(self.all_table_set))
        table = self.rng.choice(avaliable_tables)
        return table
    
    def generate_sql_query(self, schema):
        """Runs inference using an LLM model with retries and ensures a valid SQL response."""
        system_prompt, user_prompt = self.get_prompt(schema)
        is_error, response = self.call_llm(system_prompt, user_prompt)
        if not is_error:
            return response
        
        if self.total_calls >= self.max_calls:
            return {"sql": ["expired"]}
    
        return {"sql": []}

    def call_llm(self, system_prompt, user_prompt, clause_type=None, max_tokens=None):
        if max_tokens is not None:
            self.sampling_params["max_new_tokens"] = max_tokens
        response_data = requests.post(
            self.args.llm_addr,
            json={"text": [f"{system_prompt}\n\n{user_prompt}"], "sampling_params": self.sampling_params}
        ).json()
        completion = response_data[0]["text"]
        token_count = len(self.tokenizer.encode(f"{system_prompt}\n\n{user_prompt}"))
        self.total_tokens += token_count
        
        try:
            response = json.loads(completion)
        except json.JSONDecodeError:
            response = extract_json_objects(completion)
        
        self.total_calls += 1
        is_error, response = self.detect_error(response, clause_type)
        print(f"ideal_calls: {self.ideal_calls} total_calls: {self.total_calls} error_dict: {self.error_dict}")
        return is_error, response

    def get_prompt(self, schema):
        if self.args.hyperparams["is_agg"]:
            system_prompt = self.prompt_temp.system_agg_prompt
        else:
            system_prompt = self.prompt_temp.system_prompt
        user_prompt = self.prompt_temp.user_prompt.format(schema=schema)
        return system_prompt, user_prompt
    
    def get_schema(self, relations):
        schema = self.mschema.to_mschema(selected_tables=relations, example_num=5)
        return schema

    def detect_error(self, response, clause_type=None):
        response = response[0] if isinstance(response, list) and response else response

        if not isinstance(response, dict) or "sql" not in response:
            self.error_dict["parsing"] += 1
            return True, response

        sql_queries = response["sql"]
        if not isinstance(sql_queries, list):
            sql_queries = [sql_queries]

        valid_sql_queries = []
        for sql in sql_queries:
            if sql in self.generated_sql_query_set or is_new_query_equivalent(list(self.generated_sql_query_set), sql):
                self.error_dict["duplicate"] += 1
                continue
            try:
                self.data_manager.execute(sql)
                execution_results = self.data_manager.fetchall()
                result_len = len(execution_results)
                if result_len == 0:
                    self.error_dict["empty"] += 1
                    continue
                first_result = execution_results[0][0]
                if result_len == 1 and isinstance(first_result, int) and first_result == 0:
                    self.error_dict["empty"] += 1
                    continue
                self.generated_sql_query_set.add(sql)
                valid_sql_queries.append(sql)
            except:
                self.error_dict["invalid"] += 1

        response["sql"] = valid_sql_queries
        return (not valid_sql_queries), response

class CBCQueryGenerator(OneShotQueryGenerator):
    def __init__(self, args, data_manager, rng, all_table_set, table_info, dtype_dict, logger):
        super().__init__(args, data_manager, rng, all_table_set)
        self.prompt_temp = cbc
        self.dtype_dict = dtype_dict
        self.table_info = table_info
        self.logger = logger
        
    def generate(self,):
        self.sql_type_dict = select_clauses(self.args, self.rng)
        sql_clause_list = ["from"] + [clause_type for clause_type in ["where", "group", "having", "order", "limit"] if self.sql_type_dict[clause_type]] + ["select"]
        if self.sql_type_dict["group"]:
            self.ideal_calls += (len(sql_clause_list) - 2)
        else:
            self.ideal_calls += (len(sql_clause_list) - 1)

        while True:
            schema = ""
            history = []
            generated_clause_info_dict = {}
            for clause_type in sql_clause_list:
                revised_clause, relations = self.generate_clause(schema, history, clause_type, generated_clause_info_dict)
                if relations: schema = self.get_schema(relations)
                history.append(f"{revised_clause}")
            
            sql_query_list = self.get_sql_query(generated_clause_info_dict)
            if sql_query_list:
                break
        return sql_query_list

    def generate_clause(self, schema, history, clause_type, generated_clause_info_dict=None):
        if clause_type == "from":
            initial_relation = self.choose_initial_relation()
            from_clause = f"FROM {initial_relation}"
            relations = [initial_relation]
            self.update_generated_clause_info_dict(generated_clause_info_dict, clause_type, from_clause, relations)
            return from_clause, relations
        elif clause_type == "select" and self.sql_type_dict["group"]:
            group_by_column_list = generated_clause_info_dict["columns_in_group_by_clause"]
            select_columns = group_by_column_list.copy()
            response = {"select": [f"SELECT {', '.join(select_columns)}"]}
            revised_clause = self.revise_clause(response, "select")
            self.update_generated_clause_info_dict(generated_clause_info_dict, clause_type, revised_clause)
            return revised_clause, None
    
        system_prompt, user_prompt = self.get_prompt(clause_type, schema, history)
        is_error, response = self.call_llm(system_prompt, user_prompt, clause_type=clause_type)
        if not is_error:
            revised_response = self.revise_clause(response, clause_type)
            self.update_generated_clause_info_dict(generated_clause_info_dict, clause_type, revised_response)
            return revised_response, None

        if self.total_calls >= self.max_calls:
            return {"sql": ["expired"]}, None

        return {"sql": []}, None
        
    def choose_initial_relation(self,):
        df_columns = sorted([table + "." + column for table in self.all_table_set for column in self.table_info[table]])
        avaliable_tables = sorted(list(self.all_table_set))
        if self.sql_type_dict["group"]:
            candidate_root_tables = set()
            for column in df_columns:
                if is_categorical_column(self.args, column):
                    candidate_root_tables.add(column.split(".")[0])
            candidate_root_tables = sorted(list(candidate_root_tables))
            if len(candidate_root_tables) == 0:
                self.logger.warning("No table has a categorical column but we will generate GROUP BY clause")
                candidate_root_tables = avaliable_tables
            table = self.rng.choice(candidate_root_tables)
        else:
            table = self.rng.choice(avaliable_tables)
        return table

    def update_generated_clause_info_dict(self, generated_clause_info_dict, clause_type, revised_clause, relations=None):
        generated_clause_info_dict[clause_type] = revised_clause
        if relations:
            generated_clause_info_dict["relations"] = relations
        target_list = []
        if clause_type in {"where", "group", "order"}:
            for relation in generated_clause_info_dict["relations"]:
                columns = self.table_info[relation]
                for column in columns:
                    if len(generated_clause_info_dict["relations"]) > 1:
                        if column.lower() in revised_clause.lower():
                            target_list.append(f"{relation}.{column}")
                    else:
                        if f".{column.lower()}" in revised_clause.lower().replace("__", ".") or ('.' not in revised_clause.lower().replace("__", ".") and column.lower() in revised_clause.lower().replace("__", ".")):
                            target_list.append(column)
                
            if clause_type == "where":
                generated_clause_info_dict["columns_in_where_clause"] = target_list
            elif clause_type == "group":
                generated_clause_info_dict["columns_in_group_by_clause"] = target_list
            elif clause_type == "order":
                generated_clause_info_dict["columns_in_order_by_clause"] = target_list

    def revise_clause(self, generated_clause, clause_type):
        keywords = {
            "join": "JOIN",
            "where": "WHERE",
            "group": "GROUP BY",
            "having": "HAVING",
            "order": "ORDER BY",
            "limit": "LIMIT",
            "select": "SELECT",
            "from": "FROM"
        }
        clause_value = generated_clause.get(clause_type, "")[0].replace("number_of_three_point_field_goals_attemp ", "number_of_three_point_field_goals_attempted ").replace("percentage_of_three_point_field_goal_mad ", "percentage_of_three_point_field_goal_made ").replace("team_percentage_of_three_point_field_goal_ ", "team_percentage_of_three_point_field_goal_made ")
        if clause_type == "select":
            clause_value = clause_value.split('FROM')[0].split(keywords[clause_type])[-1].replace("number_of_three_point_field_goals_attemp ", "number_of_three_point_field_goals_attempted ").replace("percentage_of_three_point_field_goal_mad ", "percentage_of_three_point_field_goal_made ").replace("team_percentage_of_three_point_field_goal_ ", "team_percentage_of_three_point_field_goal_made ")
            clause_value = clause_value.split(',')[0]
        elif clause_value.strip().lower().startswith(keywords[clause_type].lower()):
                clause_value = clause_value[len(keywords[clause_type]):]
        return f"{keywords[clause_type]} {clause_value.strip()}"

    def get_prompt(self, clause_type, schema, history):
        if clause_type == "select":
            if self.args.hyperparams["is_agg"]:
                system_prompt = self.prompt_temp.system_select_agg_prompt
            else:
                system_prompt = self.prompt_temp.system_select_prompt
        elif clause_type == "where":
            system_prompt = self.prompt_temp.system_where_prompt
        elif clause_type == "group":
            system_prompt = self.prompt_temp.system_group_prompt
        elif clause_type == "having":
            system_prompt = self.prompt_temp.system_having_prompt
        elif clause_type == "order":
            system_prompt = self.prompt_temp.system_order_prompt
        elif clause_type == "limit":
            system_prompt = self.prompt_temp.system_limit_prompt
        else:
            raise ValueError(f"Invalid clause type: {clause_type}")
        
        user_prompt = self.prompt_temp.user_prompt.format(schema=schema, history=history)
        
        return system_prompt, user_prompt

    def detect_error(self, response, clause_type=None):
        if clause_type is not None:
            response = response[0] if isinstance(response, list) and response else response
            if not isinstance(response, dict) or clause_type not in response:
                self.error_dict["parsing"] += 1
                return True, response

            if response[clause_type] == "":
                self.error_dict["parsing"] += 1
                return True, response

            clauses = response[clause_type]
            if not isinstance(clauses, list):
                clauses = [clauses]

            response[clause_type] = clauses
        else:
            if response['sql'][0] in self.generated_sql_query_set or is_new_query_equivalent(list(self.generated_sql_query_set), response['sql'][0]):
                self.error_dict["duplicate"] += 1
                return True, response
            try:
                self.data_manager.execute(response['sql'][0].replace("number_of_three_point_field_goals_attemp ", "number_of_three_point_field_goals_attempted ").replace("percentage_of_three_point_field_goal_mad ", "percentage_of_three_point_field_goal_made ").replace("team_percentage_of_three_point_field_goal_ ", "team_percentage_of_three_point_field_goal_made "))
                execution_results = self.data_manager.fetchall()
                result_len = len(execution_results)
                if result_len == 0:
                    self.error_dict["empty"] += 1
                    return True, response
                first_result = execution_results[0][0]
                if result_len == 1 and isinstance(first_result, int) and first_result == 0:
                    self.error_dict["empty"] += 1
                    return True, response
                self.generated_sql_query_set.add(response['sql'][0])
            except:
                self.error_dict["invalid"] += 1
                return True, response

        return False, response

    def get_sql_query(self, generated_clause_info_dict):
        sql_query = ""
        for clause in ["select", "from", "where", "group", "having", "order", "limit"]:
            if clause in generated_clause_info_dict:
                if generated_clause_info_dict[clause] != "":
                    sql_query += f"{generated_clause_info_dict[clause].replace('__', '.')} "
        try:
            sql_query = transform_sql(sql_query.replace('__', '.'), self.dtype_dict)
        except Exception as e:
            self.logger.error(f"Error transforming SQL query: {e}")
            return None
        is_error, response = self.detect_error({"sql": [sql_query]})
        if not is_error:
            return [sql_query + "\n" for sql_query in response["sql"]]
        
        return None

class EGQueryGenerator(CBCQueryGenerator):
    def __init__(self, args, data_manager, rng, all_table_set, table_info, dtype_dict, join_key_list, logger):
        super().__init__(args, data_manager, rng, all_table_set, table_info, dtype_dict, logger)
        self.prompt_temp = eg
        self.join_key_list = join_key_list
    
    def generate(self,):
        self.sql_type_dict = select_clauses(self.args, self.rng)
        sql_clause_list = ["from"] + [clause_type for clause_type in ["where", "group", "having", "order", "limit"] if self.sql_type_dict[clause_type]] + ["select"]
        if self.sql_type_dict["group"]:
            self.ideal_calls += (len(sql_clause_list) - 2)
        else:
            self.ideal_calls += (len(sql_clause_list) - 1)
        
        while True:
            history = []
            generated_clause_info_dict = {}
            database_representation = ""
            for clause_type in sql_clause_list:
                clause_gen_count = 0
                while clause_gen_count < self.args.hyperparams["max_clause_gen_count"]:
                    clause_gen_count += 1
                    
                    # Generate the clause
                    revised_clause, _ = self.generate_clause(database_representation, history, clause_type, generated_clause_info_dict)

                    # Verify the clause with the database
                    error_type, instances = self.verify_with_database(clause_type, generated_clause_info_dict)
                    if error_type == "execution_error":
                        self.logger.error(f"Error verifying clause with the database {revised_clause}")
                        continue
                    elif error_type == "empty_result_error":
                        self.logger.error(f"Empty result error for clause {revised_clause}")
                        continue
                    
                    if clause_type not in ["select"]:
                        # Update the history
                        self.update_history(history, revised_clause)
                        filtered_rows, filtered_column_names = self.remove_all_none_columns(instances[0], instances[1])
                        if len(filtered_column_names) > 2:
                            database_representation = self.get_database_representation(filtered_rows, filtered_column_names)
                        else:
                            database_representation = self.get_database_representation(instances[0], instances[1])
                    break
                else:
                    # Delete the clause type which failed to generate
                    if clause_type in generated_clause_info_dict:
                        del generated_clause_info_dict[clause_type]
                    else:
                        self.logger.error(f"Clause type {clause_type} not found in generated_clause_info_dict")
                        return None

            sql_query_list = self.get_sql_query(generated_clause_info_dict)
            if sql_query_list:
                break
            
        return sql_query_list
    
    def get_instances(self, query_info, clause_type):
        if clause_type == "from":
            if len(query_info["relations"]) > 1:
                from_clause = query_info[clause_type]
                select_clause = generate_select_clause_for_full_outer_view(self.args, relations=query_info["relations"], use_alias=True)
                full_inner_join_sql = f"""
                    SELECT DISTINCT {select_clause}
                    {from_clause}
                """ 
            else:
                from_clause = query_info['relations'][0]
                full_inner_join_sql = f"""
                SELECT *
                FROM {from_clause}
                """
            if "nested_predicate" in query_info:
                full_inner_join_sql += f" WHERE {query_info['nested_predicate']}"
            return self.create_and_sample_view("accessed_tables", full_inner_join_sql, is_materialized=True)
    
        elif clause_type == "where":
            cond = query_info["where"][query_info["where"].find("WHERE") + 5:].strip()
            try:
                cond = transform_sql(cond, self.dtype_dict)
            except Exception as e:
                self.logger.error(f"Error transforming SQL query: {e}")
                return None
            if len(query_info['relations']) > 1:
                cond = replace_dot(cond)
            else:
                cond = cond.replace(f"{query_info['relations'][0]}.", "")
            query_info["where"] = f"WHERE {cond}"
            return self.data_manager.sample_rows_with_cond("accessed_tables", cond, 5, rng=self.rng)
        
        elif clause_type in {"group", "having"}:
            column_names = self.data_manager.fetch_column_names("accessed_tables")
            group_by_column_list = query_info.get("columns_in_group_by_clause", [])
            if len(query_info['relations']) > 1:
                filtered_group_by_column_list = []
                for col in group_by_column_list:
                    filtered_group_by_column_list.append(f"{col.split('.')[0]}__{col.split('.')[1]}")
                group_by_column_list = filtered_group_by_column_list
                column_names = [col.replace("number_of_three_point_field_goals_attemp ", "number_of_three_point_field_goals_attempted ").replace("percentage_of_three_point_field_goal_mad ", "percentage_of_three_point_field_goal_made ").replace("team_percentage_of_three_point_field_goal_ ", "team_percentage_of_three_point_field_goal_made ") for col in column_names]
                having_candidate_columns = [col for col in column_names if col not in group_by_column_list]
            else:
                having_candidate_columns = [f"{query_info['relations'][0]}.{col}" for col in column_names if col not in group_by_column_list]
            
            having_candidate_columns_with_agg = []
            for column_name in having_candidate_columns:
                if column_name.replace("__", ".").strip() in self.dtype_dict:
                    dtype = self.dtype_dict[column_name.replace('__', '.').strip()]
                else:
                    dtype = self.dtype_dict[column_name.replace('__', '.').replace('number_of_three_point_field_goals_attemp', 'number_of_three_point_field_goals_attempted').replace('percentage_of_three_point_field_goal_mad', 'percentage_of_three_point_field_goal_made').replace('team_percentage_of_three_point_field_goal_', 'team_percentage_of_three_point_field_goal_made').strip()]
                candidate_aggs = (
                    TEXTUAL_AGGS if dtype in ["bool", "str"] else
                    DATE_AGGS if dtype == "date" else
                    KEY_AGGS if is_id_column(self.args, column_name.replace("__", "."), self.join_key_list) or is_hashcode_column(self.args, column_name.replace("__", "."))
                    else NUMERIC_AGGS
                )
                having_candidate_columns_with_agg.extend((agg, column_name.split('.')[-1]) for agg in candidate_aggs)
            
            select_clause = ", ".join(
                group_by_column_list +
                [f"{agg}({col.replace('.', '__')}) as {agg.lower()}__{col.replace('.', '__')}".replace('count__', 'num__') for agg, col in having_candidate_columns_with_agg]
            )
            if len(query_info['relations']) > 1:
                query_info['group'] = replace_dot(query_info['group'])
            else:
                query_info['group'] = query_info['group'].replace(f"{query_info['relations'][0]}.", "")
            sql = f"SELECT DISTINCT {select_clause} FROM accessed_tables"
            sql += f" {query_info.get('where', '')} {query_info.get('group', '')}"
            if clause_type == "having":
                if len(query_info['relations']) > 1:
                    query_info['having'] = replace_dot(query_info['having'])
                else:
                    query_info['having'] = query_info['having'].replace(f"{query_info['relations'][0]}.", "")
                sql += f" {query_info.get('having', '')}"
            
            sample_rows, sample_schema = self.create_and_sample_view(f"accessed_tables_with_{clause_type}", sql)
            sample_schema = group_by_column_list + [f"{col.split('__')[0].upper()}({'__'.join(col.split('__')[1:])})".replace('NUM(', 'COUNT(') for col in sample_schema if col not in group_by_column_list]
            return [sample_rows, sample_schema]
        
        elif clause_type in {"order", "limit"}:
            if 'columns_in_group_by_clause' in query_info:
                group_by_column_list = query_info["columns_in_group_by_clause"]
                select_columns = group_by_column_list.copy()
                select_clause = ", ".join(select_columns).replace('.', '__')
            else:
                select_clause = "*"
            
            sql = f"SELECT {select_clause} FROM accessed_tables"
            if clause_type == "order":
                if len(query_info['relations']) > 1:
                    query_info['order'] = replace_dot(query_info['order'])
                else:
                    query_info['order'] = query_info['order'].replace(f"{query_info['relations'][0]}.", "")
            elif clause_type == "limit":
                if len(query_info['relations']) > 1:
                    query_info['limit'] = replace_dot(query_info['limit'])
                else:
                    query_info['limit'] = query_info['limit'].replace(f"{query_info['relations'][0]}.", "")
            sql += f" {query_info.get('where', '')} {query_info.get('group', '')} {query_info.get('having', '')} {query_info.get('order', '')} {query_info.get('limit', '')}"
            return self.create_and_sample_view(f"accessed_tables_with_{clause_type}", sql)

    def get_database_representation(self, sample_rows, sample_schema):
        table_representation = ""
        columns_text = " | ".join([col for col in sample_schema])
        table_representation += f"col : {columns_text}\n"
        for row_id, sample_row in enumerate(sample_rows):
            sample_row_text = " | ".join([f"'{sample_row[i]}'" if ' ' in str(sample_row[i]) else f"{sample_row[i]}" for i in range(len(sample_row))])
            table_representation += f"row {row_id + 1} : {sample_row_text}\n"
        return table_representation

    def create_and_sample_view(self, view_name, sql, is_materialized=False):
        if is_materialized:
            self.data_manager.create_view(
                self.logger, view_name, sql, type="materialized", drop_if_exists=True
            )
        else:
            self.data_manager.create_view(
                self.logger, view_name, sql, type="non_materialized", drop_if_exists=True
            )
        self.data_manager.execute("SET plan_cache_mode = force_generic_plan;")
        return self.data_manager.sample_rows(view_name, 5, rng=self.rng)

    def verify_with_database(self, clause_type, generated_clause_info_dict):
        error_type = None
        instances = None
        if clause_type not in ["nested_predicate", "select"]:
            try:
                instances = self.get_instances(generated_clause_info_dict, clause_type)
                if instances is None:
                    error_type = "execution_error"
                elif len(instances[0]) == 0:
                    error_type = "empty_result_error" 
            
            except Exception as e:
                print(e)
                error_type = "execution_error"           

            return error_type, instances
        else:
            return error_type, None

    def update_history(self, history, revised_clause):
        history.append(revised_clause)
        
    def get_sql_query(self, generated_clause_info_dict):
        sql_query = ""
        for clause in ["select", "from", "where", "group", "having", "order", "limit"]:
            if clause in generated_clause_info_dict:
                if generated_clause_info_dict[clause] != "":
                    sql_query += f"{generated_clause_info_dict[clause].replace('__', '.')} "
        try:
            sql_query = transform_sql(sql_query.replace('__', '.'), self.dtype_dict)
        except Exception as e:
            self.logger.error(f"Error transforming SQL query: {e}")
            return None

        is_error = False
        if sql_query in self.generated_sql_query_set or is_new_query_equivalent(list(self.generated_sql_query_set), sql_query):
            self.error_dict["duplicate"] += 1
            is_error = True
        else:
            self.generated_sql_query_set.add(sql_query)
        if not is_error:
            return [sql_query + "\n"]
        
        return None

    def remove_all_none_columns(self, rows, column_names):
        transposed_columns = list(zip(*rows)) if rows else [[] for _ in column_names]
        
        valid_indices = [i for i, col in enumerate(transposed_columns) if not all(val is None for val in col)]
        
        filtered_column_names = []
        for i, valid_index in enumerate(valid_indices):
            valid_column_name = column_names[valid_index]
            if 'avg__' in valid_column_name.lower() or 'sum__' in valid_column_name.lower() or 'count__' in valid_column_name.lower() or 'min__' in valid_column_name.lower() or 'max__' in valid_column_name.lower():
                filtered_column_names.append(f"{valid_column_name.split('__')[0].upper()}({valid_column_name.split('__')[-1]})")
            else:
                filtered_column_names.append(valid_column_name)
        
        filtered_column_names = [column_names[i] for i in valid_indices]
        
        # Filter rows
        filtered_rows = [tuple(row[i] for i in valid_indices) for row in rows]
        
        return filtered_rows, filtered_column_names
    
