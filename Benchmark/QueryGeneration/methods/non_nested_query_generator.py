import json
import requests
import tiktoken
from utils import *
from prompts import oneshot
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