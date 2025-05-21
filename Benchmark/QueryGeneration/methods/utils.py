import random
import psycopg2
import traceback
from json import JSONDecoder
from typing import Any, List
from sqlglot import parse_one, exp
from sqlglot.optimizer.normalize import normalize

TEXTUAL_AGGS = ["COUNT"]
DATE_AGGS = ["COUNT", "MIN", "MAX"]
KEY_AGGS = ["COUNT"]
NUMERIC_AGGS = ["COUNT", "MIN", "MAX", "AVG", "SUM"]

def is_id_column(args, col, join_key_list):
    if col == "*":
        return False
    return col in join_key_list or "id" in col.split(".")[-1].lower() or col in args.IDS

def is_hashcode_column(args, col):
    return col in args.HASH_CODES

def replace_dot(input_str: str) -> str:
    result = []
    length = len(input_str)

    for i, ch in enumerate(input_str):
        if ch == '.':
            if i > 0 and i < length - 1 and input_str[i-1].isdigit() and input_str[i+1].isdigit():
                result.append('.')
            else:
                result.append('__')
        else:
            result.append(ch)
            
    return ''.join(result)

def generate_select_clause_for_full_outer_view(args, relations=None, use_alias=True):
    select_columns = []
    # <table name>___<column name>
    for table in args.table_info.keys():
        for column in args.table_info[table]:
            if relations is not None and table not in relations:
                continue
            if use_alias:
                select_columns.append(f"{table}.{column} AS {table}__{column}")
            else:
                select_columns.append(f"{column}")

    select_clause = ", ".join(select_columns)
    return select_clause

def transform_sql(query: str, column_types: dict) -> str:
    """
    Transform the SQL query to match column data types.
    """
    tree = parse_one(query.replace('-', "123456789").replace("lb", "135792468"))
    
    def _replace_literals(expression):
        if isinstance(expression, exp.Condition):  # Conditions like WHERE clauses
            if isinstance(expression, (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE)):
                left, right = expression.args.get("this"), expression.args.get("expression")
                if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                    col_name = f"{left.table}.{left.name}" if left.table else left.name
                    if col_name.replace('__', '.') in column_types:
                        dtype = column_types[col_name.replace('__', '.')]
                        new_value = cast_value_to_dtype(right.this, dtype)
                        expression.set("expression", new_value)
        
        for e in expression.args.values():
            if isinstance(e, list):
                for sub in e:
                    _replace_literals(sub)
            elif isinstance(e, exp.Expression):
                _replace_literals(e)
    
    _replace_literals(tree)
    return tree.sql().replace('123456789', '-').replace("135792468", "lb")


def cast_value_to_dtype(value, dtype):
    """
    Convert the given value to match the corresponding SQL data type.
    """
    if dtype.lower() in ("int", "integer", "bigint", "smallint"):
        try:
            return exp.Literal.number(str(int(value)))  # Ensure integer type
        except:
            return exp.Literal.number(str(float(value))) 
    elif dtype.lower() in ("float", "double", "real", "decimal"): 
        return exp.Literal.number(str(float(value)))  # Ensure floating point type
    elif dtype.lower() in ("bool", "boolean"):
        return exp.Literal.boolean("TRUE" if value.lower() in ("1", "true", "t", "yes", "y") else "FALSE")
    elif dtype.lower() in ("text", "string", "varchar", "char", "str"):
        return exp.Literal.string(value)  # Ensure string format with quotes
    return exp.Literal.string(value)  # Default case: return as string

def is_categorical_column(args, col):
    return col in args.CATEGORIES

def select_clauses(args, rng):
    prob_where = args.hyperparams["prob_where"]
    prob_group = args.hyperparams["prob_group"]
    prob_having = args.hyperparams["prob_having"]
    prob_order = args.hyperparams["prob_order"]
    prob_limit = args.hyperparams["prob_limit"]
    sql_type = {
        "where": bool(rng.choice([0, 1], p=[1 - prob_where, prob_where])),
        "group": bool(rng.choice([0, 1], p=[1 - prob_group, prob_group])),
        "having": bool(rng.choice([0, 1], p=[1 - prob_having, prob_having])),
        "order": bool(rng.choice([0, 1], p=[1 - prob_order, prob_order])),
        "limit": bool(rng.choice([0, 1], p=[1 - prob_limit, prob_limit])),
    }
    if not sql_type["group"]:
        sql_type["having"] = False
    if not sql_type["order"]:
        sql_type["limit"] = False

    return sql_type

def set_col_info(col_info):
    IDS = col_info["ids"]
    HASH_CODES = col_info["hash_codes"]
    NOTES = col_info["notes"]
    CATEGORIES = col_info["categories"]
    FOREIGN_KEYS = col_info["foreign_keys"]

    return IDS, HASH_CODES, NOTES, CATEGORIES, FOREIGN_KEYS

class DBConnector:
    # Static attributes
    connector_cache = {}

    def __init__(self, db_id: str):
        self.db_id = db_id
        self.conn = None
        self.cur = None

    def __new__(cls, db_id):
        if not db_id in DBConnector.connector_cache.keys():
            DBConnector.connector_cache[db_id] = super(DBConnector, cls).__new__(cls)
        return DBConnector.connector_cache[db_id]

    def execute(self, sql: str) -> List[Any]:
        return self.cur.execute(sql)

    def fetchall(self) -> List[Any]:
        return self.cur.fetchall()

    def fetchone(self) -> List[Any]:
        return self.cur.fetchone()
    
    def description(self) -> List[Any]:
        return self.cur.description

    def close(self) -> None:
        self.conn.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        self.conn.close()
        return True

class PostgreSQLDatabase(DBConnector):
    def __init__(self, user_id: str, passwd: str, host: str, port: str, db_id: str):
        super(PostgreSQLDatabase, self).__init__(db_id=db_id)
        self.connect(user_id, passwd, host, port, db_id)
        self.conn.autocommit = True

    def __new__(cls, user_id, passwd, host, port, db_id):
        return super(PostgreSQLDatabase, cls).__new__(cls, db_id)

    def connect(self, user_id, passwd, host, port, db_id):
        self.conn = psycopg2.connect(f"user={user_id} password={passwd} host={host} port={port} dbname={db_id}")#, prepare_threshold=0
        self.cur = self.conn.cursor()

    def fetch_table_names(self) -> List[str]:
        sql = """
            SELECT *
            FROM pg_catalog.pg_tables
            WHERE schemaname != 'pg_catalog' AND 
                  schemaname != 'information_schema';
        """
        self.cur.execute(sql)
        tables = [
            f"{str(table[0].lower())}.{str(table[1].lower())}".replace("public.", "") for table in self.fetchall()
        ]
        return tables

    def fetch_column_names(self, table_name: str) -> List[str]:
        sql = f"""
            SELECT *
            FROM
                {table_name}
            LIMIT 0;
            """
        self.execute(sql)
        return [desc[0] for desc in self.description()]

    def fetch_all_values(self, table_name: str, column_name: str):
        sql = f"""
            SELECT {column_name}
            FROM
                {table_name};
            """
        self.execute(sql)
        return [data[0] for data in self.fetchall()]

    def fetch_distinct_values(self, table_name: str, column_name: str):
        sql = f"""
            SELECT DISTINCT {column_name}
            FROM
                {table_name}
            WHERE {column_name} IS NOT NULL;
            """
        self.execute(sql)
        return [data[0] for data in self.fetchall()]

    def check_row_exists(self, table_name: str, additional_clauses: str = None):
        sql = f"""
            SELECT COUNT(*)
            FROM
                ( SELECT * FROM {table_name}
            """
        if additional_clauses is not None:
            sql += f""" {additional_clauses}"""
        sql += "LIMIT 2 ) as T;"
        self.execute(sql)
        return (self.fetchall())[0][0]

    def check_distinct_value_exists(self, table_name: str, column_name: str, additional_clauses: str = None):
        sql = f"""
            SELECT COUNT(DISTINCT {column_name})
            FROM ( SELECT * FROM {table_name}
            """
        if additional_clauses is not None:
            sql += f""" {additional_clauses}"""
        sql += "LIMIT 2 ) as T;"
        self.execute(sql)
        return (self.fetchall())[0][0]

    def get_row_counts(self, table_name: str, additional_clauses: str = None):
        sql = f"""
            SELECT COUNT(*)
            FROM
                {table_name}
            """
        if additional_clauses is not None:
            sql += f""" {additional_clauses}"""
        sql += ";"
        self.execute(sql)
        return (self.fetchall())[0][0]

    def get_distinct_value_counts(self, table_name: str, column_name: str, additional_clauses: str = None):
        sql = f"""
            SELECT COUNT(DISTINCT {column_name})
            FROM
                {table_name}
            """
        if additional_clauses is not None:
            sql += f""" {additional_clauses}"""
        sql += ";"
        self.execute(sql)
        return (self.fetchall())[0][0]

    def get_primary_keys(self, table_name: str):
        sql = f"""select conrelid::regclass, conname AS primary_key, pg_get_constraintdef(oid) 
        from pg_constraint where conrelid::regclass::text = '{table_name}' and pg_get_constraintdef(oid) LIKE 'PRIMARY%'
        """
        self.execute(sql)
        data = self.fetchall()
        assert len(data) == 1
        key_state = (data[0][2].replace("PRIMARY KEY (", "").strip())[:-1]
        keys = key_state.split(", ")
        return keys

    def sample_rows(self, table_name: str, sample_size: int, rng=None):
        self.execute("DISCARD PLANS;")
        self.execute(f"SELECT * FROM {table_name};")
        all_data = self.fetchall()
        min_sample_size = min(sample_size, len(all_data))
        if rng is None:
            sampled_data = random.sample(all_data, min_sample_size)
        else:
            sample_indices = rng.choice(len(all_data), size=min_sample_size, replace=False)
            sampled_data = [all_data[i] for i in sample_indices]
        cols = [(desc[0]+' ').replace("number_of_three_point_field_goals_attemp ", "number_of_three_point_field_goals_attempted ").replace("percentage_of_three_point_field_goal_mad ", "percentage_of_three_point_field_goal_made ").replace("team_percentage_of_three_point_field_goal_ ", "team_percentage_of_three_point_field_goal_made ").strip() for desc in self.description()]
        return sampled_data, cols

    def sample_rows_with_cond(self, table_name: str, cond: str, sample_size: int, rng=None):
        self.execute(f"SELECT * FROM {table_name} WHERE {cond};")
        all_data = self.fetchall()
        min_sample_size = min(sample_size, len(all_data))
        if rng is None:
            sampled_data = random.sample(all_data, min_sample_size)
        else:
            sample_indices = rng.choice(len(all_data), size=min_sample_size, replace=False)
            sampled_data = [all_data[i] for i in sample_indices]
        cols = [(desc[0]+' ').replace("number_of_three_point_field_goals_attemp ", "number_of_three_point_field_goals_attempted ").replace("percentage_of_three_point_field_goal_mad ", "percentage_of_three_point_field_goal_made ").replace("team_percentage_of_three_point_field_goal_ ", "team_percentage_of_three_point_field_goal_made ").strip() for desc in self.description()]
        return sampled_data, cols
    
    def sample_rows_with_sql(self, sql: str, sample_size: int, rng=None):
        self.execute(sql)
        all_data = self.fetchall()
        min_sample_size = min(sample_size, len(all_data))
        if rng is None:
            sampled_data = random.sample(all_data, min_sample_size)
        else:
            sample_indices = rng.choice(len(all_data), size=min_sample_size, replace=False)
            sampled_data = [all_data[i] for i in sample_indices]
        cols = [(desc[0]+' ').replace("number_of_three_point_field_goals_attemp ", "number_of_three_point_field_goals_attempted ").replace("percentage_of_three_point_field_goal_mad ", "percentage_of_three_point_field_goal_made ").replace("team_percentage_of_three_point_field_goal_ ", "team_percentage_of_three_point_field_goal_made ").strip() for desc in self.description()]
        return sampled_data, cols

    def sample_rows_with_sql_with_limit(self, sql: str, sample_size: int, rng=None):
        self.execute(sql + f" LIMIT {sample_size*5};")
        all_data = self.fetchall()
        min_sample_size = min(sample_size, len(all_data))
        if rng is None:
            sampled_data = random.sample(all_data, min_sample_size)
        else:
            sample_indices = rng.choice(len(all_data), size=min_sample_size, replace=False)
            sampled_data = [all_data[i] for i in sample_indices]
        cols = [(desc[0]+' ').replace("number_of_three_point_field_goals_attemp ", "number_of_three_point_field_goals_attempted ").replace("percentage_of_three_point_field_goal_mad ", "percentage_of_three_point_field_goal_made ").replace("team_percentage_of_three_point_field_goal_ ", "team_percentage_of_three_point_field_goal_made ").strip() for desc in self.description()]
        return sampled_data, cols    

    def is_existing_view(self, view_id, type="materialized"):
        check_view_sql = "SELECT COUNT(*) "
        if type == "materialized":
            check_view_sql += f"""FROM pg_catalog.pg_matviews WHERE matviewname = '{view_id.lower()}';"""
        else:
            check_view_sql += f"""FROM pg_catalog.pg_views WHERE viewname = '{view_id.lower()}';"""
        self.execute(check_view_sql)
        count = self.fetchall()[0][0]
        assert count in (0, 1)

        return count == 1

    def create_view(self, logger, view_id, sql, type="materialized", drop_if_exists=False):
        # Check if view_id already exists
        if self.is_existing_view(view_id, type):
            if drop_if_exists:
                logger.info(f"DROP exisiting view named {view_id}")
                self.drop_view(logger, view_id, type, cascade=True)
            else:
                logger.info(f"Skip creating view named {view_id}: already exists")
                return

        if type == "materialized":
            view_sql = "CREATE MATERIALIZED VIEW "
        else:
            view_sql = "CREATE VIEW "
        view_sql += f""" {view_id} AS {sql};"""
        self.execute(view_sql)
        return

    def drop_view(self, logger, view_id, type="materialized", cascade=False):
        if not self.is_existing_view(view_id, type):
            logger.warning(f"Skip drop view named {view_id} typed {type}: not exists")
            return

        if type == "materialized":
            view_sql = f"""DROP MATERIALIZED VIEW {view_id}"""
        else:
            view_sql = f"""DROP VIEW {view_id}"""

        if cascade:
            view_sql += " CASCADE;"
        else:
            view_sql += ";"

        # logger.info(f"Drop view named {view_id} typed {type}")
        self.execute(view_sql)
        return

def connect_data_manager(ip, port, user_id, user_pw, schema):
    data_manager = PostgreSQLDatabase(user_id, user_pw, ip, port, schema["dataset"])

    table_info = {}
    tables = schema["join_tables"]
    for table in tables:
        if "columns" in schema.keys() and table in schema["columns"].keys():
            columns = schema["columns"][table]
        else:
            columns = data_manager.fetch_column_names(table)
        table_info[table] = columns

    return data_manager, table_info

def lower_case_schema_data(schema):
    new_schema = {"dataset": schema["dataset"], "use_cols": schema["use_cols"]}

    new_schema["join_tables"] = [table.lower() for table in schema["join_tables"]]

    new_schema["join_keys"] = {}
    for table in schema["join_keys"].keys():
        new_schema["join_keys"][table.lower()] = [column.lower() for column in schema["join_keys"][table]]

    new_schema["join_clauses"] = [clause.lower() for clause in schema["join_clauses"]]

    new_schema["join_root"] = schema["join_root"].lower()

    if "columns" in schema.keys():
        new_schema["columns"] = {}
        for table in schema["columns"].keys():
            new_schema["columns"][table.lower()] = [column.lower() for column in schema["columns"][table]]

    return new_schema

def is_new_query_equivalent(existing_queries: list, new_query: str) -> bool:
    """
    Check if the new query is equivalent to any query in the existing set.

    Args:
        existing_queries (list): List of existing SQL query strings.
        new_query (str): Newly generated SQL query string.

    Returns:
        bool: True if new_query is equivalent to any of existing_queries, False otherwise.
    """
    for existing_query in existing_queries:
        if are_sql_queries_equivalent(existing_query, new_query):
            return True
    return False

def are_sql_queries_equivalent(query1: str, query2: str) -> bool:
    """
    Check if two SQL queries are equivalent by normalizing their ASTs and predicates.
    """
    try:
        # Parse and normalize
        parsed_query1 = parse_one(query1)
        parsed_query2 = parse_one(query2)
        
        normalized_query1 = normalize(parsed_query1)
        normalized_query2 = normalize(parsed_query2)
        
        # Additionally normalize predicates (AND/OR sorting)
        normalized_query1 = normalize_predicates(normalized_query1)
        normalized_query2 = normalize_predicates(normalized_query2)
        
        # Compare final normalized SQL strings
        return normalized_query1.sql() == normalized_query2.sql()
    
    except Exception as e:
        print(f"Error during parsing or normalization: {e}")
        return False

def normalize_predicates(expression):
    """
    Recursively normalize the predicate expressions by sorting commutative operations (AND, OR).
    """
    if isinstance(expression, (exp.And, exp.Or)):
        # Normalize both sides first
        left = normalize_predicates(expression.args["this"])
        right = normalize_predicates(expression.args["expression"])
        
        # Sort operands based on their SQL string (or other criteria)
        operands = sorted([left, right], key=lambda x: x.sql())
        
        if isinstance(expression, exp.And):
            return exp.and_(*operands)
        else:
            return exp.or_(*operands)
    
    # If not AND/OR, just return as is
    for arg_name, arg_value in expression.args.items():
        if isinstance(arg_value, exp.Expression):
            expression.set(arg_name, normalize_predicates(arg_value))
    return expression


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