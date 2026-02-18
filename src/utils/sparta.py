import random
import psycopg2
import traceback
from json import JSONDecoder
from typing import Any, List
from sqlglot import parse_one, exp
import sqlglot
from sqlglot.optimizer.normalize import normalize

TEXTUAL_AGGS = ["COUNT"]
DATE_AGGS = ["COUNT", "MIN", "MAX"]
KEY_AGGS = ["COUNT"]
NUMERIC_AGGS = ["COUNT", "MIN", "MAX", "AVG", "SUM"]


def extract_tables_from_sql(sql_string: str, dialect: str = "mysql") -> List[str]:
    """
    Extract all table names from a nested SQL query string.

    Args:
        sql_string (str): The SQL query string to analyze
        dialect (str): SQL dialect (default: "mysql")

    Returns:
        List[str]: List of all table names accessed in the query (duplicates removed)

    Raises:
        ValueError: If SQL parsing fails
    """
    try:
        # Parse the SQL string
        parsed = sqlglot.parse_one(sql_string, dialect=dialect)

        # Set to store table names
        tables = set()

        # Use SQLGlot's find method to get all Table nodes
        for table_node in parsed.find_all(exp.Table):
            # Handle schema.table format
            if hasattr(table_node, "catalog") and table_node.catalog:
                if hasattr(table_node, "db") and table_node.db:
                    table_name = (
                        f"{table_node.catalog}.{table_node.db}.{table_node.name}"
                    )
                else:
                    table_name = f"{table_node.catalog}.{table_node.name}"
            elif hasattr(table_node, "db") and table_node.db:
                table_name = f"{table_node.db}.{table_node.name}"
            else:
                table_name = (
                    str(table_node.name) if table_node.name else str(table_node)
                )

            tables.add(table_name)

        return sorted(list(tables))

    except Exception as e:
        raise ValueError(f"SQL parsing failed: {str(e)}")


def get_columns_of_nested_predicate(predicate_sql: str):
    parsed = parse_one(predicate_sql)
    outer_columns = set()
    inner_columns = set()
    for subq in parsed.find_all(exp.Subquery):
        parent = subq.parent
        if isinstance(parent, exp.In):
            left_expr = parent.args.get("this")
            for col in left_expr.find_all(exp.Column):
                table_name = col.table
                column_name = col.name
                final_name = (
                    f"{table_name}.{column_name}" if table_name else column_name
                )
                outer_columns.add(final_name)
        elif isinstance(parent, exp.Binary):
            left_expr = parent.args.get("this")
            right_expr = parent.args.get("expression")

            if left_expr is subq:
                other_side = right_expr
            else:
                other_side = left_expr

            if other_side:
                for col in other_side.find_all(exp.Column):
                    table_name = col.table
                    column_name = col.name
                    final_name = (
                        f"{table_name}.{column_name}" if table_name else column_name
                    )
                    outer_columns.add(final_name)

        for col in subq.find_all(exp.Column):
            table_name = col.table
            column_name = col.name
            final_name = f"{table_name}.{column_name}" if table_name else column_name
            inner_columns.add(final_name)

    if not outer_columns:
        subqueries = list(parsed.find_all(exp.Select))
        for subquery in subqueries:
            subquery_tables = {t.name for t in subquery.find_all(exp.Table)}
            for col in subquery.find_all(exp.Column):
                if col.table and col.table not in subquery_tables:
                    outer_columns.add(f"{col.table}.{col.name}")

    return list(outer_columns), list(inner_columns)


def post_process_nested_predicate(sql):
    if sql.strip().lower().startswith("where"):
        return sql.strip()[len("WHERE") :].strip()
    elif sql.strip().lower().startswith("select"):
        parsed = parse_one(sql)

        from_expr = parsed.args.get("from")
        if not from_expr:
            return ""

        outer_table_name = str(from_expr.this)

        outer_alias = from_expr.alias

        where_expr = parsed.args.get("where")
        if not where_expr:
            return ""

        for col in where_expr.find_all(exp.Column):
            if (not col.table) or (outer_alias and col.table == outer_alias):
                col.set("table", outer_table_name)

        fully_qualified_predicate = str(where_expr)
        return fully_qualified_predicate[len("WHERE") :].strip()
    else:
        return sql.strip()


def get_aliases_in_select(select_node: exp.Select) -> set[str]:
    aliases = set()
    from_clause = select_node.args.get("from")
    if isinstance(from_clause, exp.From):
        for fexpr in from_clause.expressions:
            if isinstance(fexpr, exp.Table):
                if fexpr.alias:
                    aliases.add(fexpr.alias)
                else:
                    aliases.add(fexpr.name)
            elif isinstance(fexpr, exp.Subquery):
                if fexpr.alias:
                    aliases.add(fexpr.alias)

    for join in select_node.find_all(exp.Join):
        join_obj = join.this
        if isinstance(join_obj, exp.Table):
            if join_obj.alias:
                aliases.add(join_obj.alias)
            else:
                aliases.add(join_obj.name)
        elif isinstance(join_obj, exp.Subquery):
            if join_obj.alias:
                aliases.add(join_obj.alias)

    return aliases


def get_tables_in_subquery(subquery: exp.Select) -> set[str]:
    """Extract all table names/aliases referenced in a subquery's FROM clause and JOINs."""
    tables = set()

    # Get tables from FROM clause
    from_clause = subquery.args.get("from")
    if isinstance(from_clause, exp.From):
        # Handle the main table in FROM clause
        if isinstance(from_clause.this, exp.Table):
            table = from_clause.this
            tables.add(table.alias if table.alias else table.name)

        # Handle additional tables in expressions (for multiple table queries)
        for fexpr in from_clause.expressions:
            if isinstance(fexpr, exp.Table):
                # Use alias if available, otherwise use table name
                tables.add(fexpr.alias if fexpr.alias else fexpr.name)

    # Get tables from JOIN clauses
    for join in subquery.find_all(exp.Join):
        join_obj = join.this
        if isinstance(join_obj, exp.Table):
            # Use alias if available, otherwise use table name
            tables.add(join_obj.alias if join_obj.alias else join_obj.name)

    return tables


def classify_subquery(subquery: exp.Select) -> str:
    join = has_join(subquery)
    agg = has_aggregation_str_based(subquery)

    if join:
        return "Type-JA" if agg else "Type-J"
    else:
        return "Type-A" if agg else "Type-N"


def has_join(select_node: exp.Select) -> bool:
    where_expr = select_node.args.get("where")
    if not where_expr or not where_expr.this:
        return False

    # Get all tables referenced in this subquery
    subquery_tables = get_tables_in_subquery(select_node)

    for condition in walk_without_subquery(where_expr.this):
        if isinstance(condition, exp.EQ):
            left = condition.args.get("this")
            right = condition.args.get("expression")

            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                # Check if at least one side has an explicit table reference
                left_has_table = left.table is not None
                right_has_table = right.table is not None

                if not (left_has_table or right_has_table):
                    # Neither side has explicit table reference, skip
                    continue

                # Skip self-referencing predicates (same column compared to itself)
                if left.table == right.table and left.name == right.name:
                    continue

                # For meaningful predicate, we need:
                # 1. At least one side with explicit table reference
                # 2. At least one table reference external to subquery
                # 3. Not both sides referencing the same table within the subquery
                left_table = left.table if left_has_table else None
                right_table = right.table if right_has_table else None

                # Check if both sides reference tables within the subquery
                left_internal = left_table in subquery_tables if left_table else False
                right_internal = (
                    right_table in subquery_tables if right_table else False
                )

                # Skip if both sides are internal to the same table within the subquery
                if left_internal and right_internal and left_table == right_table:
                    continue

                # We have a meaningful predicate if at least one side references external table
                if (left_table and not left_internal) or (
                    right_table and not right_internal
                ):
                    return True

    return False


def walk_without_subquery(expr):
    if isinstance(expr, (exp.Subquery, exp.Exists, exp.In)):
        return
    yield expr
    for arg in expr.args.values():
        if isinstance(arg, list):
            for item in arg:
                if isinstance(item, exp.Expression):
                    yield from walk_without_subquery(item)
        elif isinstance(arg, exp.Expression):
            yield from walk_without_subquery(arg)


def has_aggregation_str_based(select_node: exp.Select) -> bool:
    for projection in select_node.expressions:
        projection = str(projection)
        if (
            projection.upper().startswith("SUM")
            or projection.upper().startswith("COUNT")
            or projection.upper().startswith("MAX")
            or projection.upper().startswith("MIN")
            or projection.upper().startswith("AVG")
        ):
            return True
    return False


def detect_type_of_nested_predicate(sql_with_nested_predicate):
    stype_list = []
    parsed = parse_one(sql_with_nested_predicate)
    subquery_nodes = list(parsed.find_all((exp.Subquery, exp.Exists)))

    for i, node in enumerate(subquery_nodes, start=1):
        if isinstance(node.this, exp.Select):
            subquery_select = node.this
            stype = classify_subquery(subquery_select)
            stype_list.append(stype.lower())

    return stype_list


def generate_from_clause_for_full_outer_view(
    args, schema=None, relations=None, join_clause_list=None, is_inner_join=False
):
    visited_join_clauses = []
    visited_tables = [schema["join_root"]] if relations is None else relations[:1]
    join_clauses = (
        schema["join_clauses"] if join_clause_list is None else join_clause_list
    )
    from_clause = (
        f"""{schema["join_root"]}""" if relations is None else f"""{relations[0]}"""
    )

    while len(visited_join_clauses) < len(join_clauses):
        if relations is not None and (len(visited_join_clauses) == len(relations) - 1):
            break
        is_updated = False
        for table in visited_tables:
            for join_clause in join_clauses:
                if join_clause in visited_join_clauses:
                    continue

                key1 = join_clause.split("=")[0]
                key2 = join_clause.split("=")[1]
                tab1 = key1.split(".")[0]
                tab2 = key2.split(".")[0]

                if relations is not None and (
                    tab1 not in relations or tab2 not in relations
                ):
                    continue

                if tab1 == table:
                    if tab2 in visited_tables:
                        from_clause += f" AND {join_clause}"
                    else:
                        from_clause += (
                            f" FULL OUTER JOIN {tab2} ON {join_clause}"
                            if not is_inner_join
                            else f" INNER JOIN {tab2} ON {join_clause}"
                        )
                        visited_tables.append(tab2)
                    is_updated = True
                    visited_join_clauses.append(join_clause)

                if tab2 == table:
                    if tab1 in visited_tables:
                        from_clause += f" AND {join_clause}"
                    else:
                        from_clause += (
                            f" FULL OUTER JOIN {tab1} ON {join_clause}"
                            if not is_inner_join
                            else f" INNER JOIN {tab1} ON {join_clause}"
                        )
                        visited_tables.append(tab1)
                    is_updated = True
                    visited_join_clauses.append(join_clause)

        if not is_updated:
            # Not connected join
            assert False

    return from_clause


def from_clause_to_dict(from_clause: str) -> dict:
    query = f"SELECT * {from_clause}"
    parsed = parse_one(query)

    table_nodes = parsed.find_all(exp.Table)

    results = {}
    for node in table_nodes:
        table_name = node.name
        alias = node.alias or table_name
        results[table_name] = alias

    return results


def extract_tables(from_clause: str) -> list[str]:

    query = f"SELECT * {from_clause}"
    parsed = parse_one(query)

    tables = [table.name for table in parsed.find_all(exp.Table)]
    return tables


def bfs(edges, parents, observed, targets):
    candidates = []
    cp_condition = {}
    for p in parents:
        childs = edges[p]
        for c, join_clause in childs:
            if c in targets:
                return [c, p], [join_clause]
            else:
                if c in observed:
                    continue
                else:
                    candidates.append(c)
                    observed.append(c)
                    cp_condition[c] = (p, join_clause)
    assert len(candidates) > 0
    path, joins = bfs(edges, candidates, observed, targets)
    src_parent, src_join = cp_condition[path[-1]]
    path.append(src_parent)
    joins.append(src_join)
    return path, joins


def find_join_path(joins, tables, used_tables):
    # Step 1. Draw schema graph
    edges = {}
    for join_clause in joins:
        t1, t2 = get_table_from_clause(join_clause)
        if t1 in edges:
            edges[t1].append((t2, join_clause))
        else:
            edges[t1] = [(t2, join_clause)]
        if t2 in edges:
            edges[t2].append((t1, join_clause))
        else:
            edges[t2] = [(t1, join_clause)]

    used_tables = list(used_tables)
    found_tables = [used_tables[0]]
    found_joins = []
    for tab in used_tables[1:]:
        path, joins = bfs(edges, [tab], [], found_tables)
        found_tables = list(set(found_tables) | set(path))
        found_joins = list(set(found_joins) | set(joins))

    return found_tables, found_joins


def get_table_from_clause(join_clause):
    A, B = join_clause.split("=")
    return A.split(".")[0], B.split(".")[0]


def get_joinable_tables(table, join_clause_list):
    """Returns a set of tables that can be joined with the given table."""
    table_set = set()
    for join_clause in sorted(join_clause_list):
        t1, t2 = get_table_from_clause(join_clause)
        if table == t1:
            table_set.update([t2])
        if table == t2:
            table_set.update([t1])

    return list(table_set)


def get_accessed_tables(sql: str) -> list[str]:
    parsed = parse_one(sql)
    if not parsed:
        return []

    from_expr = parsed.args.get("from")
    if not from_expr:
        return []

    tables = [table.name for table in from_expr.find_all(exp.Table)]

    return list(set(tables))


def map_tables_to_queries(query_blocks):
    """Helper to map tables to their corresponding query blocks."""
    table_to_queries = {}
    for query in query_blocks:
        for table in parse_one(query).find_all(exp.From):
            table_name = table.name
            table_to_queries.setdefault(table_name, []).append(query)
    return table_to_queries


def map_columns_to_tables(dtype_dict):
    """Helper to map columns to the tables they appear in."""
    column_to_tables = {}
    for table_column in dtype_dict:
        table_name, column_name = table_column.split(".", 1)
        column_to_tables.setdefault(column_name, []).append(table_name)
    return column_to_tables


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
        if ch == ".":
            if (
                i > 0
                and i < length - 1
                and input_str[i - 1].isdigit()
                and input_str[i + 1].isdigit()
            ):
                result.append(".")
            else:
                result.append("__")
        else:
            result.append(ch)

    return "".join(result)


def extract_equality_predicate_columns(sql):
    """Extract columns that are used in equality predicates (=) from WHERE clause.
    Only considers predicates in the outermost query, ignoring subqueries.
    """
    if not sql:
        return set()

    try:
        # Parse the SQL
        parsed = parse_one(sql)
        equality_columns = set()

        def extract_eq_columns(expr):
            # Skip if we're inside a subquery or subquery-related expressions
            if isinstance(expr, (exp.Select, exp.Subquery)):
                return

            # IN clause with subquery - don't traverse into the subquery
            if isinstance(expr, exp.In):
                # Only process the left side (the column being checked)
                if isinstance(expr.this, exp.Column):
                    # We don't add IN predicates as equality predicates
                    pass
                return  # Don't traverse into the subquery part

            if isinstance(expr, exp.EQ):
                # Check both sides of the equality
                if isinstance(expr.left, exp.Column):
                    col_name = expr.left.name
                    table_name = expr.left.table
                    if table_name:
                        equality_columns.add(f"{table_name}.{col_name}")
                    else:
                        equality_columns.add(col_name)

                if isinstance(expr.right, exp.Column):
                    col_name = expr.right.name
                    table_name = expr.right.table
                    if table_name:
                        equality_columns.add(f"{table_name}.{col_name}")
                    else:
                        equality_columns.add(col_name)

            # Recursively process child expressions
            for child in expr.args.values():
                if isinstance(child, exp.Expression):
                    extract_eq_columns(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, exp.Expression):
                            extract_eq_columns(item)

        # If it's a SELECT statement, only process its WHERE clause
        if isinstance(parsed, exp.Select):
            where_clause = parsed.args.get("where")
            if where_clause:
                extract_eq_columns(where_clause)
        else:
            # If it's just a WHERE clause condition
            extract_eq_columns(parsed)

        return equality_columns

    except Exception as e:
        # If parsing fails, return empty set to be safe
        print(f"Error: {e}")
        return set()


def generate_select_clause_for_full_outer_view(
    table_info, relations=None, use_alias=True
):
    select_columns = []
    for table in table_info.keys():
        for column in table_info[table]:
            if column == "name_id" or column == "patient_id" or column == "doctor_id":
                continue
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
    tree = parse_one(query.replace("-", "123456789").replace("lb", "135792468"))

    def _replace_literals(expression):
        if isinstance(expression, exp.Condition):  # Conditions like WHERE clauses
            if isinstance(expression, (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE)):
                left, right = expression.args.get("this"), expression.args.get(
                    "expression"
                )
                if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                    col_name = f"{left.table}.{left.name}" if left.table else left.name
                    if col_name.replace("__", ".") in column_types:
                        dtype = column_types[col_name.replace("__", ".")]
                        new_value = cast_value_to_dtype(right.this, dtype)
                        expression.set("expression", new_value)

        for e in expression.args.values():
            if isinstance(e, list):
                for sub in e:
                    _replace_literals(sub)
            elif isinstance(e, exp.Expression):
                _replace_literals(e)

    _replace_literals(tree)
    return tree.sql().replace("123456789", "-").replace("135792468", "lb")


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
        return exp.Literal.boolean(
            "TRUE" if value.lower() in ("1", "true", "t", "yes", "y") else "FALSE"
        )
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
    INVALID_IDS = col_info.get("invalid_ids", [])

    return IDS, HASH_CODES, NOTES, CATEGORIES, FOREIGN_KEYS, INVALID_IDS


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
        self.conn = psycopg2.connect(
            f"user={user_id} password={passwd} host={host} port={port} dbname={db_id}"
        )  # , prepare_threshold=0
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
            f"{str(table[0].lower())}.{str(table[1].lower())}".replace("public.", "")
            for table in self.fetchall()
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

    def check_distinct_value_exists(
        self, table_name: str, column_name: str, additional_clauses: str = None
    ):
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

    def get_distinct_value_counts(
        self, table_name: str, column_name: str, additional_clauses: str = None
    ):
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
            sample_indices = rng.choice(
                len(all_data), size=min_sample_size, replace=False
            )
            sampled_data = [all_data[i] for i in sample_indices]
        cols = [
            (desc[0] + " ")
            .replace(
                "number_of_three_point_field_goals_attemp ",
                "number_of_three_point_field_goals_attempted ",
            )
            .replace(
                "percentage_of_three_point_field_goal_mad ",
                "percentage_of_three_point_field_goal_made ",
            )
            .replace(
                "team_percentage_of_three_point_field_goal_ ",
                "team_percentage_of_three_point_field_goal_made ",
            )
            .strip()
            for desc in self.description()
        ]
        return sampled_data, cols

    def sample_rows_with_cond(
        self, table_name: str, cond: str, sample_size: int, rng=None
    ):
        self.execute(f"SELECT * FROM {table_name} WHERE {cond};")
        all_data = self.fetchall()
        min_sample_size = min(sample_size, len(all_data))
        if rng is None:
            sampled_data = random.sample(all_data, min_sample_size)
        else:
            sample_indices = rng.choice(
                len(all_data), size=min_sample_size, replace=False
            )
            sampled_data = [all_data[i] for i in sample_indices]
        cols = [
            (desc[0] + " ")
            .replace(
                "number_of_three_point_field_goals_attemp ",
                "number_of_three_point_field_goals_attempted ",
            )
            .replace(
                "percentage_of_three_point_field_goal_mad ",
                "percentage_of_three_point_field_goal_made ",
            )
            .replace(
                "team_percentage_of_three_point_field_goal_ ",
                "team_percentage_of_three_point_field_goal_made ",
            )
            .strip()
            for desc in self.description()
        ]
        return sampled_data, cols

    def sample_rows_with_sql(self, sql: str, sample_size: int, rng=None):
        self.execute(sql)
        all_data = self.fetchall()
        min_sample_size = min(sample_size, len(all_data))
        if rng is None:
            sampled_data = random.sample(all_data, min_sample_size)
        else:
            sample_indices = rng.choice(
                len(all_data), size=min_sample_size, replace=False
            )
            sampled_data = [all_data[i] for i in sample_indices]
        cols = [
            (desc[0] + " ")
            .replace(
                "number_of_three_point_field_goals_attemp ",
                "number_of_three_point_field_goals_attempted ",
            )
            .replace(
                "percentage_of_three_point_field_goal_mad ",
                "percentage_of_three_point_field_goal_made ",
            )
            .replace(
                "team_percentage_of_three_point_field_goal_ ",
                "team_percentage_of_three_point_field_goal_made ",
            )
            .strip()
            for desc in self.description()
        ]
        return sampled_data, cols

    def sample_rows_with_sql_with_limit(self, sql: str, sample_size: int, rng=None):
        self.execute(sql + f" LIMIT {sample_size*5};")
        all_data = self.fetchall()
        min_sample_size = min(sample_size, len(all_data))
        if rng is None:
            sampled_data = random.sample(all_data, min_sample_size)
        else:
            sample_indices = rng.choice(
                len(all_data), size=min_sample_size, replace=False
            )
            sampled_data = [all_data[i] for i in sample_indices]
        cols = [
            (desc[0] + " ")
            .replace(
                "number_of_three_point_field_goals_attemp ",
                "number_of_three_point_field_goals_attempted ",
            )
            .replace(
                "percentage_of_three_point_field_goal_mad ",
                "percentage_of_three_point_field_goal_made ",
            )
            .replace(
                "team_percentage_of_three_point_field_goal_ ",
                "team_percentage_of_three_point_field_goal_made ",
            )
            .strip()
            for desc in self.description()
        ]
        return sampled_data, cols

    def is_existing_view(self, view_id, type="materialized"):
        check_view_sql = "SELECT COUNT(*) "
        if type == "materialized":
            check_view_sql += f"""FROM pg_catalog.pg_matviews WHERE matviewname = '{view_id.lower()}';"""
        else:
            check_view_sql += (
                f"""FROM pg_catalog.pg_views WHERE viewname = '{view_id.lower()}';"""
            )
        self.execute(check_view_sql)
        count = self.fetchall()[0][0]
        assert count in (0, 1)

        return count == 1

    def create_view(
        self, logger, view_id, sql, type="materialized", drop_if_exists=False
    ):
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
        new_schema["join_keys"][table.lower()] = [
            column.lower() for column in schema["join_keys"][table]
        ]

    new_schema["join_clauses"] = [clause.lower() for clause in schema["join_clauses"]]

    new_schema["join_root"] = schema["join_root"].lower()

    if "columns" in schema.keys():
        new_schema["columns"] = {}
        for table in schema["columns"].keys():
            new_schema["columns"][table.lower()] = [
                column.lower() for column in schema["columns"][table]
            ]

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
