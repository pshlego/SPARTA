import sqlglot
from sqlglot import parse_one, exp
from typing import Tuple
from sqlglot.expressions import Column
import copy
import re

def rewrite_non_nested_query_with_groupby(sql: str) -> Tuple[str, str]:
    ast = parse_one(sql)

    table = ast.find(exp.Table)
    table_name = table.name
    table_alias = table.alias_or_name

    select_clause = ast.args["expressions"]
    agg_expr = next((e for e in select_clause if isinstance(e.this, (exp.AggFunc, exp.Sum, exp.Avg, exp.Count))), None)
    if not agg_expr:
        raise NotImplementedError("No aggregate function found.")
    agg_func = agg_expr.this
    agg_col = agg_func.this.sql() if agg_func.this else None
    agg_alias = agg_expr.alias_or_name

    having_clause = ast.args.get("having")
    having_sql = having_clause.sql() if having_clause else None
    
    group_clause = ast.args.get("group")
    group_keys = [g.sql() for g in group_clause.expressions] if group_clause else []
    group_col = group_keys[0] if group_keys else "group_col"

    order_clause = ast.args.get("order")
    order_sql = order_clause.sql() if order_clause else ""

    limit_clause = ast.args.get("limit")
    limit_sql = limit_clause.sql() if limit_clause else ""
    
    # Phase 1
    location = 0
    if order_sql or limit_sql :
        p1 = f"""
        SELECT
            writegrp({location}, ARRAY_AGG({table_alias}.tuid ORDER BY {table_alias}.tuid )) AS tuid,
            {table_alias}.{group_col} AS {group_col},
            {agg_func.sql()} AS {agg_alias}
        FROM {table_name}_1 AS {table_alias}
        GROUP BY {table_alias}.{group_col}
        {having_sql}
        {order_sql}
        {limit_sql};
        """.strip()
    else:
        p1 = f"""
        WITH grouped_{table_alias}(tuid, {group_col}, {agg_alias}) AS (
        SELECT
            writegrp({location}, ARRAY_AGG({table_alias}.tuid ORDER BY {table_alias}.tuid )) AS tuid,
            {table_alias}.{group_col},
            {agg_func.sql()} AS {agg_alias}
        FROM {table_name}_1 AS {table_alias}
        GROUP BY {table_alias}.{group_col}
        {f'{having_sql}' if having_sql else ''}
        )
        SELECT * FROM grouped_{table_alias};
        """.strip()
        

    # Phase 2
    if isinstance(agg_func, exp.Count) and agg_col == '*':
        agg_prov_expr = "wh.y"
    else:
        agg_prov_expr = f"dd({table_alias}.{agg_col}) || wh.y"

    p2 = f"""
    WITH grouped_{table_alias}(tuid, {group_col}, {agg_alias}) AS (
      SELECT
        g.tuidout AS tuid,
        dd(e.{group_col}) || wh.y AS {group_col},
        {agg_prov_expr} AS {agg_alias}
      FROM loggrp AS g
      JOIN {table_name}_2 AS e ON e.tuid = g.tuidset
      JOIN LATERAL toY(e.{group_col}) AS wh(y) ON TRUE
      WHERE g.location = {location}
    )
    SELECT
      tuid,
      dd(ARRAY_AGG({group_col})) AS {group_col},
      dd(ARRAY_AGG({agg_alias})) AS {agg_alias}
    FROM grouped_{table_alias}
    GROUP BY tuid;
    """.strip()
    
    return p1, p2

def extract_conditions(expr):
    """Recursively extract (col, op, val) triples from a WHERE clause."""
    supported_ops = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
    result = []

    if isinstance(expr, exp.And):
        result += extract_conditions(expr.left)
        result += extract_conditions(expr.right)
    elif isinstance(expr, supported_ops):
        if isinstance(expr.left, exp.Column):
            col = expr.left.name
            val = expr.right.sql()
        elif isinstance(expr.right, exp.Column):
            col = expr.right.name
            val = expr.left.sql()
        else:
            raise NotImplementedError("At least one side of the condition must be a column.")
        op_map = {
            exp.EQ: "=",
            exp.NEQ: "<>",
            exp.GT: ">",
            exp.GTE: ">=",
            exp.LT: "<",
            exp.LTE: "<=",
        }
        result.append((col, op_map[type(expr)], val))
    else:
        raise NotImplementedError("Unsupported WHERE clause structure.")
    
    return result


def rewrite_non_nested_query(sql: str) -> tuple[str, str]:
    ast = parse_one(sql)

    # 1. FROM: Get table alias and name
    table = ast.find(exp.Table)
    if not table:
        raise ValueError("No FROM clause found")
    table_name = table.name
    
    # 2. SELECT : Get Projection Columns
    select_exprs = ast.args["expressions"][0]
    select_column = select_exprs.alias_or_name
    
    # 3. WHERE
    where_expression = ast.args.get("where")
    if where_expression:
        conditions = extract_conditions(where_expression.this)  # List[(col, op, val)]
        where_sql = " AND ".join(f"{table_name}.{col} {op} {val}" for col, op, val in conditions)
        where_clause = f"WHERE {where_sql}"
    else:
        where_clause = ""

    order_clause = ast.args.get("order")
    order_sql = order_clause.sql() if order_clause else ""

    limit_clause = ast.args.get("limit")
    limit_sql = limit_clause.sql() if limit_clause else ""
    
    location = 0  # hardcoded log location
    # Phase 1 rewrite
    if order_sql or limit_sql :
        p1 = f"""
        SELECT
            writelog({location}, {table_name}.tuid) AS tuid,
            {table_name}.{select_column}
        FROM {table_name}_1 AS {table_name}
        {where_clause}
        {order_sql}
        {limit_sql}
        ;
        """.strip()
    else:
        p1 = f"""
        WITH filtered_{table_name}(tuid, {select_column}) AS (
        SELECT
            writelog({location}, {table_name}.tuid) AS tuid,
            {table_name}.{select_column}
        FROM {table_name}_1 AS {table_name}
        {where_clause}
        )
        SELECT * FROM filtered_{table_name};
        """.strip()

    # Phase 2 rewrite   
    if where_expression:
        prov_cols = f"{table_name}.{select_column} || wh.y AS {select_column}"
        toys = " || ".join(f"toY({table_name}.{col})" for col, _, _ in conditions)
        whys_clause = f",\n LATERAL (SELECT {toys}) AS wh(y)"
    else:
        prov_cols = f"{table_name}.{select_column} AS {select_column}"
        whys_clause = ""
    dd_cols = f"dd(f.{select_column})"
    
    p2 = f"""
    WITH filtered_{table_name}(tuid, {select_column}) AS (
    SELECT
        log.tuid,
        {prov_cols}
    FROM {table_name}_2 AS {table_name},
        readlog({location}, {table_name}.tuid) AS log(tuid)
        {whys_clause}
    )
    SELECT
    f.tuid,
    {dd_cols} AS {select_column}
    FROM filtered_{table_name} AS f;
    """.strip()

    return p1, p2

def extract_outer_table_names(ast) -> tuple:
    tables = set()
    join_conditions = []
    
    # FROM
    from_clause = ast.args.get("from")
    if from_clause:
        main_table = from_clause.this
        if isinstance(main_table, exp.Table):
            tables.add(main_table.name)
    # JOIN
    joins = ast.args.get("joins") or []
    for join in joins:
        table_expr = join.this
        if isinstance(table_expr, exp.Table):
            tables.add(table_expr.name)
        join_condition = join.args.get("on")
        if join_condition:
            condition_str = join_condition.sql()
            modified_condition = re.sub(r'(\b\w+\b)\.(\b\w+\b)', r'\1_outer.\2', condition_str)
            join_conditions.append(modified_condition)
    
    return tables, join_conditions

def rewrite_nested_query(sql: str) -> tuple[str, str]:
    ast = parse_one(sql)
    
    # 1. Outer table info
    outer_table_names, outer_join_conditions = extract_outer_table_names(ast)
    outer_join_columns = []
    for condition in outer_join_conditions:
        parts = condition.split("=")
        outer_join_columns.extend([part.strip() for part in parts])

    # 2. SELECT columns
    select_expr = ast.args["expressions"][0]
    select_column = select_expr.this.name
    select_table = select_expr.table
    if select_table == "":
        select_table = list(outer_table_names)[0]
    
    # 3. WHERE clause
    where_clause = ast.args.get("where")
    if not where_clause:
        raise ValueError("Missing WHERE clause")

    conditions = where_clause.this
    in_exprs = []
    eq_exprs = []
    exists_exprs = []
    nonsub_exprs = []
    
    def split_conditions(expr):
        if isinstance(expr, exp.And):
            split_conditions(expr.left)
            split_conditions(expr.right)
        elif isinstance(expr, exp.In):
            in_exprs.append(expr)
        elif isinstance(expr, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            if isinstance(expr.right, exp.Subquery):
                eq_exprs.append(expr)
            else:
                nonsub_exprs.append(expr)
        elif isinstance(expr, exp.Exists):
            exists_exprs.append(expr)
        else:
            raise NotImplementedError(f"Unsupported expression: {expr}")
    
    split_conditions(conditions)
    
    joins = []
    tuid_sources = [f"{tbl}_outer.tuid" for tbl in outer_table_names]
    readlog_args = [f"{tbl}_outer.tuid" for tbl in outer_table_names]
    from_tables_p2 = [f"{tbl}_2 AS {tbl}_outer" for tbl in outer_table_names]
    whys = [" || ".join(f"toY({c})" for c in outer_join_columns)] if outer_join_columns else []
    cte_parts = [] # for type A, JA
    
    # Type-N, J (IN subqueries)
    for in_expr in in_exprs:
        outer_col = in_expr.this.name
        outer_table_name = in_expr.this.table
        if outer_table_name == "":
            outer_table_name = list(outer_table_names)[0]
        subquery = in_expr.args["query"]
        inner_select = subquery.this
        # Subquery Table
        inner_table = inner_select.find(exp.Table)
        inner_table_name = inner_table.name
        inner_table_alias = inner_table.alias
        # Subquery Column
        inner_col_expr = inner_select.args["expressions"][0]
        inner_col = inner_col_expr.alias_or_name or inner_col_expr.name
        # Subquery Where 
        inner_where = inner_select.args.get("where")
        filters, correlations, filter_cols, correlation_cols = extract_filters(inner_where.this, outer_table_names, inner_table_name, inner_table_alias) if inner_where else ([], [], [], [])
        
        # Naming inner table
        inner_alias_name, replace_alias = generate_unique_inner_alias(inner_table_name, joins)
        # For phase1
        tuid_sources.append(f"{inner_alias_name}.tuid")
        join = f"JOIN {inner_table_name}_1 AS {inner_alias_name} ON {outer_table_name}_outer.{outer_col} = {inner_alias_name}.{inner_col}" 
        if correlations:
            join_parts = [replace_alias(f) for f in correlations] + [replace_alias(f) for f in filters]
        else:
            join_parts = [replace_alias(f) for f in filters]
        if join_parts:
            join += " AND " + " AND ".join(join_parts)
        joins.append(join)
        
        # Subquery Join
        (join_tables,
        join_columns,
        join_clause_inner,
        filters
        ) = processing_join_in_subquery(
        inner_select,
        inner_table_alias,
        inner_table_name,
        inner_alias_name,
        joins,
        filters)
        joins.extend(join_clause_inner)
        
        # For phase2
        readlog_args.append(f"{inner_alias_name}.tuid")
        from_tables_p2.append(f"{inner_table_name}_2 AS {inner_alias_name}")
        for table_name, alias in join_tables:
            readlog_args.append(f"{alias}.tuid")
            from_tables_p2.append(f"{table_name}_2 AS {alias}")
            tuid_sources.append(f"{alias}.tuid") # for phase1
        why = f"toY({outer_table_name}_outer.{outer_col}) || toY({inner_alias_name}.{inner_col})"
        for c in [replace_alias(f) for f in correlation_cols]:
            why += f" || toY({c})"
        for c in [replace_alias(f) for f in filter_cols]:
            why += f" || toY({c})"
        for c in join_columns:
            why += f" || toY({c})"
        whys.append(why)
        
    # Type-J (Exists subqueries)
    for exists_expr in exists_exprs:
        subquery = exists_expr
        inner_select = subquery.this
        inner_table = inner_select.find(exp.Table)
        inner_table_name = inner_table.name
        inner_table_alias = inner_table.alias
        
        inner_col_expr = inner_select.args["expressions"][0]
        inner_col = inner_col_expr.alias_or_name or inner_col_expr.name
        inner_where = inner_select.args.get("where")
        filters, correlations, filter_cols, correlation_cols = extract_filters(inner_where.this, outer_table_names, inner_table_name, inner_table_alias) if inner_where else ([], [], [], [])
        
        # outer_table_name = extract_outer_table_name_from_subquery(inner_select)
        inner_alias_name, replace_alias = generate_unique_inner_alias(inner_table_name, joins)

        tuid_sources.append(f"{inner_alias_name}.tuid")
        join = f"JOIN {inner_table_name}_1 AS {inner_alias_name} ON "
        if correlations:
            join_parts = [replace_alias(f) for f in correlations] + [replace_alias(f) for f in filters]
        else:
            join_parts = [replace_alias(f) for f in filters]
        if join_parts:
            join += " AND ".join(join_parts)
        joins.append(join)
        
        (join_tables,
        join_columns,
        join_clause_inner,
        filters
        ) = processing_join_in_subquery(
        inner_select,
        inner_table_alias,
        inner_table_name,
        inner_alias_name,
        joins,
        filters)
        joins.extend(join_clause_inner)
        
        readlog_args.append(f"{inner_alias_name}.tuid")
        from_tables_p2.append(f"{inner_table_name}_2 AS {inner_alias_name}")
        for table_name, alias in join_tables:
            readlog_args.append(f"{alias}.tuid")
            from_tables_p2.append(f"{table_name}_2 AS {alias}")
            tuid_sources.append(f"{alias}.tuid") # for phase1
        why_parts = [f"toY({c})" for c in [replace_alias(f) for f in correlation_cols] + [replace_alias(f) for f in filter_cols] + [replace_alias(f) for f in join_columns]]
        why = " || ".join(why_parts)
        whys.append(why)

    # Type-A, JA (=, <, > ..  subqueries)
    for eq_expr in eq_exprs:
        outer_op_str = {
            exp.EQ: "=",
            exp.NEQ: "!=",
            exp.GT: ">",
            exp.GTE: ">=",
            exp.LT: "<",
            exp.LTE: "<=",
        }.get(type(eq_expr), "UNKNOWN")
        outer_col = eq_expr.left.this.alias_or_name
        outer_table_name = eq_expr.left.table
        if outer_table_name == "":
            outer_table_name = list(outer_table_names)[0]
        subquery = eq_expr.right
        inner_select = subquery.this
        # Subquery Table
        inner_table = inner_select.find(exp.Table)
        inner_table_name = inner_table.name
        inner_table_alias = inner_table.alias
        # Subquery Column
        agg_expr = inner_select.args["expressions"][0]
        agg_func = agg_expr.__class__.__name__.lower()  # e.g., max, min
        agg_col = agg_expr.this.name
        # Subquery Where
        inner_where = inner_select.args.get("where")
        filters, correlations, filter_cols, correlation_cols  = extract_filters(inner_where.this, outer_table_names, inner_table_name, inner_table_alias) if inner_where else ([], [], [], [])
        # Subquery Join
        inner_alias_name, replace_alias = generate_unique_inner_alias(inner_table_name, joins)
        filters = [replace_alias(f) for f in filters]
        
        (join_tables,
        join_columns,
        join_clause_inner,
        filters
        ) = processing_join_in_subquery(
        inner_select,
        inner_table_alias,
        inner_table_name,
        inner_alias_name,
        joins,
        filters) 
        
        prefix = f'{inner_alias_name}.'
        join_projection = [col.split('.')[-1] for col in join_columns if col.startswith(prefix)]
        if correlations: # correlation
            for ref in correlation_cols:
                table, col = ref.split('.', 1)
                if table.endswith('_outer'):
                    cor_outer_col = col
                elif table.endswith('_inner'):
                    cor_inner_col = col
            for cor_expr in correlations:
                OPERATOR_PATTERN = re.compile(r'(>=|<=|!=|>|<|=)')
                m = OPERATOR_PATTERN.search(cor_expr)
                cor_op_str = m.group(1) if m else "UNKNOWN"    

            cte = f"""
            agg_base_{inner_alias_name} AS 
            ( SELECT {outer_table_name}_outer.{cor_outer_col} AS {cor_outer_col},
            {agg_func}({inner_alias_name}.{agg_col}) AS {agg_col}
            FROM {inner_table_name}_1 AS {inner_alias_name}
            JOIN {outer_table_name}_1 AS {outer_table_name}_outer ON {' AND '.join([replace_alias(f) for f in correlations])}
            {' AND '.join(join_clause_inner)}
            {'WHERE ' + ' AND '.join(filters) if filters else ''}
            GROUP BY {outer_table_name}_outer.{cor_outer_col}
            ), 
            agg_result_{inner_alias_name} AS 
            ( SELECT    {inner_alias_name}.tuid AS tuid,
            {outer_table_name}_outer.{cor_outer_col} AS {cor_outer_col},
            {inner_alias_name}.{agg_col} AS {agg_col} {', ' + ', '.join(f"{inner_alias_name}.{col}" for col in join_projection) if join_projection else ''}
            FROM {inner_table_name}_1 AS {inner_alias_name}
            JOIN agg_base_{inner_alias_name} ON {inner_alias_name}.{agg_col} = agg_base_{inner_alias_name}.{agg_col}
            JOIN {outer_table_name}_1 AS {outer_table_name}_outer ON {' AND '.join([replace_alias(f) for f in correlations])}
            {' AND '.join(join_clause_inner)}
            {'WHERE ' + ' AND '.join(filters) if filters else ''}
            )
            """
            joins.append(f"JOIN agg_result_{inner_alias_name} ON {outer_table_name}_outer.{outer_col} {outer_op_str} agg_result_{inner_alias_name}.{agg_col} AND {outer_table_name}_outer.{cor_outer_col} {cor_op_str} agg_result_{inner_alias_name}.{cor_inner_col}")
        else:
            cte = f"""
            agg_result_{inner_alias_name}(tuid, {agg_col}) AS 
            ( SELECT    tuid,
            {agg_col}
            {', ' + ', '.join(f"{inner_alias_name}.{col}" for col in join_projection) if join_projection else ''}
            FROM {inner_table_name}_1 AS {inner_alias_name}
            WHERE {agg_col} =
            (SELECT {agg_func.upper()}({inner_alias_name}.{agg_col}) 
            FROM {inner_table_name} AS {inner_alias_name} 
            {' AND '.join(join_clause_inner)}
            {'WHERE ' + ' AND '.join(filters) if filters else ""})
            )
            """
            joins.append(f"JOIN agg_result_{inner_alias_name} ON {outer_table_name}_outer.{outer_col} {outer_op_str} agg_result_{inner_alias_name}.{agg_col}")

        joins.extend(join_clause_inner)
        pattern = re.compile(rf"\b{re.escape(inner_alias_name)}\.(\w+)\b")
        replacement = rf"agg_result_{inner_alias_name}.\1"
        fixed = []
        for join_stmt in joins:
            if pattern.search(join_stmt):
                new_stmt = pattern.sub(replacement, join_stmt)
                fixed.append(new_stmt)
            else:
                fixed.append(join_stmt)
        joins = fixed
        
        # For phase1
        cte_parts.append(cte)
        tuid_sources.append(f"agg_result_{inner_alias_name}.tuid")
        
        # For phase2
        readlog_args.append(f"{inner_alias_name}.tuid")
        from_tables_p2.append(f"{inner_table_name}_2 AS {inner_alias_name}")
        for table_name, alias in join_tables:
            readlog_args.append(f"{alias}.tuid")
            from_tables_p2.append(f"{table_name}_2 AS {alias}")
            tuid_sources.append(f"{alias}.tuid") # for phase1
        why = f"toY({outer_table_name}_outer.{outer_col}) || toY({inner_alias_name}.{agg_col})"
        for c in filter_cols:
            why += f" || toY({replace_alias(c)})"
        for c in join_columns:
            why += f" || toY({c})" 
        whys.append(why)
        
    # Non Subquery Filter
    rewritten_exprs = []
    filter_cols = []
    where_clause_p1 = ""
    for nonsub_expr in nonsub_exprs:
        expr_copy = copy.deepcopy(nonsub_expr)
        for col in expr_copy.find_all(exp.Column):
            table = col.table
            column = col.name
            if table in outer_table_names:
                col.set("this", exp.to_identifier(column))
                col.set("table", exp.to_identifier(table + "_outer"))
                filter_cols.append(f"{table}_outer.{column}")
        rewritten_exprs.append(expr_copy.sql())
    if rewritten_exprs:    
        if rewritten_exprs:
            where_clause_p1 = " WHERE " + " AND ".join(rewritten_exprs)
        # For phase2
        if filter_cols:
            why = " || ".join(f"toY({col})" for col in filter_cols)
        whys.append(why)
                
        
    # Final query assembly
    # -- Phase 1
    location = 0
    writelog_args = ", ".join(tuid_sources)
    ctes = ',\n'.join(cte_parts) + (',\n' if cte_parts else '')
    join_tables = '\n      '.join(joins)
    select_cols = f"{select_table}_outer.{select_column}"
    outer_tables_sorted = sorted(outer_table_names)
    base_from = f"{outer_tables_sorted[0]}_1 AS {outer_tables_sorted[0]}_outer"    
    join_section = "\n".join([f"JOIN {tbl}_1 AS {tbl}_outer" for tbl in outer_tables_sorted[1:]] + [f"ON {cond}" for cond in outer_join_conditions]) if len(outer_tables_sorted) > 1 else ""
    p1 = f"""
    WITH {ctes}
     filtered_{select_table}(tuid, {select_column}) AS (
      SELECT
        writelog({location}, {writelog_args}) AS tuid,
        {select_cols}
      FROM {base_from}
      {join_section} {join_tables}
      {where_clause_p1}
    )
    SELECT * FROM filtered_{select_table};
    """.strip()
    
    # -- Phase 2
    prov_cols = f"{select_table}_outer.{select_column} || wh.y AS {select_column}"
    dd_cols = f"dd(f.{select_column}) AS {select_column}"
    readlog_str = f"readlog({location}, {', '.join(readlog_args)}) AS log(tuid)"
    from_tables_str = ',\n        '.join(from_tables_p2)
    whys_str = ' || '.join(whys)
    p2 = f"""
    WITH filtered_{select_table}(tuid, {select_column}) AS (
      SELECT
        log.tuid,
        {prov_cols}
      FROM {from_tables_str},
        {readlog_str},
        LATERAL (SELECT {whys_str}) AS wh(y)
    )
    SELECT
      f.tuid,
      {dd_cols}
    FROM filtered_{select_table} AS f;
    """.strip()

    return p1, p2

def extract_filters(expr, outer_table_names, inner_table_name, inner_table_alias):
    filter_exprs, correlation_exprs = [], []
    filter_cols, correlation_cols = [], []
    if isinstance(expr, exp.And):
        f_exprs1, c_exprs1, f_cols1, c_cols1 = extract_filters(expr.left, outer_table_names, inner_table_name, inner_table_alias)
        f_exprs2, c_exprs2, f_cols2, c_cols2 = extract_filters(expr.right, outer_table_names, inner_table_name, inner_table_alias)
        return (
            f_exprs1 + f_exprs2,
            c_exprs1 + c_exprs2,
            f_cols1 + f_cols2,
            c_cols1 + c_cols2
        )
    if isinstance(expr, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Exists)):
        cols_updated = []
        expr_copy = copy.deepcopy(expr)
        for col in expr_copy.find_all(exp.Column):
            table = col.table
            column = col.name
            if table in outer_table_names and table not in [inner_table_name, inner_table_alias]: # correlation
                col.set("this", exp.to_identifier(column))
                col.set("table", exp.to_identifier(table + "_outer"))
                cols_updated.append(f"{table}_outer.{column}")    
            elif table in [inner_table_name, inner_table_alias]:
                col.set("this", exp.to_identifier(column))
                col.set("table", exp.to_identifier(inner_table_name + "_inner"))
                cols_updated.append(f"{inner_table_name}_inner.{column}")
        modified_sql = expr_copy.sql()

        if any(any(t + "_outer" in col for t in outer_table_names) for col in cols_updated): # correlation
            correlation_exprs.append(modified_sql)
            correlation_cols.extend(cols_updated)
        else:
            filter_exprs.append(modified_sql)  # table.col = table.col
            filter_cols.extend(cols_updated) # [table.col, table.col]

    return filter_exprs, correlation_exprs, filter_cols, correlation_cols

def extract_outer_table_name_from_subquery(inner_select):
    joins_in_subquery = inner_select.args.get("joins") or []
    for join in joins_in_subquery:
        table_expr = join.this
        if isinstance(table_expr, exp.Table):
            table_name = table_expr.name 
            alias = table_expr.alias or ""
            if "_outer" in alias: 
                return table_name
    return None
        
def processing_join_in_subquery(inner_select, inner_table_alias, inner_table_name, inner_alias_name, joins, filters):
    joins_in_subquery = inner_select.args.get("joins") or []
    join_tables = []
    join_columns = []
    join_clause_inner = []
    
    main_key   = inner_table_alias or inner_table_name
    main_alias = inner_alias_name
    
    for join in joins_in_subquery:
        table_expr = join.this
        if isinstance(table_expr, exp.Table):
            table_name = table_expr.name    
            table_alias = table_expr.alias
        
        join_condition = join.args.get("on")
        if not join_condition:
            continue
        
        join_key = table_alias or table_name
        if inner_table_alias not in joins and inner_table_name == table_name:
            base_join_stmt = f"JOIN {table_name}_1 AS {table_name}_inner"
            base_join_stmt_agg = f"JOIN agg_result_{table_name}"
            existing_alias_count = sum(1 for j in joins if base_join_stmt in j or base_join_stmt_agg in j) + 1
            join_alias = f"{table_name}_inner_{existing_alias_count}"
            join_replace_alias = lambda s: s.replace(join_key, join_alias)
            filters = [join_replace_alias(f) for f in filters]
        else:
            join_alias, join_replace_alias = generate_unique_inner_alias(table_name, joins)
            filters = [join_replace_alias(f) for f in filters]
            
        replacements = {
            main_key: main_alias,
            join_key: join_alias
        }
        join_tables.append((table_name, join_alias))
        condition_str = join_condition.sql()   
        
        for old, new in replacements.items():
            pattern     = rf'\b{re.escape(old)}\.(\w+)\b'
            replacement = rf'{new}.\1'
            condition_str = re.sub(pattern, replacement, condition_str)
        
        join_stmt = f"JOIN {table_name}_1 AS {join_alias} ON {condition_str}"
        join_clause_inner.append(join_stmt)
        
        parts = [p.strip() for p in condition_str.split('=')]
        join_columns.extend(parts)
    
    return join_tables, join_columns, join_clause_inner, filters

def generate_unique_inner_alias(inner_table_name: str, joins: list):
    """
    Generate a unique alias for an inner table based on existing join statements.
    
    Parameters:
        inner_table_name (str): The name of the inner table.
        joins (list): The list of existing join statements.
    
    Returns:
        tuple: (inner_alias_name, replace_alias) where:
            - inner_alias_name (str): A unique alias for the inner table (e.g., "table_inner_1").
            - replace_alias (function): A lambda function that replaces the base alias with the unique alias in strings.
    """
    base_join_stmt = f"JOIN {inner_table_name}_1 AS {inner_table_name}_inner"
    base_join_stmt_agg = f"JOIN agg_result_{inner_table_name}"
    existing_alias_count = sum(1 for j in joins if base_join_stmt in j or base_join_stmt_agg in j)
    inner_alias_name = f"{inner_table_name}_inner_{existing_alias_count}"
    replace_alias = lambda s: s.replace(f"{inner_table_name}_inner", inner_alias_name)
    
    return inner_alias_name, replace_alias


def is_nested_query(sql: str) -> bool:
    ast = parse_one(sql)
    nested_expr_types = (exp.Subquery, exp.Exists)
    return any(isinstance(node, nested_expr_types) for node in ast.walk())


def rewrite_query(sql: str, deleted_columns=[], outer_col="") -> tuple[str, str]:

    # change the projected column of the sql to the column in the deleted_predicate.
    if deleted_columns == [] and outer_col == "":
        revised_sql = sql
    elif outer_col == "":
        revised_sql = revise_select_clause(sql, deleted_columns)
    else:
        revised_sql = revise_select_clause(sql, [outer_col])
    
    if is_nested_query(revised_sql):
        return rewrite_nested_query(revised_sql)
    else:
        ast = parse_one(revised_sql)
        if ast.args.get("group"):
            return rewrite_non_nested_query_with_groupby(revised_sql)
        else:
            return rewrite_non_nested_query(revised_sql)

def revise_select_clause(subquery: str, column_names: list):
    parsed = parse_one(subquery)
    select_node = parsed.find(exp.Select)
    new_select_expressions = []
    for column_name in column_names:
        if "." in column_name:
            table, col = column_name.split(".", 1)
            new_projection = exp.Column(
                this=exp.Identifier(this=col),
                table=exp.Identifier(this=table)
            )
        else:
            new_projection = exp.Column(this=exp.Identifier(this=column_name))
        new_select_expressions.append(new_projection)
    select_node.set("expressions", new_select_expressions)
    return parsed.sql()