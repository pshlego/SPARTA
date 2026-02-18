import re
import copy
import random
import sqlglot
from sqlglot import exp, parse_one
from src.benchmark.sparta.QueryGeneration.methods.query_rewriting import (
    rewrite_query,
)


class TableCellHighlighter:
    def __init__(self, data_manager, rng):
        self.data_manager = data_manager
        self.rng = rng
        self.missing_tables = []

    def get_database_representation(
        self, sample_rows, sample_schema, is_outer_table=False
    ):
        table_representation = ""
        columns_text = " | ".join([col for col in sample_schema])
        table_representation += f"col : {columns_text}\n"
        for row_id, sample_row in enumerate(sample_rows):
            sample_row_text = " | ".join(
                [str(sample_row[i]) for i in range(len(sample_row))]
            )
            if row_id == 0 and is_outer_table:
                table_representation += f"row {row_id + 1} : {sample_row_text} **VERY IMPORTANT** Please add an additional predicate in the subquery Q to include this important row.\n"
            else:
                table_representation += f"row {row_id + 1} : {sample_row_text}\n"
        return table_representation

    def generate_selected_table_representation(
        self,
        selected_cell_ids,
        projected_cell_ids,
        cause_str,
        projected_subquery,
        outer_table,
        inner_table,
        projected_column_of_inner_query,
        deleted_columns_of_inner_query,
        operator,
        aggregate_function,
        deleted_subquery_str,
        nested_predicate_type,
    ):
        selected_cell_ids = set(selected_cell_ids)
        projected_cell_ids = set(projected_cell_ids)
        schema_query = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{outer_table}'
            ORDER BY ordinal_position
        """
        self.data_manager.execute(schema_query)
        sample_schema = [row[0] for row in self.data_manager.fetchall()]

        self.data_manager.execute(f"SELECT * FROM {outer_table}")
        sample_rows = self.data_manager.fetchall()

        self.data_manager.execute(f"SELECT * FROM {outer_table}_2")
        cell_id_rows = self.data_manager.fetchall()

        updated_rows = []
        selected_row_ids = set()
        used_column_ids = set()

        col_val_list = []

        for row_idx, (data_row, id_row) in enumerate(zip(sample_rows, cell_id_rows)):

            new_row = []
            for col_idx, (val, id_info) in enumerate(zip(data_row, id_row[1:])):
                cell_id = id_info[0]
                sel = cell_id in selected_cell_ids
                proj = cell_id in projected_cell_ids

                if sel and proj:
                    new_row.append(f"{val} (selected & filtered out by '{cause_str}').")
                    selected_row_ids.add(row_idx)
                    used_column_ids.add(col_idx)
                    col_val_list.append((row_idx, val))
                elif proj:
                    new_row.append(f"{val} (filtered out by '{cause_str}').")
                    selected_row_ids.add(row_idx)
                    used_column_ids.add(col_idx)
                    col_val_list.append((row_idx, val))
                elif sel:
                    new_row.append(f"{val} (selected)")
                    used_column_ids.add(col_idx)
                else:
                    new_row.append(val)
            updated_rows.append(new_row)

        if nested_predicate_type in ["type-a", "type-ja"]:
            selected_indices = self.sample_row_indices_with_max_min(col_val_list, 5)

        else:
            if isinstance(
                [val for row_idx, val in col_val_list if val is not None][0], str
            ):
                filtered_row_values = list(
                    set(
                        [
                            f"""'{val.replace("'", "''")}'"""
                            for row_idx, val in col_val_list
                            if val is not None
                        ]
                    )
                )
            else:
                filtered_row_values = list(
                    set([f"{val}" for row_idx, val in col_val_list if val is not None])
                )
            inner_projected_query = f"""
                SELECT {projected_column_of_inner_query}
                FROM {inner_table}
                WHERE {projected_column_of_inner_query} {operator} ({', '.join(filtered_row_values)})
            """

            self.data_manager.execute(inner_projected_query)
            inner_rows = self.data_manager.fetchall()
            inner_values = [row[0] for row in inner_rows]
            if inner_values == []:
                self.missing_tables.append(inner_table)
                return None
            else:
                selected_indices = self.sample_row_indices_with_max_min(
                    col_val_list, 5, inner_values=inner_values
                )

        used_column_ids = sorted(used_column_ids)
        final_rows = [updated_rows[i] for i in selected_indices]
        final_rows = [
            [updated_rows[i][j] for j in used_column_ids] for i in selected_indices
        ]
        updated_schema = sample_schema
        updated_schema = [sample_schema[j] for j in used_column_ids]
        outer_table_representation = self.get_database_representation(
            final_rows, updated_schema, is_outer_table=True
        )
        outer_table_representation = (
            f"Table Name: {outer_table}\n{outer_table_representation}"
        )
        inner_projected_columns = [projected_column_of_inner_query]
        for deleted_column in deleted_columns_of_inner_query:
            if deleted_column.split(".")[-1] not in inner_projected_columns:
                inner_projected_columns.append(deleted_column.split(".")[-1])

        if nested_predicate_type in ["type-a", "type-ja"]:
            ast = parse_one(deleted_subquery_str)
            ast.set("expressions", inner_projected_columns)
            if nested_predicate_type == "type-ja":
                where_node = ast.args.get("where")
                if where_node:
                    join_pred, updated_pred = self.remove_join_predicate(where_node)
                    if updated_pred is not None:
                        ast.set("where", exp.Where(this=updated_pred))
                    else:
                        del ast.args["where"]
            inner_info_query = str(ast)

        else:
            if isinstance(
                [val for row_idx, val in col_val_list if val is not None][0], str
            ):
                filtered_row_values = list(
                    set(
                        [
                            f"""'{val.replace("'", "''")}'"""
                            for row_idx, val in col_val_list
                            if row_idx in selected_indices and val is not None
                        ]
                    )
                )
            else:
                filtered_row_values = list(
                    set(
                        [
                            f"{val}"
                            for row_idx, val in col_val_list
                            if row_idx in selected_indices and val is not None
                        ]
                    )
                )

            inner_info_query = f"""
                SELECT {', '.join(inner_projected_columns)}
                FROM {inner_table}
                WHERE {projected_column_of_inner_query} {operator} ({', '.join(filtered_row_values)})
            """

        self.data_manager.execute(inner_info_query)
        sampled_rows = self.data_manager.fetchall()
        inner_col_val_list = [
            (row_idx, row[0]) for row_idx, row in enumerate(sampled_rows)
        ]

        sampled_indices = self.sample_row_indices_with_max_min(inner_col_val_list, 5)
        filtered_rows = []
        if nested_predicate_type in ["type-a", "type-ja"]:
            for row_idx, row_values in enumerate(sampled_rows):
                if len(row_values) == 1:
                    if row_idx in sampled_indices:
                        if (
                            row_idx == sampled_indices[0]
                            and aggregate_function.lower() == "max"
                            and operator in [">", ">="]
                            and nested_predicate_type == "type-a"
                        ):
                            filtered_rows.append(
                                [
                                    f"{row_value} (add a predicate to filter this out)"
                                    for row_value in row_values
                                ]
                            )
                        elif (
                            row_idx == sampled_indices[0]
                            and aggregate_function.lower() == "max"
                            and nested_predicate_type == "type-ja"
                        ):
                            filtered_rows.append(
                                [
                                    f"{row_value} (revise the logical operator from '> (Q)' to '>= (Q)')"
                                    for row_value in row_values
                                ]
                            )
                        else:
                            filtered_rows.append(
                                [f"{row_value} (selected)" for row_value in row_values]
                            )
                else:
                    if (
                        row_idx == sampled_indices[0]
                        and aggregate_function.lower() == "max"
                        and operator in [">", ">="]
                        and nested_predicate_type == "type-a"
                    ):
                        filtered_rows.append(
                            [f"{row_values[0]} (add a predicate to filter this out)"]
                            + [f"{row_value}" for row_value in row_values[1:]]
                        )
                    elif (
                        row_idx == sampled_indices[0]
                        and aggregate_function.lower() == "max"
                        and nested_predicate_type == "type-ja"
                    ):
                        filtered_rows.append(
                            [
                                f"{row_values[0]} (revise the logical operator from '> (Q)' to '>= (Q)')"
                            ]
                            + [f"{row_value}" for row_value in row_values[1:]]
                        )
                    elif row_idx in sampled_indices[2:]:
                        filtered_rows.append(
                            [f"{row_value}" for row_value in row_values]
                        )

        else:
            filtered_rows = [
                [row_values[0]]
                + [
                    f"{row_value} (filtered out by the WHERE clause of Q)"
                    for row_value in row_values[1:]
                ]
                for row_idx, row_values in enumerate(sampled_rows)
                if row_idx in sampled_indices
            ]

        inner_table_representation = self.get_database_representation(
            filtered_rows, inner_projected_columns
        )
        inner_table_representation = (
            f"Table Name: {inner_table}\n{inner_table_representation}"
        )

        return outer_table_representation + "\n\n" + inner_table_representation

    def sample_row_indices_with_max_min(
        self, col_val_list, n, seed=None, inner_values=[]
    ):
        if seed is not None:
            random.seed(seed)

        if not col_val_list or len(col_val_list) == 0:
            return []

        if inner_values != []:
            col_val_list = [
                x for x in col_val_list if x[1] is not None and x[1] in inner_values
            ]
        else:
            col_val_list = [x for x in col_val_list if x[1] is not None]

        max_row = max(col_val_list, key=lambda x: x[1])
        min_row = min(col_val_list, key=lambda x: x[1])

        max_idx = max_row[0]
        min_idx = min_row[0]

        remaining = [
            row_idx
            for row_idx, val in col_val_list
            if row_idx != max_idx and row_idx != min_idx
        ]

        sampled = random.sample(remaining, min(n, len(remaining)))

        return [max_idx, min_idx] + sampled

    def remove_join_predicate(self, where_node):
        if not where_node or not where_node.this:
            return None, None

        pred = where_node.this

        if (
            isinstance(pred, exp.EQ)
            and isinstance(pred.this, exp.Column)
            and isinstance(pred.expression, exp.Column)
        ):
            return pred, None

        if not isinstance(pred, exp.And):
            return None, pred

        def recurse_remove_join_predicate(expr):
            if not isinstance(expr, exp.And):
                if (
                    isinstance(expr, exp.EQ)
                    and isinstance(expr.this, exp.Column)
                    and isinstance(expr.expression, exp.Column)
                ):
                    return expr, None
                return None, expr

            left = expr.args.get("this")
            right = expr.args.get("expression")

            removed, new_right = recurse_remove_join_predicate(right)
            if removed:
                if new_right is None:
                    return removed, left
                new_and = expr.copy()
                new_and.set("expression", new_right)
                return removed, new_and

            removed, new_left = recurse_remove_join_predicate(left)
            if removed:
                if new_left is None:
                    return removed, right
                new_and = expr.copy()
                new_and.set("this", new_left)
                return removed, new_and

            return None, expr

        removed_pred, updated_expr = recurse_remove_join_predicate(pred)
        return removed_pred, updated_expr if updated_expr is not None else None


class ProvQueryFixer:
    def __init__(self, data_manager, rng):
        self.data_manager = data_manager
        self.highlighter = TableCellHighlighter(data_manager, rng)
        self.rng = rng
        self.p0_sql = """
        truncate table log0;
        truncate table log1;
        truncate table log2;
        truncate table log3;
        """

    def fix_query(self, query, nested_predicate_type=""):

        subquery, removed_preds = self.find_non_empty_subquery(query)
        diff = self.diff_sql_parts(query, subquery)
        if diff == {}:
            return f"Fixed query which has execution results: {subquery}", "", "", ""

        deleted_level = min([removed_pred[-1] for removed_pred in removed_preds])
        deleted_columns_of_inner_query = [
            removed_pred[1]
            for removed_pred in removed_preds
            if removed_pred[-1] == deleted_level
        ]
        target_predicate = diff["WHERE"][deleted_level - 2]
        outer_col, operator, deleted_subquery_str = (
            self.extract_outer_column_and_subquery(target_predicate)
        )

        aggregate_function = None
        inner_table = None
        projected_column_of_inner_query = None
        execution_result_str = ""
        try:
            self.data_manager.execute(deleted_subquery_str)
            deleted_subquery_execution_results = self.data_manager.fetchall()
            deleted_subquery_execution_results = [
                row[0] for row in deleted_subquery_execution_results
            ]
            if len(deleted_subquery_execution_results) == 1:
                cause_str = f"{outer_col} {operator} (Execution result of Q)"
                execution_result_str = deleted_subquery_execution_results[0]
            else:
                cause_str = f"{outer_col} {operator} (Execution results of Q)"
                execution_result_str = ", ".join(deleted_subquery_execution_results)
            inner_table = self.get_accessed_tables(deleted_subquery_str)[0]
            if nested_predicate_type not in ["type-a", "type-ja"]:
                projected_column_of_inner_query = parse_one(
                    deleted_subquery_str
                ).named_selects[0]
            else:
                projected_column_of_inner_query = str(
                    parse_one(deleted_subquery_str).selects[0].this.this
                )
                aggregate_function = parse_one(deleted_subquery_str).expressions[0].key
        except Exception as e:
            cause_str = f"{outer_col} {operator} ({deleted_subquery_str})"
            try:
                inner_table = self.get_accessed_tables(deleted_subquery_str)[0]
            except Exception as e:
                inner_table = None
            try:
                if nested_predicate_type not in ["type-a", "type-ja"]:
                    projected_column_of_inner_query = parse_one(
                        deleted_subquery_str
                    ).named_selects[0]
                else:
                    projected_column_of_inner_query = str(
                        parse_one(deleted_subquery_str).selects[0].this.this
                    )
                    aggregate_function = (
                        parse_one(deleted_subquery_str).expressions[0].key
                    )
            except Exception as e:
                projected_column_of_inner_query = None
                aggregate_function = None

        if outer_col is None:
            return (
                f"Fixed query which has execution results: {subquery}",
                target_predicate,
                deleted_subquery_str,
                execution_result_str,
            )

        projected_subquery = self.project_only_column(subquery, outer_col)
        outer_table = self.get_accessed_tables(projected_subquery)[0]
        p1_sql, p2_sql = rewrite_query(projected_subquery)
        try:
            self.data_manager.execute(self.p0_sql)
            p1_sql = p1_sql.replace("{'", "").replace("'}", "")
            self.data_manager.execute(p1_sql)
            p1_results = self.data_manager.fetchall()
            p2_sql = (
                p2_sql.replace("_outer_outer", "_outer")
                .replace("{'", "")
                .replace("'}", "")
            )
            self.data_manager.execute(p2_sql)
            p2_results = self.data_manager.fetchall()
            selected_cell_ids = []
            projected_cell_ids = []
            for tuple_id, cell_ids in p2_results:
                for cell_id in cell_ids:
                    if cell_id < 0:
                        selected_cell_ids.append(abs(cell_id))
                    else:
                        projected_cell_ids.append(cell_id)

            result_representation = (
                self.highlighter.generate_selected_table_representation(
                    selected_cell_ids,
                    projected_cell_ids,
                    cause_str,
                    projected_subquery,
                    outer_table,
                    inner_table,
                    projected_column_of_inner_query,
                    deleted_columns_of_inner_query,
                    operator,
                    aggregate_function,
                    deleted_subquery_str,
                    nested_predicate_type,
                )
            )
            if result_representation is None:
                return (
                    f"Fixed query which has execution results: {subquery}",
                    target_predicate,
                    deleted_subquery_str,
                    execution_result_str,
                )
            return (
                result_representation,
                target_predicate,
                deleted_subquery_str,
                execution_result_str,
            )
        except Exception as e:
            return (
                f"Fixed query which has execution results: {subquery}",
                target_predicate,
                deleted_subquery_str,
                execution_result_str,
            )

    def find_non_empty_subquery(self, query):
        execution_result_rows, execution_result_schema = (
            self.data_manager.sample_rows_with_sql(str(query), 10, self.rng)
        )
        if len(execution_result_rows) != 0:
            return query, []

        parsed = sqlglot.parse_one(query, read="mysql")
        if not parsed:
            raise ValueError("Could not parse SQL query")

        removed_predicates = []

        still_empty = True
        while still_empty:
            target_select, target_depth = self.get_rightmost_select(parsed)
            if not target_select:
                break

            where_expr = target_select.args.get("where")
            removed_pred, updated_pred = self.remove_rightmost_predicate(where_expr)
            if removed_pred is None:
                changed = self.remove_subquery_predicate_if_where_empty(parsed)
                if changed:
                    execution_result_rows, execution_result_schema = (
                        self.data_manager.sample_rows_with_sql(
                            str(parsed), 10, self.rng
                        )
                    )
                    if len(execution_result_rows) != 0:
                        removed_predicates.append(("", "", target_depth))
                        break

            if removed_pred is not None:
                if isinstance(removed_pred, exp.Where):
                    if isinstance(removed_pred.this.this, exp.Column):
                        column_name = str(removed_pred.this.this)
                    elif isinstance(removed_pred.this.expression, exp.Column):
                        column_name = str(removed_pred.this.expression)
                    else:
                        column_name = str(removed_pred.this)
                    removed_predicates.append(
                        (str(removed_pred.this), column_name, target_depth)
                    )
                else:
                    if isinstance(removed_pred.this, exp.Column):
                        column_name = str(removed_pred.this)
                    elif isinstance(removed_pred.expression, exp.Column):
                        column_name = str(removed_pred.expression)
                    else:
                        column_name = str(removed_pred)
                    removed_predicates.append(
                        (str(removed_pred), column_name, target_depth)
                    )

            if updated_pred is None:
                try:
                    del target_select.args["where"]
                except:
                    pass
            else:
                target_select.set("where", exp.Where(this=updated_pred))

            execution_result_rows, execution_result_schema = (
                self.data_manager.sample_rows_with_sql(str(parsed), 10, self.rng)
            )
            if len(execution_result_rows) != 0:
                changed = self.remove_subquery_predicate_if_where_empty(parsed)
                break

            changed = self.remove_subquery_predicate_if_where_empty(parsed)
            if changed:
                execution_result_rows, execution_result_schema = (
                    self.data_manager.sample_rows_with_sql(str(parsed), 10, self.rng)
                )
                if len(execution_result_rows) != 0:
                    break

        final_query = parsed.sql(dialect="mysql")
        return final_query, removed_predicates

    def get_rightmost_select(self, expression):
        selects_with_depth = []

        def _visit(node, depth):
            if depth <= 2:
                if isinstance(node, exp.Select):
                    selects_with_depth.append((node, depth))

                for child in node.args.values():
                    if isinstance(child, list):
                        for sub in child:
                            if isinstance(sub, exp.Expression):
                                _visit(
                                    sub,
                                    (
                                        depth + 1
                                        if isinstance(node, exp.Select)
                                        else depth
                                    ),
                                )
                    elif isinstance(child, exp.Expression):
                        _visit(
                            child, depth + 1 if isinstance(node, exp.Select) else depth
                        )

        _visit(expression, 1)

        return selects_with_depth[-1] if selects_with_depth else (None, None)

    def remove_rightmost_predicate(self, where_node):
        if not where_node:
            return None, None

        pred = where_node.this
        if not pred:
            return None, where_node.this

        if not isinstance(pred, exp.And):
            return where_node, None

        left_side = pred.args.get("this")
        right_side = pred.args.get("expression")

        if (
            right_side
            and not isinstance(right_side, exp.And)
            and not (
                isinstance(right_side, exp.EQ)
                and isinstance(right_side.this, exp.Column)
                and isinstance(right_side.expression, exp.Column)
            )
        ):
            removed_pred = right_side
            return removed_pred, left_side

        if right_side and isinstance(right_side, exp.And):
            temp_where = exp.Where(this=right_side)
            removed_pred, new_right_expr = self.remove_rightmost_predicate(temp_where)
            if removed_pred is None:
                return None, pred
            new_and = pred.copy()
            new_and.set("expression", new_right_expr)
            return removed_pred, new_and

        if (
            right_side
            and isinstance(right_side, exp.EQ)
            and isinstance(right_side.this, exp.Column)
            and isinstance(right_side.expression, exp.Column)
        ):
            temp_where = exp.Where(this=exp.And(this=right_side, expression=left_side))
            removed_pred, new_right_expr = self.remove_rightmost_predicate(temp_where)
            if removed_pred is None:
                return None, pred
            return removed_pred, new_right_expr

        return None, pred

    def remove_subquery_predicate_if_where_empty(self, parsed_ast):
        removed_any = False

        def _remove_empty_where_clauses(node):
            for key, child in list(node.args.items()):
                if isinstance(child, list):
                    for sub in child:
                        if isinstance(sub, exp.Expression):
                            _remove_empty_where_clauses(sub)
                elif isinstance(child, exp.Expression):
                    _remove_empty_where_clauses(child)

            if isinstance(node, exp.Select):
                where_node = node.args.get("where")
                if where_node and not where_node.this:
                    del node.args["where"]

        def _remove_expression_from_parent(expr_node, parent, parent_key):
            """
            Remove 'expr_node' from its parent's boolean expression if possible.
            E.g., if parent is AND(A, expr_node), remove expr_node => becomes A.
            """
            if not parent or not parent_key:
                return

            if (
                isinstance(parent, exp.And)
                or isinstance(parent, exp.Or)
                or isinstance(parent, exp.Not)
            ):
                left = parent.args.get("this")
                right = parent.args.get("expression")

                if left is expr_node:
                    del parent.args["this"]
                elif right is expr_node:
                    del parent.args["expression"]

            elif isinstance(parent, exp.Where):
                if parent.this is expr_node:
                    parent.set("this", None)
            else:
                pass

        def _cleanup_incomplete_ands(node, parent=None, parent_key=None):
            for key, child in list(node.args.items()):
                if isinstance(child, list):
                    for idx, sub in enumerate(child):
                        if isinstance(sub, exp.Expression):
                            _cleanup_incomplete_ands(sub, node, (key, idx))
                elif isinstance(child, exp.Expression):
                    _cleanup_incomplete_ands(child, node, key)

            if (
                isinstance(node, exp.And)
                or isinstance(node, exp.Or)
                or isinstance(node, exp.Not)
            ):
                left = node.args.get("this")
                right = node.args.get("expression")

                if left and not right:
                    _replace_node_in_parent(node, left, parent, parent_key)
                elif right and not left:
                    _replace_node_in_parent(node, right, parent, parent_key)
                elif not left and not right:
                    _replace_node_in_parent(node, None, parent, parent_key)

        def _replace_node_in_parent(old_node, new_node, parent, parent_key):
            if not parent or not parent_key:
                return

            if isinstance(parent_key, tuple):
                key_name, idx = parent_key
                if isinstance(parent.args.get(key_name), list):
                    siblings = parent.args[key_name]
                    if idx < len(siblings) and siblings[idx] is old_node:
                        if new_node is None:
                            siblings.pop(idx)
                        else:
                            siblings[idx] = new_node
            else:
                child_val = parent.args.get(parent_key)
                if child_val is old_node:
                    if new_node is None:
                        del parent.args[parent_key]
                    else:
                        parent.set(parent_key, new_node)

        def _visit(node, parent=None, parent_key=None):
            for key, child in list(node.args.items()):
                if isinstance(child, list):
                    for idx, sub in enumerate(child):
                        if isinstance(sub, exp.Expression):
                            _visit(sub, node, (key, idx))
                elif isinstance(child, exp.Expression):
                    _visit(child, node, key)

            if isinstance(
                node, (exp.In, exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Exists)
            ):
                rhs = (
                    node.args.get("expression")
                    or node.args.get("query")
                    or node.args.get("expressions")
                )
                if isinstance(rhs, exp.Subquery):
                    sub_select = rhs.this
                    if isinstance(sub_select, exp.Select):
                        if (
                            "where" not in sub_select.args
                            or sub_select.args["where"] is None
                        ):
                            if not (
                                len(str(parent).split(" AND ")) > 1
                                and str(parent).split(" AND ")[0] == str(node)
                            ):
                                _remove_expression_from_parent(node, parent, parent_key)
                                nonlocal removed_any
                                removed_any = True

        _visit(parsed_ast)

        _remove_empty_where_clauses(parsed_ast)
        _cleanup_incomplete_ands(parsed_ast)
        return removed_any

    def diff_sql_parts(self, sql_a: str, sql_b: str) -> dict:
        a_expr = parse_one(sql_a)
        b_expr = parse_one(sql_b)
        result = {}

        def extract_expressions(expr_list):
            return [str(e).strip() for e in expr_list] if expr_list else []

        def extract_from(expr):
            return [str(e).strip() for e in expr.expressions] if expr else []

        def extract_joins(expr):
            return [str(j).strip() for j in expr.find_all(exp.Join)]

        def collect_all_conditions(expr):
            if expr is None:
                return []

            conditions = []

            if isinstance(
                expr,
                (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Exists),
            ):
                conditions.append(expr)

                for arg in expr.args.values():
                    if isinstance(arg, (exp.Expression, exp.Select, exp.Subquery)):
                        conditions += collect_all_conditions(arg)
                    elif isinstance(arg, list):
                        for e in arg:
                            if isinstance(e, exp.Expression):
                                conditions += collect_all_conditions(e)

            elif isinstance(expr, (exp.And, exp.Or, exp.Paren)):
                for key in ["this", "expression", "left", "right"]:
                    sub = expr.args.get(key)
                    if sub:
                        conditions += collect_all_conditions(sub)

            elif isinstance(expr, exp.Select):
                for key in ["where", "having"]:
                    sub = expr.args.get(key)
                    if sub:
                        conditions += collect_all_conditions(sub)

            elif isinstance(expr, exp.Subquery):
                if expr.args.get("this"):
                    conditions += collect_all_conditions(expr.args["this"])

            elif isinstance(expr, exp.Expression):
                for arg in expr.args.values():
                    if isinstance(arg, (exp.Expression, exp.Select, exp.Subquery)):
                        conditions += collect_all_conditions(arg)
                    elif isinstance(arg, list):
                        for e in arg:
                            if isinstance(e, exp.Expression):
                                conditions += collect_all_conditions(e)

            return conditions

        def is_equivalent(e1, e2):
            return e1 == e2 or str(e1).strip() == str(e2).strip()

        def find_missing_conditions(expr_a, expr_b):
            a_conds = collect_all_conditions(expr_a)
            b_conds = collect_all_conditions(expr_b) if expr_b else []

            missing = []
            for a in a_conds:
                if not any(is_equivalent(a, b) for b in b_conds):
                    missing.append(str(a).strip())

            return missing

        select_diff = set(extract_expressions(a_expr.args.get("expressions"))) - set(
            extract_expressions(b_expr.args.get("expressions"))
        )
        if select_diff:
            result["SELECT"] = list(select_diff)

        from_diff = set(extract_from(a_expr.args.get("from"))) - set(
            extract_from(b_expr.args.get("from"))
        )
        if from_diff:
            result["FROM"] = list(from_diff)

        join_diff = set(extract_joins(a_expr)) - set(extract_joins(b_expr))
        if join_diff:
            result["JOIN"] = list(join_diff)

        where_diff = find_missing_conditions(
            a_expr.args.get("where"), b_expr.args.get("where")
        )
        if where_diff:
            result["WHERE"] = where_diff

        group_diff = set(extract_expressions(a_expr.args.get("group"))) - set(
            extract_expressions(b_expr.args.get("group"))
        )
        if group_diff:
            result["GROUP BY"] = list(group_diff)

        having_diff = find_missing_conditions(
            a_expr.args.get("having"), b_expr.args.get("having")
        )
        if having_diff:
            result["HAVING"] = having_diff

        order_diff = set(extract_expressions(a_expr.args.get("order"))) - set(
            extract_expressions(b_expr.args.get("order"))
        )
        if order_diff:
            result["ORDER BY"] = list(order_diff)

        a_limit = (
            str(a_expr.args.get("limit")).strip() if a_expr.args.get("limit") else None
        )
        b_limit = (
            str(b_expr.args.get("limit")).strip() if b_expr.args.get("limit") else None
        )
        if a_limit and a_limit != b_limit:
            result["LIMIT"] = [a_limit]

        a_ctes = [str(e).strip() for e in a_expr.find_all(exp.CTE)]
        b_ctes = [str(e).strip() for e in b_expr.find_all(exp.CTE)]
        with_diff = set(a_ctes) - set(b_ctes)
        if with_diff:
            result["WITH"] = list(with_diff)

        return result

    def extract_outer_column_and_subquery(self, predicate: str):
        expr = sqlglot.parse_one(predicate)

        if isinstance(expr, exp.In):
            outer_col = str(expr.args["this"])
            subquery_str = str(expr.args["query"])
        elif isinstance(expr, exp.Exists):
            outer_col = None
            subquery_str = str(expr.args["this"])
        else:
            outer_col = str(expr.args.get("this"))
            subquery_expr = expr.args.get("expression")
            subquery_str = str(subquery_expr) if subquery_expr else None

        operator_map = {
            "eq": "=",
            "neq": "!=",
            "gt": ">",
            "gte": ">=",
            "lt": "<",
            "lte": "<=",
            "in": "IN",
            "not_in": "NOT IN",
            "exists": "EXISTS",
            "not_exists": "NOT EXISTS",
        }

        operator_class_name = expr.__class__.__name__
        operator = operator_map.get(operator_class_name.lower(), operator_class_name)

        if subquery_str.startswith("(") and subquery_str.endswith(")"):
            subquery_str = subquery_str[1:-1]

        return outer_col, operator, subquery_str

    def project_only_column(self, sql_query: str, column_name: str) -> str:
        parsed = sqlglot.parse_one(sql_query)

        # Split table and column if dot notation is used
        if "." in column_name:
            table, col = column_name.split(".", 1)
            new_projection = exp.Column(
                this=exp.Identifier(this=col), table=exp.Identifier(this=table)
            )
        else:
            new_projection = exp.Column(this=exp.Identifier(this=column_name))

        # Replace the projection in the SELECT clause
        parsed.set("expressions", [new_projection])

        return str(parsed)

    def is_correlated_subquery_with_sqlglot(self, subquery_str):
        ast = sqlglot.parse_one(subquery_str)

        def extract_table_aliases_from_ast(ast):
            tables = set()

            from_clause = ast.args.get("from")
            if from_clause:
                main_table = from_clause.this
                if isinstance(main_table, exp.Table):
                    tables.add(main_table.alias_or_name)

            for join in ast.find_all(exp.Join):
                table_expr = join.this
                if isinstance(table_expr, exp.Table):
                    tables.add(table_expr.alias_or_name)

            return tables

        def extract_referenced_aliases_from_ast(ast):
            referenced_aliases = set()
            for column in ast.find_all(exp.Column):
                table_part = column.args.get("table")
                if table_part:
                    referenced_aliases.add(table_part.alias_or_name)
            return referenced_aliases

        inner_aliases = extract_table_aliases_from_ast(ast)
        referenced_aliases = extract_referenced_aliases_from_ast(ast)
        correlated_aliases = referenced_aliases - inner_aliases
        if len(correlated_aliases) == 1:
            correlated_alias = next(iter(correlated_aliases))
            return True, correlated_alias
        else:
            return (len(correlated_aliases) > 0), ""

    def fix_subquery(self, original_query: str) -> str:

        original_query = original_query.strip()
        if original_query.startswith("(") and original_query.endswith(")"):
            ast = sqlglot.parse_one(original_query)
            if isinstance(ast, exp.Subquery):
                original_query = ast.this.sql()

        is_cor, correlated_alias = self.is_correlated_subquery_with_sqlglot(
            original_query
        )
        if not is_cor:
            return original_query

        select_ast = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        where_expr = select_ast.args.get("where")
        correlation_expr = None
        remaining_predicates = []
        if where_expr:

            def collect_conditions(expr):
                if isinstance(expr, exp.And):
                    return collect_conditions(expr.left) + collect_conditions(
                        expr.right
                    )
                else:
                    return [expr]

            predicates = collect_conditions(where_expr.this)
            for pred in predicates:
                is_corr_pred = False
                for col in pred.find_all(exp.Column):
                    if col.table:
                        if col.table.upper() == correlated_alias.upper().strip():
                            is_corr_pred = True
                            correlation_expr = pred
                            break
                    if is_corr_pred:
                        break
                if not is_corr_pred:
                    remaining_predicates.append(pred.sql())
            if remaining_predicates:
                new_where_sql = " AND ".join(remaining_predicates)
            else:
                new_where_sql = ""
        else:
            new_where_sql = ""

        join_strings = []
        expr_copy = copy.deepcopy(correlation_expr)
        for col in expr_copy.find_all(exp.Column):
            table = col.table
            column = col.name

            if table == correlated_alias:
                col.set("this", exp.to_identifier(column))
                col.set("table", exp.to_identifier(table + "_outer"))
        correlation_expr_sql = expr_copy.sql()
        join_str = f" JOIN {correlated_alias} AS {correlated_alias}_outer ON {correlation_expr_sql} "
        join_strings.append(join_str)
        join_clause = " ".join(join_strings)

        from_clause = select_ast.args.get("from")
        primary_table_ast = from_clause.args.get("this")
        primary_table_sql = primary_table_ast.sql()
        pattern = re.compile(
            r"(.*\bFROM\b)(.*?)(\bWHERE\b.*)", re.IGNORECASE | re.DOTALL
        )
        m = pattern.search(original_query)
        if m:
            before_from = m.group(1)
            new_from_clause = " FROM " + primary_table_sql + " " + join_clause + " "
            before_from_split = before_from.split("FROM")[0]
            new_query = before_from_split + new_from_clause + " WHERE " + new_where_sql
        else:
            pattern2 = re.compile(r"(.*\bFROM\b)(.*)", re.IGNORECASE | re.DOTALL)
            m2 = pattern2.search(original_query)
            if m2:
                before_from = m2.group(1)
                new_from_clause = " FROM " + primary_table_sql + " " + join_clause + " "
                new_query = before_from.split("FROM")[0] + new_from_clause
                if new_where_sql:
                    new_query += " WHERE " + new_where_sql
            else:
                new_query = original_query

        return new_query

    def move_correlated_predicates_to_join(self, query):
        tree = sqlglot.parse_one(query)

        def collect_subqueries(expression, subqueries):
            if isinstance(expression, exp.Subquery):
                subqueries.append(expression)
            for arg_key, arg_value in expression.args.items():
                if isinstance(arg_value, list):
                    for item in arg_value:
                        collect_subqueries(item, subqueries)
                elif isinstance(arg_value, exp.Expression):
                    collect_subqueries(arg_value, subqueries)

        subqueries = []
        collect_subqueries(tree, subqueries)

        for sub in subqueries:
            fixed_subquery_str = self.fix_subquery(str(sub))
            fixed_ast = sqlglot.parse_one(fixed_subquery_str)
            fixed_subquery_expr = exp.Subquery(this=fixed_ast)
            sub.replace(fixed_subquery_expr)

        return str(tree)

    def get_accessed_tables(self, sql: str):
        parsed = sqlglot.parse_one(sql)
        if not parsed:
            return []
        from_expr = parsed.args.get("from")
        if not from_expr:
            return []
        tables = [table.name for table in from_expr.find_all(exp.Table)]
        return list(set(tables))
