system_from_prompt = """
You are both an Medical professional and an SQL expert. Given the database and the inner query block, generate a FROM clause of the outer query block that reflects authentic Medical-related curiosity.

Ensure the following requirements:
1) Output Structure: Return a JSON object containing a single key, "from", with its value being **a single-table** FROM clause of the outer query block from the provided database (i.e., do not include any sub-selects or nested queries directly in the FROM clause).
2) Ensure Medical professional Relevance: Generate the FROM clause of the outer query block that aligns naturally with realistic and meaningful **multi-hop** queries that Medical professionals are likely to ask.
3) Ensure Synthetic Correctness: Generate the FROM clause that is syntactically correct and executable on the provided database.
4) Separate Inner Query: The inner query block must remain separate (i.e., it should later be incorporated into the WHERE clause instead of being nested within the FROM clause).
5) Ensure Natural Connection: Choose an outer table whose columns can be naturally referenced or filtered against the results of the inner query block.

**IMPORTANT**: If the inner query block performs aggregation in the SELECT clause and no outer table includes the aggregated columns, reuse the table referenced in the inner query as the outer table.
"""

system_select_inner_query_block_prompt = """
You are both an Medical professional and an SQL expert. Given the provided database, generated clauses and the candidate inner query blocks, select the most appropriate inner query block for generating a nested predicate that reflects authentic Medical-related curiosity.

Select the most appropriate inner query block to generate a nested predicate that aligns naturally with realistic and meaningful **multi-hop** queries Medical professionals are likely to ask.

Your output must be in JSON format with the key:
- "inner_query_block": Select the most appropriate inner query block from the Candidate Inner Query Blocks.

**IMPORTANT**: Do not select the inner query block that has already been used in the generated clauses and not in the candidate inner query blocks.
"""

user_select_inner_query_block_prompt = """
Database: 
{schema}

Generated FROM Clause: 
{generated_from_clause}

Generated WHERE Clause: 
{generated_where_clause}

Candidate Inner Query Blocks:
{candidate_inner_query_blocks}

Return the results in a FLAT JSON format.

*DO NOT include any explanations or notes in the output. ONLY return JSON.*
"""


user_from_prompt = """
Database: 
{schema}

Inner Query Block: 
{subquery}

Return the results in a FLAT JSON format.

*DO NOT include any explanations or notes in the output. ONLY return JSON.*
"""

system_generate_nested_predicate_type_n_prompt = """
You are both an Medical professional and an SQL expert. Based on the given database, generated clauses, selected inner query block Q, and its execution result, generate a "type-n" nested predicate that reflects authentic Medical-related curiosity.

Ensure the following requirements:
1) Ensure "type-n" Nesting: The inner query block Q must not contain a join predicate that references the relation of the outer query block, and its SELECT clause must project a column without an aggregate function. 
2) Ensure Medical professional Relevance: Generate the nested predicate that aligns naturally with realistic and meaningful **multi-hop** queries that Medical professionals are likely to ask.
3) Ensure Synthetic Correctness: Generate the nested predicate that is syntactically correct and executable on the provided database.
4) Ensure Semantic Alignment: If the inner query's SELECT column does not semantically match any column in the outer query’s table, revise it for consistency.

The "type-n" nested predicate must be in the form: OuterTable.column [IN | NOT IN] ( Q ).

Your output must be in JSON format with the keys:
- "nested_predicate": Generate only the "type-n" nested predicate based on the selected inner query block. Do not generate a full SELECT statement.
- "logical_operator": If a WHERE clause exists, return 'AND' or 'OR' to connect it.

**IMPORTANT**: 
- Ensure that the nesting level of the inner query block is correctly preserved. The expected nesting level of the provided inner query is {height}.
- Do not modify the nesting level of the provided inner query block.
"""

system_generate_nested_predicate_type_j_prompt = """
You are both an Medical professional and an SQL expert. Based on the given database, generated clauses, selected inner query block, and its execution result, generate a "type-j" nested predicate that reflects authentic Medical-related curiosity.

Ensure the following requirements:
1) Ensure "type-j" Nesting: Revise the inner query block Q to ensure it includes a join predicate in its WHERE clause that references the outer query's relation, and its SELECT clause must project a column without an aggregate function. 
2) Ensure Medical professional Relevance: Generate the nested predicate that aligns naturally with realistic and meaningful **multi-hop** queries that Medical professionals are likely to ask.
3) Ensure Synthetic Correctness: Generate the nested predicate that is syntactically correct and executable on the provided database.
4) Ensure Semantic Alignment: If the inner query's SELECT column does not semantically match any column in the outer query’s table, revise it for consistency.

The "type-j" nested predicate must be in one of the following forms with the join predicate appearing inside the WHERE clause of the subquery Q:
- OuterTable.column [IN | NOT IN] (SELECT ... FROM ... WHERE ... [join predicate on the outer query's relation] ...)
- [EXISTS | NOT EXISTS] (SELECT ... FROM ... WHERE ... [join predicate on the outer query's relation] ...)

Your output must be in JSON format with the keys:
- "nested_predicate": Generate only the "type-j" nested predicate based on the selected inner query block. Do not generate a full SELECT statement.
- "logical_operator": If a WHERE clause exists, return 'AND' or 'OR' to connect it.

**IMPORTANT**: 
- The join predicate involving the outer query's relation must appear in the **WHERE clause** of the subquery Q, not in its **FROM clause**.
- Ensure that the nesting level of the inner query block is correctly preserved. The expected nesting level of the provided inner query is {height}.
- Do not modify the nesting level of the provided inner query block.
"""

system_generate_nested_predicate_type_a_prompt = """
You are both an Medical professional and an SQL expert. Based on the given database, generated clauses, selected inner query block, and its execution result, generate a "type-a" nested predicate that reflects authentic Medical-related curiosity.

Ensure the following requirements:
1) Ensure "type-a" Nesting: The inner query block Q must not contain a join predicate that references the relation of the outer query block, and its SELECT clause must indicate an aggregate function associated with the column name.
2) Ensure Medical professional Relevance: Generate the nested predicate that aligns naturally with realistic and meaningful **multi-hop** queries that Medical professionals are likely to ask.
3) Ensure Synthetic Correctness: Generate the nested predicate that is syntactically correct and executable on the provided database.

The "type-a" nested predicate must be in the form: OuterTable.column [= | != | < | <= | > | >=] ( Q with aggregate function ).

Your output must be in JSON format with the keys:
- "nested_predicate": Generate only the "type-a" nested predicate based on the selected inner query block. Do not generate a full SELECT statement.
- "logical_operator": If a WHERE clause exists, return 'AND' or 'OR' to connect it.

**IMPORTANT**: 
- Do not revise the SELECT clause of the Q.
- Ensure that the nesting level of the inner query block is correctly preserved. The expected nesting level of the provided inner query is {height}.
- Do not modify the nesting level of the provided inner query block.
"""

system_generate_nested_predicate_type_ja_prompt = """
You are both an Medical professional and an SQL expert. Based on the given database, generated clauses, selected inner query block, and its execution result, generate a "type-ja" nested predicate that reflects authentic Medical-related curiosity.

Ensure the following requirements:
1) Ensure "type-ja" Nesting: Revise the inner query block Q to ensure it includes a join predicate in its WHERE clause that references the outer query's relation, and its SELECT clause must indicate an aggregate function associated with the column name. 
2) Ensure Medical professional Relevance: Generate the nested predicate that aligns naturally with realistic and meaningful **multi-hop** queries that Medical professionals are likely to ask.
3) Ensure Synthetic Correctness: Generate the nested predicate that is syntactically correct and executable on the provided database.

The "type-ja" nested predicate must be in one of the following forms, where the subquery Q contains a join predicate in its WHERE clause and an aggregate function in its SELECT clause:
- [OuterTable.column | Constant] [= | != | < | <= | > | >=] (SELECT [aggregate function] ... FROM ... WHERE ... [join predicate on the outer query's relation] ...)
- [EXISTS | NOT EXISTS] (SELECT [aggregate function] ... FROM ... WHERE ... [join predicate on the outer query's relation] ...)

Your output must be in JSON format with the keys:
- "nested_predicate": Generate only the "type-ja" nested predicate based on the selected inner query block. Do not generate a full SELECT statement.
- "logical_operator": If a WHERE clause exists, return 'AND' or 'OR' to connect it.

**IMPORTANT**: 
- The join predicate involving the outer query's relation must appear in the **WHERE clause** of the subquery Q, not in its **FROM clause** and do not revise the SELECT clause of the Q.
- Ensure that the nesting level of the inner query block is correctly preserved. The expected nesting level of the provided inner query is {height}.
- Do not modify the nesting level of the provided inner query block.
"""

system_generate_multiple_nested_predicates_prompt = """
You are both an Medical professional and an SQL expert. Based on the given database, generated clauses, selected inner query block, and its execution result, generate an WHERE clause that contains {nested_predicate_types} nested predicates that reflects authentic Medical-related curiosity.

Ensure the following requirements:
1) Ensure Medical professional Relevance: Generate the nested predicate that aligns naturally with realistic and meaningful **multi-hop** queries that Medical professionals are likely to ask.
2) Ensure Synthetic Correctness: Generate the nested predicate that is syntactically correct and executable on the provided database.
3) Ensure Semantic Alignment: If the inner query's SELECT clause  does not semantically match with the outer table's column, revise the SELECT clause of the Q to ensure alignment.
{nested_predicate_requirements}

{nested_predicate_formats}

Your output must be in JSON format with the keys:
- "nested_predicate": Generate only the WHERE clause that contains {nested_predicate_types} nested predicates based on the selected inner query block. Do not generate a full SELECT statement. The where clause must contain {num_child_queries} nested predicates.
"""

user_generate_multiple_nested_predicates_prompt = """
Schema: 
{schema}

Generated Clauses: 
{history}

Candidate Inner Query Blocks:
{candidate_inner_query_blocks}

Return the results in a FLAT json.

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""

user_generate_nested_predicate_prompt = """
Database: 
{schema}

Generated FROM Clause of the Outer Query: 
{generated_from_clause}

Generated WHERE Clause of the Outer Query: 
{generated_where_clause}

Selected Inner Query Block Q: 
{selected_inner_query_block}

Return the results in a FLAT JSON format.

*DO NOT include any explanations or notes in the output. ONLY return JSON.*
"""
