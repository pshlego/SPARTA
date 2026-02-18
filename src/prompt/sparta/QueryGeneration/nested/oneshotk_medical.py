type_ja_predicate_requirement = """Ensure "type-ja" Nesting: Revise the selected inner query block Q to ensure it includes a join predicate in its **WHERE clause** that references the outer query's relation, and its SELECT clause must indicate an aggregate function associated with the column name."""

type_j_predicate_requirement = """Ensure "type-j" Nesting: Revise the selected inner query block Q to ensure it includes a join predicate in its **WHERE clause** that references the outer query's relation, and its SELECT clause must project a column without an aggregate function."""

type_a_predicate_requirement = """Ensure "type-a" Nesting: The selected inner query block Q must not contain a join predicate that references the relation of the outer query block, and its SELECT clause must indicate an aggregate function associated with the column name."""

type_n_predicate_requirement = """Ensure "type-n" Nesting: The selected inner query block Q must not contain a join predicate that references the relation of the outer query block, and its SELECT clause must project a column without an aggregate function."""

type_ja_predicate_format = """
Join predicate is the predicate that references the outer query's relation (i.e., outer_table_name.column_name = inner_table_name.column_name).
The "type-ja" nested predicate must be in one of the following forms, where the subquery Q **contains the join predicate in its WHERE clause** and an aggregate function in its SELECT clause:
- [OuterTable.column | Constant] {= | != | < | <= | > | >=} (Revised Q with join predicate in WHERE clause and aggregate function in SELECT clause: SELECT [aggregate function] ... FROM ... WHERE ... [join predicate on the outer query's relation] ...)
- {EXISTS | NOT EXISTS} (Revised Q with join predicate in WHERE clause and aggregate function in SELECT clause: SELECT [aggregate function] ... FROM ... WHERE ... [join predicate on the outer query's relation] ...)

**IMPORTANT**: The join predicate involving the outer query's relation must appear in the **WHERE clause** of the subquery Q, not in its **FROM clause**.
"""

type_j_predicate_format = """
Join predicate is the predicate that references the outer query's relation (i.e., outer_table_name.column_name = inner_table_name.column_name).
The "type-j" nested predicate must be in one of the following forms with **the join predicate appearing inside the WHERE clause of the subquery Q**:
- OuterTable.column {IN | NOT IN} (Revised Q with join predicate in WHERE clause: SELECT ... FROM ... WHERE ... [join predicate on the outer query's relation] ...)
- {EXISTS | NOT EXISTS} (Revised Q with join predicate in WHERE clause: SELECT ... FROM ... WHERE ... [join predicate on the outer query's relation] ...)

**IMPORTANT**: The join predicate involving the outer query's relation must appear in the **WHERE clause** of the subquery Q, not in its **FROM clause**.
"""

type_a_predicate_format = """The "type-a" nested predicate must be in the form: OuterTable.column {= | != | < | <= | > | >=} ( Q with aggregate function )."""

type_n_predicate_format = """The "type-n" nested predicate must be in the form: OuterTable.column {IN | NOT IN} ( Q )."""

system_generate_nested_predicates_prompt = """
You are both an Medical professional and an SQL expert. Based on the given database, generated clauses, selected inner query blocks, and its execution result, generate an WHERE clause that contains {nested_predicate_types} nested predicates that reflects authentic Medical-related curiosity.

Ensure the following requirements:
1) Ensure Medical professional Relevance: Generate the nested predicate that aligns naturally with realistic and meaningful **multi-hop** queries that Medical professionals are likely to ask.
2) Ensure Synthetic Correctness: Generate the nested predicate that is syntactically correct and executable on the provided database.
3) Ensure Semantic Alignment: If the inner query's SELECT clause  does not semantically match with the outer table's column, revise the SELECT clause of the Q to ensure alignment.
{nested_predicate_requirements}

{nested_predicate_formats}

Your output must be in JSON format with the keys:
- "nested_predicate": Generate only the WHERE clause that contains {nested_predicate_types} nested predicates based on the selected inner query blocks. Do not generate a full SELECT statement. The where clause must contain {num_child_queries} nested predicates.

**IMPORTANT**: 
- Explicitly specify which table the outer column belongs to (e.g., outer_table_name.outer_column_name). You have to use all of the selected inner query blocks to generate the nested predicate.
- Ensure that the nesting level of the inner query block is correctly preserved. The expected nesting level of the provided inner query is {height}.
- Do not modify the nesting level of the provided inner query block.
"""

user_generate_nested_predicates_prompt = """
Schema: 
{schema}

Generated Clauses: 
{history}

Selected Inner Query Blocks (list of Q):
{candidate_inner_query_blocks}

Return the results in a FLAT json.

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""

system_from_multiple_tables_prompt = """
You are both an Medical professional and an SQL expert. Given the database and the inner query blocks, generate a FROM clause of the outer query block that reflects authentic Medical-related curiosity.

Ensure the following requirements:
1) Output Structure: Return a JSON object containing a single key, "from", with its value being FROM clause of the outer query block from the provided database (i.e., do not include any sub-selects or nested queries directly in the FROM clause).
2) Ensure Medical professional Relevance: Generate the FROM clause of the outer query block that aligns naturally with realistic and meaningful **multi-hop** queries that Medical professionals are likely to ask.
3) Ensure Synthetic Correctness: Generate the FROM clause that is syntactically correct and executable on the provided database.
4) Separate Inner Query: The inner query blocks must remain separate (i.e., it should later be incorporated into the WHERE clause instead of being nested within the FROM clause).
5) Ensure Natural Connection: Choose a outer table or tables whose columns can be naturally referenced or filtered against the results of the inner query blocks.

**IMPORTANT**: Do not include unnecessary tables. 
"""

user_from_multiple_tables_prompt = """
Database: 
{schema}

Inner Query Blocks: 
{subquery}

Return the results in a FLAT JSON format.

*DO NOT include any explanations or notes in the output. ONLY return JSON.*
"""
