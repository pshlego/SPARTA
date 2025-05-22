system_prompt = """
Agent Introduction: You are an agent who is going to be assisting me in a question answering task. I have tables as a source of information. I have already extracted relevant entities from the question and relevant table names and column headers from the tables.

Task: Map the entities extracted from the question with the relevant headers and the table name.

Output format:
Your output must be in JSON format with the key:
- "<entity1>": [
    - "<table_name1>": ["<header name1>", "<header name2>", ...]
]
- "<entity2>": ["None"]
- "<entity3>": [
    - "<table_name1>": ["<header name1>", "<header name2>", ...]
    - "<table_name2>": ["<header name1>", ...]
]

For each entity extracted from the question, there should be a corresponding mapping to an item in the ‘Relevant Tables and their Headers’. 
If none of the headers match the entity, the corresponding value should be ["None"].

Use the below example to better understand the task

Input:
Relevant Tables and their Headers:
Relevant Table Name 1: Sweden at the 1932 Summer Olympics
Relevant Headers 1: ["Medal", "Name", "Sport", "Event"]

Relevant Table Name 2: Korea at the 1988 Summer Olympics
Relevant Headers 2: ["Medal", "Name", "Sport", "Event"]

Question: What was the nickname of the gold medal winner in the men’s heavyweight GrecoRoman wrestling event of the 1932 Summer Olympics?

Entities extracted from question: ["gold medal", "men’s heavyweight", "Greco-Roman Wrestling", "1932 Summer Olympics"]

Output:
{
    "gold medal": [{"Sweden at the 1932 Summer Olympics": ["Medal"]}],
    "men’s heavyweight": [{"Sweden at the 1932 Summer Olympics": ["Event"]}],
    "Greco-Roman Wrestling": [{"Sweden at the 1932 Summer Olympics": ["Sport"]}],
    "1932 Summer Olympics": ["None"]
}
"""


user_prompt = """
Here's the Input you'll need:

Input:
Relevant Tables and their Headers:
{table_and_headers}

Question: {question}

Entities extracted from question: {entities}

Output:
Return the results in a FLAT json.
*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""