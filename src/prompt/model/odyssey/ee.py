system_prompt = """
Agent Introduction: You are an agent who is going to be assisting me in a question answering task. For this task, I need to first identify the named entities in the question.

Task: Identify the named entities in the provided question. Theses entities will serve as key elements for extracting pertinent information from the available sources, which include table names and its headers.

Output format:
Your output must be in JSON format with the key:
- Entities: ["<entity1>", "<entity2>", .....]

Use the below example to better understand the task

Input:
Question: What was the nickname of the gold medal winner in the men's heavyweight Greco-Roman wrestling event of the 1932 Summer Olympics?

Database Schema:
Table Name 1: Sweden at the 1932 Summer Olympics
Table Headers 1: ["Medal", "Name", "Sport", "Event"]

Table Name 2: Korea at the 1988 Summer Olympics
Table Headers 2: ["Medal", "Name", "Sport", "Event"]


Output:
{
    "Entities": ["nickname", "medal", "gold", "men's heavyweight", "Greco-Roman Wrestling event", "1932 Summer Olympics"]
}
"""


user_prompt = """
Here's the Input you'll need:

Input:
Question: {question}

Database Schema:
{table_and_headers}

Output:
Return the results in a FLAT json.

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""

system_prompt_passage_only = """
Agent Introduction: You are an agent who is going to be assisting me in a question answering task. For this task, I need to first identify the named entities in the question.

Task: Identify the named entities in the provided question. Theses entities will serve as key elements for extracting pertinent information from the available sources, which include passages.

Output format:
Your output must be in JSON format with the key:
- Entities: ["<entity1>", "<entity2>", .....]

Use the below example to better understand the task

Input:
Question: What was the nickname of the gold medal winner in the men's heavyweight Greco-Roman wrestling event of the 1932 Summer Olympics?

Output:
{
    "Entities": ["nickname", "medal", "gold", "men's heavyweight", "Greco-Roman Wrestling event", "1932 Summer Olympics"]
}
"""

user_prompt_passage_only = """
Here's the Input you'll need:

Input:
Question: {question}

Output:
Return the results in a FLAT json.

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""
