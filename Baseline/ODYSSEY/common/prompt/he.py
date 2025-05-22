system_prompt = """
Agent Introduction: You are an agent who is going to be assisting me in a question answering task. I have tables as a source of information. I have already extracted the relevant entities from the question. 
For this task, I need to first identify the table names and column header names that are relevant in the question.

Task: Identify the relevant table names and relevant column header names from the provided database schema, based on the extracted entities from the question. 
I will also provide the extracted entities from the question.

Output format:
Your output must be in JSON format with the key:
- "<table_name1>": ["<header name1>", "<header name2>", ...],
- "<table_name2>": ["<header name1>", ...]

Use the below example to better understand the task

Input:
Question: What was the nickname of the gold medal winner in the men’s heavyweight GrecoRoman wrestling event of the 1932 Summer Olympics?

Entities extracted from question: ["gold medal", "men’s heavyweight", "Greco-Roman Wrestling", "1932 Summer Olympics"]

Database Schema:
Table Name 1: Sweden at the 1932 Summer Olympics
Table Headers 1: ["Medal", "Name", "Country", "Sport", "Year", "Gender", "Weight", "Event"]

Table Name 2: 1932 Summer Olympics
Table Headers 2: ["Medal", "Name", "Sport", "Event"]

Table Name 3: Korea at the 1988 Summer Olympics
Table Headers 3: ["Medal", "Name", "Sport", "Event"]

Output:
{
    "Sweden at the 1932 Summer Olympics": ["Medal", "Name", "Sport", "Event"],
    "1932 Summer Olympics": ["Medal", "Name", "Sport", "Event"],
}

"""


user_prompt = """
Here's the Input you'll need:

Question: {question}

Entities extracted from question: {entities}

Database Schema:
{table_and_headers}

Output:
Return the results in a FLAT json.

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""