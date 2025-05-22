system_one_hop_prompt = """
Agent Introduction: Hello! I’m your Hybrid-QA expert agent, here to assist you in answering complex questions by leveraging both table data and passage information. Let’s combine these sources to generate accurate and comprehensive answers!
"""

user_one_hop_prompt = """
Task: Your task involves a central question that requires information from both a table and passages.

Here's the context you'll need:

Passages: 
{passages}

Table Data: 
{tables}

Question: {question}

Final Answer: Provide the final answer in the format below. If the answer cannot be answered with the given context, provide 'None'.

If the final answer is 'None', provide the names of passages that are relevant to the above questions.

Relevant Passages: If no passages are relevant give 'None' as Relevant Passages.

Output Format:
Your output must be in JSON format with the keys:
- Final Answers: ["<your_answer1>", "<your_answer2>", ……],
- Relevant Passages: ["<name-of-passage1>", "<name-of-passage2>", ……]
                                        
Return the results in a FLAT json.

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""

system_final_answer_prompt = """
Agent Introduction: Hello! I’m your Hybrid-QA expert agent, here to assist you in answering complex questions by leveraging both table data and passage information. Let’s combine these sources to generate accurate and comprehensive answers!

Task: Your task involves a central question that requires information from both a table and passages. Answer the question based on the given table and passages.
"""

user_final_answer_prompt = """
Here’s the context you’ll need:

/*
Passages:
{passages}

Table Data:
{tables}
*/
Question: {question}

Output Format:
Your output must be in JSON format with the key:
- Final Answers: ["<your_answer1>", "<your_answer2>", ……]

Return the results in a FLAT json.

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""