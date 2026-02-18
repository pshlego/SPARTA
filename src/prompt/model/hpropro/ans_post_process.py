ANS_POST_PROCESS_PROMPT = """Based on the question and the executed result, what is the final answer
Executed result:
[CONTEXT]
Query:
[QUERY]
Just give me the answer. Don't give me any explanation. If you can't find the answer, return "NOT_AVAILABLE"
The [answer] should be as short as possible, like a word, a number or a shot span."""
