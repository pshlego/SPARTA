CHECK_PROMPT = """Please verify whether the semantics of the two strings meet the given conditions.
For example:
Q: ten > 9
Check: True
Q: 21 Nov, 2030 < 09-31-2021
Check: False
Q: Beijing == The capital of China
Check: True
Q: 2022/10/01 == Oct 1st, 2022
Check: True

Q: [STRING1] [REL] [STRING2]
Check:"""
