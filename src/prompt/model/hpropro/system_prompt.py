SYSTEM_PROMPT = """Read the table and write python code to answer the given question. Note that sometimes answering questions requires extracting information in the hyperlinks. [HYPER] means these are hyperlinks in the cell. Assume we has defined the following functions to extract information from hyperlink text. You can use these functions directly as black box. You don't need to process the table. 

Attention: 
    1. The given table is in the pandas DataFrame format.
    2. The table has already contained the hyperlink in the cell. 
       a. If you want GET or SORT the content in the cell, you MUST use cell[0]. All data in cell[0] are string.
       b. If you want get the hyperlink string in the cell, you use cell[1]
    3. If you want to extract information from the hyperlink passages, you can use the function extract_info(cell[1], target information) defined above.
    4. If you want to compare two objects, you can use the function check(str1, str2, operation) defined above, don't use '==' or '<' or '>'.
"""

SYSTEM_PROMPT_NOT_TABLE = """You are given a question and some relevant knowledge retrieved from related passages or documents. 
Your task is to answer the question as accurately and concisely as possible based on the given information.
"""
