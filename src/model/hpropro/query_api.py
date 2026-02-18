import os
from openai import OpenAI


def query_API(message, model, extract_info_flag=False, temperature=0, n=1):
    """Construct the message to the standard format, and call the api to solve.

    Args:
        message (str/list): The input message.
            The method could construct the message by judging the type of the input.
        model (str, optional): The name of api called. Defaults to 'gpt-3.5-turbo'.

    Returns:
        str: The result.
    """

    system_prompt = "You are a helpful assistant."

    if type(message) == str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
    elif type(message) == list:
        messages = message

    else:
        print("The type of input message is wrong(neither 'str' or 'list').")
        return None
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    try:
        if model == "gpt-5":
            if extract_info_flag:  # minimal reasoning for extract_info
                llm_hyperparams = {
                    "reasoning_effort": "minimal",
                    "verbosity": "low",
                }
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **llm_hyperparams,
                )
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
    except Exception as e:
        print(f"[Error] Unexpected error: {e}")
        raise e

    result = resp.choices[0].message.content.strip()

    if type(result) == str:
        return result
    return result.choices[0].message.content
