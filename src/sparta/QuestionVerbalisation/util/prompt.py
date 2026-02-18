def create_incontext_prompt2(*args):
    if len(args) % 2 == 0:
        raise ValueError("The number of arguments must be odd.")
    
    formatted_text = ""
    for i in range(0, len(args)-1, 2):
        input_str = args[i]
        output_str = args[i + 1]
        formatted_text += 'Input: {0}\nOutput: {1}\n###\n'.format(input_str, output_str)
    formatted_text += 'Input: {0}\nOutput:'.format(args[-1])
    return formatted_text