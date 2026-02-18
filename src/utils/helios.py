import unicodedata
import re

from src.utils.tokenizers import SimpleTokenizer


def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text


def has_answer(answers, text) -> bool:
    """Check if a document contains an answer string."""
    tokenizer = SimpleTokenizer()

    # Answer is a list of possible strings
    text_tokens = tokenizer.tokenize(remove_accents_and_non_ascii(text)).words(uncased=True)

    for single_answer in answers:
        single_answer_tokens = tokenizer.tokenize(remove_accents_and_non_ascii(single_answer)).words(uncased=True)

        for i in range(0, len(text_tokens) - len(single_answer_tokens) + 1):
            if single_answer_tokens == text_tokens[i : i + len(single_answer_tokens)]:
                return True
                
    return False