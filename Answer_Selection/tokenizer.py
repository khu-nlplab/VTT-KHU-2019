import tensorflow as tf
import nltk

nltk.download('punkt')

def sentence2token(input, lower=True):
    input.strip()
    if lower:
        input = str(input).lower()

    tokens = nltk.word_tokenize(str(input))

    if '' in tokens:
        tokens = tokens.remove('')

    return tokens

def sentence2char(input, lower=True):
    input.strip()
    if lower:
        input = input.lower()

    tokens = sentence2token(input, lower)

    char_list = []

    for token in tokens:
        chars = list(token)
        if '' in chars:
            chars = chars.remove('')
        char_list.append(chars)

    return char_list