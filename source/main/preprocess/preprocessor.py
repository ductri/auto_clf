from nltk.tokenize import word_tokenize
import re


p1 = re.compile('[0-9]+')
p2 = re.compile('\n+')


def __tokenize_single_doc(doc):
    return ' '.join(word_tokenize(doc))


def __cut_off(doc, length):
    return ' '.join(doc.split()[:length])


def __replace_digit(doc):
    return p1.sub('__D__', doc)


def __replace_breakline(doc):
    return p2.sub(' ', doc)


def preprocess_text(doc):
    doc = __tokenize_single_doc(doc)
    doc = __replace_digit(doc)
    doc = __replace_breakline(doc)
    doc = doc.lower()
    return doc


def infer_preprocess(doc):
    doc = preprocess_text(doc)
    return doc


def train_preprocess(doc, max_length):
    doc = preprocess_text(doc)
    doc = __cut_off(doc, max_length)
    return doc
