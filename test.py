import pandas as pd
import sklearn
from nltk.corpus import stopwords
import itertools
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

df = pd.read_csv('IMDB-Dataset.csv')

def by_indexes(iterable):
    output = {}
    for index, key in enumerate(iterable):
        output.setdefault(key, []).append(index)
    return output

def generate_co_occurrence_matrix(corpus, window_size, vocab):
    # Create a dictionary to store co-occurrence counts
    # vocab ,vocab_size, vocab_to_ix, ix_to_vocab, text_as_int = word_processor(corpus)
    
    new_corpus = corpus
    new_corpus['tokens'] = new_corpus['review'].apply(lambda row: word_tokenize(row))
    new_corpus['unigrams'] = new_corpus['review'].apply(lambda row: list(nltk.ngrams(row.split(), 1)))

    co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
    for sentence in corpus:
        words = sentence.split()

        for i, target_word in enumerate(words):
            for j in range(max(i - window_size, 0), min(i + window_size + 1, len(words))):
                context_word = words[j]

                if target_word in vocab and context_word in vocab:
                    co_occurrence_matrix[target_word][context_word] += 1

    return co_occurrence_matrix


def split_and_preprocess_data(data):
    data['review'] = data['review'].map(lambda x: x.lower())
    data['review'] = data['review'].apply(lambda x: re.sub(r'[^\w\s]', '',x))
    return data.iloc[:int(data.shape[0]/2)], data.iloc[int(data.shape[0]/2):]


def train_with_glove_repo(corpus):
    
    return 