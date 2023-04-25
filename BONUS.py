# PMI calculation
import nltk
from nltk.corpus import brown
from collections import Counter
import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import pickle # to save to dictonary
import codecs

tokens = brown.words()
sentences = brown.sents()

alpha_tokens = [token.lower() for token in tokens if any(c.isalpha() for c in token)]

words = set(alpha_tokens)

word_freqs = Counter(words)

# print(word_freqs.most_common(10))

#create the vocabulary
word_index_dict={}
i=0
for word in words:
    word_index_dict[word] = i
    i += 1

# print(word_index_dict)

counts = np.zeros(len(word_index_dict))
countsbigrams = np.zeros((len(word_index_dict), len(word_index_dict)))







