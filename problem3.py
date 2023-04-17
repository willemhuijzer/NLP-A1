#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs

# Load the word-to-index dictionary
vocab = codecs.open("brown_vocab_100.txt", encoding="utf-8")
word_index_dict = {}
for i, line in enumerate(vocab):
    word = line.rstrip()
    word_index_dict[word] = i
vocab.close()

# Open the brown_100.txt file
f = codecs.open("brown_100.txt", encoding="utf-8")

# Initialize the counts to a numpy matrix of zeros
counts = np.zeros((len(word_index_dict), len(word_index_dict)))

f = codecs.open("brown_100.txt")

# Iterate through the file and update counts
previous_word = "<s>"
for line in f:
    words = line.lower().split()
    for word in words:
        counts[word_index_dict[previous_word], word_index_dict[word]] += 1
        previous_word = word
f.close()

# Normalize the counts
probs = normalize(counts, norm='l1', axis=1)

np.save("unigram_probs.npy", probs)

# Write the bigram probabilities to a file
with open("bigram_probs.txt", "w") as out_file:
    out_file.write(f"p(the | all): {probs[word_index_dict['all'], word_index_dict['the']]:.6f}\n")
    out_file.write(f"p(jury | the): {probs[word_index_dict['the'], word_index_dict['jury']]:.6f}\n")
    out_file.write(f"p(campaign | the): {probs[word_index_dict['the'], word_index_dict['campaign']]:.6f}\n")
    out_file.write(f"p(calls | anonymous): {probs[word_index_dict['anonymous'], word_index_dict['calls']]:.6f}\n")

# Generate sentences using the bigram model
generated_sentences = [GENERATE(word_index_dict, probs, model_type='bigram', max_words=50, start_word='<s>') for _ in range(5)]

# Print the generated sentences
for i, sentence in enumerate(generated_sentences, start=1):  # HELP heel veel <s> <s> <s>
    print(f"Sentence {i}: {sentence}")
