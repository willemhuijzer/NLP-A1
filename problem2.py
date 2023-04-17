#!/usr/bin/env python3
# UNIGRAM MODEL

import numpy as np
from generate import GENERATE


vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word = line.rstrip()
    word_index_dict[word] = i
vocab.close()

brown_file = open("brown_100.txt")

# Initialize the counts to a zero vector
counts = np.zeros(len(word_index_dict))

# Iterate through the file and update counts
for line in brown_file:
    # Split the sentence into a list of words and convert each word to lowercase
    words = line.lower().split()

    # Increment the count for each word in the sentence
    for word in words:
        index = word_index_dict[word]
        counts[index] += 1

brown_file.close()

# Normalize the counts
probs = counts / np.sum(counts)

np.save("unigram_probs.npy", probs)

# Print the normalized counts
print("Test: \n")
print("The probability of the word 'all' is: ", probs[word_index_dict['all']])
print("The probability of the word 'resolution' is: ", probs[word_index_dict['resolution']])


