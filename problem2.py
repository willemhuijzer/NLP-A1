# UNIGRAM MODEL
import numpy as np
from generate import GENERATE

word_index_dict = {}
# vf = vocab file in "r" mode (read mode)
with open("brown_vocab_100.txt", "r") as vf:
    for index, line in enumerate(vf):
        word = line.rstrip()
        word_index_dict[word] = index
    

# Initialize the counts to a zero vector
counts = np.zeros(len(word_index_dict))

brown_file = open("brown_100.txt")
# Iterate through the file and update counts
for line in brown_file:
    words = line.lower().split()

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


