# TRIGRAM MODEL
import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import codecs

# Load the word-to-index dictionary
vocab = codecs.open("brown_vocab_100.txt")
word_index_dict = {}
for i, line in enumerate(vocab):
    word = line.rstrip()
    word_index_dict[word] = i
vocab.close()

# Initialize the counts to a numpy matrix of zeros
counts_2 = np.zeros((len(word_index_dict), len(word_index_dict)))
counts_3 = np.zeros((len(word_index_dict), len(word_index_dict), len(word_index_dict)))
f = codecs.open("brown_100.txt")

# Iterate through the file and update counts
for line in f:
    previous_word = "<s>"
    words = line.lower().split()
    for i, word in enumerate(words[1: -2]):
        next_word = words[i + 2]
        counts_3[word_index_dict[previous_word], word_index_dict[word], word_index_dict[next_word]] += 1
        counts_2[word_index_dict[previous_word], word_index_dict[word]] += 1
        previous_word = word
f.close()

upper = counts_3[word_index_dict['in'], word_index_dict['the'], word_index_dict['past']]
down = counts_2[word_index_dict['in'], word_index_dict['the']]
probs = upper / down 

print(probs)
# Write the bigram probabilities to a file
# print(probs[word_index_dict['in'], word_index_dict['the'], word_index_dict['past']])