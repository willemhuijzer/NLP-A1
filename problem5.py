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

# To calculate the trigrams, divide count(w1, w2, w3) by count(w1, w2)
trigrams = [('in', 'the', 'past'), ('in', 'the', 'time'), ('the', 'jury', 'said'), ('the', 'jury', 'recommended'),
            ('jury', 'said', 'that'), ('agriculture', 'teacher', ',')]

for trigram in trigrams:
    word1, word2, word3 = trigram
    numerator = counts_3[word_index_dict[word1], word_index_dict[word2], word_index_dict[word3]]
    denominator = counts_2[word_index_dict[word1], word_index_dict[word2]]
    trigram_probs = numerator / denominator
    print(f"Trigram probability for {trigram}: {trigram_probs:.6f}")

# Add alpha smoothing (cannot use Normalize function for counts_3 as it only accepts 2-dimensional arrays as input)
# We use the following formula: p(w3 | w1, w2) = (count(w1, w2, w3) + alpha) / (count(w1, w2) + alpha * vocab)
alpha = 0.1
for trigram in trigrams:
    word1, word2, word3 = trigram
    numerator = counts_3[word_index_dict[word1], word_index_dict[word2], word_index_dict[word3]] + alpha
    denominator = counts_2[word_index_dict[word1], word_index_dict[word2]] + alpha * len(word_index_dict)
    trigram_probs = numerator / denominator
    print(f"Smoothed trigram probability for {trigram}: {trigram_probs:.6f}")
