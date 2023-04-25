# ANALYSIS OF THE BROWN CORPUS
import nltk
from nltk.corpus import brown
from nltk import FreqDist
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
 
# nltk.download("brown")
# nltk.download("averaged_perceptron_tagger")

sentences = brown.sents()
tokens = brown.words()
words = [token.lower() for token in tokens if any(c.isalpha() for c in token)]
words_10 = [token for token in words if words.count(token) >= 10]
words_10_unique = set(words_10)
N = len(words) # number of words in the whole corpus
print("kka")
word_index_dict = {}
for i, word in enumerate(words_10_unique):
    word_index_dict[word] = i

# Initialize the counts of all successive pairs (w1, w2) to a numpy matrix of zeros
counts = np.zeros((len(word_index_dict), len(word_index_dict)))

# Count the number of occurrences of all successive pairs (w1, w2) in sentences
for sentence in sentences:
    for i in range(len(sentence) - 1):
        w1 = sentence[i].lower()
        w2 = sentence[i + 1].lower()
        if w1 in word_index_dict and w2 in word_index_dict:
            counts[word_index_dict[w1]][word_index_dict[w2]] += 1

# calculate pmi
pmi = np.zeros((len(word_index_dict), len(word_index_dict)))

for w1 in word_index_dict:
    for w2 in word_index_dict:
        print(word_index_dict[w1], word_index_dict[w2])
        i = word_index_dict[w1]
        j = word_index_dict[w2]
        if counts[i][j] > 0:  # Add this condition to avoid log(0)
                pmi[i][j] = np.log2((counts[i][j] * N) / (words.count(w1) * words.count(w2)))

# calculate pmi
pmi = np.zeros((len(word_index_dict), len(word_index_dict)))

for w1 in word_index_dict:
    for w2 in word_index_dict:
        i = word_index_dict[w1]
        j = word_index_dict[w2]
        pmi[i][j] = np.log((counts[i][j] * N) / (words.count(w1) * words.count(w2)))


# Find the indices of the 20 highest and lowest PMI values
highest_pmi_indices = np.unravel_index(np.argsort(pmi.ravel())[-20:], pmi.shape)
lowest_pmi_indices = np.unravel_index(np.argsort(pmi.ravel())[:20], pmi.shape)

print("20 word pairs with the highest PMI values:")
for i, j in zip(*highest_pmi_indices):
    w1 = list(word_index_dict.keys())[list(word_index_dict.values()).index(i)]
    w2 = list(word_index_dict.keys())[list(word_index_dict.values()).index(j)]
    print(f"{w1}, {w2}, PMI: {pmi[i, j]}")

print("\n20 word pairs with the lowest PMI values:")
for i, j in zip(*lowest_pmi_indices):
    w1 = list(word_index_dict.keys())[list(word_index_dict.values()).index(i)]
    w2 = list(word_index_dict.keys())[list(word_index_dict.values()).index(j)]
    print(f"{w1}, {w2}, PMI: {pmi[i, j]}")