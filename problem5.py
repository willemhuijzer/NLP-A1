import numpy as np
import pickle  # to open a dictionary

with open("word_index_dict.pkl", "rb") as f:
    word_index_dict = pickle.load(f)

unigram_probs = np.load("unigram_probs.npy")
bigram_probs = np.load("bigram_probs.npy")

# Chain rule P(w1, w2, w3) = P(w3 | w2) * P(w2 | w1) * P(w1)
def trigram_probability(trigram, unigram_probs, bigram_probs, word_index_dict):
    w1, w2, w3 = trigram
    unigram_w1 = unigram_probs[word_index_dict[w1]]
    bigram_w2_w1 = bigram_probs[word_index_dict[w1], word_index_dict[w2]]
    bigram_w3_w2 = bigram_probs[word_index_dict[w2], word_index_dict[w3]]

    trigram_prob =  bigram_w3_w2 * bigram_w2_w1 * unigram_w1 
    return trigram_prob

trigram = ("in", "the", "past")
trigram_prob = trigram_probability(trigram, unigram_probs, bigram_probs, word_index_dict)

print(f"P({trigram[0]}, {trigram[1]}, {trigram[2]}) = {trigram_prob:.7f}")
