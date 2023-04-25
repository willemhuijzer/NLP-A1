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
for sentence in sentences:
    sentence.insert(0,'<s>')
    sentence.append('<\s>')

# print(sentences[1])


alpha_tokens = [token.lower() for token in tokens if any(c.isalpha() for c in token)]
# print(alpha_tokens)

words = set(alpha_tokens)

word_freqs = Counter(words)

freq_threshold = 10
freq_filtered = [word for word, freq in word_freqs.items() if freq > freq_threshold]
# print(freq_filtered)


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

for sentence in sentences:
    previous_word = '<s>'
    words = [word.lower() for word in sentence if any(c.isalpha() for c in word)]
    for word in words:
        index = word_index_dict[word]
        counts[index] += 1
        countsbigrams[word_index_dict[previous_word], word_index_dict[word]] +=1 
        previous_word = word

print(counts)
#het klopt nog niet met de <s> want bij de unigram count wordt deze nu niet meegenomen 





