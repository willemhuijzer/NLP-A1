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

# Generate sentences using the unigram model
generated_sentences = [GENERATE(word_index_dict, probs, model_type='unigram', max_words=50, start_word='<s>') for _ in range(10)]

# Store the generated sentences in a file unigram_generation.txt
with open("unigram_generation.txt", "w") as f:
    for sentence in generated_sentences:
        f.write(f"{sentence}")
                 
for i, sentence in enumerate(generated_sentences, start=1):  # HELP heel veel <s> <s> <s>
    print(f"Sentence {i}: {sentence}")

#Iterate through every sentence in the toy corpus
toycorpus_file = open("toy_corpus.txt")

with open("unigram_eval.txt", "w") as f:
    for line in toycorpus_file:
        words = line.lower().split()
        sentprob = 1
        sent_len = len(words)
        for word in words:
            index = word_index_dict[word]
            wordprob = probs[index]
            sentprob *= wordprob
        perplexity = 1/(pow(sentprob, 1.0/sent_len))
        
        
        f.write(f"{perplexity}\n")
f.close()
toycorpus_file.close()
                 


