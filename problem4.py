# SMOOTHING
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

# Add-α smoothing
counts += 0.1

# Normalize the counts
probs = normalize(counts, norm='l1', axis=1)

# Write the bigram probabilities to a file
print(f"p(the | all): {probs[word_index_dict['all'], word_index_dict['the']]:.7f}\n")
print(f"p(jury | the): {probs[word_index_dict['the'], word_index_dict['jury']]:.7f}\n")
print(f"p(campaign | the): {probs[word_index_dict['the'], word_index_dict['campaign']]:.7f}\n")
print(f"p(calls | anonymous): {probs[word_index_dict['anonymous'], word_index_dict['calls']]:.7f}\n")

# Generate sentences using the bigram model
generated_sentences = [GENERATE(word_index_dict, probs, model_type='bigram', max_words=50, start_word='<s>') for _ in range(10)]

# Store the generated sentences in a file smoothed_generation.txt
with open("smoothed_generation.txt", "w") as f:
    for sentence in generated_sentences:
        f.write(f"{sentence}\n")
                 
for i, sentence in enumerate(generated_sentences, start=1):  # HELP heel veel <s> <s> <s>
    print(f"Sentence {i}: {sentence}\n")

#Iterate through every sentence in the toy corpus
toycorpus_file = open("toy_corpus.txt")


with open("smoothed_eval.txt", "w") as f:
    for line in toycorpus_file:
        numbigrams = []
        previous_word = "<s>"
        words = line.lower().split()
        sentprob = 1
        # sent_len = len(words)
        for word in words[1:]:
            bigram = (previous_word, word)
            numbigrams.append(bigram)
            sent_len = len(numbigrams)
            wordprob = probs[word_index_dict[previous_word], word_index_dict[word]]
            sentprob *= wordprob
            previous_word = word
        perplexity = 1/(pow(sentprob, 1.0/sent_len))
        
        f.write(f"{perplexity}\n")
f.close()
toycorpus_file.close()
