import nltk
from nltk.corpus import brown
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Helper functions
def compute_stats_and_freqs(corpus):
    joined_corpus = " ".join(corpus)  # join sentences into a single string
    tokens = nltk.word_tokenize(joined_corpus)
    freqs = nltk.FreqDist(tokens)
    stats = {"num_tokens": len(tokens), "num_types": len(freqs)}
    return stats, freqs

def pos_tagging(corpus):
    tokens = nltk.word_tokenize(" ".join(corpus))
    tagged_tokens = nltk.pos_tag(tokens)
    pos_tags = [tag for word, tag in tagged_tokens]
    most_common_tags = Counter(pos_tags).most_common(10)
    return most_common_tags

def plot_freq_curves(freqs, title, log_axes=False):
    plt.figure()
    ranks = range(1, len(freqs) + 1)
    frequencies = [f for _, f in freqs]

    if log_axes:
        plt.loglog(ranks, frequencies, marker='.')
    else:
        plt.plot(ranks, frequencies, marker='.')

    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

# Analyzing the whole corpus and chosen genres
whole_corpus = brown.sents()
whole_corpus_stats = compute_stats_and_freqs(whole_corpus)

genre1 = brown.sents(categories='news')
genre1_stats = compute_stats_and_freqs(genre1)

genre2 = brown.sents(categories='romance')
genre2_stats = compute_stats_and_freqs(genre2)

print("Whole corpus statistics:", whole_corpus_stats)
print("Genre 1 (news) statistics:", genre1_stats)
print("Genre 2 (romance) statistics:", genre2_stats)

# POS tagging
whole_corpus_pos_tags = pos_tagging(whole_corpus)
print("Ten most frequent POS tags in the whole corpus:", whole_corpus_pos_tags)

# Plot frequency curves
plot_freq_curves(whole_corpus_stats['freqs'], 'Whole corpus frequency curve')
plot_freq_curves(whole_corpus_stats['freqs'], 'Whole corpus frequency curve (log-log)', log_axes=True)
plot_freq_curves(genre1_stats['freqs'], 'Genre 1 (news) frequency curve')
plot_freq_curves(genre1_stats['freqs'], 'Genre 1 (news) frequency curve (log-log)', log_axes=True)
plot_freq_curves(genre2_stats['freqs'], 'Genre 2 (romance) frequency curve')
plot_freq_curves(genre2_stats['freqs'], 'Genre 2 (romance) frequency curve (log-log)', log_axes=True)
