import nltk
from nltk.corpus import brown
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

nltk.download("brown")
nltk.download("averaged_perceptron_tagger")

def analyze_corpus(corpus, categories=None):
    # Load the data from the corpus
    if categories:
        tokens = brown.words(categories=categories)
        words = [token for token in tokens if token.isalnum()] # of isalpha ????? is een getal een woord?
        sents = brown.sents(categories=categories)
    else:
        tokens = brown.words()
        words = [token for token in tokens if token.isalnum()] # of isalpha ????? is een getal een woord?
        sents = brown.sents()

    # Calculate the statistics
    num_tokens = len(tokens)
    num_types = len(set(tokens))
    num_words = len(words) 
    num_sents = len(sents)
    avg_words_per_sentence = num_words / num_sents
    avg_word_length = sum(len(word) for word in words) / num_words

    # POS tagging
    pos_tags = nltk.pos_tag(tokens)
    pos_tag_counts = Counter(tag for _, tag in pos_tags).most_common(10)

    # Frequency distribution
    word_freq = Counter(tokens)
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    return {
        "num_tokens": num_tokens,
        "num_types": num_types,
        "num_words": num_words,
        "num_sents": num_sents,
        "avg_words_per_sentence": avg_words_per_sentence,
        "avg_word_length": avg_word_length,
        "pos_tag_counts": pos_tag_counts,
        "sorted_word_freq": sorted_word_freq,
    }
    
def plot_frequency_curves(data, title):
    # Extract frequencies from the sorted_word_freq data
    frequencies = [freq for _, freq in data["sorted_word_freq"]]
    
    # Create a linear plot
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies)
    plt.title(f"Frequency Curve (Linear) - {title}")
    plt.xlabel("Position in Frequency List")
    plt.ylabel("Frequency")
    plt.show()

    # Create a log-log plot
    plt.figure(figsize=(10, 5))
    plt.loglog(frequencies)
    plt.title(f"Frequency Curve (Log-Log) - {title}")
    plt.xlabel("Position in Frequency List")
    plt.ylabel("Frequency")
    plt.show()


whole_corpus_stats = analyze_corpus(brown)
news_stats = analyze_corpus(brown, categories="news")
romance_stats = analyze_corpus(brown, categories="romance")

print("Whole Corpus:")
for key, value in whole_corpus_stats.items():
    if key != "sorted_word_freq":
        print(f"{key}: {value}")

print("\nNews:")
for key, value in news_stats.items():
    if key != "sorted_word_freq":
        print(f"{key}: {value}")

print("\nRomance:")
for key, value in romance_stats.items():
    if key != "sorted_word_freq":
        print(f"{key}: {value}")

# Plot frequency curves for the whole corpus
plot_frequency_curves(whole_corpus_stats, "Whole Corpus")

# Plot frequency curves for the news genre
plot_frequency_curves(news_stats, "News")

# Plot frequency curves for the romance genre
plot_frequency_curves(romance_stats, "Romance")
