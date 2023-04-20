# ANALYSIS OF THE BROWN CORPUS
import nltk
from nltk.corpus import brown
from collections import Counter
import matplotlib.pyplot as plt

nltk.download("brown")
nltk.download("averaged_perceptron_tagger")

def plot_frequency_curves(data, title):
    # Extract frequencies from the sorted_word_freq data
    frequencies = [freq for _, freq in data["sorted_word_freq"]]
    
    # Linear plot creation
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies)
    plt.title(f"Frequency Curve (Linear) - {title}")
    plt.xlabel("Position in Frequency List")
    plt.ylabel("Frequency")
    plt.show()

    # Log-log plot creation
    plt.figure(figsize=(10, 5))
    plt.loglog(frequencies)
    plt.title(f"Frequency Curve (Log-Log) - {title}")
    plt.xlabel("Position in Frequency List")
    plt.ylabel("Frequency")
    plt.show()


def analyze_corpus(corpus, categories=None):
    # Load the data from the corpus
    if categories:
        tokens = brown.words(categories=categories)
        words = [token for token in tokens if token.isalnum()] # of isalpha ????? is een getal een woord?
        sentences = brown.sents(categories=categories)
    else:
        tokens = brown.words()
        words = [token for token in tokens if token.isalnum()] # of isalpha ????? is een getal een woord?
        sentences = brown.sents()

    # Calculate the statistics
    number_tokens = len(tokens)
    number_types = len(set(tokens))
    number_words = len(words) 
    number_sentences = len(sentences)
    avg_words_per_sentence = number_words / number_sentences
    avg_word_length = sum(len(word) for word in words) / number_words

    # POS tagging
    pos_tags = nltk.pos_tag(tokens)
    pos_tag_counts = Counter(tag for _, tag in pos_tags).most_common(10)

    # Frequency distribution
    word_freq = Counter(tokens)
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    return {    "number_tokens": number_tokens,
                "number_types": number_types,
                "number_words": number_words,
                "number_sentences": number_sentences,
                "avg_words_per_sentence": avg_words_per_sentence,
                "avg_word_length": avg_word_length,
                "pos_tag_counts": pos_tag_counts,
                "sorted_word_freq": sorted_word_freq,}
    

whole_corpus_stats = analyze_corpus(brown)
news_stats = analyze_corpus(brown, categories="humor")
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

# Plot the frequency curves
plot_frequency_curves(whole_corpus_stats, "Whole Corpus")
plot_frequency_curves(news_stats, "Humor")
plot_frequency_curves(romance_stats, "Romance")
