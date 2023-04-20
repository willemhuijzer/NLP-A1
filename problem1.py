# LOAD THE VOCABULARY FILE

word_index_dict = {}
# vf = vocab file in "r" mode (read mode)
with open("brown_vocab_100.txt", "r") as vf:
    for index, line in enumerate(vf):
        word = line.rstrip()
        word_index_dict[word] = index

print(f"Index of the word 'all': {word_index_dict['all']}")
print(f"Index of the word 'resolution': {word_index_dict['resolution']}")
print(f"Length of the dictionary: {len(word_index_dict)}")
