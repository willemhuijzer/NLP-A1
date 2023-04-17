#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

word_index_dict = {}

# vf = vocab file in "r" mode (read mode)
with open("brown_vocab_100.txt", "r") as vf:
    word_index_dict = {}

    for index, line in enumerate(vf):
        word = line.rstrip()
        word_index_dict[word] = index

# wf = word_to_index file in "w" mode (write mode)
with open("word_to_index_100.txt", "w") as wf:
    # Convert the dictionary to a string using the str() function
    dict_str = str(word_index_dict)
    wf.write(dict_str)



print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
