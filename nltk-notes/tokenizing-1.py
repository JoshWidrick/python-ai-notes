import nltk


# nltk.download()

"""
overview
---

here, we are taking a body of text, and we need to pull it apart to be able to get it to a point that we can use it
for something. we do this through organizing the text through tokenizing.

terms
---

tokenizing - word tokeinzers, sentence tokenizers. system / process of seperating large bodies of text by word, 
sentence, etc.
lexicon - words and their meanings.
corpora - body of text. ex. medical journals.
"""

from nltk.tokenize import sent_tokenize, word_tokenize


example_text = 'Hello there, how are you doing today? The weather is great and Python is awesome. The sky is pinkish-blue. You should not eat cardboard.'

print(sent_tokenize(example_text))
print(word_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)























