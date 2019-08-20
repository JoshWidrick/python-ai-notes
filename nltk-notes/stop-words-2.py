from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


example_text = 'This is an example showing off stop word filtration.'

stop_words = set(stopwords.words('english'))

# print(stop_words

words = word_tokenize(example_text)

# filtered_text = []
# for w in words:
#     if w not in stop_words:
#         filtered_text.append(w)

filtered_text = [w for w in words if w not in stop_words]

print(filtered_text)





























