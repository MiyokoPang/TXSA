import time
import json
import pprint
from pprint import pprint
import nltk
from textblob import TextBlob
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk import pos_tag, bigrams, FreqDist, ConditionalFreqDist
from nltk.tag import RegexpTagger
from nltk.probability import ConditionalProbDist, MLEProbDist, LidstoneProbDist

# Downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function for cleaner display format - Miyoko
def print_in_chunks(token_list, chunk_size=5):
    for i in range(0, len(token_list), chunk_size):
        print(token_list[i:i+chunk_size])

# -----------------------------
# Data_1.txt Import
# -----------------------------
with open("Data_1.txt", "r", encoding="utf-8") as file:
    data_1 = file.read()

# Q1 Word Tokenization
# split() function - Miyoko Pang
print('Q1 split function')
start = time.time()
tokenization_1 = data_1.split()
end = time.time()
print_in_chunks(tokenization_1, 5)
print(f"Time taken: {(end - start) * 1000000:.2f} ms")

# Regular Expression function - Yi Jing
print('Q1 RE function')
start_time = time.time()
tokens = re.findall(r'\b\w+\b', data_1)
end_time = time.time()

for i in range(0, len(tokens), 10):
    print(', '.join(tokens[i:i+10]))

print(f"\nTotal number of tokens: {len(tokens)}")
print(f"Time taken: {(end_time - start_time) * 1000000:.2f} ms")

# NLTK function - Shu Hui
print('Q1 NLTK function')
start = time.time()
tokenization = word_tokenize(data_1)
end = time.time()
print_in_chunks(tokenization, 5)
print(f"Time taken: {(end - start) * 1000000:.2f} ms")

print('-----------------------------')

# Q1.3: Stop Words & Punctuation Removal - Yi Jing


# Q2 Form Word Stemming
# Regular Expression function - Yi Jing
print('Q2 RE function')
start_time = time.time()
tokens = re.findall(r'\b\w+\b', data_1)

def simple_stem(word):
    return re.sub(r'(ing|ed|ly|es|s)$', '', word)

stemmed_tokens = [simple_stem(token.lower()) for token in tokens]
end_time = time.time()

for i in range(0, len(stemmed_tokens), 10):
    print(', '.join(stemmed_tokens[i:i+10]))

print(f"Time taken: {(end_time - start_time) * 1000000:.2f} ms")

# NLTK PorterStemmer function - Miyoko Pang
print('Q2 PorterStemmer function')
start = time.time()
ps = PorterStemmer()
porter_stems = [ps.stem(w) for w in word_tokenize(data_1)]
end = time.time()
print_in_chunks(porter_stems, 5)
print(f"Time taken: {(end - start) * 1000000:.2f} ms")

# NLTK LancasterStemmer function - Shu Hui
print('Q2 LancasterStemmer function')
start = time.time()
ls = LancasterStemmer()
lancaster_stems = [ls.stem(w) for w in word_tokenize(data_1)]
end = time.time()
print_in_chunks(lancaster_stems, 5)
print(f"Time taken: {(end - start) * 1000000:.2f} ms")

# -----------------------------
# Data_2.txt Import
# -----------------------------
with open("Data_2.txt", "r", encoding="utf-8") as file:
    data_2 = file.read()

print('-----------------------------')

# Q3 POS Taggers and Syntactic Analysers
# NLTK POS tagger function - Shu Hui
print('Q3 NLTK POS Tagger')
start = time.time()
pos_nltk = pos_tag(word_tokenize(data_2))
end = time.time()
print_in_chunks(pos_nltk, 5)
print(f"Time taken: {(end - start) * 1000000:.2f} ms")

# TextBlob POS Tagger - Miyoko Pang
print('Q3 TextBlob POS Tagger')
start = time.time()
blob = TextBlob(data_2)
blob_pos = blob.tags
end = time.time()
print_in_chunks(blob_pos, 5)
print(f"Time taken: {(end - start) * 1000000:.2f} ms")

# Regular Expression tagger - Yi Jing
print('Q3 Regular Expression POS Tagger')
start_time = time.time()
tokens = word_tokenize(data_2)

patterns = [
    (r'.*ing$', 'VBG'),                 # gerunds
    (r'.*ed$', 'VBD'),                  # past tense verbs
    (r'.*es$', 'VBZ'),                  # 3rd person singular present
    (r'.*ould$', 'MD'),                 # modals
    (r'.*\'s$', 'NN$'),                 # possessive nouns
    (r'.*s$', 'NNS'),                   # plural nouns
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),   # numbers
    (r'^(the|a|an)$', 'DT'),            # articles
    (r'^(and|or|but)$', 'CC'),          # conjunctions
    (r'^(at|in|on|with|away)$', 'IN'),  # prepositions
    (r'.*', 'NN')                       # default
]

regexp_tagger = RegexpTagger(patterns)
tagged = regexp_tagger.tag(tokens)
end_time = time.time()

for word, tag in tagged:
    print(f"{word:<10} => {tag}")
print(f"\nTime taken: {(end_time - start_time) * 1000000:.2f} ms")

print('-----------------------------')

# Q4 Sentence Probabilities - Bigram Models
print('Q4 Unsmoothed and Smoothed Bigram Model')

# Q5 Individual Work Assignments
# Miyoko Pang 


# Yi Jing     


# Shu Hui     
