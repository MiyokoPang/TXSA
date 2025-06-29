import time
import json
import pprint
from pprint import pprint

# Function for cleaner display format - Miyoko
def print_in_chunks(token_list, chunk_size=5):
    for i in range(0, len(token_list), chunk_size):
        print(token_list[i:i+chunk_size])


# Data_1.txt Import
with open("Data_1.txt", "r", encoding="utf-8") as file:
    data_1 = file.read()

# Q1 Word Tokenization
# split() function
print('Q1 split function')
start = time.time()
tokenization_1 = data_1.split()
end = time.time()
print_in_chunks(tokenization_1, 5)
print(f"Time taken: {end - start:.4f} seconds")

# Regular Expression function - Yi Jing
print('Q1 RE function')


# NLTK function - Shu Hui 
print('Q1 NLTK function')
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
tokenization = word_tokenize(data_1)
print(tokenization)




print('-----------------------------')

# Q2 Form Word Stemming
# Regular Expression function - Yi Jing
print('Q2 RE function')


# NLTK PorterStemmer function
print('Q2 PorterStemmer function')
import nltk
nltk.download('punkt_tab')
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize

start = time.time()
ps = PorterStemmer()
porter_stems = [ps.stem(w) for w in word_tokenize(data_1)]
end = time.time()
print_in_chunks(porter_stems, 5)
print(f"Time taken: {end - start:.4f} seconds")

# NLTK LancasterStemmer function - Shu Hui
import nltk
from nltk.stem import LancasterStemmer
print('Q2 LancasterStemmer function')
ls = LancasterStemmer()
words = word_tokenize(data_1)
print([ls.stem(w) for w in words])



# Data_2.txt Import
with open("Data_2.txt", "r", encoding="utf-8") as file:
    data_2 = file.read()

print('-----------------------------')
# Q3 POS Taggers and Syntactic Analysers
# NLTK POS tagger function - Shu Hui
print('Q3 NLTK POS Tagger')
import nltk
from nltk import pos_tag, word_tokenize
pos_nltk = pos_tag(data_2)
print(pos_tag)


# TextBlob POS Tagger
print('Q3 TextBlob POS Tagger')
nltk.download('averaged_perceptron_tagger_eng')
from textblob import TextBlob

start = time.time()
blob = TextBlob(data_2)
blob_pos = blob.tags
end = time.time()
print_in_chunks(blob_pos, 5)
print(f"Time taken: {end - start:.4f} seconds")

# Regular Expression tagger - Yi Jing
print('Q3 Regular Expression POS Tagger')


print('-----------------------------')
# Q4 Sentence Probabilities
# Unsmoothed and smoothed bigram model 
print('Q5 Unsmoothed and Smoothed Bigram Model')


# Q5 Indvidual
# Miyoko Pang



# Yi Jing



# Shu Hui



