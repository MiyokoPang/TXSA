# Data_1.txt Import
with open("Data_1.txt", "r", encoding="utf-8") as file:
    data_1 = file.read()

# Q1 Word Tokenization
# split() function
print('Q1 split function')
tokenization_1 = data_1.split()
print(tokenization_1)

# Regular Expression function - Yi Jing
print('Q1 RE function')


# NLTK function - Shu Hui 
print('Q1 NLTK function')


print('-----------------------------')

# Q2 Form Word Stemming
# Regular Expression function - Yi Jing
print('Q2 RE function')


# NLTK PorterStemmer function
print('Q2 PorterStemmer function')
import nltk
nltk.download('punkt_tab')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
words = word_tokenize(data_1)
print([ps.stem(w) for w in words])

# NLTK LancasterStemmer function - Shu Hui
print('Q2 LancasterStemmer function')


# Data_2.txt Import
with open("Data_2.txt", "r", encoding="utf-8") as file:
    data_2 = file.read()

print('-----------------------------')
# Q3 POS Taggers and Syntactic Analysers
# NLTK POS tagger function - Shu Hui
print('Q3 NLTK POS Tagger')


# TextBlob POS Tagger
print('Q3 TextBlob POS Tagger')
nltk.download('averaged_perceptron_tagger_eng')
from textblob import TextBlob
blob = TextBlob(data_2)
print(blob.tags)

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



