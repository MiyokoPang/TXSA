import time
import json
import pprint
from pprint import pprint
import nltk
from textblob import TextBlob
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk import pos_tag, FreqDist, ConditionalFreqDist, CFG
from nltk.util import bigrams
from nltk.tag import RegexpTagger
from nltk.probability import ConditionalProbDist, MLEProbDist, LidstoneProbDist
from nltk.parse.chart import ChartParser

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

print_in_chunks(tokens)
print(f"\nTotal number of tokens: {len(tokens)}")
print(f"Time taken: {(end_time - start_time) * 1000000:.2f} µs")

# NLTK function - Shu Hui
print('Q1 NLTK function')
start = time.time()
tokenization = word_tokenize(data_1)
end = time.time()
print_in_chunks(tokenization, 5)
print(f"Time taken: {(end - start) * 1000000:.2f} ms")

print('-----------------------------')

# Q1.3: Stop Words & Punctuation Removal - Yi Jing
print("Q1.3 Stop Words & Punctuation Removal")
start_time = time.time()

# Tokenize
tokens = word_tokenize(data_1)

# Prepare stop words and punctuation
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Remove stop words and punctuation
filtered_tokens = []
found_stopwords = []

for word in tokens:
    word_lower = word.lower()
    if word_lower in stop_words:
        found_stopwords.append(word_lower)
    elif word not in punctuation:
        filtered_tokens.append(word)

end_time = time.time()

# Print results
print("Original Token Count:", len(tokens))
print("Filtered Token Count:", len(filtered_tokens))

print("\nFiltered Tokens:")
for i in range(0, len(filtered_tokens), 10):
    print(' '.join(filtered_tokens[i:i+10]))

print("\nStop Words Found and Removed:")
for i in range(0, len(found_stopwords), 10):
    print(', '.join(found_stopwords[i:i+10]))

print(f"\nTotal Stop Words Removed: {len(found_stopwords)}")
print(f"Time taken: {(end_time - start_time) * 1000000:.2f} µs")

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

# Possible Parse Trees - Miyoko and Yi Jing
sentence = re.sub(r'[^\w\s]', '', data_2).lower().split()
print("Tokens:", sentence)
grammar = CFG.fromstring("""
  S -> NP VP
  NP -> Det Adj Adj N
  NP -> Det Adj N
  VP -> V PP Conj V Adv
  PP -> P NP
  Det -> 'the'
  Adj -> 'big' | 'black' | 'white'
  N -> 'dog' | 'cat'
  V -> 'barked' | 'chased'
  P -> 'at'
  Conj -> 'and'
  Adv -> 'away'
""")
parser = ChartParser(grammar)
start = time.time()
trees = list(parser.parse(sentence))
end = time.time()
if not trees:
    print("No valid parse tree could be generated.")
else:
    for tree in trees:
        tree.pretty_print()
        tree.draw()
print(f"\nTime taken to generate parse tree: {(end - start) * 1000000:.2f} ms")

# -----------------------------
# Data_3.txt Import
# -----------------------------
with open("Data_3.txt", "r", encoding="utf-8") as file:
    data_3 = file.read()

print('-----------------------------')

# Q4 Sentence Probabilities - Bigram Models
print('Q4 Unsmoothed and Smoothed Bigram Model')
all_sentences = re.findall(r"<s>.*?</s>", data_3)
test_sentence = all_sentences[-1]
train_sentences = all_sentences[:-1]
train_data = []
for s in train_sentences:
    tokens = s.replace('<s>', '').replace('</s>', '').strip().split()
    train_data.append(["<s>"] + tokens + ["</s>"])
train_tokens = [token for sent in train_data for token in sent]

train_bigrams = list(bigrams(train_tokens))
cfd = ConditionalFreqDist(train_bigrams)
unsmoothed_model = ConditionalProbDist(cfd, MLEProbDist)
smoothed_model = ConditionalProbDist(cfd, lambda fd: LidstoneProbDist(fd, 0.1))
test_tokens = test_sentence.replace('<s>', '').replace('</s>', '').strip().split()
test_tokens = ["<s>"] + test_tokens + ["</s>"]

def calc_prob(sentence_tokens, model, label=""):
    prob = 1.0
    print(f"\n{label} Bigram Probabilities:")
    for w1, w2 in bigrams(sentence_tokens):
        p = model[w1].prob(w2)
        print(f"P({w2} | {w1}) = {p:.10f}")
        prob *= p
    print(f"{label} Sentence Probability = {prob:.10f}")

calc_prob(test_tokens, unsmoothed_model, "[Unsmoothed MLE]")
calc_prob(test_tokens, smoothed_model, "[Smoothed Lidstone]")

print('-----------------------------')

# Q5 Individual Work Assignments
# Miyoko Pang
print('Q5 TreebankWordTokenizer function')
from nltk.tokenize import TreebankWordTokenizer
data_1_clean = data_1.replace('\n', ' ').strip()
start = time.time()
treebank_tokenizer = TreebankWordTokenizer()
tokens = treebank_tokenizer.tokenize(data_1_clean)
end = time.time()
print_in_chunks(tokens, 5)
print(f"Time taken: {(end - start) * 1000000:.2f} ms")

# Yi Jing     


# Shu Hui     
print('Q5 Alternative Tokenizer - WordPunctTokenizer')
from nltk.tokenize import WordPunctTokenizer
import time

# Clean the data
data_1_clean2 = data_1.replace('\n', ' ').strip()

start = time.time()
wordpunct_tokenizer = WordPunctTokenizer()
tokens = wordpunct_tokenizer.tokenize(data_1_clean2)
end = time.time()

print_in_chunks(tokens, 5)
print(f"Time taken: {(end - start) * 1000000:.2f} ms")
