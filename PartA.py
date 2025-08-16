



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
print_in_chunks(filtered_tokens)

print("\nStop Words Found and Removed:")
print_in_chunks(found_stopwords)

print(f"\nTotal Stop Words Removed: {len(found_stopwords)}")
print(f"Time taken: {(end_time - start_time) * 1000000:.2f} µs")

print('-----------------------------')

# Q2 Form Word Stemming
# Regular Expression function - Yi Jing
print('Q2 RE function')
start_time = time.time()
tokens = re.findall(r'\b\w+\b', data_1)

def simple_stem(word):
    return re.sub(r'(ing|ed|ly|es|s)$', '', word)

stemmed_tokens = [simple_stem(token.lower()) for token in tokens]
end_time = time.time()

print_in_chunks(stemmed_tokens)

print(f"Time taken: {(end_time - start_time) * 1000000:.2f} µs")

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

print('-----------------------------')

# -----------------------------
# Data_3.txt Import
# -----------------------------
with open("Data_3.txt", "r", encoding="utf-8") as file:
    data_3 = file.read()

print('-----------------------------')

# Q4 Sentence Probabilities - Bigram Models
print('Q4 Unsmoothed and Smoothed Bigram Model')
all_sentences = re.findall(r"<s>.*?</s>", data_3) # Extract wrapped sentences
test_sentence = all_sentences[-1]
train_sentences = all_sentences[:-1]
tokenized_train = [['<s>'] + s.replace('<s>', '').replace('</s>', '').strip().split() + ['</s>'] for s in train_sentences]

# Bigram Models (MLE and Laplace)
start = time.time()
n = 2
train_data_mle, padded_sents_mle = padded_everygram_pipeline(n, tokenized_train)
train_data_laplace, padded_sents_laplace = padded_everygram_pipeline(n, tokenized_train)
mle_model = MLE(n)
laplace_model = Laplace(n)
mle_model.fit(train_data_mle, padded_sents_mle)
laplace_model.fit(train_data_laplace, padded_sents_laplace)
test_tokens = ['<s>'] + test_sentence.replace('<s>', '').replace('</s>', '').strip().split() + ['</s>']
test_ngrams = list(ngrams(test_tokens, n))

# Calculate probabilities
prob_mle = 1.0
prob_laplace = 1.0
for w1, w2 in test_ngrams:
    mle_score = mle_model.score(w2, [w1])
    laplace_score = laplace_model.score(w2, [w1])
    print(f"Bigram ({w1}, {w2}): MLE={mle_score:.10f}, Laplace={laplace_score:.10f}")  # Debug line
    prob_mle *= mle_score
    prob_laplace *= laplace_score
end = time.time()
print(f"Test Sentence: {' '.join(test_tokens)}")
print(f"MLE Bigram Probability: {prob_mle:.10f}")
print(f"Laplace-smoothed Bigram Probability: {prob_laplace:.10f}")
print(f"\nTime taken to generate bigram probabilities: {(end - start) * 1000:.2f} ms")

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
print("Q5 spaCy Tokenizer")
import spacy
nlp = spacy.load("en_core_web_sm")

# Clean newline and extra whitespace
clean_data_1 = re.sub(r'\s+', ' ', data_1.strip())

start = time.time()
doc = nlp(clean_data_1)
tokens = [token.text for token in doc]
end = time.time()

print_in_chunks(tokens)
print(f"\nTotal Tokens: {len(tokens)}")
print(f"Time taken: {(end - start) * 1000000:.2f} µs")

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
