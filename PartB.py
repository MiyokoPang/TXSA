import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("C:/Users/User/Downloads/sentiment_tweets3.csv/sentiment_tweets3.csv")
df.columns = ['Index', 'message', 'label']

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources 
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Clean raw text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)                 # Remove @
    text = re.sub(r'[^\w\s]', '', text)                  # Remove punctuation
    text = text.lower()                                  # Convert to lowercase
    return text

df['clean_message'] = df['message'].astype(str).apply(clean_text)

# Step 2: Remove stopwords + apply stemming
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Add new column with processed message
df['processed_message'] = df['clean_message'].apply(preprocess_text)

# 1. Basic class distribution

label_counts = df['label'].value_counts()
print("\nLabel Distribution:")
for label, count in label_counts.items():
    label_name = "Not Depressed" if label == 0 else "Depressed"
    print(f"{label} ({label_name}): {count} tweets")

# Pie chart view
plt.figure(figsize=(6,6))
plt.pie(label_counts, labels=['Not Depressed (0)', 'Depressed (1)'], autopct='%1.1f%%', startangle=90)
plt.title("Tweet Label Distribution")
plt.axis('equal')
plt.show()

# 2. Average tweet length by label

df['tweet_length'] = df['message'].astype(str).apply(len)
avg_lengths = df.groupby('label')['tweet_length'].mean()
print("\nAverage Tweet Length by Label:")
print(f"Not Depressed (0): {avg_lengths[0]:.2f} characters")
print(f"Depressed (1): {avg_lengths[1]:.2f} characters")

# Visual comparison
plt.figure(figsize=(6,4))
sns.barplot(x=avg_lengths.index, y=avg_lengths.values)
plt.xticks([0, 1], ['Not Depressed', 'Depressed'])
plt.ylabel("Average Tweet Length")
plt.title("Average Tweet Length by Label")
plt.show()

# 3. Show Top 20 Most Frequent Words

from collections import Counter
import matplotlib.pyplot as plt

# Combine all processed tweets into one big string and split into words
all_words = ' '.join(df['processed_message']).split()

# Count the 20 most common words
common_words = Counter(all_words).most_common(20)

# Print top 20 most frequent words
print("\nTop 20 Most Frequent Words:")
for word, count in common_words:
    print(f"{word}: {count}")

# Separate words and their counts for plotting
words = [word for word, count in common_words]
counts = [count for word, count in common_words]

# Plot horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(words[::-1], counts[::-1], color='skyblue')  # Reverse for highest word on top
plt.xlabel('Frequency')
plt.title('Top 20 Most Frequent Words in Tweets')
plt.tight_layout()
plt.show()