# -----------------------------
# Q1: Preprocessing and Exploratory Data Analysis
# -----------------------------
# Shu Hui
print("Q1: Preprocessing and Exploratory Data Analysis - Shu Hui")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter

# Download required resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
print("Loading dataset")
path_dir = Path(__file__).resolve().parent
df_dir = path_dir / "sentiment_tweets3.csv"
df = pd.read_csv(df_dir)
df.columns = ['Index', 'message', 'label']

# Step 1: Clean raw text 
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

print("Cleaning text...")
df['clean_message'] = df['message'].astype(str).apply(clean_text)

# Step 2: Stopwords removal and stemming 
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

print("Preprocessing text (stopword removal + stemming)")
df['processed_message'] = df['clean_message'].apply(preprocess_text)

# 1. Class Distribution 
print("\nQ1.1: Class Distribution")
label_counts = df['label'].value_counts()
for label, count in label_counts.items():
    label_name = "Not Depressed" if label == 0 else "Depressed"
    print(f"{label} ({label_name}): {count} tweets")

# Pie chart
plt.figure(figsize=(6,6))
plt.pie(label_counts, labels=['Not Depressed (0)', 'Depressed (1)'], autopct='%1.1f%%', startangle=90)
plt.title("Tweet Label Distribution")
plt.axis('equal')
plt.show()

# 2. Average Tweet Length 
print("\nQ1.2: Average Tweet Length by Label")
df['tweet_length'] = df['message'].astype(str).apply(len)
avg_lengths = df.groupby('label')['tweet_length'].mean()
print(f"Not Depressed (0): {avg_lengths[0]:.2f} characters")
print(f"Depressed (1): {avg_lengths[1]:.2f} characters")

# Bar chart
plt.figure(figsize=(6,4))
sns.barplot(x=avg_lengths.index, y=avg_lengths.values)
plt.xticks([0, 1], ['Not Depressed', 'Depressed'])
plt.ylabel("Average Tweet Length")
plt.title("Average Tweet Length by Label")
plt.show()

# 3. Top 20 Frequent Words 
print("\nQ1.3: Top 20 Most Frequent Words")
all_words = ' '.join(df['processed_message']).split()
common_words = Counter(all_words).most_common(20)

for word, count in common_words:
    print(f"{word}: {count}")

# Plot
words = [word for word, count in common_words]
counts = [count for word, count in common_words]

plt.figure(figsize=(10, 6))
plt.barh(words[::-1], counts[::-1], color='skyblue')
plt.xlabel('Frequency')
plt.title('Top 20 Most Frequent Words in Tweets')
plt.tight_layout()
plt.show()

print("-------------------")

# -----------------------------
# Individual Section: Q2: Supervised Text Classification Model + Q3: Hyper Parameter Selection
# -----------------------------

print("Individual Sections: Q2: Supervised Text Classification Model + Q3: Hyper Parameter Selection")

# Miyoko Pang: Logistic Regression
print("Miyoko Pang (TP067553)")
print("\nQ2: Supervised Text Classification Model: Logistic Regression")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

X = df['processed_message']
y = df['label']

# Train-test split
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
end_time = time.time()
print(f"Data split. Time taken: {(end_time - start_time) * 1000000:.2f} µs")

# TF-IDF vectorization
start_time = time.time()
vectorizer = TfidfVectorizer( max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
end_time = time.time()
print(f"TF-IDF vectorization complete. Time taken: {(end_time - start_time) * 1000000:.2f} µs")

# Train logistic regression
start_time = time.time()
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
end_time = time.time()
print(f"Model trained. Time taken: {(end_time - start_time) * 1000000:.2f} µs")

# Predict
start_time = time.time()
y_pred = model.predict(X_test_vec)
end_time = time.time()
print(f"Prediction complete. Time taken: {(end_time - start_time) * 1000000:.2f} µs")

# Results
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nQ3: Hyper Parameter Selection - Logistic Regression")

from sklearn.model_selection import GridSearchCV

params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200]
}

start_time = time.time()
grid = GridSearchCV(LogisticRegression(), params, cv=5, scoring='f1', verbose=1)
grid.fit(X_train_vec, y_train)
end_time = time.time()

print("Best Params:", grid.best_params_)
print(f"Grid Search Time: {(end_time - start_time) * 1000:.2f} ms")

print("-------------------")

# Shu Hui: KNN
print("Shu Hui")
print("\nQ2: Supervised Text Classification Model: KNN")

print("\nQ3: Hyper Parameter Selection - KNN")

print("-------------------")

# Yi Jing
print("Yi Jing")
print("\nQ2: Supervised Text Classification Model: ?")

print("\nQ3: Hyper Parameter Selection - ?")

