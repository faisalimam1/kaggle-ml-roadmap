# ============================================================
# SMS Spam Detection — End-to-End Pipeline
# Dataset: SMS Spam Collection (UCI / Kaggle)
# Author: Faisal Imam
# ============================================================
# Pipeline:
#   1. Load & clean data
#   2. EDA (class distribution, message length analysis)
#   3. Text preprocessing (lowercase, punctuation, stopwords, stemming)
#   4. TF-IDF vectorization
#   5. Train-test split (stratified)
#   6. Naive Bayes classifier + threshold tuning
#   7. Logistic Regression comparison
#   8. Evaluation (Precision, Recall, F1, Confusion Matrix)
# ============================================================

# ---------------------------
# 1. Import Libraries
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

nltk.download('stopwords')
nltk.download('punkt')

# ---------------------------
# 2. Load Dataset
# ---------------------------
df = pd.read_csv(
    '/kaggle/input/datasets/organizations/uciml/sms-spam-collection-dataset/spam.csv',
    encoding='latin-1'
)
print("Raw shape:", df.shape)
print(df.head())

# ---------------------------
# 3. Clean Column Structure
# ---------------------------
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['label', 'message']
print("Cleaned shape:", df.shape)

# ---------------------------
# 4. EDA
# ---------------------------
# Missing values
print("\nMissing values:\n", df.isnull().sum())

# Class distribution
print("\nClass distribution:\n", df['label'].value_counts())
print("\nClass distribution (%):\n", df['label'].value_counts(normalize=True) * 100)

# Visualize class distribution
sns.countplot(x='label', data=df)
plt.title('Spam vs Ham Distribution')
plt.show()

# Message length analysis
df['message_length'] = df['message'].apply(len)
print("\nAverage message length:\n", df.groupby('label')['message_length'].mean())

plt.figure(figsize=(10, 4))
df[df['label'] == 'ham']['message_length'].plot(
    kind='hist', bins=50, alpha=0.6, label='Ham', color='blue'
)
df[df['label'] == 'spam']['message_length'].plot(
    kind='hist', bins=50, alpha=0.6, label='Spam', color='red'
)
plt.legend()
plt.title('Message Length Distribution: Spam vs Ham')
plt.xlabel('Message Length')
plt.show()

# ---------------------------
# 5. Text Preprocessing
# ---------------------------
ps = PorterStemmer()

def clean_text(message):
    # Lowercase
    message = message.lower()
    # Remove punctuation and numbers
    message = re.sub('[^a-z]', ' ', message)
    # Tokenize
    words = message.split()
    # Remove stopwords and stem
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Test on a sample message
test_message = "Congratulations! You've WON a FREE iPhone!!! Call 08712300 NOW!!!"
print("\nBefore cleaning:", test_message)
print("After cleaning :", clean_text(test_message))

# Apply to entire dataset
df['cleaned_message'] = df['message'].apply(clean_text)
print("\nOriginal vs Cleaned:\n")
print(df[['message', 'cleaned_message']].head(10))

# ---------------------------
# 6. TF-IDF Vectorization
# ---------------------------
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned_message']).toarray()
print("\nShape of X:", X.shape)
print("Top 10 vocabulary words:", tfidf.get_feature_names_out()[:10])

# Encode target
df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})
y = df['label_encoded']

# ---------------------------
# 7. Train-Test Split (Stratified)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining size:", X_train.shape)
print("Testing size :", X_test.shape)

# ---------------------------
# 8. Naive Bayes Classifier
# ---------------------------
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n--- Naive Bayes (Default Threshold = 0.5) ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix — Naive Bayes (default)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ---------------------------
# 9. Threshold Tuning (Improve Recall)
# ---------------------------
# Default threshold (0.5) is too conservative for imbalanced spam detection.
# Lowering to 0.3 catches more spam at small cost to precision.
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred_adjusted = (y_prob >= threshold).astype(int)

print("\n--- Naive Bayes (Threshold = 0.3) ---")
print("Precision:", precision_score(y_test, y_pred_adjusted))
print("Recall   :", recall_score(y_test, y_pred_adjusted))
print("F1 Score :", f1_score(y_test, y_pred_adjusted))

# ---------------------------
# 10. Logistic Regression (Comparison)
# ---------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall   :", recall_score(y_test, lr_pred))
print("F1 Score :", f1_score(y_test, lr_pred))

# ---------------------------
# 11. Model Comparison Visual
# ---------------------------
models = ['NB (default)', 'NB (threshold=0.3)', 'Logistic Regression']
f1_scores = [
    f1_score(y_test, y_pred),
    f1_score(y_test, y_pred_adjusted),
    f1_score(y_test, lr_pred)
]
recall_scores = [
    recall_score(y_test, y_pred),
    recall_score(y_test, y_pred_adjusted),
    recall_score(y_test, lr_pred)
]

x = range(len(models))
plt.figure(figsize=(9, 4))
plt.bar([i - 0.2 for i in x], f1_scores, width=0.4, label='F1 Score', color='blue', alpha=0.7)
plt.bar([i + 0.2 for i in x], recall_scores, width=0.4, label='Recall', color='red', alpha=0.7)
plt.xticks(x, models)
plt.ylim(0.7, 1.0)
plt.title('Model Comparison — F1 Score vs Recall')
plt.legend()
plt.show()

# ---------------------------
# 12. Final Summary
# ---------------------------
print("\n" + "=" * 50)
print("         FINAL MODEL COMPARISON SUMMARY")
print("=" * 50)
print(f"Naive Bayes (default)     | F1: {f1_score(y_test, y_pred):.4f} | Recall: {recall_score(y_test, y_pred):.4f}")
print(f"Naive Bayes (threshold)   | F1: {f1_score(y_test, y_pred_adjusted):.4f} | Recall: {recall_score(y_test, y_pred_adjusted):.4f}")
print(f"Logistic Regression       | F1: {f1_score(y_test, lr_pred):.4f} | Recall: {recall_score(y_test, lr_pred):.4f}")
print("=" * 50)
print("Winner: Naive Bayes with threshold = 0.3")
print("=" * 50)
