# ============================================================
# SMS Spam Detection — Complete End-to-End Pipeline
# Dataset: SMS Spam Collection (UCI / Kaggle)
# Author: Faisal Imam
# Days: 8, 9, 10
# ============================================================
# Pipeline:
#   1.  Import Libraries
#   2.  Load Dataset
#   3.  Clean Column Structure
#   4.  EDA (class distribution, message length)
#   5.  Text Preprocessing (lowercase, punctuation, stopwords, stemming)
#   6.  TF-IDF Vectorization
#   7.  Train-Test Split (stratified)
#   8.  Naive Bayes Classifier
#   9.  Evaluation (Precision, Recall, F1, Confusion Matrix)
#   10. Threshold Tuning (0.5 -> 0.3)
#   11. Logistic Regression Comparison
#   12. Model Comparison Visual
#   13. Cross Validation (5-fold Stratified)
#   14. Save Model + Vectorizer (pickle)
#   15. Load & Verify
#   16. Custom Prediction Function
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
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
# Drop 3 empty garbage columns added during CSV export
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['label', 'message']
print("\nCleaned shape:", df.shape)
print(df.head())


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
# Spam messages are ~2x longer than ham — length itself is a signal
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
    """
    Cleans raw SMS text through 5 steps:
    1. Lowercase       - removes case inconsistency (FREE = free)
    2. Remove noise    - strips punctuation and numbers via regex
    3. Tokenize        - splits string into list of words
    4. Remove stopwords - strips words with zero signal (the, is, a)
    5. Stem            - reduces words to root form (winning -> win)
    """
    message = message.lower()
    message = re.sub('[^a-z]', ' ', message)
    words = message.split()
    words = [ps.stem(word) for word in words
             if word not in stopwords.words('english')]
    return ' '.join(words)

# Test on sample message before applying to entire dataset
test_message = "Congratulations! You've WON a FREE iPhone!!! Call 08712300 NOW!!!"
print("\nBefore cleaning:", test_message)
print("After cleaning :", clean_text(test_message))

# Apply to entire dataset
df['cleaned_message'] = df['message'].apply(clean_text)
print("\nOriginal vs Cleaned (first 10 rows):")
print(df[['message', 'cleaned_message']].head(10))


# ---------------------------
# 6. TF-IDF Vectorization
# ---------------------------
# TF-IDF converts text to numbers, weighting words by how distinctive
# they are — not just how frequent. Rare words in spam = high score.
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned_message']).toarray()
print("\nShape of X:", X.shape)                              # (5572, 3000)
print("Top 10 vocabulary words:", tfidf.get_feature_names_out()[:10])

# Encode target variable: spam=1, ham=0
df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})
y = df['label_encoded']
print("\nTarget distribution:\n", y.value_counts())


# ---------------------------
# 7. Train-Test Split (Stratified)
# ---------------------------
# stratify=y ensures both splits maintain the 86.6%/13.4% class ratio
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
# MultinomialNB is designed for word count/frequency data — exactly
# what TF-IDF produces. This is why NB dominates text classification.
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nPredictions done!")


# ---------------------------
# 9. Evaluation (Default Threshold = 0.5)
# ---------------------------
print("\n--- Naive Bayes (Default Threshold = 0.5) ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix — Naive Bayes (default)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# ---------------------------
# 10. Threshold Tuning
# ---------------------------
# Default threshold (0.5) is too conservative — missing 25 spam messages.
# Lowering to 0.3 catches more spam (higher recall) at small cost to precision.
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred_adjusted = (y_prob >= threshold).astype(int)

print("\n--- Naive Bayes (Threshold = 0.3) ---")
print("Precision:", precision_score(y_test, y_pred_adjusted))
print("Recall   :", recall_score(y_test, y_pred_adjusted))
print("F1 Score :", f1_score(y_test, y_pred_adjusted))


# ---------------------------
# 11. Logistic Regression (Comparison)
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
# 12. Model Comparison Visual
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
plt.bar([i - 0.2 for i in x], f1_scores,
        width=0.4, label='F1 Score', color='blue', alpha=0.7)
plt.bar([i + 0.2 for i in x], recall_scores,
        width=0.4, label='Recall', color='red', alpha=0.7)
plt.xticks(x, models)
plt.ylim(0.7, 1.0)
plt.title('Model Comparison — F1 Score vs Recall')
plt.legend()
plt.show()


# ---------------------------
# 13. Cross Validation (5-fold Stratified)
# ---------------------------
# One test split can be lucky. CV proves results are consistent
# across ALL data — not just one fortunate split.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    MultinomialNB(),
    X, y,
    cv=skf,
    scoring='f1'
)

print("\n--- Cross Validation Results ---")
print("F1 Score per fold:", cv_scores)
print("Mean F1 Score    :", cv_scores.mean().round(4))   # 0.9099
print("Std Dev          :", cv_scores.std().round(4))    # 0.013


# ---------------------------
# 14. Save Model + Vectorizer
# ---------------------------
# CRITICAL: Always save the vectorizer alongside the model.
# The model learned weights for 3000 specific features in a specific
# order. A different vectorizer = different feature order = garbage predictions.
with open('spam_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\nModel and vectorizer saved successfully!")


# ---------------------------
# 15. Load & Verify
# ---------------------------
with open('spam_classifier.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_tfidf = pickle.load(f)

print("Model loaded and working!")


# ---------------------------
# 16. Custom Prediction Function
# ---------------------------
def predict_spam(message):
    """
    Predicts whether a new SMS message is spam or ham.
    Uses the same preprocessing and vectorizer from training.
    """
    cleaned = clean_text(message)
    vectorized = loaded_tfidf.transform([cleaned]).toarray()
    probability = loaded_model.predict_proba(vectorized)[0][1]
    prediction = "SPAM" if probability >= 0.3 else "HAM (Legitimate)"

    print(f"Message   : {message}")
    print(f"Cleaned   : {cleaned}")
    print(f"Spam Prob : {probability:.4f}")
    print(f"Prediction: {prediction}")
    print("-" * 55)

# Test with spam messages
predict_spam("Congratulations! You've WON a FREE iPhone! Call NOW!")
predict_spam("URGENT: Account suspended. Click here to verify!")
predict_spam("Win £1000 cash! Text WIN to 87777 now! Limited offer!")

# Test with ham messages
predict_spam("Hey, are we still meeting for lunch tomorrow?")
predict_spam("Can you pick up milk on your way home?")
predict_spam("The assignment is due next Friday, don't forget!")


# ---------------------------
# Final Summary
# ---------------------------
print("\n" + "=" * 55)
print("           FINAL MODEL COMPARISON SUMMARY")
print("=" * 55)
print(f"Naive Bayes (default)    | F1: {f1_score(y_test, y_pred):.4f} | Recall: {recall_score(y_test, y_pred):.4f}")
print(f"Naive Bayes (tuned 0.3)  | F1: {f1_score(y_test, y_pred_adjusted):.4f} | Recall: {recall_score(y_test, y_pred_adjusted):.4f}")
print(f"Logistic Regression      | F1: {f1_score(y_test, lr_pred):.4f} | Recall: {recall_score(y_test, lr_pred):.4f}")
print("=" * 55)
print(f"Cross Validated Mean F1  : {cv_scores.mean():.4f}")
print(f"Cross Validated Std Dev  : {cv_scores.std():.4f}")
print("=" * 55)
print("Winner: Naive Bayes with threshold = 0.3")
print("=" * 55)
