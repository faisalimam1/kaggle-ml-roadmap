# ============================================================
# Day 8 - SMS Spam Detection | Text Preprocessing
# Dataset: SMS Spam Collection (UCI / Kaggle)
# Author: Faisal
# ============================================================

# ---------------------------
# Step 1: Import Libraries
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

# ---------------------------
# Step 2: Load Dataset
# ---------------------------
df = pd.read_csv(
    '/kaggle/input/datasets/organizations/uciml/sms-spam-collection-dataset/spam.csv',
    encoding='latin-1'
)
print("Raw shape:", df.shape)
print(df.head())

# ---------------------------
# Step 3: Clean Column Structure
# ---------------------------
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['label', 'message']
print("Cleaned shape:", df.shape)

# ---------------------------
# Step 4: EDA
# ---------------------------
# Missing values
print("\nMissing values:\n", df.isnull().sum())

# Class distribution
print("\nClass distribution:\n", df['label'].value_counts())
print("\nClass distribution (%):\n", df['label'].value_counts(normalize=True) * 100)

# Visualize distribution
sns.countplot(x='label', data=df)
plt.title('Spam vs Ham Distribution')
plt.show()

# Message length analysis
df['message_length'] = df['message'].apply(len)
print("\nAverage message length:\n", df.groupby('label')['message_length'].mean())

# Visualize message length
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
# Step 5: Text Preprocessing
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
print("After cleaning:", clean_text(test_message))

# Apply to entire dataset
df['cleaned_message'] = df['message'].apply(clean_text)

# Verify
print("\nOriginal vs Cleaned:\n")
print(df[['message', 'cleaned_message']].head(10))