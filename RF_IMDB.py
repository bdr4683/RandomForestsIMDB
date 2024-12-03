import pandas as pd
import numpy as np
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

data = pd.read_csv("IMDB_Dataset.csv")

# Encode sentiment labels: 'positive' as 1, 'negative' as 0
data['label'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Preprocessing the reviews
def preprocess_review(text):
    text = str(text).replace('<br />', ' ')  # Remove HTML tags
    text = ''.join([c.lower() if c.isalnum() or c.isspace() else '' for c in text])  # Remove punctuation and lowercase
    return text

data['review'] = data['review'].apply(preprocess_review)

# Select 5,000 for training and 5,000 for testing
train_data = data[:5000]  # First 5,000 rows for training
test_data = data[5000:10000]  # Next 5,000 rows for testing

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Preprocessing function: Tokenize and convert to bag-of-words
def preprocess_data(df, vocab=None):
    if vocab is None:
        # Create a vocabulary by counting word frequencies
        vocab = Counter(" ".join(df['review'].fillna("").values).split())
        vocab = {word: idx for idx, word in enumerate(vocab.keys())}

    # Convert text to feature vector (bag-of-words)
    def text_to_vector(text):
        vector = np.zeros(len(vocab))
        for word in text.split():
            if word in vocab:
                vector[vocab[word]] += 1
        return vector

    # Create a new DataFrame to avoid warnings
    new_df = df.copy()
    new_df['features'] = new_df['review'].fillna("").apply(text_to_vector)
    return new_df, vocab

# Preprocess the training data
train_data, vocab = preprocess_data(train_data)

# Preprocess the test data using the same vocabulary
test_data, _ = preprocess_data(test_data, vocab)

# Split the data into features and labels
X_train = np.stack(train_data['features'].values)
y_train = np.array(train_data['label'])

X_test = np.stack(test_data['features'].values)
y_test = np.array(test_data['label'])

