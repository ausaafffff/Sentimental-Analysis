import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data.csv")
print("✅ Data loaded:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3))

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

df['cleaned'] = df['Overview'].apply(clean_text)

threshold = df['IMDB_Rating'].median()
print("Threshold:", threshold)
df['label'] = df['IMDB_Rating'].map(lambda x: 1 if x >= threshold else 0)
print("Label balance:\n", df['label'].value_counts())

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)
X = vectorizer.fit_transform(df['cleaned']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("model.pkl created!")
print("vectorizer.pkl created!")