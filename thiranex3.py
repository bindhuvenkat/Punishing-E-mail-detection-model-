import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

# Load dataset
df = pd.read_csv("emails.csv")

# Extract URL & keyword features
def extract_features(text):

    urls = re.findall(
        r'https?://\S+|www\.\S+',
        str(text)
    )

    num_urls = len(urls)

    suspicious_keywords = [
        'verify',
        'bank',
        'login',
        'password',
        'account',
        'urgent',
        'click'
    ]

    keyword_count = sum(
        word in str(text).lower()
        for word in suspicious_keywords
    )

    return pd.Series([num_urls, keyword_count])

df[['num_urls','keyword_count']] = df['email_text'].apply(
    extract_features
)

# TF-IDF
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_text = tfidf.fit_transform(df['email_text'])

# Combine features
X = hstack([
    X_text,
    df[['num_urls','keyword_count']]
])

y = df['label']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))