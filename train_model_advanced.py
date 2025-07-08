import pandas as pd
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("phishing_site_urls.csv")

# Feature extraction functions
def has_ip(url):
    ip_pattern = r"(https?:\/\/)?(\d{1,3}\.){3}\d{1,3}"
    return 1 if re.search(ip_pattern, url) else 0

def count_dots(url):
    return url.count(".")

def has_at(url):
    return 1 if "@" in url else 0

def uses_https(url):
    return 1 if url.startswith("https") else 0

def uses_shortener(url):
    shorteners = ["bit.ly", "tinyurl", "goo.gl", "ow.ly", "is.gd", "buff.ly", "adf.ly"]
    return 1 if any(s in url for s in shorteners) else 0

def extract_features(url):
    return [
        len(url),
        count_dots(url),
        has_at(url),
        has_ip(url),
        uses_https(url),
        uses_shortener(url)
    ]

# Extract features
features = df["url"].apply(extract_features)
X = pd.DataFrame(features.tolist(), columns=["length", "dots", "has_at", "has_ip", "https", "shortener"])
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {acc:.2f}")

# Save
joblib.dump(model, "phishing_model.pkl")
print("✅ phishing_model.pkl saved")
