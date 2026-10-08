import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

data = pd.read_csv(
    "dataset/spam.csv",
    encoding="latin-1"
)

print("\nOriginal column names:")
print(data.columns)


# ============================================================
# STEP 2: SELECT REQUIRED COLUMNS
# ============================================================

# Dataset may contain v1/v2 or label/message columns

if "v1" in data.columns and "v2" in data.columns:
    data = data[["v1", "v2"]]
    data.columns = ["label", "message"]

elif "label" in data.columns and "message" in data.columns:
    data = data[["label", "message"]]

else:
    raise ValueError("Dataset columns not recognized!")


# Remove missing values
data = data.dropna()

# Remove duplicate messages
data = data.drop_duplicates()

print("\nDataset shape:")
print(data.shape)

print("\nFirst 5 messages:")
print(data.head())


# ============================================================
# STEP 3: CONVERT LABELS
# ============================================================

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

data = data.dropna()

data["label"] = data["label"].astype(int)


print("\nLabel counts:")
print(data["label"].value_counts())


# ============================================================
# STEP 4: SPLIT DATA
# ============================================================

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data size:", len(X_train))
print("Testing data size:", len(X_test))


# ============================================================
# STEP 5: TF-IDF
# ============================================================

# Word features
word_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    max_features=10000,
    sublinear_tf=True
)

# Character features
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=1,
    max_features=15000,
    sublinear_tf=True
)


# Combine word + character features
vectorizer = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])


X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# ============================================================
# STEP 6: TRAIN MODEL
# ============================================================

model = LinearSVC(
    C=2.0,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed successfully!")


# ============================================================
# STEP 7: TEST MODEL
# ============================================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL ACCURACY:", round(accuracy * 100, 2), "%")
print("===================================")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["HAM", "SPAM"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# STEP 8: ADD SPAM PATTERN DETECTION
# ============================================================

def spam_pattern_score(message):

    message = message.lower()

    score = 0

    # --------------------------------------------------------
    # Money / financial words
    # --------------------------------------------------------

    money_patterns = [
        r"\bwin\b",
        r"\bwinner\b",
        r"\bwon\b",
        r"\bprize\b",
        r"\breward\b",
        r"\bcash\b",
        r"\bmoney\b",
        r"\blo[a-z]*n\b",
        r"\bloan\b",
        r"\bcredit\b",
        r"\bjackpot\b",
        r"\blottery\b",
    ]

    for pattern in money_patterns:
        if re.search(pattern, message):
            score += 2


    # --------------------------------------------------------
    # Promotional words
    # --------------------------------------------------------

    promotional_patterns = [
        r"\bfree\b",
        r"\boffer\b",
        r"\bdiscount\b",
        r"\bdeal\b",
        r"\bbonus\b",
        r"\bexclusive\b",
        r"\bpromo\b",
        r"\bpromotion\b",
        r"\bsubscribe\b",
        r"\bclaim\b",
        r"\bavail\b",
        r"\bget\b.*\bfree\b",
    ]

    for pattern in promotional_patterns:
        if re.search(pattern, message):
            score += 1


    # --------------------------------------------------------
    # Recharge / telecom promotional messages
    # --------------------------------------------------------

    recharge_patterns = [
        r"\brecharge\b",
        r"\bunlimited calls?\b",
        r"\b\d+\s*gb\b",
        r"\b\d+\s*mb\b",
        r"\b\d+\s*sms\b",
        r"\b\d+\s*days?\b",
        r"\bvalidity\b",
        r"\bdata\b.*\bcall",
        r"\bcall\b.*\bdata",
    ]

    recharge_found = 0

    for pattern in recharge_patterns:
        if re.search(pattern, message):
            recharge_found += 1

    # Multiple telecom promotional terms = strong spam signal
    if recharge_found >= 2:
        score += 4


    # --------------------------------------------------------
    # Urgency
    # --------------------------------------------------------

    urgent_patterns = [
        r"\burgent\b",
        r"\bact now\b",
        r"\bcall now\b",
        r"\bhurry\b",
        r"\blimited time\b",
        r"\btoday only\b",
        r"\bexpires?\b",
        r"\blast chance\b",
    ]

    for pattern in urgent_patterns:
        if re.search(pattern, message):
            score += 2


    # --------------------------------------------------------
    # Suspicious links
    # --------------------------------------------------------

    if re.search(r"https?://", message):
        score += 2

    if re.search(r"\bwww\.", message):
        score += 2

    # Short suspicious-looking links
    if re.search(r"\b[a-zA-Z0-9-]+\.(in|com|net|xyz|tk)\b", message):
        score += 1


    # --------------------------------------------------------
    # Phone / missed call promotions
    # --------------------------------------------------------

    if re.search(r"\bmissed call\b", message):
        score += 2

    if re.search(r"\bcall\s+(now|on)\b", message):
        score += 2


    # --------------------------------------------------------
    # OTP + recharge/offer combination
    # --------------------------------------------------------

    if "otp" in message and (
        "recharge" in message
        or "offer" in message
        or "free" in message
        or "get" in message
    ):
        score += 3


    return score


# ============================================================
# STEP 9: FINAL PREDICTION FUNCTION
# ============================================================

def predict_message(message):

    # Machine learning prediction
    transformed_message = vectorizer.transform([message])

    ml_prediction = model.predict(transformed_message)[0]

    # Spam pattern score
    pattern_score = spam_pattern_score(message)

    # --------------------------------------------------------
    # Combine ML + rule based detection
    # --------------------------------------------------------

    if pattern_score >= 4:
        final_prediction = 1

    elif pattern_score >= 2 and ml_prediction == 1:
        final_prediction = 1

    else:
        final_prediction = ml_prediction

    return final_prediction, pattern_score


# ============================================================
# STEP 10: TEST SOME EXAMPLE MESSAGES
# ============================================================

test_messages = [

    "URGENT! You have won a FREE prize. Call now to claim your reward!",

    "Congratulations! You won 1000000 Rs. Claim your prize now.",

    "Need extra funds? Apply for a Personal Loan at 1kx.in",

    "You're missing out on incoming calls & OTPs! Recharge with Rs199 & get unlimited calls, 2GB data & 100 SMS/day for 28 days.",

    "Recharge now and get unlimited calls and 2GB data.",

    "Hi, how are you doing today?",

    "Can you send me the notes?",

    "I will meet you tomorrow.",

    "Please call me when you reach home."

]


print("\n===================================")
print("EXAMPLE PREDICTIONS")
print("===================================")

for message in test_messages:

    prediction, score = predict_message(message)

    if prediction == 1:
        result = "🚨 SPAM"
    else:
        result = "✅ HAM / NOT SPAM"

    print("\nMessage:", message)
    print("Pattern score:", score)
    print("Prediction:", result)


# ============================================================
# STEP 11: USER INPUT
# ============================================================

print("\n===================================")
print("EMAIL / SMS SPAM DETECTOR")
print("===================================")

while True:

    message = input(
        "\nEnter an email/message (or type 'exit' to quit): "
    )

    if message.lower() == "exit":
        print("Program closed.")
        break

    if message.strip() == "":
        print("Please enter a message.")
        continue

    prediction, score = predict_message(message)

    print("Spam pattern score:", score)

    if prediction == 1:
        print("Prediction: 🚨 SPAM")
    else:
        print("Prediction: ✅ HAM / NOT SPAM")
        # Save the trained model and vectorizer
        import joblib

        joblib.dump(model, "models/spam_model.pkl")
        joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

        print("\nModel saved successfully!")
        print("Vectorizer saved successfully!")