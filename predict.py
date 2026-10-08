import joblib
import re


# ============================================================
# LOAD SAVED MODEL
# ============================================================

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# ============================================================
# SPAM PATTERN DETECTION
# ============================================================

def spam_pattern_score(message):

    message = message.lower()
    score = 0

    # Money / financial words
    money_patterns = [
        r"\bwin\b",
        r"\bwinner\b",
        r"\bwon\b",
        r"\bprize\b",
        r"\breward\b",
        r"\bcash\b",
        r"\bmoney\b",
        r"\bloan\b",
        r"\bcredit\b",
        r"\bjackpot\b",
        r"\blottery\b"
    ]

    for pattern in money_patterns:
        if re.search(pattern, message):
            score += 2


    # Promotional words
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
        r"\bavail\b"
    ]

    for pattern in promotional_patterns:
        if re.search(pattern, message):
            score += 1


    # Recharge / telecom messages
    recharge_patterns = [
        r"\brecharge\b",
        r"\bunlimited calls?\b",
        r"\d+\s*gb",
        r"\d+\s*mb",
        r"\d+\s*sms",
        r"\d+\s*days?",
        r"\bvalidity\b"
    ]

    recharge_found = 0

    for pattern in recharge_patterns:
        if re.search(pattern, message):
            recharge_found += 1

    if recharge_found >= 2:
        score += 4


    # Urgency
    urgent_patterns = [
        r"\burgent\b",
        r"\bact now\b",
        r"\bcall now\b",
        r"\bhurry\b",
        r"\blimited time\b",
        r"\btoday only\b",
        r"\bexpires?\b",
        r"\blast chance\b"
    ]

    for pattern in urgent_patterns:
        if re.search(pattern, message):
            score += 2


    # Links
    if re.search(r"https?://", message):
        score += 2

    if re.search(r"\bwww\.", message):
        score += 2


    # Missed call
    if re.search(r"\bmissed call\b", message):
        score += 2


    # OTP + promotional message
    if "otp" in message and (
        "recharge" in message
        or "offer" in message
        or "free" in message
    ):
        score += 3


    return score


# ============================================================
# FINAL PREDICTION
# ============================================================

def predict_message(message):

    # ML prediction
    message_tfidf = vectorizer.transform([message])

    ml_prediction = model.predict(message_tfidf)[0]

    # Pattern score
    pattern_score = spam_pattern_score(message)

    # Combine ML + pattern detection
    if pattern_score >= 4:
        final_prediction = 1

    elif pattern_score >= 2 and ml_prediction == 1:
        final_prediction = 1

    else:
        final_prediction = ml_prediction

    return final_prediction, pattern_score


# ============================================================
# SPAM DETECTOR
# ============================================================

print("===================================")
print("       EMAIL SPAM DETECTOR")
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