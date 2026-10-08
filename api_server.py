from flask import Flask, request, jsonify
from flask_cors import CORS

import csv
import io
import json
import joblib
import math
import os
import re
from urllib.parse import urlparse


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "spam_model.pkl",
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl",
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "spam.csv",
)

REGISTRY_PATH = os.path.join(
    BASE_DIR,
    "model_registry.json",
)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    MODEL_LOADED = True

    print("====================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("Model:", type(model).__name__)
    print("Vectorizer:", type(vectorizer).__name__)
    print("====================================")

except Exception as e:
    model = None
    vectorizer = None
    MODEL_LOADED = False

    print("====================================")
    print("MODEL LOAD ERROR")
    print(str(e))
    print("====================================")


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):
    text = str(text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text, terms):
    return any(term in text for term in terms)


def count_terms(text, terms):
    return sum(1 for term in terms if term in text)


# ============================================================
# URL HELPERS
# ============================================================

def extract_urls(text):
    """Extract normal URLs plus URLs surrounded by punctuation."""
    urls = re.findall(
        r"(?:https?://|www\.)[^\s<>\]\[(){}]+",
        text,
        flags=re.IGNORECASE,
    )
    return [u.rstrip(".,!?;:'\"") for u in urls]


def get_domain(url):
    try:
        clean = url.strip().strip("[](){}<>")

        if clean.startswith("www."):
            clean = "https://" + clean
        elif not re.match(r"^https?://", clean, flags=re.IGNORECASE):
            clean = "https://" + clean

        domain = urlparse(clean).netloc.lower().split(":")[0]

        if domain.startswith("www."):
            domain = domain[4:]

        return domain
    except Exception:
        return ""


def url_is_suspicious(url):
    domain = get_domain(url)

    if not domain:
        return True

    # Explicitly suspicious/shortened/demo-looking domains.
    suspicious_fragments = [
        "bit.ly",
        "tinyurl.",
        "t.co",
        "rb.gy",
        "shorturl.",
        "goo.gl",
        "track-link",
        "tracking-link",
        "secure-check",
        "account-check",
        "verify-account",
        "support-check",
        "refund-support",
        "reward-center",
        ".example",
    ]

    if contains_any(domain, suspicious_fragments):
        return True

    # Raw IP addresses are suspicious in notification messages.
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", domain):
        return True

    # Punycode is commonly used for lookalike domains.
    if "xn--" in domain:
        return True

    # Excessive hyphenation often appears in deceptive lookalike domains.
    if domain.count("-") >= 3:
        return True

    return False


def has_suspicious_url(text):
    return any(url_is_suspicious(url) for url in extract_urls(text))


# ============================================================
# OFFICIAL SERVICE DOMAINS
# These are only used for a positive legitimacy signal.
# ============================================================

OFFICIAL_DOMAINS = {
    "jio": {"jio.com", "reliancejio.com"},
    "airtel": {"airtel.in", "airtel.com"},
    "vi": {"myvi.in", "vodafone.in"},
    "bsnl": {"bsnl.co.in"},
}


def domain_matches(domain, allowed_domains):
    if not domain:
        return False

    return any(
        domain == allowed
        or domain.endswith("." + allowed)
        for allowed in allowed_domains
    )


def detect_known_provider(text):
    if re.search(r"\breliance\s+jio\b|\bjio\b", text):
        return "jio"

    if re.search(r"\bairtel\b", text):
        return "airtel"

    if re.search(r"\bvoda\s*fone\b|\bvi\b", text):
        return "vi"

    if re.search(r"\bbsnl\b", text):
        return "bsnl"

    return None


# ============================================================
# HIGH-RISK PHISHING / SCAM SIGNALS
# ============================================================

STRONG_SCAM_TERMS = [
    "you have won",
    "you won",
    "winner",
    "lottery",
    "jackpot",
    "free prize",
    "cash prize",
    "claim your prize",
    "claim reward",
    "selected for a reward",
    "earn money",
    "make money",
    "double your money",
    "guaranteed profit",
    "guaranteed return",
    "investment opportunity",
    "prize money",
]


PHISHING_ACTION_TERMS = [
    "update your details",
    "update details",
    "enter your details",
    "confirm your details",
    "verify your details",
    "update your address",
    "confirm your address",
    "verify your address",
    "enter password",
    "enter your password",
    "share password",
    "share otp",
    "send otp",
    "enter otp",
    "enter cvv",
    "share cvv",
    "verify your account",
    "confirm your account",
    "verify account",
    "confirm bank account",
    "share bank details",
    "enter bank details",
]


URGENCY_TERMS = [
    "immediately",
    "urgent",
    "act now",
    "within 24 hours",
    "within 48 hours",
    "today",
    "last chance",
    "before it expires",
    "avoid cancellation",
    "avoid return",
    "avoid return to sender",
    "avoid suspension",
    "avoid closure",
]


def has_strong_scam_signal(text):
    return contains_any(text, STRONG_SCAM_TERMS)


def has_phishing_action(text):
    return contains_any(text, PHISHING_ACTION_TERMS)


def has_urgency(text):
    return contains_any(text, URGENCY_TERMS)


# ============================================================
# GENERIC PHISHING DETECTOR
# ============================================================

def detect_phishing(text):
    """Return (True, reason) only for strong combinations."""

    urls = extract_urls(text)
    suspicious_url = has_suspicious_url(text)
    phishing_action = has_phishing_action(text)
    urgency = has_urgency(text)

    delivery_context = contains_any(
        text,
        [
            "parcel",
            "package",
            "delivery",
            "shipment",
            "tracking",
            "return to sender",
            "address details",
        ],
    )

    account_context = contains_any(
        text,
        [
            "account",
            "bank",
            "card",
            "wallet",
            "payment",
            "refund",
        ],
    )

    credential_context = contains_any(
        text,
        [
            "password",
            "otp",
            "cvv",
            "bank details",
            "login",
        ],
    )

    # Strongest case: suspicious link + request for sensitive details.
    if suspicious_url and (phishing_action or credential_context):
        return True, "Suspicious link with sensitive-action request"

    # Delivery phishing: fake hold/address/update + suspicious link.
    if (
        suspicious_url
        and delivery_context
        and (phishing_action or urgency)
    ):
        return True, "Suspicious delivery/address link"

    # Account/payment phishing: suspicious link + urgency + sensitive context.
    if (
        suspicious_url
        and account_context
        and urgency
    ):
        return True, "Suspicious account/payment link"

    # Explicit phishing wording without URL can still be high-risk when it
    # asks for credentials or financial details urgently.
    if (
        credential_context
        and urgency
        and phishing_action
    ):
        return True, "Credential phishing request"

    # Strong scam language is spam even without a URL.
    if has_strong_scam_signal(text):
        return True, "Strong scam/promotional signal"

    return False, ""


# ============================================================
# LEGITIMATE TRANSACTION DETECTOR
# ============================================================

def detect_transaction_message(text):

    transaction_groups = [
        [
            "debited",
            "credited",
            "transaction",
            "txn",
        ],
        [
            "a/c",
            "account",
            "avl bal",
            "available balance",
            "balance",
        ],
        [
            "ref no",
            "reference no",
            "reference number",
            "transaction id",
            "txn id",
        ],
        [
            "payment successful",
            "payment received",
            "transfer successful",
            "transfer completed",
        ],
    ]

    groups_matched = sum(
        1
        for group in transaction_groups
        if contains_any(text, group)
    )

    risky_credentials = contains_any(
        text,
        [
            "share otp",
            "send otp",
            "enter cvv",
            "share cvv",
            "enter password",
            "share password",
            "share bank details",
        ],
    )

    # A transaction alert should not receive a HAM override if it contains
    # a suspicious link or an explicit scam/phishing request.
    if (
        groups_matched >= 2
        and not risky_credentials
        and not has_suspicious_url(text)
        and not has_strong_scam_signal(text)
    ):
        return True

    return False


# ============================================================
# LEGITIMATE KNOWN PROVIDER SERVICE MESSAGE
# ============================================================

def detect_known_service_message(text):

    provider = detect_known_provider(text)

    if not provider:
        return False

    service_terms = [
        "plan",
        "recharge",
        "validity",
        "service",
        "services",
        "data pack",
        "talktime",
        "renew",
        "renewal",
        "expired",
        "expires",
        "stopped",
        "bill",
        "billing",
    ]

    service_hits = count_terms(
        text,
        service_terms,
    )

    # A legitimate provider message must not ask for passwords/OTP/CVV or
    # personal-detail updates.
    unsafe_action = contains_any(
        text,
        [
            "update your details",
            "update details",
            "enter your details",
            "share otp",
            "send otp",
            "enter otp",
            "share password",
            "enter password",
            "share cvv",
            "enter cvv",
            "share bank details",
        ],
    )

    if unsafe_action:
        return False

    if has_strong_scam_signal(text):
        return False

    urls = extract_urls(text)

    # If there is a URL, require it to match the detected provider.
    if urls:
        has_official_url = any(
            domain_matches(
                get_domain(url),
                OFFICIAL_DOMAINS[provider],
            )
            for url in urls
        )

        if not has_official_url:
            return False

    return service_hits >= 2


# ============================================================
# LEGITIMATE ORDER / DELIVERY MESSAGE
# ============================================================

def detect_legitimate_order_message(text):

    order_terms = [
        "order",
        "order id",
        "order number",
        "shipment",
        "delivery",
        "package",
        "parcel",
        "tracking",
        "tracking number",
    ]

    status_terms = [
        "confirmed",
        "shipped",
        "dispatched",
        "delivered",
        "out for delivery",
        "arriving",
        "expected delivery",
    ]

    order_hits = count_terms(
        text,
        order_terms,
    )

    status_hits = count_terms(
        text,
        status_terms,
    )

    # IMPORTANT:
    # Never mark a delivery message as legitimate when it contains a
    # suspicious link or asks for address/details urgently.
    unsafe_request = contains_any(
        text,
        [
            "update your details",
            "update your address",
            "confirm your address",
            "verify your address",
            "enter your details",
            "pay now",
        ],
    )

    if has_suspicious_url(text):
        return False

    if unsafe_request:
        return False

    if has_strong_scam_signal(text):
        return False

    return (
        order_hits >= 1
        and status_hits >= 1
    )


# ============================================================
# LEGITIMATE BILL / RECEIPT
# ============================================================

def detect_legitimate_billing_message(text):

    bill_terms = [
        "invoice",
        "receipt",
        "bill",
        "amount due",
        "payment received",
        "payment successful",
        "paid successfully",
        "statement",
    ]

    identifier_terms = [
        "invoice no",
        "invoice number",
        "receipt no",
        "receipt number",
        "transaction id",
        "reference number",
        "customer id",
        "account",
    ]

    bill_hits = count_terms(
        text,
        bill_terms,
    )

    identifier_hits = count_terms(
        text,
        identifier_terms,
    )

    if has_suspicious_url(text):
        return False

    if has_phishing_action(text):
        return False

    if has_strong_scam_signal(text):
        return False

    return (
        bill_hits >= 1
        and identifier_hits >= 1
    )


# ============================================================
# SPAM PATTERN SCORE
# This is explanatory/supporting information only.
# It NEVER decides the final label by itself.
# ============================================================

def spam_pattern_score(text):

    text = normalize_text(text)
    score = 0

    if extract_urls(text):
        score += 1

    spam_words = [
        "free",
        "cash",
        "money",
        "prize",
        "winner",
        "won",
        "reward",
        "bonus",
        "offer",
        "claim",
        "lottery",
        "selected",
        "congratulations",
        "limited",
        "deal",
        "jackpot",
    ]

    for word in spam_words:
        if word in text:
            score += 1

    security_words = [
        "password",
        "verify",
        "verification",
        "login",
        "signin",
        "sign in",
        "otp",
        "cvv",
    ]

    for word in security_words:
        if word in text:
            score += 1

    financial_words = [
        "upi",
        "payment",
        "transfer",
        "loan",
        "investment",
        "refund",
    ]

    for word in financial_words:
        if word in text:
            score += 1

    money_patterns = [
        r"₹\s?\d+",
        r"\$\s?\d+",
        r"\b\d+\s?(usd|inr|rupees|dollars)\b",
    ]

    for pattern in money_patterns:
        if re.search(pattern, text):
            score += 1

    if re.search(r"\b\d{10}\b", text):
        score += 1

    urgent_words = [
        "urgent",
        "immediately",
        "act now",
        "act immediately",
        "limited time",
        "hurry",
        "expire",
        "expires",
        "last chance",
    ]

    for word in urgent_words:
        if word in text:
            score += 1

    return score


# ============================================================
# MODEL CONFIDENCE
# ============================================================

def calculate_model_confidence(tfidf, ml_spam):

    try:
        decision = float(
            model.decision_function(tfidf)[0]
        )

        magnitude = abs(decision)

        confidence = (
            1.0
            /
            (1.0 + math.exp(-magnitude))
        ) * 100.0

        return round(float(confidence), 2), decision

    except Exception:
        return (
            100.0 if ml_spam else 0.0,
            0.0,
        )


# ============================================================
# MAIN PREDICTION LOGIC
# ============================================================

def predict_email(message):

    if not message:
        raise ValueError(
            "Email message cannot be empty"
        )

    message = str(message).strip()

    if not message:
        raise ValueError(
            "Email message cannot be empty"
        )

    if model is None or vectorizer is None:
        raise RuntimeError(
            "ML model is not loaded"
        )

    text = normalize_text(message)

    # --------------------------------------------------------
    # ML MODEL
    # --------------------------------------------------------

    tfidf = vectorizer.transform([message])
    prediction = model.predict(tfidf)[0]

    if isinstance(prediction, str):
        ml_spam = prediction.lower() in [
            "spam",
            "1",
            "true",
        ]
    else:
        ml_spam = int(prediction) == 1

    raw_ml_prediction = (
        "SPAM" if ml_spam else "HAM"
    )

    # --------------------------------------------------------
    # SUPPORTING SIGNALS
    # --------------------------------------------------------

    pattern_score = spam_pattern_score(message)

    phishing_detected, phishing_reason = detect_phishing(text)

    transaction_alert = detect_transaction_message(text)

    service_alert = detect_known_service_message(text)

    order_alert = detect_legitimate_order_message(text)

    billing_alert = detect_legitimate_billing_message(text)

    legitimate_notification = any([
        transaction_alert,
        service_alert,
        order_alert,
        billing_alert,
    ])

    if transaction_alert:
        notification_type = "transaction"
    elif service_alert:
        notification_type = "service"
    elif order_alert:
        notification_type = "order"
    elif billing_alert:
        notification_type = "billing"
    else:
        notification_type = ""

    # --------------------------------------------------------
    # FINAL DECISION PRIORITY
    # --------------------------------------------------------
    # 1. Strong phishing/scam ALWAYS wins.
    # 2. Clear legitimate structured notification can override
    #    a raw ML false positive.
    # 3. Otherwise use the trained LinearSVC output.

    if phishing_detected:

        final_prediction = "SPAM"
        decision_basis = phishing_reason

    elif legitimate_notification:

        final_prediction = "HAM"
        decision_basis = (
            "Structured legitimate "
            + notification_type
            + " notification"
        )

    else:

        final_prediction = (
            "SPAM" if ml_spam else "HAM"
        )
        decision_basis = "LinearSVC model"

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    model_confidence, decision_score = (
        calculate_model_confidence(
            tfidf,
            ml_spam,
        )
    )

    if final_prediction == "SPAM":

        if phishing_detected:
            confidence = max(
                85.0,
                model_confidence,
            )
        else:
            confidence = model_confidence

    else:

        if legitimate_notification and ml_spam:
            confidence = max(
                75.0,
                100.0 - model_confidence,
            )
        else:
            confidence = model_confidence

    return {
        "prediction": final_prediction,
        "confidence": round(float(confidence), 2),
        "spam_pattern_score": pattern_score,

        # Raw trained model output remains visible for transparency.
        "ml_prediction": raw_ml_prediction,
        "raw_ml_prediction": raw_ml_prediction,

        "legitimate_notification": legitimate_notification,
        "notification_type": notification_type,

        "transaction_alert": transaction_alert,
        "service_alert": service_alert,
        "order_alert": order_alert,
        "billing_alert": billing_alert,

        "phishing_detected": phishing_detected,
        "phishing_reason": phishing_reason,
        "suspicious_url": has_suspicious_url(text),
        "strong_scam_signal": has_strong_scam_signal(text),

        "model_decision_score": round(
            float(decision_score),
            4,
        ),

        "decision_basis": decision_basis,
    }


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "Email Spam Detection API",
        "version": "1.0",
    })


# ============================================================
# HEALTH
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
    })


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Request body is required",
            }), 400

        message = data.get("message")

        if not message:
            message = data.get("body")

        if not message:
            subject = data.get("subject", "")
            body = data.get("email_body", "")
            message = f"{subject}\n{body}".strip()

        if not message:
            return jsonify({
                "success": False,
                "error": "Email message is required",
            }), 400

        result = predict_email(message)

        return jsonify({
            "success": True,
            **result,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# ============================================================
# BATCH CSV PREDICTION
# ============================================================

@app.route("/batch/predict", methods=["POST"])
def predict_batch():

    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "CSV file is required",
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected",
            }), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({
                "success": False,
                "error": "Only CSV files are supported",
            }), 400

        content = file.read().decode(
            "utf-8-sig",
            errors="replace",
        )

        reader = csv.DictReader(
            io.StringIO(content)
        )

        if not reader.fieldnames:
            return jsonify({
                "success": False,
                "error": "CSV file has no columns",
            }), 400

        results = []

        for index, row in enumerate(reader, start=1):

            clean_row = {
                str(key).strip(): value
                for key, value in row.items()
            }

            message = ""

            possible_columns = [
                "message",
                "body",
                "text",
                "email",
                "content",
                "email_body",
            ]

            for column in possible_columns:
                value = clean_row.get(column, "")

                if (
                    value is not None
                    and str(value).strip()
                ):
                    message = str(value).strip()
                    break

            if not message:
                value = clean_row.get("v2", "")

                if (
                    value is not None
                    and str(value).strip()
                ):
                    message = str(value).strip()

            if not message:
                subject = clean_row.get("subject", "")
                body = clean_row.get("email_body", "")
                message = f"{subject}\n{body}".strip()

            if not message:
                continue

            result = predict_email(message)

            results.append({
                "id": index,
                "message": message,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "spam_pattern_score": result["spam_pattern_score"],
                "ml_prediction": result["ml_prediction"],
                "legitimate_notification": result[
                    "legitimate_notification"
                ],
                "notification_type": result[
                    "notification_type"
                ],
                "phishing_detected": result[
                    "phishing_detected"
                ],
                "suspicious_url": result["suspicious_url"],
                "decision_basis": result["decision_basis"],
            })

        spam_count = sum(
            1 for item in results
            if item["prediction"] == "SPAM"
        )

        ham_count = sum(
            1 for item in results
            if item["prediction"] == "HAM"
        )

        total_count = len(results)

        if total_count:
            average_confidence = (
                sum(
                    float(item["confidence"])
                    for item in results
                )
                / total_count
            )
        else:
            average_confidence = 0

        return jsonify({
            "success": True,
            "total": total_count,
            "spam": spam_count,
            "ham": ham_count,
            "average_confidence": round(
                average_confidence,
                2,
            ),
            "results": results,
        })

    except UnicodeDecodeError:
        return jsonify({
            "success": False,
            "error": "Unable to read CSV encoding",
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# ============================================================
# DATASET
# ============================================================

@app.route("/dataset", methods=["GET"])
def dataset_info():

    try:
        if not os.path.exists(DATASET_PATH):
            return jsonify({
                "success": False,
                "error": "dataset/spam.csv not found",
            }), 404

        with open(
            DATASET_PATH,
            "r",
            encoding="latin-1",
        ) as file:
            reader = csv.reader(file)
            rows = list(reader)

        if not rows:
            return jsonify({
                "success": False,
                "error": "Dataset is empty",
            }), 400

        header = rows[0]

        label_column = (
            "v1" if "v1" in header else None
        )

        text_column = (
            "v2" if "v2" in header else None
        )

        spam_count = 0
        ham_count = 0

        if label_column:
            label_index = header.index(label_column)

            for row in rows[1:]:
                if len(row) <= label_index:
                    continue

                label = str(
                    row[label_index]
                ).strip().lower()

                if label == "spam":
                    spam_count += 1
                elif label == "ham":
                    ham_count += 1

        total_samples = spam_count + ham_count

        return jsonify({
            "success": True,
            "total_samples": total_samples,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "label_column": label_column,
            "text_column": text_column,
            "columns": header,
            "source": "dataset/spam.csv",
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# ============================================================
# MODEL METRICS
# ============================================================

@app.route("/metrics", methods=["GET"])
def metrics():

    return jsonify({
        "success": True,
        "accuracy": 99.03,
        "precision": 100.00,
        "recall": 92.00,
        "f1": 96.00,
        "samples": 5169,
        "training_samples": 4135,
        "testing_samples": 1034,
        "model": {
            "name": "Email Spam Detection Model",
            "version": "v1.0",
            "status": "Production",
            "algorithm": "LinearSVC",
            "framework": "scikit-learn",
            "features": "Word + Character TF-IDF",
        },
        "classification_report": {
            "ham": {
                "precision": 99.0,
                "recall": 100.0,
                "f1": 99.0,
                "support": 903,
            },
            "spam": {
                "precision": 100.0,
                "recall": 92.0,
                "f1": 96.0,
                "support": 131,
            },
            "macro_avg": {
                "precision": 99.0,
                "recall": 96.0,
                "f1": 98.0,
            },
            "weighted_avg": {
                "precision": 99.0,
                "recall": 99.0,
                "f1": 99.0,
            },
        },
        "confusion_matrix": [
            [903, 0],
            [10, 121],
        ],
    })


# ============================================================
# MODEL REGISTRY
# ============================================================

@app.route("/model-registry", methods=["GET"])
def model_registry():

    try:
        if not os.path.exists(REGISTRY_PATH):
            return jsonify({
                "success": False,
                "error": "model_registry.json not found",
            }), 404

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            registry = json.load(file)

        return jsonify({
            "success": True,
            "registry": registry,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("====================================")
    print("EMAIL SPAM DETECTION API")
    print("====================================")
    print("Server: http://localhost:8000")
    print("Health: http://localhost:8000/health")
    print("Predict: http://localhost:8000/predict")
    print("Batch: http://localhost:8000/batch/predict")
    print("Dataset: http://localhost:8000/dataset")
    print("Metrics: http://localhost:8000/metrics")
    print("Registry: http://localhost:8000/model-registry")
    print("====================================")
    print()

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True,
    )