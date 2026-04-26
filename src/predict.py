# ================================================================
#  Grievance Prediction with Confidence Scores
# ================================================================
#  This script loads pre-trained models and predicts:
#    - Category  (e.g., Roads, Healthcare, Electricity...)
#    - Priority  (High / Medium / Low)
#  It also shows HOW CONFIDENT the model is in each prediction.
# ================================================================

import os
import pickle
import warnings
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------
#  STEP 0: Download NLTK resources (only runs once)
# ----------------------------------------------------------------
print("=" * 60)
print("  Grievance Prediction System with Confidence Scores")
print("=" * 60)

print("\n[Step 0] Loading NLTK resources...")
for resource in ["wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"]:
    nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
print("  [OK] NLTK ready.")


# ----------------------------------------------------------------
#  STEP 1: Define the SAME preprocessing used during training
#
#  WHY same preprocessing?
#  The TF-IDF vectorizer was fitted on cleaned text.
#  If we pass raw/dirty text now, the words won't match the
#  vocabulary it learned → near-zero feature vector → bad result.
#  Identical preprocessing = same input space = reliable output.
# ----------------------------------------------------------------

# Spelling correction dictionary (same as training)
SPELLING_MAP = {
    "watar": "water", "watear": "water", "watr": "water",
    "shorttage": "shortage", "shoratge": "shortage",
    "pipline": "pipeline", "supplm": "supply", "problm": "problem",
    "potholles": "potholes", "potholz": "potholes", "pothols": "potholes",
    "rods": "roads", "raods": "roads", "brken": "broken",
    "dangerus": "dangerous", "vehiclz": "vehicles",
    "eletricity": "electricity", "electrcity": "electricity",
    "frqunt": "frequent", "frequnt": "frequent",
    "fluctuaton": "fluctuation", "apliances": "appliances",
    "deli": "delhi",
    "doctrs": "doctors", "doctr": "doctor", "docttor": "doctor",
    "avlble": "available", "availble": "available",
    "medcal": "medical", "facilty": "facility",
    "busses": "buses", "buzes": "buses", "buss": "bus",
    "buze": "bus", "commut": "commute", "anoyying": "annoying",
    "convnient": "convenient", "inconvnient": "inconvenient",
    "dumpng": "dumping", "unhygenic": "unhygienic",
    "dirtiness": "dirty", "evrywhre": "everywhere",
    "awfl": "awful", "smells": "smell",
    "certificat": "certificate", "certifcate": "certificate",
    "appliaction": "application", "appoinment": "appointment",
    "verificaton": "verification", "stuk": "stuck", "genrtd": "generated",
    "urgntly": "urgently", "urgenlty": "urgently",
    "imediate": "immediate", "immidiate": "immediate",
    "pleese": "please", "pls": "please", "plz": "please",
    "dayz": "days", "singl": "single", "propaly": "properly",
    "continuosly": "continuously", "everywhre": "everywhere",
    "problm": "problem", "prob": "problem",
    "hydrabad": "hyderabad", "chenai": "chennai",
    "luckow": "lucknow", "nagr": "nagar",
    "sever": "severe",
}


def correct_spelling(text: str) -> str:
    """Replace known messy words with correct versions."""
    for wrong, right in SPELLING_MAP.items():
        text = re.sub(
            r"\b" + re.escape(wrong) + r"\b",
            right, text, flags=re.IGNORECASE
        )
    return text


def get_wordnet_pos(tag: str):
    """Map POS tag first letter to WordNet POS constant."""
    tag_map = {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_map.get(tag[0].upper(), wordnet.NOUN)


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline — MUST be identical to training:
      1. Lowercase
      2. Remove reference codes (RefH123 etc.)
      3. Correct spelling
      4. Remove punctuation / digits
      5. Collapse extra spaces
      6. Lemmatize with POS tagging
    """
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove ref codes
    text = re.sub(r"\bref[hml]\d+\b", "", text, flags=re.IGNORECASE)

    # Step 3: Spelling correction
    text = correct_spelling(text)

    # Step 4: Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 5: Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Step 6: Lemmatize (sentence-level POS tagging — faster)
    tokens = text.split()
    if tokens:
        pos_tags = nltk.pos_tag(tokens)
        tokens = [
            lemmatizer.lemmatize(tok, get_wordnet_pos(pos))
            for tok, pos in pos_tags
        ]

    return " ".join(tokens)


# ----------------------------------------------------------------
#  Duration Extraction Feature
# ----------------------------------------------------------------
def extract_duration(text: str) -> float:
    """Extracts duration mentioned in text and converts it to DAYS."""
    duration_days = 0.0
    text_lower = str(text).lower()
    
    # Match hours / ghante
    hour_match = re.search(r'(\d+)\s*(hour|hours|hr|hrs|ghante|ghanta)', text_lower)
    if hour_match:
        return float(hour_match.group(1)) / 24.0
        
    # Match days / din / dino
    day_match = re.search(r'(\d+)\s*(day|days|din|dino)', text_lower)
    if day_match:
        return float(day_match.group(1))
        
    return duration_days


# ----------------------------------------------------------------
#  STEP 2: Load pre-trained models and SentenceTransformer
#
#  We load the saved LogisticRegression models.
#  We also reload the 'all-MiniLM-L6-v2' embedder.
# ----------------------------------------------------------------
print("\n[Step 1] Loading models and embedder...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

with open(os.path.join(MODEL_DIR, "category_model.pkl"), "rb") as f:
    cat_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "priority_model.pkl"), "rb") as f:
    pri_model = pickle.load(f)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("  [OK] category_model.pkl  loaded")
print("  [OK] priority_model.pkl  loaded")
print("  [OK] sentence embedder   loaded")

# Show classes each model knows
print(f"\n  Category classes : {list(cat_model.classes_)}")
print(f"  Priority classes : {list(pri_model.classes_)}")


# ----------------------------------------------------------------
#  STEP 3: Define the prediction + confidence function
#
#  HOW predict_proba() WORKS:
#  - model.predict_proba(X) returns a 2D array: shape (1, n_classes)
#    e.g. [[0.05, 0.87, 0.03, 0.02, 0.01, 0.01, 0.01]]
#  - Each value = probability that the input belongs to that class
#  - model.classes_ tells you which class maps to which column
#  - max of that array = confidence score of the winning class
# ----------------------------------------------------------------

def predict_with_confidence(complaint_text: str):
    """
    Given raw (possibly messy) complaint text:
      1. Preprocess it (same as training)
      2. Transform with TF-IDF (no re-fitting!)
      3. Predict category and priority
      4. Get confidence scores via predict_proba()
      5. Print results with interpretation
    """

    print("\n" + "=" * 60)
    print("  INPUT & PREDICTION")
    print("=" * 60)

    # --- 3a. Preprocess ---
    print(f"\n  Raw Input   : {complaint_text}")
    cleaned = preprocess(complaint_text)
    duration = extract_duration(complaint_text)
    print(f"  Cleaned     : {cleaned} | Extracted Duration: {duration} days")

    # --- 3b. Convert to sentence embedding & combine with duration ---
    embedding = embedder.encode([cleaned])
    combined = np.hstack([embedding, np.array([[duration]])])

    print("\n" + "-" * 60)

    # ── Model 1: Category (Embeddings Only) ──────────────────────
    cat_pred        = cat_model.predict(embedding)[0]
    cat_proba_array = cat_model.predict_proba(embedding)[0]   # all class probs
    cat_confidence  = cat_proba_array.max()            # highest prob

    # ── Model 2: Priority (Embeddings + Duration) ────────────────
    pri_pred        = pri_model.predict(combined)[0]
    pri_proba_array = pri_model.predict_proba(combined)[0]
    pri_confidence  = pri_proba_array.max()
    
    rule_applied = ""
    if duration > 2:
        pri_pred = "High"
        pri_confidence = 1.0
        rule_applied = " (Rule Override: Duration > 2 days)"

    # ── Print results ─────────────────────────────────────────────
    print(f"\n  Category : {cat_pred}")
    print(f"             Confidence : {cat_confidence * 100:.1f}%")

    print(f"\n  Priority : {pri_pred}{rule_applied}")
    print(f"             Confidence : {pri_confidence * 100:.1f}%")

    # ── Interpretation ───────────────────────────────────────────
    #  If confidence < 60% the model is unsure — treat with caution.
    #  Threshold of 0.60 is a common starting point; adjust as needed.
    print("\n" + "-" * 60)

    cat_level = "Low confidence  - treat with caution" \
                if cat_confidence < 0.60 else \
                "High confidence - reliable prediction"

    pri_level = "Low confidence  - treat with caution" \
                if pri_confidence < 0.60 else \
                "High confidence - reliable prediction"

    print(f"  Category assessment : [{cat_level}]")
    print(f"  Priority assessment : [{pri_level}]")

    # ── Full probability breakdown (bonus detail) ─────────────────
    print("\n  -- Full probability breakdown --")
    print("  Category model:")
    for cls, prob in sorted(
        zip(cat_model.classes_, cat_proba_array),
        key=lambda x: x[1], reverse=True
    ):
        bar = "#" * int(prob * 30)
        print(f"    {cls:<28} {prob*100:5.1f}%  {bar}")

    print("  Priority model:")
    for cls, prob in sorted(
        zip(pri_model.classes_, pri_proba_array),
        key=lambda x: x[1], reverse=True
    ):
        bar = "#" * int(prob * 30)
        print(f"    {cls:<10} {prob*100:5.1f}%  {bar}")

    print("=" * 60)


# ----------------------------------------------------------------
#  STEP 4: Test Execution
# ----------------------------------------------------------------
if __name__ == "__main__":
    test_text = "bijli nahi aa rahi since 3 days"
    predict_with_confidence(test_text)
    print("\n[Done] Prediction pipeline with confidence scores complete.")
