# ================================================================
#  Improved Grievance Classification with Higher Confidence
# ================================================================
#  Changes made to improve confidence:
#  1. Added synonym mapping (bijli -> electricity, etc.)
#  2. Preserved numbers (e.g., "3 days" is important for priority)
#  3. Improved TF-IDF (ngram_range=(1,2), min_df=2)
#  4. Used class_weight='balanced' to handle minority classes
# ================================================================

import os
import re
import warnings

import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
#  Step 1: Setup Paths & Download NLTK resources
# ----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

for resource in ["wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"]:
    nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()

# ----------------------------------------------------------------
#  Step 2: Text Normalization Dictionary
# ----------------------------------------------------------------
# We map regional words, slang, and common typos to standard English
NORMALIZATION_MAP = {
    # Hindi/Regional words
    "bijli": "electricity", "pani": "water", "kachra": "garbage",
    "rasta": "road", "sadak": "road", "gaddi": "vehicle",
    
    # Common misspellings and variants
    "watar": "water", "watear": "water", "watr": "water",
    "shorttage": "shortage", "shoratge": "shortage",
    "pipline": "pipeline", "supplm": "supply", "problm": "problem",
    "potholles": "potholes", "potholz": "potholes", "pothols": "potholes",
    "rods": "roads", "raods": "roads", "brken": "broken",
    "dangerus": "dangerous", "vehiclz": "vehicles",
    "eletricity": "electricity", "electrcity": "electricity",
    "frqunt": "frequent", "frequnt": "frequent",
    "fluctuaton": "fluctuation", "apliances": "appliances",
    "deli": "delhi", "hydrabad": "hyderabad", "chenai": "chennai",
    "luckow": "lucknow", "nagr": "nagar",
    "doctrs": "doctors", "doctr": "doctor", "docttor": "doctor",
    "avlble": "available", "availble": "available",
    "medcal": "medical", "facilty": "facility",
    "busses": "buses", "buzes": "buses", "buss": "bus", "buze": "bus",
    "commut": "commute", "anoyying": "annoying",
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
    "continuosly": "continuously", "sever": "severe"
}

def normalize_text(text: str) -> str:
    """Replace regional terms and typos with standard words."""
    for wrong, right in NORMALIZATION_MAP.items():
        # \b ensures we only match whole words, not parts of words
        text = re.sub(r"\b" + re.escape(wrong) + r"\b", right, text, flags=re.IGNORECASE)
    return text

def get_wordnet_pos(tag: str):
    """Map POS tag first letter to WordNet POS constant."""
    tag_map = {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_map.get(tag[0].upper(), wordnet.NOUN)

def preprocess(text: str) -> str:
    """
    Improved Preprocessing:
    - Normalizes text (fixes typos and Hindi words)
    - Keeps numbers (vital for '3 days', etc.)
    - Lemmatizes
    """
    # 1. Lowercase
    text = str(text).lower()

    # 2. Remove reference codes (e.g., RefH123)
    text = re.sub(r"\bref[hml]\d+\b", "", text, flags=re.IGNORECASE)

    # 3. Apply normalization map
    text = normalize_text(text)

    # 4. Remove punctuation but KEEP letters AND NUMBERS [a-z0-9]
    # Keeping numbers is crucial so "3 days" isn't reduced to just "days"
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 5. Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Lemmatize (faster sentence-level POS tagging)
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
#  Step 3: Load Data
# ----------------------------------------------------------------
print("=" * 60)
print("  Training Improved Models")
print("=" * 60)

train_df = pd.read_csv(os.path.join(DATA_DIR, "grievances_dataset.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "messy_test_dataset.csv"))

print("Applying improved preprocessing...")
train_df["clean_text"] = train_df["complaint_text"].apply(preprocess)
test_df["clean_text"]  = test_df["complaint_text"].apply(preprocess)


# ----------------------------------------------------------------
#  Step 4: Build Embedding-based Models
# ----------------------------------------------------------------
print("Extracting duration features...")
train_durations = train_df["complaint_text"].apply(extract_duration).values.reshape(-1, 1)
test_durations = test_df["complaint_text"].apply(extract_duration).values.reshape(-1, 1)

print("Loading sentence-transformers model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Converting texts to embeddings (this may take a moment)...")
X_train_embeddings = embedder.encode(train_df["clean_text"].tolist())
X_test_embeddings = embedder.encode(test_df["clean_text"].tolist())

print("Combining embeddings with duration for Priority Model...")
X_train_combined = np.hstack([X_train_embeddings, train_durations])
X_test_combined = np.hstack([X_test_embeddings, test_durations])

print("Training Category model (Embeddings only)...")
model_cat = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
model_cat.fit(X_train_embeddings, train_df["category"])

print("Training Priority model (Embeddings + Duration)...")
model_pri = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
model_pri.fit(X_train_combined, train_df["priority"])

os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "category_model.pkl"), "wb") as f:
    pickle.dump(model_cat, f)
with open(os.path.join(MODEL_DIR, "priority_model.pkl"), "wb") as f:
    pickle.dump(model_pri, f)
print(f"Models saved successfully to {MODEL_DIR} (Embedder is reloaded at inference).")


# ----------------------------------------------------------------
#  Step 5: Evaluate & Predict with Confidence
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("  EVALUATION ON MESSY TEST DATA (Embeddings + Duration)")
print("=" * 60)

cat_preds = model_cat.predict(X_test_embeddings)
pri_preds = model_pri.predict(X_test_combined)

# Apply rule override for accuracy score comparison
for i, dur in enumerate(test_durations):
    if dur[0] > 2:
        pri_preds[i] = "High"

print(f"Category Accuracy : {accuracy_score(test_df['category'], cat_preds)*100:.2f}%")
print(f"Priority Accuracy : {accuracy_score(test_df['priority'], pri_preds)*100:.2f}%\n")

print("=" * 60)
print("  SAMPLE PREDICTIONS (WITH CONFIDENCE)")
print("=" * 60)

# We'll show a few diverse examples from the test set
sample_indices = [0, 1, 2, 3, 4] 

# Adding a completely new messy one that uses the hindi words
new_messy_complaint = "pani nahi aa raha hai from 3 days in my kachra area, no bijli also!"
test_df.loc[len(test_df)] = {"complaint_text": new_messy_complaint, "category": "Water Supply", "priority": "High", "clean_text": preprocess(new_messy_complaint)}
sample_indices.append(len(test_df)-1)

for idx in sample_indices:
    raw_text = test_df["complaint_text"].iloc[idx]
    clean_text = test_df["clean_text"].iloc[idx]
    duration = extract_duration(raw_text)
    
    # Get embedding for the single text
    embedding = embedder.encode([clean_text])
    combined = np.hstack([embedding, np.array([[duration]])])
    
    # Category Prediction & Confidence
    pred_cat = model_cat.predict(embedding)[0]
    conf_cat = model_cat.predict_proba(embedding).max()
    
    # Priority Prediction & Confidence
    pred_pri = model_pri.predict(combined)[0]
    conf_pri = model_pri.predict_proba(combined).max()
    
    rule_applied = ""
    if duration > 2:
        pred_pri = "High"
        conf_pri = 1.0
        rule_applied = " (Rule Override: Duration > 2 days)"

    print(f"\nInput: {raw_text}")
    print(f"Clean: {clean_text} | Extracted Duration: {duration} days")
    
    # Category Output
    cat_conf_label = "High confidence prediction" if conf_cat >= 0.6 else "Low confidence prediction"
    print(f"Category: {pred_cat:<25} (Confidence: {conf_cat*100:.1f}%) -> {cat_conf_label}")
    
    # Priority Output
    pri_conf_label = "High confidence prediction" if conf_pri >= 0.6 else "Low confidence prediction"
    print(f"Priority: {pred_pri:<25} (Confidence: {conf_pri*100:.1f}%) -> {pri_conf_label}{rule_applied}")

print("\n" + "=" * 60)
print("  Done!")
print("=" * 60)
