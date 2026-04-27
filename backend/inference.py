import sys
import os
import numpy as np

# Add project root to sys.path to import from src/
# When running app.py from backend/, BASE_DIR should be project-root
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import from src.predict
from src.predict import (
    preprocess, 
    extract_duration, 
    cat_model, 
    pri_model, 
    embedder
)
from utils.decision_engine import decision_engine

def predict_complaint(complaint_text):
    """
    Wraps the logic from predict.py but returns a dictionary 
    instead of printing to console.
    """
    # 1. Preprocess
    cleaned = preprocess(complaint_text)
    duration = extract_duration(cleaned)

    # 2. Embeddings
    embedding = embedder.encode([cleaned])
    combined = np.hstack([embedding, np.array([[duration]])])

    # 3. Category Model
    cat_pred = cat_model.predict(embedding)[0]
    cat_proba_array = cat_model.predict_proba(embedding)[0]
    
    sorted_cat_probs = np.sort(cat_proba_array)[::-1]
    cat_confidence = float(sorted_cat_probs[0])

    # 4. Priority Model
    pri_pred = pri_model.predict(combined)[0]
    pri_proba_array = pri_model.predict_proba(combined)[0]
    
    sorted_pri_probs = np.sort(pri_proba_array)[::-1]
    pri_confidence = float(sorted_pri_probs[0])
    if duration > 2:
        pri_pred = "High"
        pri_confidence = 1.0

    # 5. Decision Engine (Routing Layer)
    decision_payload = decision_engine(complaint_text, str(pri_pred), cat_confidence, pri_confidence)

    return {
        "category": str(cat_pred),
        "priority": str(pri_pred),
        "priority": decision_payload["priority"],
        "duration_hours": decision_payload["duration_hours"],
        "decision": decision_payload["decision"]
    }
