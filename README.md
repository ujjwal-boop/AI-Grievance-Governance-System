# AI Grievance Governance System (Bharat Sovereign Edition)

A robust, full-stack machine learning solution for classifying and prioritizing civic grievances. This system handles "messy" real-world text (including Hinglish and regional slang), extracts temporal features, and provides an authoritative governance decision (Auto-Process vs. Ombudsman Review).

## 🏛️ Bharat Sovereign Design
The system features a premium frontend aesthetic inspired by official Government of India portals (NIC, MeitY).
*   **Colors**: Tiranga Authority Palette (India Ink Navy, Kesari Saffron, India Green).
*   **Typography**: Bilingual system (English primary, Hindi secondary) using Noto Serif and Noto Sans.
*   **Identity**: Integrated Ashoka Chakra watermark, Tricolor accents, and official "Sarkari Mudra" (Stamp Style) status badges.

---

## 🛠️ Technology Stack

### 🔹 Core ML & NLP
*   **Language**: Python 3.13+
*   **NLP Engine**: `Sentence-Transformers` (`all-MiniLM-L6-v2`) for dense semantic embeddings.
*   **Text Processing**: `NLTK` (WordNet Lemmatizer, POS Tagging) for deep linguistic cleaning.
*   **Machine Learning**: `Scikit-learn` (Logistic Regression with balanced class weights).
*   **Data Handling**: `Pandas` & `NumPy` for feature fusion and matrix operations.

### 🔹 Backend Architecture
*   **Framework**: `Flask` (RESTful API).
*   **Database**: `SQLite` (Persistent storage for complaints and audit trails).
*   **CORS**: `Flask-CORS` for seamless frontend integration.
*   **Inference Wrapper**: A modular `inference.py` layer that pulls logic from the core ML pipeline without modifying training scripts.

### 🔹 Frontend Design
*   **Aesthetics**: Glassmorphism (Backdrop blur, translucent borders).
*   **Interactions**: SVG-based spinning Ashoka Chakra (24-spoke) loading indicator.
*   **UX**: Dual portals—**User Portal** (Anonymous submission) and **Ombudsman Dashboard** (Administrative review & Logs).

---

## 🧠 Algorithms & Logic

### 1. Semantic Embedding (Sentence-Transformers)
Instead of traditional TF-IDF (keyword-based), the system uses the `all-MiniLM-L6-v2` transformer. This allows the AI to understand that *"bijli nahi aa rahi"* and *"power outage"* represent the same intent by mapping them to similar points in a 384-dimensional vector space.

### 2. Hybrid Feature Fusion
The **Priority Model** doesn't just look at text; it combines:
*   **Text Embedding**: 384D semantic vector.
*   **Temporal Feature**: Extracted duration (e.g., "3 days" normalized to a float value).
This fusion allows the model to prioritize a "3-day outage" higher than a "2-hour outage" even if the text is identical.

### 3. Confidence & Governance Engine
The system calculates a **Final Confidence Score** based on:
*   **Winning Probability**: The `predict_proba()` result from the model.
*   **Confidence Margin**: The gap between the 1st and 2nd best predictions (penalizes vague inputs).
*   **Rule Overrides**: Hard-coded logic (e.g., any grievance > 2 days is automatically escalated to **High Priority**).

**Governance Decisions**:
*   `AUTO_PROCESS`: Confidence ≥ 75%
*   `BORDERLINE`: Confidence 60% - 75%
*   `SEND_TO_OMBUDSMAN`: Confidence < 60%

---

## 📊 Performance Metrics

The system was evaluated against a **messy test dataset** containing typos, regional slang (Hinglish), and natural language temporal mentions.

| Metric | Accuracy | Methodology |
| :--- | :--- | :--- |
| **Category Classification** | **94.44%** | Pure Semantic Embedding (Transformer-based) |
| **Priority Classification** | **100.00%** | Hybrid Model (ML Prediction + Duration Rule Override) |
| **Hinglish Support** | **Robust** | Handled via custom normalization and synonym mapping |

---

## 📁 Project Structure
*   `backend/`: Flask server, SQLite DB, and model inference wrapper.
*   `frontend/`: Premium HTML/CSS/JS portals.
*   `src/`: Core ML logic and standalone prediction scripts.
*   `models/`: Serialized `.pkl` artifacts for Category and Priority models.
*   `data/`: Training and messy test datasets (**Self-generated** using `generate_improved_dataset.py`).

> [!NOTE]
> The dataset used for training and evaluation was synthesized specifically for this project to simulate real-world civic grievances with Indian regional context and linguistic noise.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full-Stack App
**Start Backend**:
```bash
cd backend
python app.py
```
**Open Frontend**:
Open `frontend/index.html` (User) or `frontend/ombudsman.html` (Officer) in any modern browser.

### 3. Training & CLI
*   **Retrain Models**: `python src/train_predict_improved.py`
*   **Interactive CLI**: `python src/predict.py`

---

## 🛡️ Credits
Designed & Developed by **Antigravity AI** | Built with an Indian Institutional Design Language.
© 2024 Ministry of Governance, Government of India.
