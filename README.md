# AI Grievance Governance System

A robust machine learning backend to classify civic grievances by **Category** and **Priority**. It handles real-world, messy text, extracts temporal features (like "3 days"), and provides confidence scores for every prediction.

## 🚀 Key Features

*   **Semantic Understanding:** Uses `all-MiniLM-L6-v2` sentence embeddings instead of basic TF-IDF to truly understand the context of the complaint.
*   **Duration Extraction:** Automatically identifies timeframes (e.g., "10 hours", "3 dino se") and factors them into Priority predictions.
*   **Confidence Scores:** Employs `predict_proba` to tell you exactly how confident the model is (e.g., 98% vs 45%), allowing for human-in-the-loop review on vague inputs.
*   **Robust Preprocessing:** Built-in spelling correction and synonym mapping for regional slang (`bijli` -> `electricity`).

## 📁 Project Structure

*   `data/` : Clean and messy datasets used for training and evaluation.
*   `models/` : Serialized Logistic Regression models (`category_model.pkl` and `priority_model.pkl`).
*   `src/train_predict_improved.py` : Script to train models, generate embeddings, and evaluate accuracy.
*   `src/predict.py` : Standalone prediction script to run inference on new text.

## 💻 How to Install

1. Clone or download this project.
2. Ensure you have Python 3.9+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 How to Run

To train the models from scratch:
```bash
python src/train_predict_improved.py
```

To run a quick prediction test:
```bash
python src/predict.py
```

## 📊 Example Output
```text
  Raw Input   : bijli nahi aa rahi since 3 days
  Cleaned     : electricity nahi aa rahi since day | Extracted Duration: 3.0 days
  Category    : Electricity (Confidence: 0.98)
  Priority    : High (Confidence: 1.00) -> (Rule Override: Duration > 2 days)
```
