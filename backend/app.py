import os
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from inference import predict_complaint

app = Flask(__name__)
CORS(app)

DB_PATH = os.path.join(os.path.dirname(__file__), 'database.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            category TEXT,
            priority TEXT,
            decision TEXT,
            duration_hours REAL,
            status TEXT DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    ensure_columns(conn, cursor)
    conn.commit()
    conn.close()

def ensure_columns(conn, cursor):
    cursor.execute("PRAGMA table_info(complaints)")
    existing = {row[1] for row in cursor.fetchall()}
    columns_to_add = [
        ("duration_hours", "REAL")
    ]

    for column_name, column_type in columns_to_add:
        if column_name not in existing:
            cursor.execute(f"ALTER TABLE complaints ADD COLUMN {column_name} {column_type}")

init_db()

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    # ML Prediction
    prediction = predict_complaint(text)

    if prediction.get('decision') == "AUTO_PROCESS":
        status = "AUTHORITIES INFORMED"
    else:
        status = "PENDING"
    
    # Store in DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO complaints (
            text, category, priority, duration_hours, decision, status
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        text, 
        prediction['category'], 
        prediction['priority'], 
        prediction.get('duration_hours'),
        prediction['decision'],
        status
    ))
    conn.commit()
    conn.close()
    
    return jsonify(prediction)

@app.route('/ombudsman/pending', methods=['GET'])
def get_pending():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM complaints 
        WHERE decision = 'SEND_TO_OMBUDSMAN' AND status = 'PENDING'
        ORDER BY created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    return jsonify([dict(row) for row in rows])

@app.route('/ombudsman/logs', methods=['GET'])
def get_logs():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM complaints 
        ORDER BY created_at DESC 
        LIMIT 15
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    return jsonify([dict(row) for row in rows])

@app.route('/ombudsman/update', methods=['POST'])
def update_complaint():
    data = request.json
    comp_id = data.get('id')
    category = data.get('category')
    priority = data.get('priority')
    status = data.get('status')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE complaints 
        SET category = ?, priority = ?, status = ?
        WHERE id = ?
    ''', (category, priority, status, comp_id))
    conn.commit()
    conn.close()
    
    return jsonify({"message": "Updated successfully", "id": comp_id})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
