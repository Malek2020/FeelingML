import os
import joblib
import numpy as np
import pandas as pd
import MySQLdb  # ✅ Using MySQLdb now
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

app = Flask(__name__)
MODEL_FILE = "sentiment_model.pkl"

# ✅ MySQL Database Configuration (Using MySQLdb)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # Change if needed
    'passwd': '',  # Add your password if necessary
    'db': 'sentiment_db',
    'port': 3306,
    'connect_timeout': 10
}

def get_db_connection():
    """Establish a MySQLdb connection and ensure the table exists."""
    try:
        print("🔄 Connecting to MySQL...")
        conn = MySQLdb.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tweets (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text TEXT NOT NULL,
                positive TINYINT(1) NOT NULL,
                negative TINYINT(1) NOT NULL
            )
        """)
        conn.commit()
        print("✅ Table `tweets` checked/created successfully!")

        return conn
    except MySQLdb.Error as err:
        print(f"❌ MySQL Connection Error: {err}")
        return None 

def load_data():
    """Fetch tweets from MySQL database inside Flask."""
    conn = get_db_connection()
    if conn is None:
        print("❌ MySQL connection failed.")
        return pd.DataFrame() 

    try:
        print("✅ Running SQL query to fetch tweets...")
        query = "SELECT text, positive, negative FROM tweets"
        df = pd.read_sql(query, conn)  
        print(f"✅ Successfully fetched {len(df)} rows from MySQL.")
        return df

    except Exception as e:
        print(f"❌ MySQL Query Error: {e}")
        return pd.DataFrame()
    
    finally:
        conn.close()
        print("🔌 Closed MySQL connection.")

FRENCH_STOP_WORDS = set([
    "alors", "au", "aussi", "avec", "bon", "car", "ce", "cela", "ces", "comme",
    "dans", "des", "du", "elle", "en", "encore", "est", "et", "eu", "fait",
    "faites", "fois", "haut", "ici", "il", "ils", "je", "juste", "la", "le", "les",
    "leur", "mais", "mes", "moi", "mon", "mot", "ni", "nous", "on", "ou", "par",
    "pas", "peut", "pour", "quand", "que", "qui", "sa", "sans", "ses", "seulement",
    "si", "son", "sont", "sous", "sur", "ta", "tandis", "tant", "te", "tes", "ton",
    "tous", "tout", "trop", "tu", "un", "une", "vos", "votre", "vous", "vu", "ça",
    "étaient", "était", "étions", "été", "être"
])

def train_model():
    """Trains a Logistic Regression model using MySQL data."""
    df = load_data()

    if df.empty:
        print("⚠️ No data available for training.")
        return

    print(f"✅ Training model on {len(df)} samples from MySQL.")

    vectorizer = TfidfVectorizer(
        stop_words=list(FRENCH_STOP_WORDS),
        sublinear_tf=True,
        ngram_range=(1, 3)  
    )

    X = vectorizer.fit_transform(df['text'])
    y_positive = df['positive']
    y_negative = df['negative']

    model_pos = LogisticRegression(C=10, max_iter=500).fit(X, y_positive)
    model_neg = LogisticRegression(C=10, max_iter=500).fit(X, y_negative)

    joblib.dump((vectorizer, model_pos, model_neg), MODEL_FILE)
    print("✅ Model trained and saved successfully.")

@app.route('/analyze', methods=['POST'])
def analyze_sentiments():
    """API to analyze sentiment of given tweets."""
    data = request.json.get("tweets")
    if not data or not isinstance(data, list):
        return jsonify({"error": "Please provide a list of tweets."}), 400

    vectorizer, model_pos, model_neg = joblib.load(MODEL_FILE)
    X = vectorizer.transform(data)

    positive_scores = model_pos.predict_proba(X)[:, 1]
    negative_scores = model_neg.predict_proba(X)[:, 1]

    results = {tweet: round(2 * (pos_score - neg_score), 2)
               for tweet, pos_score, neg_score in zip(data, positive_scores, negative_scores)}

    return jsonify(results)

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """API to retrain the model using latest MySQL data."""
    train_model()
    return jsonify({"message": "Model retrained successfully with latest database data."})

@app.route('/evaluate', methods=['GET'])
def evaluate_model():
    """API to evaluate model performance."""
    df = load_data()
    if df.empty:
        return jsonify({"error": "No data available for evaluation."})

    vectorizer, model_pos, model_neg = joblib.load(MODEL_FILE)
    X = vectorizer.transform(df['text'])

    y_pos_pred = model_pos.predict(X)
    y_neg_pred = model_neg.predict(X)

    report_pos = classification_report(df['positive'], y_pos_pred, output_dict=True)
    report_neg = classification_report(df['negative'], y_neg_pred, output_dict=True)

    matrix_pos = confusion_matrix(df['positive'], y_pos_pred).tolist()
    matrix_neg = confusion_matrix(df['negative'], y_neg_pred).tolist()

    evaluation = {
        "positive_class": {
            "confusion_matrix": matrix_pos,
            "report": report_pos
        },
        "negative_class": {
            "confusion_matrix": matrix_neg,
            "report": report_neg
        }
    }
    return jsonify(evaluation)

if __name__ == '__main__':
    print("🔹 Training model using MySQL dataset.")
    train_model()
    print("🌐 Starting Flask server on http://127.0.0.1:5000/ 🚀")
    app.run(host='0.0.0.0', port=5000, debug=True)
