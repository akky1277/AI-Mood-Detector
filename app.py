"""
================================================================
  AI MOOD DETECTOR - Flask Web Interface
  File: app.py

  This file creates the web server.
  It has two routes:
    GET  /        → Show the main webpage
    POST /predict → Receive text, run AI, return result as JSON
================================================================
"""

from flask import Flask, render_template, request, jsonify
from mood_engine import predict_mood, load_model, train_and_save
import os

app = Flask(__name__)

# Load the model once when the server starts (faster responses)
print("🚀 Starting AI Mood Detector Web App...")
if not os.path.exists("model.pkl"):
    train_and_save()
MODEL, VECTORIZER = load_model()
print("✅ Model ready. Visit http://127.0.0.1:5000\n")


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive text via AJAX, run prediction, return JSON.
    The frontend JavaScript calls this endpoint and
    updates the page without any reload.
    """
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    result = predict_mood(text, MODEL, VECTORIZER)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
