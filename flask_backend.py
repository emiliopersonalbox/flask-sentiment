from flask import Flask, request, jsonify
from flask_cors import CORS  # Importa CORS
from transformers import pipeline
import os


app = Flask(__name__)

# Carica i modelli di Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis", model="Siebert/sentiment-roberta-large-english")
toxicity_pipeline = pipeline("text-classification", model="unitary/toxic-bert")
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Funzione per analizzare il testo
def analyze_text(text):
    results = {}

    # Analizza il sentiment
    sentiment = sentiment_pipeline(text)[0]
    results["Sentiment"] = sentiment['label']

    # Analizza la tossicità
    toxicity = toxicity_pipeline(text)[0]
    results["Tossicità"] = f"{toxicity['label']} ({toxicity['score']:.2f})"

    # Analizza le emozioni negative
    emotions = emotion_pipeline(text)
    negative_emotions = ['anger', 'sadness', 'fear', 'disgust']
    results["Emozioni Negative"] = [e['label'] for e in emotions if e['label'] in negative_emotions]

    # Blocca messaggi tossici o negativi
    if results["Sentiment"] != "POSITIVE" or results["Tossicità"].startswith("toxic") or results["Emozioni Negative"]:
        results["Status"] = "❌ Messaggio rifiutato: contiene elementi negativi."
    else:
        results["Status"] = "✅ Messaggio approvato!"

    return results

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        data = request.get_json()
        text = data.get("text", "") if data else ""
    else:
        text = request.args.get("text", "")

    return jsonify({'message': 'API funzionante!', 'text': text})


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Usa la porta assegnata da Render
    app.run(host="0.0.0.0", port=port)  # Rendi accessibile il server



