from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Abilita CORS per permettere richieste da altri domini

# **Pipeline per diverse analisi del Sentiment**
pipelines = {
    "sentiment-analysis": pipeline("sentiment-analysis"),  # Positivo/Negativo/Neutro
    "sarcasm-detection": pipeline("text-classification", model="mvanvlasselaer/robbert-sarcasm-dutch"),  # Sarcasmo
    "emotion-analysis": pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base"),  # Emozioni
    "contextual-meaning": pipeline("zero-shot-classification"),  # Contesto e significato
    "subjectivity-detection": pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-subjective-objective"),  # Oggettivo/Soggettivo
}

@app.route("/")
def home():
    return jsonify({"message": "API funzionante!"})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "Nessun testo fornito"}), 400

        # **Esegui tutte le analisi sentimentali**
        results = {}
        for model_name, model in pipelines.items():
            try:
                if model_name == "contextual-meaning":
                    results[model_name] = model(text, candidate_labels=["gioia", "tristezza", "rabbia", "sorpresa", "sarcasmo", "neutro"])
                else:
                    results[model_name] = model(text)
            except Exception as e:
                results[model_name] = {"error": str(e)}

        return jsonify({"text": text, "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
