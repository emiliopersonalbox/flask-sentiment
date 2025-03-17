from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Abilita CORS per richieste esterne

# **Definiamo le pipeline con modelli specifici**
pipelines = {
    "sentiment-analysis": pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment"),  # Classifica il sentiment (1-5 stelle)
    "emotion-analysis": pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base"),  # Rileva emozioni (gioia, tristezza, rabbia...)
    "sarcasm-detection": pipeline("text-classification", model="tonygallovich/sarcasm-detection"),  # Rileva sarcasmo
    "subjectivity-detection": pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-subjective-objective"),  # Oggettivo vs Soggettivo
    "contextual-meaning": pipeline("zero-shot-classification", model="facebook/bart-large-mnli"),  # Analizza il significato contestuale
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
                    results[model_name] = model(text, candidate_labels=["positivo", "negativo", "sarcasmo", "gioia", "tristezza", "rabbia", "neutro"])
                else:
                    results[model_name] = model(text)
            except Exception as e:
                results[model_name] = {"error": str(e)}

        return jsonify({"text": text, "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
