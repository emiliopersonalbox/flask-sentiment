from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Abilita CORS per richieste esterne

# **Usiamo un unico modello avanzato di DeepSeek per tutte le analisi**
deepseek_model = pipeline("text-generation", model="deepseek-ai/deepseek-llm-7b")

@app.route("/")
def home():
    return jsonify({"message": "API DeepSeek funzionante!"})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "Nessun testo fornito"}), 400

        # **Chiediamo a DeepSeek di fare l'analisi del sentimento e altro**
        prompt = f"Analizza il sentimento del testo seguente e forniscimi un'analisi dettagliata: {text}"
        response = deepseek_model(prompt, max_length=500, do_sample=True)

        return jsonify({"text": text, "analysis": response[0]["generated_text"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
