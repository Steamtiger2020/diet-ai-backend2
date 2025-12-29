from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import os

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

app = Flask(__name__)
CORS(app)  # Libera acesso Web/Expo — evita "Failed to fetch"

HF_API_KEY = os.getenv("HF_API_KEY") or "COLOQUE_SUA_KEY_AQUI"

# Modelo recomendado para custo ZERO e menor erro
MODEL_URL = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# ==========================================================
# ROTA PRINCIPAL — TESTE (abre no navegador)
# ==========================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Diet Backend Online ✔", "usage": "/analyze para análise de alimentos"})


# ==========================================================
# ROTA DE ANÁLISE — Seu app irá usar esta!
# ==========================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        image_base64 = data.get("image")

        if not image_base64:
            return jsonify({"error": "Envie { image: base64 } no body"}), 400

        # -------- Envio ao HuggingFace -------- #
        payload = {
            "inputs": {
                "image": image_base64,
                "prompt": (
                    "USER: <image>\n"
                    "Identifique o alimento, estime calorias, proteínas e carboidratos.\n"
                    "Retorne SOMENTE um JSON no formato:\n"
                    "{\"name\":\"Nome\",\"cal\":0,\"p\":0,\"c\":0,\"dica\":\"texto curto\"}\n"
                    "ASSISTANT:"
                )
            }
        }

        response = requests.post(MODEL_URL, headers=headers, json=payload)

        if response.status_code != 200:
            return jsonify({"error": "Modelo carregando ou indisponível no momento. Tente em 20s."}), 503

        result = response.json()

        # Hugging Face retorna lista com generated_text
        text = result[0]["generated_text"]

        # Extrair somente o JSON retornado pela IA
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return jsonify({"error": "IA respondeu, mas sem JSON válido"}), 500

        food_data = text[start:end + 1]

        return jsonify({"result": food_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# INICIALIZAÇÃO LOCAL
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
