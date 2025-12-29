from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import base64

app = Flask(__name__)
CORS(app)  

HF_TOKEN = os.getenv("HF_API_KEY") or "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"

MODEL_URL = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"

@app.route("/")
def home():
    return "Backend online ✔ IA pronta para uso!"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        img = data.get("image", None)

        if not img:
            return jsonify({"error": "Nenhuma imagem recebida"}), 400

        print("📥 Imagem recebida — enviando para HuggingFace...")

        response = requests.post(
            MODEL_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "inputs": {
                    "image": img,
                    "prompt": (
                        "Analyze the food image and return ONLY JSON like this: "
                        "{\"name\":\"\",\"cal\":0,\"p\":0,\"c\":0,\"dica\":\"\"}"
                    )
                }
            },
            timeout=90
        )

        if response.status_code != 200:
            return jsonify({"error": "Modelo carregando ou falhou"}), 503

        text = response.text

        # Limpa resposta para extrair JSON
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            return jsonify({"error": "Sem JSON retornado"}), 500

        result = text[start:end+1]
        return jsonify(eval(result))  # <-- converte para JSON válido

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
