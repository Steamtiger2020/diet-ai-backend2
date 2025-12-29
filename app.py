from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("HF_API_KEY")  # Configure no Render
HF_MODEL = "llava-hf/llava-1.5-7b-hf"

@app.route("/")
def home():
    return "Backend online ✔ IA pronta para uso!"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Imagem não recebida"}), 400

        base64_image = data["image"]

        payload = {
            "inputs": {
                "prompt": (
                    "Analyze the food image realistically. "
                    "Return ONLY JSON in PT-BR: "
                    "{\"name\":\"\",\"cal\":0,\"p\":0,\"c\":0,\"dica\":\"\"}"
                ),
                "image": base64_image
            }
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
            timeout=60
        )

        if response.status_code == 503:
            return jsonify({"error": "modelo carregando"}), 503

        try:
            raw = response.json()
        except:
            return jsonify({"error": "Resposta não JSON da HuggingFace"}), 500

        text = ""
        if isinstance(raw, list) and raw[0].get("generated_text"):
            text = raw[0]["generated_text"]
        else:
            return jsonify({"error": "Sem generated_text"}), 500

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return jsonify({"error": "JSON não encontrado"}), 500

        final_json = json.loads(text[start:end+1])
        return jsonify({"result": final_json})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
