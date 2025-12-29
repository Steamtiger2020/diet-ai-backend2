from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_API_KEY") or "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"

# Modelo mais leve e estável → melhor para plano free
HF_MODEL = "llava-hf/llava-v1.6-mistral-7b"

@app.route("/")
def home():
    return "🚀 Backend ativo e pronto!"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        base64_img = data.get("image")

        if not base64_img:
            return jsonify({"error": "Nenhuma imagem recebida"}), 400

        print("📥 Imagem recebida — enviando para HuggingFace...")

        payload = {
            "inputs": {
                "image": base64_img,
                "prompt":
                    "Analyze the food image and return ONLY JSON like:"
                    "{\"name\":\"\",\"cal\":0,\"p\":0,\"c\":0,\"dica\":\"\"}"
            }
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            print("❌ Modelo respondeu erro:", response.text[:200])
            return jsonify({"error": "Modelo carregando, tente novamente"}), 503

        text = response.text
        print("📄 Resposta recebida:", text[:200])

        # Extração do JSON dentro da resposta
        s,e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1:
            return jsonify({"error":"Nenhum JSON encontrado"}), 500

        json_clean = text[s:e+1]

        import json
        return jsonify(json.loads(json_clean))

    except Exception as e:
        print("🔥 ERRO NO SERVIDOR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
