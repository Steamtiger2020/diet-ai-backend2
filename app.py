from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, base64, time, os

app = Flask(__name__)
CORS(app)

HF_API_KEY = os.getenv("HF_TOKEN", "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")

MODEL_URL = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

@app.route("/")
def home():
    return jsonify({
        "status": "DietAI Backend Online ✔",
        "usage": "/analyze para análise de alimentos"
    })

@app.route("/analyze", methods=["POST"])
def analyze_food():
    try:
        data = request.json
        base64_image = data.get("image")

        if not base64_image:
            return jsonify({"error": "Imagem não recebida"}), 400

        payload = {
            "inputs": {
                "image": base64_image,
                "prompt": (
                    "USER: <image>\nAnalise o alimento na imagem. "
                    "Retorne APENAS JSON: {\"name\":\"\",\"cal\":0,\"p\":0,\"c\":0,\"dica\":\"\"}\nASSISTANT:"
                )
            }
        }

        # 🔥 Tentativas automáticas (resolve o 503)
        for tentativa in range(5):
            print(f"Tentando HuggingFace... tentativa {tentativa+1}/5")

            response = requests.post(MODEL_URL, headers=headers, json=payload)

            # Modelo acordando ⏳
            if response.status_code == 503 or "loading" in response.text:
                time.sleep(10)  # espera o modelo ligar
                continue

            # Sucesso
            result = response.json()
            text = result[0].get("generated_text", "")

            # 💾 Extrair apenas JSON
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                return jsonify({"result": text[start:end+1]})

            return jsonify({"error": "IA respondeu sem JSON válido"}), 500

        return jsonify({"error": "Modelo não respondeu após 5 tentativas"}), 503

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
