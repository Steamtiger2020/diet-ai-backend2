from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_API_KEY")

# Modelo LLaVA (Visão)
HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/llava-hf/llava-1.5-7b-hf"

@app.route("/")
def home():
    return "🚀 Backend de Diagnóstico Online!"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        base64_img = data.get("image")

        if not base64_img:
            return jsonify({"error": "Sem imagem"}), 400

        print(f"📥 Enviando para HF com token: {HF_TOKEN[:4]}...{HF_TOKEN[-4:]}")

        prompt_text = "USER: <image>\nAnalise a comida. Retorne APENAS um JSON puro: {\"name\": \"Nome do Prato\", \"cal\": 0, \"p\": 0, \"c\": 0, \"dica\": \"Dica\"}\nASSISTANT:"

        payload = {
            "inputs": {
                "image": base64_img,
                "prompt": prompt_text
            }
        }

        # Requisição direta sem Retry (para ver o erro real)
        response = requests.post(
            HF_MODEL_URL, 
            headers={"Authorization": f"Bearer {HF_TOKEN}"}, 
            json=payload,
            timeout=60 # Aumentei para 60 segundos
        )

        # SE DER ERRO NA HUGGING FACE, RETORNA O ERRO EXATO PARA O CELULAR
        if response.status_code != 200:
            error_msg = response.text
            print(f"❌ Erro Real da HF ({response.status_code}): {error_msg}")
            return jsonify({"error": f"HF Error {response.status_code}: {error_msg}"}), 503

        # Se funcionou, processa
        result = response.json()
        raw_text = ""
        
        if isinstance(result, list) and len(result) > 0:
            raw_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            raw_text = result.get("generated_text", "")

        if "ASSISTANT:" in raw_text:
            raw_text = raw_text.split("ASSISTANT:")[1]

        # Extração JSON simples
        s = raw_text.find("{")
        e = raw_text.rfind("}")
        
        if s != -1 and e != -1:
            try:
                final_json = json.loads(raw_text[s:e+1])
                return jsonify(final_json)
            except:
                return jsonify({"error": "Erro ao ler JSON", "raw": raw_text})
        else:
            # Fallback se não vier JSON
            return jsonify({
                "name": "Prato Identificado",
                "cal": 300, "p": 15, "c": 20, 
                "dica": f"A IA respondeu: {raw_text[:50]}..."
            })

    except Exception as e:
        print("🔥 Erro Crítico:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
