import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import json

app = Flask(__name__)
CORS(app)

# Pega a chave das variáveis de ambiente ou usa a sua hardcoded (Cuidado ao expor!)
HF_TOKEN = os.getenv("HF_API_KEY") or "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"

# Modelo LLaVA 1.5 7B (O mais estável e rápido para visão gratuita)
HF_MODEL_URL = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"

def query_huggingface(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Tenta até 5 vezes se o modelo estiver carregando
    for i in range(5):
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        data = response.json()

        # Se tiver erro de carregamento (Model is loading)
        if "error" in data and "loading" in data["error"]:
            wait_time = data.get("estimated_time", 10)
            print(f"⏳ Modelo carregando... Esperando {wait_time:.1f}s (Tentativa {i+1}/5)")
            time.sleep(wait_time)
            continue # Tenta de novo
        
        return data # Retorna sucesso ou outro erro real
    
    return {"error": "Timeout: Modelo demorou muito para carregar"}

@app.route("/")
def home():
    return "🚀 Backend DietAI rodando com LLaVA!"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        base64_img = data.get("image")

        if not base64_img:
            return jsonify({"error": "Nenhuma imagem recebida"}), 400

        print("📥 Recebendo imagem para análise...")

        # Prompt específico para o LLaVA (USER / ASSISTANT)
        prompt_text = "USER: <image>\nAnalise a comida. Retorne APENAS um JSON puro (sem markdown) no formato: {\"name\": \"Nome do Prato\", \"cal\": 0, \"p\": 0, \"c\": 0, \"dica\": \"Dica curta\"}\nASSISTANT:"

        payload = {
            "inputs": {
                "image": base64_img,
                "prompt": prompt_text
            }
        }

        # Chama a função inteligente com retry
        result = query_huggingface(payload)

        # Se retornou erro da API
        if isinstance(result, dict) and "error" in result:
            print("❌ Erro HF:", result["error"])
            return jsonify(result), 503

        # O LLaVA retorna uma lista com 'generated_text'
        raw_text = ""
        if isinstance(result, list) and len(result) > 0:
            raw_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            raw_text = result.get("generated_text", "")

        # Limpeza: O LLaVA repete o prompt, pegamos só o que vem depois de ASSISTANT:
        if "ASSISTANT:" in raw_text:
            raw_text = raw_text.split("ASSISTANT:")[1]

        print("🤖 Resposta Bruta:", raw_text)

        # Extrair JSON da resposta
        s = raw_text.find("{")
        e = raw_text.rfind("}")
        
        if s != -1 and e != -1:
            json_str = raw_text[s:e+1]
            try:
                final_json = json.loads(json_str)
                return jsonify(final_json)
            except:
                return jsonify({"error": "IA não retornou JSON válido", "raw": raw_text}), 500
        else:
            return jsonify({"error": "JSON não encontrado na resposta", "raw": raw_text}), 500

    except Exception as e:
        print("🔥 Erro no Servidor:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
