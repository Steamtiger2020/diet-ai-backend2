import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import json

app = Flask(__name__)
CORS(app)

# --- CARREGAMENTO E VERIFICAÇÃO DO TOKEN ---
HF_TOKEN = os.getenv("HF_API_KEY")

print("--- INICIANDO SERVIDOR ---")
if not HF_TOKEN:
    print("❌ ERRO GRAVE: A variável HF_API_KEY não foi encontrada no sistema!")
else:
    # Mostra só o começo e o fim para segurança (não vaza o token inteiro)
    print(f"✅ Token encontrado: {HF_TOKEN[:3]}...{HF_TOKEN[-3:]}")
    # Remove espaços em branco acidentais que causam erro
    HF_TOKEN = HF_TOKEN.strip()

HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/llava-hf/llava-1.5-7b-hf"

def query_huggingface(payload):
    # Verifica de novo antes de enviar
    if not HF_TOKEN:
        return {"error": "Servidor sem Token configurado (HF_API_KEY missing)"}

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    for i in range(5):
        try:
            response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=30)
            data = response.json()

            if isinstance(data, dict) and "error" in data:
                error_msg = data["error"]
                if "loading" in error_msg.lower():
                    wait_time = data.get("estimated_time", 10)
                    print(f"⏳ Carregando... {wait_time}s")
                    time.sleep(wait_time)
                    continue
                # Se der erro de autenticação, mostra no log
                print(f"❌ Erro da API HF: {error_msg}")
                return data
            
            return data
            
        except Exception as e:
            print(f"⚠️ Erro de conexão: {e}")
            time.sleep(2)
            continue
    
    return {"error": "Timeout: Hugging Face não respondeu"}

@app.route("/")
def home():
    token_status = "✅ Configurado" if HF_TOKEN else "❌ FALTANDO"
    return f"🚀 Backend DietAI Online! Token: {token_status}"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        base64_img = data.get("image")

        if not base64_img:
            return jsonify({"error": "Nenhuma imagem recebida"}), 400

        print("📥 Processando imagem...")

        prompt_text = "USER: <image>\nAnalise a comida. Retorne APENAS um JSON puro: {\"name\": \"Nome\", \"cal\": 0, \"p\": 0, \"c\": 0, \"dica\": \"Dica\"}\nASSISTANT:"

        payload = {
            "inputs": {
                "image": base64_img,
                "prompt": prompt_text
            }
        }

        result = query_huggingface(payload)

        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 503

        # Lógica de extração do texto
        raw_text = ""
        if isinstance(result, list) and len(result) > 0:
            raw_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            raw_text = result.get("generated_text", "")

        if "ASSISTANT:" in raw_text:
            raw_text = raw_text.split("ASSISTANT:")[1]

        s = raw_text.find("{")
        e = raw_text.rfind("}")
        
        if s != -1 and e != -1:
            try:
                final_json = json.loads(raw_text[s:e+1])
                return jsonify(final_json)
            except:
                return jsonify({"error": "Erro ao ler JSON", "raw": raw_text}), 500
        else:
            return jsonify({"error": "JSON não encontrado", "raw": raw_text}), 500

    except Exception as e:
        print("🔥 Erro Crítico:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
