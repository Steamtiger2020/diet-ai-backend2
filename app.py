import os  # <--- ESTAVA FALTANDO ESTA LINHA IMPORTANTE!
import time
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURAÇÃO ---
# Tenta pegar do Render (Variável de Ambiente). 
HF_TOKEN = os.getenv("HF_API_KEY")

# Verificação de segurança no Log (sem mostrar o token inteiro)
print("--- INICIANDO SERVIDOR ---")
if not HF_TOKEN:
    print("❌ ERRO GRAVE: A variável HF_API_KEY não foi encontrada!")
    # Fallback de emergência (só se quiser testar local, não recomendado p/ produção)
    # HF_TOKEN = "hf_..." 
else:
    print(f"✅ Token carregado: {HF_TOKEN[:3]}...{HF_TOKEN[-3:]}")

# URL do Modelo LLaVA (Visão) - Estável e Rápido
HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/llava-hf/llava-1.5-7b-hf"

def query_huggingface(payload):
    if not HF_TOKEN:
        return {"error": "Token de API não configurado no servidor"}

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Tenta 3 vezes (Retry)
    for i in range(3):
        try:
            response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=40)
            
            # Se der erro 503 (Loading), espera e tenta de novo
            if response.status_code == 503:
                error_data = response.json()
                wait_time = error_data.get("estimated_time", 20)
                print(f"⏳ Modelo carregando... Esperando {wait_time}s")
                time.sleep(wait_time)
                continue
            
            if response.status_code != 200:
                print(f"❌ Erro API HF ({response.status_code}): {response.text}")
                return {"error": f"Erro HF: {response.text}"}

            return response.json()
            
        except Exception as e:
            print(f"⚠️ Erro conexão: {e}")
            time.sleep(2)
    
    return {"error": "Timeout: Hugging Face não respondeu"}

@app.route("/")
def home():
    return "🚀 Backend DietAI Online e Corrigido!"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        base64_img = data.get("image")

        if not base64_img:
            return jsonify({"error": "Nenhuma imagem recebida"}), 400

        print("📥 Recebendo imagem...")

        # Prompt otimizado para LLaVA
        prompt_text = "USER: <image>\nAnalise a comida. Retorne APENAS um JSON puro: {\"name\": \"Nome\", \"cal\": 0, \"p\": 0, \"c\": 0, \"dica\": \"Dica\"}\nASSISTANT:"

        payload = {
            "inputs": {
                "image": base64_img,
                "prompt": prompt_text
            }
        }

        result = query_huggingface(payload)

        # Se retornou erro direto da função query
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 503

        # Processar resposta do LLaVA (pode vir lista ou dict)
        raw_text = ""
        if isinstance(result, list) and len(result) > 0:
            raw_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            raw_text = result.get("generated_text", "")

        # Limpar o prompt da resposta (LLaVA repete o que perguntou)
        if "ASSISTANT:" in raw_text:
            raw_text = raw_text.split("ASSISTANT:")[1]

        # Extrair JSON
        s = raw_text.find("{")
        e = raw_text.rfind("}")
        
        if s != -1 and e != -1:
            try:
                final_json = json.loads(raw_text[s:e+1])
                return jsonify(final_json)
            except:
                return jsonify({"error": "Erro ao ler JSON da IA", "raw": raw_text}), 500
        else:
            return jsonify({"error": "JSON não encontrado", "raw": raw_text}), 500

    except Exception as e:
        print("🔥 Erro Crítico no Servidor:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
