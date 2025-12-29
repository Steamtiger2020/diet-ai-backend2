import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import json

app = Flask(__name__)
CORS(app)

# Pega a chave das variáveis de ambiente
HF_TOKEN = os.getenv("HF_API_KEY") 

# --- URL ATUALIZADA (CORREÇÃO DO ERRO) ---
# Mudamos de api-inference para router.huggingface.co/hf-inference
HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/llava-hf/llava-1.5-7b-hf"

def query_huggingface(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Retry automático caso o modelo esteja carregando
    for i in range(5):
        try:
            response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=30)
            data = response.json()

            # Se for erro de URL antiga (apenas precaução) ou Loading
            if isinstance(data, dict) and "error" in data:
                error_msg = data["error"]
                if "loading" in error_msg.lower():
                    wait_time = data.get("estimated_time", 10)
                    print(f"⏳ Modelo carregando... Esperando {wait_time:.1f}s (Tentativa {i+1}/5)")
                    time.sleep(wait_time)
                    continue 
                return data # Retorna o erro real (como o da URL) para debug
            
            return data # Sucesso
            
        except Exception as e:
            print(f"⚠️ Erro de conexão na tentativa {i+1}: {e}")
            time.sleep(2)
            continue
    
    return {"error": "Timeout: Falha ao conectar com Hugging Face"}

@app.route("/")
def home():
    return "🚀 Backend DietAI Atualizado (Router URL)!"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        base64_img = data.get("image")

        if not base64_img:
            return jsonify({"error": "Nenhuma imagem recebida"}), 400

        print("📥 Recebendo imagem...")

        prompt_text = "USER: <image>\nAnalise a comida. Retorne APENAS um JSON puro (sem markdown) no formato: {\"name\": \"Nome do Prato\", \"cal\": 0, \"p\": 0, \"c\": 0, \"dica\": \"Dica curta\"}\nASSISTANT:"

        payload = {
            "inputs": {
                "image": base64_img,
                "prompt": prompt_text
            }
        }

        result = query_huggingface(payload)

        # Tratamento de erro que vem da HF
        if isinstance(result, dict) and "error" in result:
            print("❌ Erro HF:", result["error"])
            return jsonify(result), 503

        # Processamento da resposta LLaVA
        raw_text = ""
        if isinstance(result, list) and len(result) > 0:
            raw_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            raw_text = result.get("generated_text", "")

        if "ASSISTANT:" in raw_text:
            raw_text = raw_text.split("ASSISTANT:")[1]

        print("🤖 Resposta Bruta:", raw_text[:100], "...")

        # Extrair JSON
        s = raw_text.find("{")
        e = raw_text.rfind("}")
        
        if s != -1 and e != -1:
            json_str = raw_text[s:e+1]
            try:
                final_json = json.loads(json_str)
                return jsonify(final_json)
            except:
                return jsonify({"error": "Erro ao ler JSON da IA", "raw": raw_text}), 500
        else:
            return jsonify({"error": "JSON não encontrado", "raw": raw_text}), 500

    except Exception as e:
        print("🔥 Erro Crítico:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
