import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

# --- CONFIGURAÇÃO ---
HF_TOKEN = os.getenv("HF_API_KEY")

print("--- INICIANDO SERVIDOR DIET AI ---")
if not HF_TOKEN:
    print("❌ ERRO: HF_API_KEY não encontrada!")
else:
    print("✅ Token carregado.")

# Usaremos o modelo Qwen-2.5, que é INCRÍVEL para visão e aceita JSON mode
# Se ele estiver muito lento, podemos voltar para o "meta-llama/Llama-3.2-11B-Vision-Instruct"
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

def analyze_image_with_client(base64_img):
    if not HF_TOKEN:
        return {"error": "Token não configurado"}

    # Inicializa o cliente oficial (Ele resolve as URLs sozinho)
    client = InferenceClient(api_key=HF_TOKEN)

    # Prompt para garantir JSON
    prompt = (
        "Identifique o prato nesta imagem. Estime as calorias e macronutrientes. "
        "Responda EXATAMENTE neste formato JSON, sem texto antes ou depois: "
        "{\"name\": \"Nome do Prato\", \"cal\": 500, \"p\": 30, \"c\": 50, \"dica\": \"Dica curta\"}"
    )

    try:
        # Chamada Chat Completion (Padrão moderno)
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.7
        )

        # Pega a resposta
        content = response.choices[0].message.content
        print("🤖 Resposta IA:", content)

        # Limpeza para garantir JSON (remove ```json ... ```)
        clean_content = content.replace("```json", "").replace("```", "").strip()
        
        # Tenta achar o JSON no meio do texto se houver lixo
        s = clean_content.find("{")
        e = clean_content.rfind("}")
        if s != -1 and e != -1:
            clean_content = clean_content[s:e+1]

        return json.loads(clean_content)

    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Erro Client: {error_msg}")
        
        # Tratamento de erro de Loading
        if "503" in error_msg or "loading" in error_msg.lower():
            return {"error": "IA está acordando (Loading). Tente em 30s."}
        
        return {"error": f"Erro na IA: {error_msg}"}

@app.route("/")
def home():
    return "🚀 DietAI Backend (Client Version) Online!"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        base64_img = data.get("image")

        if not base64_img:
            return jsonify({"error": "Nenhuma imagem recebida"}), 400

        print("📥 Recebendo imagem...")
        
        # Chama a função nova usando a biblioteca oficial
        result = analyze_image_with_client(base64_img)

        # Se retornou erro nosso
        if "error" in result:
            status_code = 503 if "acordando" in result["error"] else 500
            return jsonify(result), status_code

        return jsonify(result)

    except Exception as e:
        print("🔥 Erro Crítico:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
