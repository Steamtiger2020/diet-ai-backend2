import base64
import requests
import json
import os

# Pega o token do ambiente (Render)
HF_API_KEY = os.getenv("HF_API_KEY") 

# ✅ CORREÇÃO 1: Usar LLaVA (Mais leve e aceita conta grátis sem travar)
MODEL_URL = "https://router.huggingface.co/hf-inference/models/llava-hf/llava-1.5-7b-hf"

def analyze_food(base64_image):
    if not HF_API_KEY:
        return {"error": "Token da API não configurado"}

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    # ✅ CORREÇÃO 2: Formato de prompt específico para LLaVA
    prompt_text = "USER: <image>\nAnalise a refeição e retorne apenas JSON puro: {\"name\":\"Nome\",\"cal\":0,\"p\":0,\"c\":0,\"dica\":\"Dica\"}\nASSISTANT:"
    
    payload = {
        "inputs": {
            "image": base64_image,
            "prompt": prompt_text
        }
    }

    try:
        # Timeout de 30s para não travar o servidor
        res = requests.post(MODEL_URL, headers=headers, json=payload, timeout=30)
        
        # Se o modelo estiver carregando (erro comum), retornamos erro específico
        if res.status_code == 503:
            return {"error": "Modelo carregando, tente em 20s"}
            
        if res.status_code != 200:
            return {"error": f"Erro HuggingFace: {res.text}"}

        # O LLaVA retorna lista ou dicionário
        response_json = res.json()
        
        raw_text = ""
        if isinstance(response_json, list) and len(response_json) > 0:
            raw_text = response_json[0].get("generated_text", "")
        elif isinstance(response_json, dict):
            raw_text = response_json.get("generated_text", "")

        # Limpeza da resposta do LLaVA
        if "ASSISTANT:" in raw_text:
            raw_text = raw_text.split("ASSISTANT:")[1]

        # Extração do JSON
        s, e = raw_text.find("{"), raw_text.rfind("}")
        if s >= 0 and e >= 0:
            return json.loads(raw_text[s:e+1])
            
        return {"error": "IA não retornou JSON válido", "raw": raw_text}

    except Exception as e:
        return {"error": f"Erro interno: {str(e)}"}
