import base64
import requests
import json
import os

HF_API_KEY = os.getenv("HF_API_KEY")  # variável será configurada no RENDER!

MODEL_URL = "https://router.huggingface.co/hf-inference/meta-llama/Llama-3.2-11B-Vision-Instruct"

def analyze_food(base64_image):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"} 
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analise a refeição e retorne apenas JSON {\"name\":\"\",\"cal\":0,\"p\":0,\"c\":0,\"dica\":\"\"}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 300
    }

    res = requests.post(MODEL_URL, headers=headers, json=payload)

    try:
        data = res.json()
    except:
        return {"error": "Resposta inválida da IA"}

    # tenta achar o JSON na resposta
    if "choices" in data:
        txt = data["choices"][0]["message"]["content"][0]["text"]
        s, e = txt.find("{"), txt.rfind("}")
        if s>=0 and e>=0:
            return json.loads(txt[s:e+1])
    
    return {"error": "IA não retornou JSON válido"}
