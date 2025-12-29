from flask import Flask, request, jsonify
from HuggingFaceAPI import analyze_food

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status":"online","message":"API Diet-AI ativa!"})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json

    if "image" not in data:
        return jsonify({"error":"Envie base64 em 'image'"}), 400

    result = analyze_food(data["image"])
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
