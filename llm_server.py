from flask import Flask, request, jsonify
from gpt_oss import GPTOSSForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch

app = Flask(__name__)

# Automatically resolve model path relative to project structure
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../models/gpt-oss-120b/original")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# BitsAndBytes quantization to fit large model across GPUs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model across available GPUs
model = GPTOSSForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto",
)

@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_new_tokens", 256)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
