from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Local model path
model_path = ("/project/def-khisti/navel/models/gpt-oss-120b/original").resolve()

print("Loading model from:", model_path)
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
print("Model loaded successfully!")

prompt = "What are some ecommerce companies?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
