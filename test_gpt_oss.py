from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Local model path
model_path = "/project/def-khisti/navel/models/gpt-oss-120b/original"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain why the sky appears blue during the day."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating response...")
outputs = model.generate(**inputs, max_new_tokens=100)

print("Response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
