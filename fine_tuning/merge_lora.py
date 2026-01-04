import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/home/ehe/scratch/phi3"
LORA_MODEL = "/home/ehe/scratch/phi3-finetuned"
OUT_DIR = "/home/ehe/scratch/phi3-merged"

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(base, LORA_MODEL)

print("Merging LoRA...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(OUT_DIR, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(OUT_DIR)

print("DONE")