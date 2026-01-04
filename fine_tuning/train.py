import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

BASE_MODEL_PATH = "/home/ehe/scratch/phi3"
FINETUNED_MODEL_PATH = "/home/ehe/scratch/phi3-finetuned"
DATASET_PATH = "/home/ehe/scratch/dataset.json"

print("Loading Tokenizer ...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer Loaded")

print("Loading Model ...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True
)

model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )
)

print("Model Loaded")

print("Loading Dataset ...")

train_ds = load_dataset(
    "json",
    data_files=DATASET_PATH,
    field="train"
)["train"]

eval_ds = load_dataset(
    "json",
    data_files=DATASET_PATH,
    field="validation"
)["train"]

print("Dataset Loaded")

print("Tokenizing Dataset ...")

def tokenize_example(example):
    prompt = (
        f"<s>Question: {example['question']}\n"
        f"Context: {example['context']}\n"
        f"Answer:"
    )

    full_text = prompt + " " + example["answer"] + "</s>"

    return tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding=False
    )

train_ds = train_ds.map(tokenize_example, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(tokenize_example, remove_columns=eval_ds.column_names)

print("Dataset Tokenized")

print("Setting Training Arguments ...")

training_args = TrainingArguments(
    output_dir=FINETUNED_MODEL_PATH,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=3e-4,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none",
    remove_unused_columns=False
)

print("Training Arguments Ready")

print("Loading Trainer ...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

print("Trainer Loaded")

print("Start Fine Tuning ...")

trainer.train()

print("Fine Tuning Completed")

print("Saving Model ...")

trainer.save_model(FINETUNED_MODEL_PATH)

print("Model Saved")