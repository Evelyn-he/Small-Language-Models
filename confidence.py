import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import math

HF_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"  # HF model for log-probs

def load_tokenizer():
    print("Loading Hugging Face model for evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    hf_model = AutoModelForCausalLM.from_pretrained(HF_MODEL).to(device)
    print("Model loaded.\n")

    return {
        "tokenizer": tokenizer,
        "device": device,
        "hf_model": hf_model
    }

def get_verbalized_confidence(answer):
    answer = answer.lower()
    answer = re.sub(r'[^a-zA-Z0-9\s]', '', answer)


    confident = True
    if "sorry" in answer:
        confident = False
    if "restricted" in answer:
        confident = False
    if "microsoft" in answer:
        confident = False
    if "as an ai" in answer:
        confident = False
    if "as an artificial intelligence" in answer:
        confident = False
    if "unsure" in answer:
        confident= False

    return confident

def get_log_probs(prompt, answer, log_probs_eval):
    tokenizer = log_probs_eval["tokenizer"]
    device = log_probs_eval["device"]
    hf_model = log_probs_eval["hf_model"]

    # tokenize both parts
    full_text = prompt + answer
    inputs = tokenizer(full_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hf_model(**inputs)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    shift_labels = inputs["input_ids"][:, 1:]
    token_logprobs = log_probs[:, :-1, :].gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # figure out where the answer tokens start
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    answer_start = prompt_ids.shape[1] - 1  # offset because of shift

    # only average over answer tokens
    answer_logprobs = token_logprobs[:, answer_start:]
    avg_logprob = answer_logprobs.mean().item()
    avg_prob = math.exp(avg_logprob)

    print(f"Avg log-probability (answer | prompt): {avg_logprob:.4f} | Approx. confidence: {avg_prob:.4f}")

    if avg_prob < 0.1:
        return False
    return True

def evaluate_confidence(prompt, answer, log_probs_eval):
    
    confident = True

    # if not get_log_probs(prompt, answer, log_probs_eval):
    #     print("SLM is not confident on account of log probabilities")
    #     confident = False

    if not get_verbalized_confidence(answer):
        print("SLM is not confident on account of language used")
        confident = False
    
    return confident
    