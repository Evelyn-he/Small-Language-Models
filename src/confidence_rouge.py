import re
import requests
import time
from rouge_score import rouge_scorer
import numpy as np

OLLAMA_API = "http://localhost:11434/api/generate" # ollama API endpoint

def get_verbalized_confidence(answer):
    answer = answer.lower()
    answer = re.sub(r'[^a-zA-Z0-9\s]', '', answer)

    if re.search(r'\bas a\b', answer) or re.search(r'\bas an\b', answer) or re.search(r'\bas the\b', answer):
            return False

    verbalized_nonconfidence = ["sorry", "restricted", "microsoft", "unsure"]

    confident = True
    for w in verbalized_nonconfidence:
        if w in answer:
            confident = False
            break

    return confident

def generate_multiple_responses(model, prompt, num_samples=2):
    responses = []
    
    payload_base = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 100,
            "stop": ["\n\n", "You:"],
            "temperature": 0.7,  # sampling temperature (higher = more diverse)
        }
    }
    
    for i in range(num_samples):
        try:
            response = requests.post(OLLAMA_API, json=payload_base)
            response.raise_for_status()
            
            data = response.json()
            if "response" in data:
                responses.append(data["response"].strip())
            
            time.sleep(0.1)  # small delay to avoid overwhelming the API
            
        except Exception as e:
            print(f"Failed to generate sample {i+1}: {e}")
            continue
    
    return responses


def calculate_rouge_confidence(responses):
    rouge_types=['rouge1', 'rouge2', 'rougeL']
    
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    similarities = []
    
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            scores = scorer.score(responses[i], responses[j])
            
            # average F1 scores across all rouge types
            avg_f1 = np.mean([scores[rouge_type].fmeasure for rouge_type in rouge_types])
            similarities.append(avg_f1)
    
    # overall confidence is the average similarity
    confidence_score = np.mean(similarities)
    
    return confidence_score


def evaluate_rouge_confidence(model, prompt, original_response, num_samples, 
                              rouge_threshold=0.5, verbose=False):
    if not get_verbalized_confidence(original_response):
        if verbose:
            print("Failed verbalized confidence check")
        return False
 
    additional_responses = generate_multiple_responses(model, prompt, num_samples)
    
    if not additional_responses:
        if verbose:
            print("Cannot not generate additional samples")
        return False
    
    all_responses = [original_response] + additional_responses
    
    confidence_score = calculate_rouge_confidence(all_responses)
    
    if verbose:
        print(f"Rouge Confidence Score: {confidence_score:.4f} (threshold: {rouge_threshold})")
    
    confident = confidence_score >= rouge_threshold
    
    return confident