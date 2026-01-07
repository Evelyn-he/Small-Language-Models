import requests
import ollama
import time
import json

# from confidence import evaluate_confidence
from confidence_rouge import evaluate_rouge_confidence

OLLAMA_API = "http://localhost:11434/api/generate" # ollama API endpoint
MODEL = "phi3:3.8b"
CHAR_DELAY = 0  # delay between characters for printing out AI response

def warmup_model():
    """Dummy request to load the model into memory"""
    payload = {
        "model": MODEL,
        "prompt": "Hi",
        "stream": False,
        "options": {"num_predict": 1}
    }
    requests.post(OLLAMA_API, json=payload)

def stream_response(args, messages, log_probs_eval=None):

    start_time = time.time()
    prompt = ""
    for msg in messages:
        role = "You" if msg["role"] == "user" else "AI"
        prompt += f"{role}: {msg['content']}\n"
    
    prompt += "AI: "

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True, # This allows us to access AI's response before it's done
        "options": {
            "num_predict": 100,  # Maximum number of tokens for the AI response length
            "stop": ["\n\n", "You:"],
            "top_k": 1
        }
    }

    response_text = ""
    start_time = time.time()

    # This prints out the AI's response as it's responding
    # (as opposed to printing once the AI is done responding).
    with requests.post(OLLAMA_API, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            
            try:
                data = json.loads(line.decode("utf-8"))
                
                if "response" in data:
                    chunk = data["response"]
                    
                    for char in chunk:
                        print(char, end="", flush=True)
                        time.sleep(CHAR_DELAY)
                    
                    response_text += chunk
                
                if data.get("done", False):
                    break
                    
            except json.JSONDecodeError:
                continue


    print()

    end_time = time.time()

    if(args.verbose):
        print("\t[DEBUG] SLM response time: ", end_time - start_time)


    start_time = time.time()
    
    # confidence = evaluate_confidence(prompt, response_text, log_probs_eval)

    confidence = evaluate_rouge_confidence(
        model=MODEL,
        prompt=prompt,
        original_response=response_text,
        num_samples=2,  # generate 2 additional responses for comparison
        rouge_threshold=0.25,  # confidence threshold (adjustable)
        verbose=args.verbose
    )
    
    if not confidence:
        print("*** SLM is not confident ***")
    end_time = time.time()

    if(args.verbose):
        print("\t[DEBUG] Confidence evaluation time: ", end_time - start_time)

    return response_text, confidence