import requests
import ollama
import time
import json
import re

# from confidence import evaluate_confidence
from src.confidence_rouge import evaluate_rouge_confidence

OLLAMA_API = "http://localhost:11434/api/generate" # ollama API endpoint
MODEL = "phi3-q8_0:latest"
MODEL_FALLBACK = "phi3:3.8b"
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

def stream_response(args, messages):

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

    if args.verbose:
        print(f"\t[DEBUG] payload model: {payload['model']}")

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
    
    # confidence = evaluate_confidence(prompt, response_text)

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

def should_use_fallback(args, user_query):
    #currently fall back only if confidence is less than 50%

    print(f"\t[DEBUG] In Fallback checker")
    
    confidence_prompt = f"""
        You are estimating the probability that a SMALL e-commerce
        customer-service model (SLM) could answer a user question.

        The model is specifically trained for e-commerce support.

        It is good at:
        - answering questions about orders, shipping, returns, refunds, and accounts
        - looking up information from orders, products, FAQs, and store policies
        - combining a few simple facts
        - following basic business rules

        Most normal customer-service questions should receive HIGH probability.

        Limitations:
        - cannot perform deep reasoning or complex multi-step analysis
        - cannot answer philosophical, legal, or opinion-based questions
        - may struggle with extremely ambiguous or unrelated requests

        Your task:
        Estimate the probability (0-1) that this model would likely produce
        a helpful answer to the question.

        Calibration examples:

        Question: Where is my order?
        Answer: 0.92

        Question: How do I return an item I bought last week?
        Answer: 0.90

        Question: What is the warranty on this product?
        Answer: 0.88

        Question: Why do humans value material possessions?
        Answer: 0.14

        Question: If shipping delays increase by 15% next year, how will that affect market demand?
        Answer: 0.18

        Rules:
        - Output ONLY a decimal number
        - Format: 0.xx
        - Exactly two decimal places
        - Do not output 0.00, 0.25, 0.50, 0.75, or 1.00

        Question:
        {user_query}

        Answer with only the number.
        AI: """

    payload = {
        "model": MODEL_FALLBACK,
        "prompt": confidence_prompt,
        "stream": False,  # Don't stream 
        "options": {
            "num_predict": 20,  # Short response, just a number
            "stop": ["\n\n", "You:"], 
            "temperature": 0.6  # Lower temp -> deterministic response
        }
    }
    
    try:
        response = requests.post(OLLAMA_API, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if "response" in data:
            response_text = data["response"].strip()
            
            if args.verbose:
                print(f"\t[DEBUG] SLM confidence response: {response_text}")
            
            # Extract a number
            find_num = re.search(r"-?\d+(?:\.\d+)?", response_text)
            
            if find_num:
                # Try to parse the first number found
                try:
                    confidence = float(find_num.group())
                    # Clamp to [0, 1] range
                    confidence = max(0.0, min(1.0, confidence))
                    
                    if args.verbose:
                        print(f"\t[DEBUG] Parsed confidence score: {confidence}")

                    # If confidence > 0.5, use fallback (LLM)
                    fallback = confidence <= 0.25
                    return fallback
                except ValueError:
                    if args.verbose:
                        print(f"\t[DEBUG] Could not parse number: {find_num}")
            
            # If no valid number found, default to False (use SLM)
            if args.verbose:
                print(f"\t[DEBUG] No valid confidence number found in response, default to SLM")
            return False
            
    except Exception as e:
        if args.verbose:
            print(f"\t[DEBUG] Error in confidence check: {e}, default to SLM")
        # Default to False (use SLM) if there's an error
        return False
    
    return False