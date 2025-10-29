import requests
import json
import time
import re


OLLAMA_API = "http://localhost:11434/api/generate" # ollama API endpoint
MODEL = "phi3:3.8b"
CHAR_DELAY = 0.015  # delay between characters for printing out AI response

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
            "num_predict": 100  # Maximum number of tokens for the AI response length
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

    total_response_time = time.time()

    if args.verbose:
        print(f"    [DEBUG] SLM Response Time: {total_response_time - start_time} seconds")

    return response_text

def user_input_filter(user_input):
    patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?\d{0,3})?[-.\s()]*(?:\d{3})[-.\s()]*(?:\d{3})[-.\s()]*(?:\d{4})\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "ssn": r"\b\d{3}[-\s]*\d{2}[-\s]*\d{4}\b", #US
        "SIN": r"\b\d{3}[-\s]*\d{3}[-\s]*\d{3}\b"
    } 
    
    #I'm just testing now if my sensitive info is filtereed correctley with my function before you recieve the inputs, please tell me if you see the info (thats bad) or if you just see the redacted. Email: test@gmail.com phone: 249-294-3849 credit_card: 3948 2834 2834 2837 ssn: 293-23-2940 SIN: 294-284-248
    
    filtered_user_input = user_input
    for label, pattern in patterns.items():
        filtered_user_input = re.sub(pattern, f"[REDACTED {label.upper()}]", filtered_user_input)

    #debug
    print("\nFiltered input: ", filtered_user_input, "\n")

    return filtered_user_input

def main_loop(args):
    conversation = []
    warmup_model()


    print("Chat with Ollama (type 'exit' or 'quit' to end)")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting ...")
            break
        user_input = user_input_filter(user_input)

        conversation.append({"role": "user", "content": user_input})

        print("AI: ", end="", flush=True)
        reply = stream_response(args, conversation)
        
        # Add AI response to conversation history
        conversation.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main_loop()