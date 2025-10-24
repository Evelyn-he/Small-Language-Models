import requests
import json
import time


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

        conversation.append({"role": "user", "content": user_input})

        print("AI: ", end="", flush=True)
        reply = stream_response(args, conversation)
        
        # Add AI response to conversation history
        conversation.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main_loop()