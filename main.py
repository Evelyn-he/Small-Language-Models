import re
from slm import warmup_model, stream_response
from llm import llm_response
from confidence import load_tokenizer

from data_retriever import get_user_context, write_to_output_txt

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
    
    #I'm just testing now if my sensitive info is filtereed correctley with my function before you recieve the inputs, please tell me if you see the info (thats bad) or if you just see the redacted. Email: test@gmail.com phone: 249-294-3849 credit_card: 3948 2834 2834 2837 ssn: 293-23-2940 SIN: 294-284-248
    
    filtered_user_input = user_input
    for label, pattern in patterns.items():
        filtered_user_input = re.sub(pattern, f"[REDACTED {label.upper()}]", filtered_user_input)

    #debug
    #print("\nFiltered input: ", filtered_user_input, "\n")

    return filtered_user_input


def main_loop(args):
    try:
        user_id = int(input("Enter user ID: ").strip())
    except ValueError:
        print("Invalid user ID. Please enter a number.")
        return
    redact = True
    user_context = get_user_context(user_id, redact)
    write_to_output_txt(user_context)

    conversation = []
    filtered_convo = []
    warmup_model()
    log_probs_eval = load_tokenizer()


    print("Chat with Ollama (type 'exit' or 'quit' to end)")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting ...")
            break

        filtered_input = user_input_filter(user_input)

        filtered_convo.append({"role": "user", "content": filtered_input})
        conversation.append({"role": "user", "content": f"answer question as a customer agent based on the relavant user context (do not provide unnecessary details): {user_input} (User context: {user_context})\n"})

        print("AI: ", end="", flush=True)
        reply, confidence = stream_response(args, conversation, log_probs_eval)

        if not confidence:
            print(f"Filtered input: {filtered_input}") #can comment this out later
            reply = llm_response(args, filtered_convo)

        conversation.append({"role": "assistant", "content": reply})
        filtered_convo.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main_loop()