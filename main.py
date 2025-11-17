import re
from slm import warmup_model, stream_response
from llm import llm_response
from confidence import load_tokenizer
from data_retriever import get_user_context, write_to_output_txt
# from query_augmentation import get_user_context
from embeddings import embed_text, get_order_embeddings, get_relevant_orders

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
    #print("\nFiltered input: ", filtered_user_input, "\n")

    return filtered_user_input

def initialize():
    warmup_model()
    log_probs_eval = load_tokenizer()
    return log_probs_eval

def create_user_session(user_id, redact=True):
    user_context = get_user_context(user_id, redact)
    conversation = []
    filtered_convo = []
    return user_context, conversation, filtered_convo

def process_message(user_id, user_input, args, conversation, filtered_convo, user_context, log_probs_eval):
    filtered_input = user_input_filter(user_input)

    conversation.append({"role": "user", "content": f"Answer question as a customer agent based on the relavant user context (do not provide unnecessary details). User input: {user_input} (User context: {user_context})\n"})
    filtered_convo.append({"role": "user", "content": filtered_input})

    print("AI: ", end="", flush=True)
    reply, confidence = stream_response(args, conversation, log_probs_eval)

    if not confidence:
        print(f"Filtered input: {filtered_input}") #can comment this out later
        reply = llm_response(args, filtered_convo)

    conversation.append({"role": "assistant", "content": reply})
    filtered_convo.append({"role": "assistant", "content": reply})

    return reply


def main_loop(args):
    log_probs_eval = initialize()

    try:
        user_id = int(input("Enter user ID: ").strip())
    except ValueError:
        print("Invalid user ID. Please enter a number.")
        return
    
    user_context, conversation, filtered_convo = create_user_session(user_id)
    # write_to_output_txt(user_context)

    print("Chat with Ollama (type 'exit' or 'quit' to end)")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting ...")
            break

        process_message(user_id, user_input, args, conversation, filtered_convo, user_context, log_probs_eval)


if __name__ == "__main__":
    main_loop()