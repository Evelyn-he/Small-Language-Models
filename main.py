
from slm import warmup_model, stream_response
from llm import llm_response
from confidence import load_tokenizer

def main_loop(args):
    conversation = []
    warmup_model()
    log_probs_eval = load_tokenizer()


    print("Chat with Ollama (type 'exit' or 'quit' to end)")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting ...")
            break

        conversation.append({"role": "user", "content": user_input})

        print("AI: ", end="", flush=True)
        reply, confidence = stream_response(args, conversation, log_probs_eval)

        if not confidence:
            reply = llm_response(args, conversation)

        conversation.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main_loop()