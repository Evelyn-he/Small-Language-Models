import re
import time
import spacy
from slm import warmup_model, stream_response
from llm import llm_response
from confidence import load_tokenizer
from data_retriever import get_user_data, write_to_output_txt
from context import AggregatedUserData, OrderVectorStore, get_query_context

def user_input_filter(user_input):
    #REGEX PATTERNS
    patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?\d{0,3})?[-.\s()]*(?:\d{3})[-.\s()]*(?:\d{3})[-.\s()]*(?:\d{4})\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "ssn": r"\b\d{3}[-\s]*\d{2}[-\s]*\d{4}\b", #US
        "SIN": r"\b\d{3}[-\s]*\d{3}[-\s]*\d{3}\b",
        #be careful with diff credit card formats, Visa, mastercard, american express etc
        #Postal codes
        "postal_code_canada": r"\b[ABCEGHJ-NPRSTVXY][0-9][ABCEGHJ-NPRSTV-Z]\s?[0-9][ABCEGHJ-NPRSTV-Z][0-9]\b",
        #URLS
        "URL": r"(?:https?://)?(?:www\.)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,63}(?::\d{1,5})?(?:/[^\s]*)?"
    } 
    
    # Handle both single string and list of strings
    if isinstance(user_input, list):
        return [user_input_filter(x) for x in user_input]

    filtered_user_input = user_input
    for label, pattern in patterns.items():
        filtered_user_input = re.sub(pattern, f"[REDACTED {label.upper()}]", filtered_user_input)

    return filtered_user_input


def entity_recognition_filter(user_input):
    #python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
    inputs = nlp(user_input)
    for ent in inputs.ents:
        if ent.label_ in {"PERSON", "GPE", "LOC", "ORG"}:
            user_input = user_input.replace(ent.text,f"[REDACTED {ent.label_}]")
    #print("\nNLP Spacy filtered input: ", user_input, "\n")
    return user_input

def initialize():
    warmup_model()
    log_probs_eval = load_tokenizer()
    return log_probs_eval

def create_user_session(user_id, redact=True):
    user_context = get_user_context(user_id, redact)
    conversation = []
    filtered_convo = []
    # write_to_output_txt(user_context)

    # Create FAISS store for THIS user only
    vector_store = OrderVectorStore(user_context)

    return user_context, conversation, filtered_convo, vector_store

def process_message(user_id, user_input, args, conversation, filtered_convo, user_context, log_probs_eval, vector_store):

    start_time = time.time()
    user_data = get_user_data(user_id, redact)
    end_time = time.time()

    if (args.verbose):
        print("\t[DEBUG] Time to collect all user data: ", end_time - start_time)
    
    conversation = []
    filtered_convo = []


    # Create FAISS store for THIS user only
    start_time = time.time()
    vector_store = OrderVectorStore(user_data)
    end_time = time.time()
    
    if(args.verbose):
        print("\t[DEBUG] Time to create FAISS store: ", end_time - start_time)

    start_time = time.time()
    aggregated_user_data = AggregatedUserData(user_data)
    end_time = time.time()

    if(args.verbose):
        print("\t[DEBUG] Time to Aggregate data: ", end_time - start_time)

    return user_data, conversation, filtered_convo, vector_store, aggregated_user_data

def process_message(
        user_id,
        user_input,
        args,
        conversation,
        filtered_convo,
        user_data,
        log_probs_eval,
        vector_store,
        aggregated_user_data
):

    query_context = get_query_context(
        args,
        user_input=user_input,
        user_data=user_data,
        aggregated_user_data=aggregated_user_data,
        vector_store=vector_store,
    )
    #top_orders =  vector_store.search_and_mask(user_input, top_k=20)
    if (args.verbose):
        print(f"\t[DEBUG] User context:{query_context.replace("\n", "\\n")}") # Note: Remove the .replace to make the debug message prettier.

    filtered_input = user_input_filter(user_input)
    filtered_input = entity_recognition_filter(filtered_input)

    print("\nNLP Spacy filtered input: ", filtered_input, "\n")
    
    filtered_query_context = user_input_filter(query_context)

    conversation.append({
        "role": "user",
        "content": f"Answer question as a customer agent based on the relavant user context (do not provide unnecessary details). If you're unsure, say you're unsure. Negative quantities are returns. User input: {user_input} (User context:\n {query_context})\n"
    })
    filtered_convo.append({
        "role": "user",
        "content": f"Answer question as a customer agent based on the relavant user context (do not provide unnecessary details). Negative quantities are returns. User input: {filtered_input} (User context: {filtered_query_context})\n"
    })

    print("AI: ", end="", flush=True)

    reply, confidence = stream_response(args, conversation, log_probs_eval)

    if not confidence:

        if (args.verbose):
            print(f"\t[DEBUG] Filtered input: {filtered_input}")

        start_time = time.time()
        reply = llm_response(args, filtered_convo)
        end_time = time.time()

        if(args.verbose):
            print("\t[DEBUG] LLM response time: ", end_time - start_time)

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
    
    user_data, conversation, filtered_convo, vector_store, aggregated_user_data = create_user_session(args, user_id)
    # write_to_output_txt(user_data)

    print("Chat with Ollama (type 'exit' or 'quit' to end)")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting ...")
            break

        process_message(user_id, user_input, args, 
            conversation, filtered_convo, user_data, 
            log_probs_eval, vector_store, aggregated_user_data)

        print() 

        print("=" * 60)

if __name__ == "__main__":
    main_loop()