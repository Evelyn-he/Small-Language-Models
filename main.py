import re
import time
import spacy
import os
import certifi
from pymongo import MongoClient

from slm import warmup_model, stream_response
from llm import llm_response

from context_augmentation.context import get_query_context
from context_augmentation.routing import Router
from context_augmentation.augment_purchase_query import PurchaseRetriever
from context_augmentation.augment_product_query import ProductRetriever
from context_augmentation.augment_faq_query import FAQRetriever

DB_NAME = "slm-capstone-proj"
CUSTOMER_TABLE = "purchases"
BUSINESS_TABLE = "products"
CUSTOMER_COL = "purchases_columns_meta"

# Create a shared environment across all connections
_client = MongoClient(os.environ["MONGO_URI"], tls=True, tlsCAFile=certifi.where())

def create_user_session(args):
    # Retrieve tables from MongoDB
    start_time = time.time()

    db = _client[DB_NAME]

    product_retriever = ProductRetriever(db["products"])
    purchase_retriever = PurchaseRetriever(db["purchases"], db["purchases_columns_meta"])
    faq_retriever = FAQRetriever(db["faqs"])
    router = Router(db["collection_routing"], db["purchases_routing"])

    retrievers = {
        "purchases": purchase_retriever,
        "products": product_retriever,
        "faq": faq_retriever
    }

    # data = {
    #     "purchases": db["purchases"],
    #     "purchases_col_embeddings": db["purchases_columns_meta"],
    #     "purchases_routing": db["purchase_routing"],
    #     "products": db["products"],
    # }

    end_time = time.time()

    if (args.verbose):
        print("\t[DEBUG] Time to get data from MongoDB: ", end_time - start_time)

    return retrievers, router

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


def process_message(user_id, user_input, args, conversation, filtered_convo, retrievers, router):

    query_context = get_query_context(
        args=args,
        user_id=user_id,
        query=user_input,
        retrievers=retrievers,
        router=router
    )
    
    if (args.verbose):
        print(f"\t[DEBUG] User context:\n{query_context}")

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

    reply, confidence = stream_response(args, conversation)

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

    max_saved_prompt = 0 # currently disable any saved prompt
    max_length = max_saved_prompt * 2 # 1 for user content & 1 for assistant content
    if max_length == 0:
        conversation.clear()
        filtered_convo.clear()
    elif len(conversation) > max_length:
        del conversation[:-max_length]
        del filtered_convo[:-max_length]

    return reply


def main_loop(args):
    warmup_model()

    try:
        user_id = int(input("Enter user ID: ").strip())
    except ValueError:
        print("Invalid user ID. Please enter a number.")
        return
    
    retrievers, router = create_user_session(args)
    
    conversation = []
    filtered_convo = []
    # write_to_output_txt(user_data)

    print("Chat with Ollama (type 'exit' or 'quit' to end)")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting ...")
            break

        process_message(user_id, user_input, args, conversation, filtered_convo, retrievers, router)

        print() 

        print("=" * 60)

if __name__ == "__main__":
    main_loop()