import re
import requests

from src.models.slm import OLLAMA_API, MODEL_FALLBACK


def self_prompted_confidence(args, user_query: str) -> bool:
    print(f"\t[DEBUG] In Fallback checker")

    confidence_prompt = f"""
    You are estimating the probability that a SMALL e-commerce
    customer-service model (SLM) could answer a user question.

    The model is specifically trained for e-commerce support.

    It is good at:
    - answering typical e-commerce customer service questions
    - retrieving information from store systems such as orders, shipments, products, inventory, accounts, FAQs, and policies
    - looking up customer-specific order history (e.g., whether an item was ordered, quantity purchased, order timing, or price at time of purchase)
    - checking delivery or shipment details (e.g., delivery status, delivery date, delivery location, or shipping destination)
    - checking product availability or stock levels
    - retrieving store policies or FAQ information
    - retrieving and combining a few fields from structured store records (e.g., item, date, location, status, price, quantity)
    - answering questions that require simple lookups across orders, deliveries, inventory, or policies
    - applying basic business rules

    Most normal customer-service questions involving orders, shipping,
    deliveries (status, time, or location), products, inventory (availability or quantity),
    order history (quantity, timing, or price), accounts, or store policies
    should receive HIGH probability.

    If the answer can be found by retrieving information from store data
    (orders, shipments, inventory, or policies), the probability should
    usually be above 0.85.

    Limitations:
    - cannot perform deep reasoning or complex multi-step analysis
    - cannot answer philosophical, speculative, or opinion-based questions
    - may struggle with highly ambiguous or unrelated requests
    - may struggle with opinionated questions such as recommendations or inferences

    Note:
    Questions that involve checking order details (such as what, when, where,
    how many, how much, or status) are NOT considered complex reasoning,
    even if multiple pieces of information are involved.

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

    Question: Did I order this item before?
    Answer: 0.90

    Question: Where was this item delivered?
    Answer: 0.90

    Question: Has this item been delivered yet?
    Answer: 0.88

    Question: When did this order arrive?
    Answer: 0.91

    Question: What was the price of this item when I bought it?
    Answer: 0.90

    Question: Is this item in stock?
    Answer: 0.92

    Question: Do you ship internationally?
    Answer: 0.91

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
    AI:
    """

    payload = {
        "model": MODEL_FALLBACK,
        "prompt": confidence_prompt,
        "stream": False,
        "options": {
            "num_predict": 20,
            "stop": ["\n\n", "You:"],
            "temperature": 0.6,
        },
    }

    try:
        response = requests.post(OLLAMA_API, json=payload)
        response.raise_for_status()
        data = response.json()

        if "response" in data:
            response_text = data["response"].strip()

            if args.verbose:
                print(f"\t[DEBUG] SLM confidence response: {response_text}")

            find_num = re.search(r"-?\d+(?:\.\d+)?", response_text)

            if find_num:
                try:
                    confidence = float(find_num.group())
                    confidence = max(0.0, min(1.0, confidence))

                    if args.verbose:
                        print(f"\t[DEBUG] Parsed confidence score: {confidence}")

                    confident = confidence > 0.25
                    if not confident:
                        print("*** SLM is not confident ***")
                    return confident
                except ValueError:
                    if args.verbose:
                        print(f"\t[DEBUG] Could not parse number: {find_num}")

            if args.verbose:
                print(
                    f"\t[DEBUG] No valid confidence number found in response, default to SLM"
                )
                #default to true (confident) if no valid confidence number found
            return True

    except Exception as e:
        if args.verbose:
            print(f"\t[DEBUG] Error in confidence check: {e}, default to SLM")
        return True

    #Default to confident
    return True

