"""
Self-prompted confidence: uses a fallback model to estimate whether the SLM
could answer the user question. Returns True if fallback (LLM) should be used (low confidence).
"""
import re
import requests

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_FALLBACK = "phi3:3.8b"


def self_prompted_confidence(args, user_query):
    """
    Ask the fallback model to estimate the probability that the SLM could answer the question.
    Returns True if fallback (LLM) should be used (i.e. estimated confidence <= 0.25).
    """
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

                    fallback = confidence <= 0.25
                    return fallback
                except ValueError:
                    if args.verbose:
                        print(f"\t[DEBUG] Could not parse number: {find_num}")

            if args.verbose:
                print(
                    f"\t[DEBUG] No valid confidence number found in response, default to SLM"
                )
            return False

    except Exception as e:
        if args.verbose:
            print(f"\t[DEBUG] Error in confidence check: {e}, default to SLM")
        return False

    return False
