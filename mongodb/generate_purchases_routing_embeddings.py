"""
This file is used to generate the embeddings used by our pipeline.

If the higher-level routing algorithm (see `generate_collection_routing_embeddings.py`) has decided
to use the `purchases` table, there are two types of queries that a consumer can ask:
    (1) Questions about their most recent purchase.
    (2) Questions about a specific purchase or of a specific item they've purchased in the past.

This file generates embeddings so that we can compute the cosine similarity and decide on which 'type'
of purchase question a specific query is.
"""

import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv


load_dotenv()

DB_NAME = "slm-capstone-proj"
COLLECTION_NAME = "purchases_routing"

# Initialize MongoDB connection
client = MongoClient(os.environ["MONGO_URI"])
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define routing examples
routing_data = {
    "latest_purchase": [
        "Tell me about my latest order",
        "What was my most recent purchase?",
        "When did I put in the invoice for my latest purchase?",
        "What is the invoice number for my most recent purchase?",
        "What items do I have in my latest order?",
        "How much did I spend on my latest order?",
        "How many items did I buy in my latest order?",
        "Where was my most recent order shipped to?",
        "Show me my last invoice",
        "What did I just buy?",
        "Details about my newest order",
        "My most recent transaction"
    ],
    "specific_item": [
        "When was the last time I ordered socks?",
        "Tell me about my invoice 56332",
        "How much have I spent on halloween decorations in total?",
        "How many candles have I bought?",
        "how many vases did I buy in invoice 32345?",
        "have I bought a pan before?",
        "What was the last time I ordered a pencil sharpener?",
        "Show me all my candle purchases",
        "Find purchases containing 'heart'",
        "How many times have I ordered the DOGGY RUBBER?",
        "What did I buy in December 2010?",
        "History of all my lantern purchases",
        "When was my latest order of pet food?"
    ]
}

collection.delete_many({})

for route, examples in routing_data.items():
    embeddings = model.encode(examples)
    avg_embedding = np.mean(embeddings, axis=0).tolist()

    document = {
        "route": route,
        "examples": examples,
        "embedding": avg_embedding,
    }
    collection.insert_one(document)

print(f"Created {collection.count_documents({})} aggregated route embeddings")