import numpy as np 
from typing import List, Dict

from sentence_transformers import SentenceTransformer
hf_model = SentenceTransformer("all-MiniLM-L6-v2")  


def embed_text(text):
    return hf_model.encode(text)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dict_to_text(d: dict) -> str:
    """
    Convert any dictionary to a string for embeddings.
    """
    parts = [f"{k}: {v}" for k, v in d.items()]
    return " | ".join(parts)

def get_order_embeddings(orders: List[Dict]):
    embeddings = []

    for order in orders:
        text = dict_to_text(order)
        embeddings.append(embed_text(text))
    return embeddings

def get_relevant_orders(orders, order_embeddings, query, num_return=5) -> List[Dict]:

    query_embedding = embed_text(query)

    # Compute similarity
    similarities = [cosine_similarity(np.array(order_emb), np.array(query_embedding))
                    for order_emb in order_embeddings]

    # Attach temporarily for sorting
    for order, sim in zip(orders, similarities):
        order['_sim'] = sim

    # Sort descending and take top N
    top_orders = sorted(orders, key=lambda x: x['_sim'], reverse=True)[:num_return]

    # Remove temporary similarity key
    for order in top_orders:
        order.pop('_sim', None)

    return top_orders