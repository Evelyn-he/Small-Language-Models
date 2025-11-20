import numpy as np 
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import faiss
#hf_model = SentenceTransformer("all-MiniLM-L6-v2")  
hf_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def embed_text(text):
    #return hf_model.encode(text)
    return hf_model.encode(text).astype("float32")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#Get dynamic relevant fields from user input
def select_relevant_fields(query: str, sample_order: Dict, threshold=0.1):
    query_emb = embed_text(query)

    fields = []
    for col_name in sample_order.keys():
        col_emb = embed_text(col_name)
        sim = cosine_similarity(query_emb, col_emb)

        if sim >= threshold:
            fields.append(col_name)

    #common query mappings:
    alias_map = {
        "price": ["UnitPrice"],
        "cost": ["UnitPrice"],
        "much": ["UnitPrice"],
        "qty": ["Quantity"],
        "quantity": ["Quantity"],
        "number": ["Quantity"],
        "many": ["Quantity"],
        "latest":["InvoiceDate"]
    }

    query_lower = query.lower()

    for keyword, field_list in alias_map.items():
        if keyword in query_lower:
            for f in field_list:
                if f in sample_order and f not in fields:
                    fields.append(f)

    # Always keep Description for semantic product matching
    if "Description" not in fields:
        fields.append("Description")

    return fields

def dict_to_text(order: Dict, fields: List[str]) -> str:
    """
    Convert any dictionary to a string for embeddings.
    """
    return " | ".join(f"{f}: {order[f]}" for f in fields if f in order)

#FAISS vector store for orders
class OrderVectorStore:
    def __init__(self, orders: List[Dict]):
        self.orders = orders

        # Precompute embeddings per field for each order
        # List[Dict[field_name -> embedding]]
        self.field_embeddings = []
        for order in orders:
            emb_dict = {}
            for field, value in order.items():
                emb_dict[field] = embed_text(f"{field}: {value}")
            self.field_embeddings.append(emb_dict)

    def _build_vectors_for_query(self, fields: List[str]) -> np.ndarray:
        vectors = []
        for order_emb_dict in self.field_embeddings:
            # take only selected fields and average their embeddings
            selected_embs = [order_emb_dict[f] for f in fields if f in order_emb_dict]
            if selected_embs:
                avg_emb = np.mean(selected_embs, axis=0)
            else:
                # fallback if no fields matched
                avg_emb = np.zeros_like(list(order_emb_dict.values())[0])
            vectors.append(avg_emb)
        return np.array(vectors, dtype="float32")

    def search_and_mask(self, query: str, top_k=5, threshold=0.25) -> List[str]:
        if not self.orders:
            return []

        # Step 1: select relevant fields dynamically
        fields = select_relevant_fields(query, self.orders[0], threshold=threshold)

        # Step 2: build vectors for these fields
        vectors = self._build_vectors_for_query(fields)

        # Step 3: build FAISS index (once per query)
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        # Step 4: embed query
        q_emb = embed_text(query).reshape(1, -1)

        # Step 5: search top-k
        distances, indices = index.search(q_emb, top_k)

        # Step 6: convert top orders to masked text using selected fields
        top_orders = [self.orders[idx] for idx in indices[0] if idx != -1]
        return [dict_to_text(o, fields) for o in top_orders]