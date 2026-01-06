import time
import numpy as np 
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import faiss
#hf_model = SentenceTransformer("all-MiniLM-L6-v2")  
hf_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


################################################################### (STEP 1) Functions to compute semantic similarity & Top-k orders ###################################################################

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
        "latest":["PurchaseDate"],
        "last":["PurchaseDate"],
        "tracking":["TrackingNumber"],
        "where":["Country"]
    }

    query_lower = query.lower()

    for keyword, field_list in alias_map.items():
        if keyword in query_lower:
            for f in field_list:
                if f in sample_order and f not in fields:
                    fields.append(f)

    # Always keep PurchasedItemDescription for semantic product matching
    if "PurchasedItemDescription" not in fields:
        fields.append("PurchasedItemDescription")

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

    def search_and_mask(self, query: str, top_k=15, threshold=0.25) -> List[dict]:
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
        top_orders = []
        for idx in indices[0]:
            if idx == -1:
                continue
            order = self.orders[idx]
            masked_order = {k: v for k,v in order.items() if k in fields}
            top_orders.append(masked_order)
        
        # Step 7: remove identical key-value pairs
        seen = set()
        unique_top_orders = []

        for d in top_orders:
            key = tuple(d.items())
            if key not in seen:
                seen.add(key)
                unique_top_orders.append(d)

        return unique_top_orders


################################################################### (STEP 2) Functions to Compute Aggregated Data ###################################################################
class AggregatedUserData:
    def __init__(self, orders: List[Dict]):
        self.item_stats = {}

        for order in orders:
            desc = order["PurchasedItemDescription"]

            # Initialize on the first encounter of a unique item
            # (This will also be the most recent purchase because "orders" are
            # sorted from most to least recent.)
            if desc not in self.item_stats:
                self.item_stats[desc] = {
                    "description": desc,
                    "total_entries": 0,
                    "total_quantity": 0,
                    "most_recent_purchase": order["PurchaseDate"],
                    "unit_price": order["UnitPrice"],
                }
            
            self.item_stats[desc]["total_entries"] += 1
            self.item_stats[desc]["total_quantity"] += int(order["Quantity"])
    
    def get_aggregated_top_orders(self, top_orders:List[dict]) -> List[dict]:
        unique_items = {item["PurchasedItemDescription"] for item in top_orders}
        aggregated_top_orders = []

        for item in unique_items:
            aggregated_top_orders.append(self.item_stats[item])
        
        return aggregated_top_orders



################################################################### (STEP 3) Functions to Compute Most Recent Orders ###################################################################

def get_recent_orders(user_data: List[dict], top_orders: List[dict], recent_entries:int=5) -> List[dict]:
    
    if not top_orders:
        return []

    # Use the same columns as top order
    key_order = list(top_orders[0].keys())

    recent = user_data[:recent_entries]

    return [
        {k: order[k] for k in key_order if k in order}
        for order in recent
    ]  

################################################################### Main Function to Augment Query ###################################################################
def get_query_context(
        args,
        user_input: str,
        user_data:List[dict],
        aggregated_user_data:List[dict],
        vector_store:OrderVectorStore,
        top_k:int=25,
        recent_entries:int=5
) -> str:
    
    # Step 1: Get the top-k orders with most relevancy to the input query
    start_time = time.time()
    top_orders = vector_store.search_and_mask(user_input, top_k=15)
    end_time = time.time()
    if(args.verbose):
        print("\t[DEBUG] Top Order Retrieval Time: ", end_time - start_time)


    # Step 2: Based on the unique entries in the top-k orders, collect aggregate data to add to context
    start_time = time.time()
    aggregated_top_orders = aggregated_user_data.get_aggregated_top_orders(top_orders)
    end_time = time.time()
    if(args.verbose):
        print("\t[DEBUG] Aggregated Order Retrieval Time: ", end_time - start_time)


    # Step 3: Append data on most recent orders, regardless of relevancy
    start_time = time.time()
    recent_orders = get_recent_orders(user_data=user_data, top_orders=top_orders, recent_entries=recent_entries)
    end_time = time.time()
    if(args.verbose):
        print("\t[DEBUG] Most Recent Order Retrieval Time: ", end_time - start_time)


    # Step 4: Combine Step 1-3 together to return as context for input query
    def format_list_of_dicts(lst: list[dict], title: str) -> str:
        lines = [f"--- {title} ---"]
        for i, d in enumerate(lst, 1):
            # create a key: value string for each dict
            entries = ", ".join(f"{k}: {v}" for k, v in d.items())
            lines.append(f"{i}. {entries}")
        return "\n".join(lines)
    
    context_parts = [
        format_list_of_dicts(top_orders, "Top Relevant Orders"),
        format_list_of_dicts(aggregated_top_orders, "Aggregated Top Orders"),
        format_list_of_dicts(recent_orders, "Most Recent Orders"),
    ]
    
    context_str = "\n\n".join(context_parts)
    return context_str