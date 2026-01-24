import time
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
hf_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> np.ndarray:
    """
    Given some text, return an encodded array using the 'all-MiniLM-L6-v2' huggingface model.
    """
    return hf_model.encode(text, normalize_embeddings=True).astype("float32")
   
def get_query_context(args, user_id, query, retrievers, router, top_k=10):

    query_embedding = embed_text(query)
    
    start_time = time.time()
    route = router.route_collection(args, query_embedding)
    end_time = time.time()

    if (args.verbose):
        print("\t[DEBUG] Routing Time: ", end_time-start_time)

    start_time = time.time()
    context = ""

    if route == "purchases":
        purchases_retriever = retrievers["purchases"]
        context = purchases_retriever.search(args, query, query_embedding, user_id, router)
    elif route == "products":
        products_retriever = retrievers["products"]
        context = products_retriever.search(query_embedding)
    elif route == "faq":
        faq_retriever = retrievers["faq"]
        context = faq_retriever.search(query_embedding)
        
    else:
        print(f"Warning: Route {route} does not exist. Skipping data augmentation")
        return query
    end_time = time.time()
    if(args.verbose):
        print("\t[DEBUG] Context Retrieval Time: ", end_time-start_time)

    return context

    

