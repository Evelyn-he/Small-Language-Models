from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import time
import os
import certifi

from mongodb.insert_faq_data import DATABASE_NAME, COLLECTION_NAME, CATEGORY_KEYWORDS

MONGO_URI = os.environ["MONGO_URI"]

model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class FAQRetriever:
    def __init__(self, mongo_uri, db_name, collection_name):
        self.client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.model = model
    
    def category_vector_search(self, query, top_k = 3):
        """
        category label filtering + vector similarity
        slower than two tier approach
        """
        # start_time = time.time()
        
        # Detect category label from query
        query_lower = query.lower()
        detected_categories = []
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                detected_categories.append(category)
        
        # Filter by category if detected
        filter_query = {}
        if detected_categories:
            filter_query = {"categories": {"$in": detected_categories}}
        
        candidates = list(self.collection.find(filter_query).limit(20))
        
        # Calculate similarity for candidates
        query_embedding = self.model.encode(query)
        
        scored_results = []
        for doc in candidates:
            similarity = cosine_similarity(query_embedding, doc['embedding'])
            scored_results.append({
                'question': doc['question'],
                'answer': doc['answer'],
                'categories': doc['categories'],
                'score': similarity,
                '_id': doc['_id']
            })
        
        # Sort by similarity and return top k
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # elapsed = time.time() - start_time
        # print(f"Query time: {elapsed*1000:.2f}ms")
        
        return scored_results[:top_k]
    
    def keyword_vector_search(self, query, top_k = 3):
        """
        keyword match + vector refinement
        fastest approach
        """
        # start_time = time.time()
        
        # Keyword match (get 10-15 candidates)
        candidates = list(self.collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(15))
        
        # If no keyword matches, fall back to all docs
        if not candidates:
            candidates = list(self.collection.find().limit(20))
        
        # Calculate vector similarity on candidates
        query_embedding = self.model.encode(query)
        
        scored_results = []
        for doc in candidates:
            similarity = cosine_similarity(query_embedding, doc['embedding'])
            scored_results.append({
                'question': doc['question'],
                'answer': doc['answer'],
                'categories': doc.get('categories', []),
                'score': similarity
            })
        
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # elapsed = time.time() - start_time
        # print(f"Query time: {elapsed*1000:.2f}ms")
        
        return scored_results[:top_k]
    
    def search(self, query, top_k = 3, method = "keyword"):
        if method == "category":
            return self.category_vector_search(query, top_k)
        elif method == "keyword":
            return self.keyword_vector_search(query, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def close(self):
        self.client.close()


def augment_faqs_query(user_query):
    retriever = FAQRetriever(MONGO_URI, DATABASE_NAME, COLLECTION_NAME)
    
    num_ret_faq = 1
    
    results = retriever.search(user_query, top_k=num_ret_faq, method="keyword")
    
    faq_context = "Relevant FAQs:\n\n"
    for i, result in enumerate(results, num_ret_faq):
        faq_context += f"{i}. Q: {result['question']}\n"
        faq_context += f"   A: {result['answer']}\n\n"
    
    retriever.close()
    
    return faq_context

if __name__ == "__main__":
    user_input = "I need help tracking my shipment"
    faq_context = augment_faqs_query(user_input)
    
    print(f"\nUser Query: {user_input}")
    print(f"\n{faq_context}")
    