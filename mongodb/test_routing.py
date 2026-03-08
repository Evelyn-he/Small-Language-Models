"""
A gpt-generated script to test the accuracy of our routing algorithm.
"""

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DB_NAME = "slm-capstone-proj"
COLLECTION_NAME = "collection_routing"

# Initialize MongoDB connection
client = MongoClient(os.environ["MONGO_URI"])
db = client[DB_NAME]
aggregated_collection = db[COLLECTION_NAME]

# Initialize embedding model (use the same model as in setup)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load route embeddings from MongoDB
route_embeddings = {}
route_info = {}

for doc in aggregated_collection.find():
    route_embeddings[doc['route']] = np.array(doc['embedding'])
    route_info[doc['route']] = {
        'examples': doc['examples']
    }

print("Loaded routes:", list(route_embeddings.keys()))
print("-" * 80)

def route_query(query, threshold=0):
    """
    Routes a query to the appropriate database based on semantic similarity
    
    Args:
        query: The user's question
        threshold: Minimum similarity score to consider a match (0-1)
    
    Returns:
        dict with route, confidence, and all scores
    """
    # Embed the query
    query_embedding = model.encode(query)
    
    # Calculate similarity with each route
    similarities = {}
    for route, embedding in route_embeddings.items():
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            embedding.reshape(1, -1)
        )[0][0]
        similarities[route] = similarity
    
    # Find best match
    best_route = max(similarities, key=similarities.get)
    best_score = similarities[best_route]
    
    # Check if confidence is above threshold
    if best_score < threshold:
        return {
            'route': 'uncertain',
            'confidence': best_score,
            'all_scores': similarities,
            'message': f'Low confidence ({best_score:.3f}). Consider clarifying the query.'
        }
    
    return {
        'route': best_route,
        'confidence': best_score,
        'all_scores': similarities
    }

def query_database(route, query):
    """
    Dummy function that simulates querying the appropriate database
    """
    responses = {
        'faq': f"[FAQ Database] Searching general information for: '{query}'",
        'purchase_history': f"[Purchase History Database] Searching customer orders for: '{query}'",
        'product_stock': f"[Product Stock Database] Searching inventory for: '{query}'",
        'uncertain': f"[No Database] Unable to determine appropriate database for: '{query}'"
    }
    return responses.get(route, "Unknown route")

def evaluate_routing(test_queries):
    """
    Evaluate routing for a list of test queries
    """
    print("\n" + "="*80)
    print("SEMANTIC ROUTING EVALUATION")
    print("="*80 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: '{query}'")
        print("-" * 80)
        
        # Route the query
        result = route_query(query)
        
        # Display routing decision
        print(f"✓ Routed to: {result['route'].upper()}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"\n  Similarity Scores:")
        for route, score in sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 50)
            print(f"    {route:20s}: {score:.3f} {bar}")
        
        # Simulate database query
#        print(f"\n  Database Response:")
#        print(f"    {query_database(result['route'], query)}")
        print("\n" + "="*80 + "\n")

# Test queries covering all three routes and edge cases
test_queries = [
    # FAQ queries
    "What is your return policy?",
    "How long does shipping take?",
    "What methods of payment do you accept?",
    
    # Purchase history queries
    "When did I last order some mittens?",
    "Have I bought christmas decorations before?",
    "How much have I spent on jugs?",
    "Tell me about my latest order.",
    "When was my latest order?",
    "How many labubus have I bought?",
    "What is the invoice number on my labubu shipment?",
    
    # Product stock queries
    "Do you have any socks in stock?",
    "How much does the textbook cost?",
    "Are there any christmas products available?",
    "What is the stock code of the water bottles?"
]

# Run evaluation
if __name__ == "__main__":
    # evaluate_routing(test_queries)
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE - Enter your own queries (type 'quit' to exit)")
    print("="*80 + "\n")
    
    while True:
        user_query = input("Enter query: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        if not user_query:
            continue
            
        result = route_query(user_query)
        print(f"\n→ Routed to: {result['route'].upper()} (confidence: {result['confidence']:.3f})")
        #print(f"  {query_database(result['route'], user_query)}\n")
        
        # Show all scores
        print("  All scores:")
        for route, score in sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {route}: {score:.3f}")
        print()
    
    print("\nGoodbye!")
    client.close()