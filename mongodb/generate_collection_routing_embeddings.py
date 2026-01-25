"""
This file is used to generate the embeddings used by our pipeline to route the context-augmentation
algorithm to one of 3 collections (faq, purchases, products) in our database.

The approach is as follows:
    (1) For each collection (faq, purchases, products), we generate 15-20 example queries that would
        we want to be routed to that particular collection.
    (2) Using hf model 'all-MiniLM-L6-v2', we generate embeddings for each of the queries.
    (3) Average together all the queries corresponding to a particular collection.
    (4) Upload the averaged embeddings & the collection they belong to to MongoDB documents

Then, during runtime, we can compute the cosine-similarity between the user's query and each of the 3
averaged embeddings. The embedding with the highest cosine-similarity will correspond to the collection
that we route to for context-augmentation. 
"""

import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DB_NAME = "slm-capstone-proj"
COLLECTION_NAME = "collection_routing"

# Initialize MongoDB connection
client = MongoClient(os.environ["MONGO_URI"])
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define routing examples
routing_data = {
    "faq": [
       "How can I create an account?",
       "What payment methods do you accept?",
       "How can I track my order?",
       "What is your return policy?",
       "Can I cancel my order?",
       "How long does shipping take?",
       "Do you offer international shipping?",
       "What should I do if my package is lost or damaged?",
       "Can I change my shipping address after placing an order?",
       "How can I contact customer support?",
       "Can I order by phone?",
       "Do you have a loyalty program?",
       "Can I change or cancel an item in my order?",
       "How can I leave a product review?",
       "What should I do if I receive the wrong item?",
       "Can I order a product that is out of stock?",
       "Can I return a product if I changed my mind?",
       "Can I return a product without a receipt?",
       "Can I order a product for delivery to a different country?",
       "Can I return a product if it was damaged due to mishandling during shipping?"
    ],
    "purchases": [
        "What did I order last month?",
        "Show me my purchase history",
        "How many CREAM CUPID HEARTS COAT HANGER did I buy?",
        "What was my last order?",
        "When did I buy the WHITE METAL LANTERN?",
        "What orders did I place in December 2010?",
        "What's my total spending this year?",
        "Did I order from you before?",
        "How much did I spend on my last order?",
        "Can I see my order from last week?",
        "What did I purchase on December 1st?",
        "Have I ever bought any heart-themed products?",
        "What quantity of items did I buy in my last order?",
        "When was my first purchase with you?",
        "When did I last order a mug?",
        "What is the tracking number of my lastest order?",
        "What is the order number of my red mugs?",
        "What country was my glasses shipped to?",
        "When was my latest order?",
        "Has my order 536367 arrived?",
        "Has my latest order arrived yet?",
        "Has my latest order been delivered yet?",
        "How many wooden frames have I bought before?",
        "How much have I spent on christmas decorations?",
        "How many measuring tapes have I bought before?",
        "What types of measuring tapes have I bought before?",
        "How much have I spent on measuring tapes in total?",
    ],
    "products": [
        "Do you have INFLATABLE POLITICAL GLOBE in stock?",
        "How much does the GROOVY CACTUS INFLATABLE cost?",
        "What's the price of DOGGY RUBBER?",
        "Is the HEARTS WRAPPING TAPE available?",
        "How many SPOTS ON RED BOOKCOVER TAPE do you have in stock?",
        "Show me products with 'HEART' in the name",
        "What inflatable products do you sell?",
        "Do you have any wrapping tape in stock?",
        "What's the cheapest product you have?",
        "Are there any cactus-themed items available?",
        "How much stock do you have of the DOGGY RUBBER?",
        "What products are currently out of stock?",
        "Do you sell any political or globe items?"
        "What is the stock code of the antique frames?",
        "What's the difference between a gold and black tape measure?"
    ]
}

# Clear existing collection if it exists
collection.delete_many({})

# Create aggregated route embeddings
print("\nCreating aggregated route embeddings...")
for route, examples in routing_data.items():
    # Generate embeddings for all examples
    embeddings = model.encode(examples)
    
    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0).tolist()
    
    document = {
        "route": route,
        "examples": examples,
        "embedding": avg_embedding,
    }
    collection.insert_one(document)

print(f"Created {collection.count_documents({})} aggregated route embeddings")

# Create vector search indexes (requires MongoDB Atlas or MongoDB 6.0+)
# print("\nNote: To enable vector search, create an Atlas Search index with the following configuration:")
# print("""
# {
#   "mappings": {
#     "dynamic": true,
#     "fields": {
#       "embedding": {
#         "type": "knnVector",
#         "dimensions": 384,
#         "similarity": "cosine"
#       }
#     }
#   }
# }
# """)