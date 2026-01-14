import json
from pymongo import MongoClient, ASCENDING, TEXT
from sentence_transformers import SentenceTransformer
import re
import os
from pathlib import Path
import certifi

FAQ_FILE_PATH = Path(__file__).parent.parent / "data" / "FAQ.json"

DATABASE_NAME = "slm-capstone-proj"
COLLECTION_NAME = "faqs"

CATEGORY_KEYWORDS = {
    'account': ['account', 'sign up', 'registration', 'login', 'profile'],
    'payment': ['payment', 'credit card', 'debit card', 'paypal', 'pay', 'invoice', 'price'],
    'shipping': ['shipping', 'delivery', 'ship', 'expedited', 'international', 'track', 'package', 'country'],
    'returns': ['return', 'refund', 'exchange', 'cancel', 'change'],
    'orders': ['order', 'purchase', 'checkout', 'cart', 'buy', 'pre-order', 'backorder'],
    'products': ['product', 'item', 'stock', 'available', 'inventory'],
    'support': ['support', 'contact', 'help', 'chat', 'phone', 'email'],
    'policies': ['policy', 'terms', 'conditions'],
    'promotions': ['discount', 'promo', 'code', 'sale', 'loyalty', 'gift card'],
    'services': ['service', 'services', 'gift wrap', 'installation', 'demonstration', 'custom'],
    'reviews': ['reviews', 'review', 'rate'],
    'error': ['damaged', 'wrong', 'improper', 'incorrect'],
    'sales': ['sales', 'sale', 'out of stock', 'coming soon', 'limited-edition', 'discontinued', 'clearance'],
}

def categorize_question(question):
    question_lower = question.lower()
    categories = []
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            categories.append(category)
    
    # Default category if none found
    if not categories:
        categories.append('general')
    
    return categories

def extract_keywords(question):
    """
    Extract important keywords from the question for faster text search
    """
    # Remove common stop words and extract meaningful terms
    stop_words = {'how', 'can', 'what', 'do', 'you', 'i', 'is', 'the', 'a', 'an', 'to', 'for', 'if', 'my', 'your'}
    
    # Tokenize and clean
    words = re.findall(r'\b\w+\b', question.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    return keywords

def prepare_faq_document(faq_item, model):
    question = faq_item['question']
    answer = faq_item['answer']
    
    embedding = model.encode(question).tolist()
    categories = categorize_question(question)
    keywords = extract_keywords(question)
    
    document = {
        'question': question,
        'answer': answer,
        'embedding': embedding,
        'categories': categories,
        'keywords': keywords,
        'question_lower': question.lower()
    }
    
    return document

def create_indexes(collection):
    """
    Create indexes for optimal query performance
    """
    
    # Text index for full-text search on questions
    collection.create_index([('question', TEXT)], name='question_index')
    
    # Category index for filtering
    collection.create_index([('categories', ASCENDING)], name='categories_index')
    
    # Keywords index for fast keyword matching
    collection.create_index([('keywords', ASCENDING)], name='keywords_index')
    
    # Compound index for category + text search
    collection.create_index([
        ('categories', ASCENDING),
        ('question_lower', ASCENDING)
    ], name='category_question_index')

def insert_faqs(faq_file_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = MongoClient(os.environ["MONGO_URI"], tls=True, tlsCAFile=certifi.where())
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    with open(faq_file_path, 'r') as f:
        data = json.load(f)
    
    faqs = data['questions']
    
    documents = []
    
    for i, faq in enumerate(faqs, 1):
        doc = prepare_faq_document(faq, model)
        documents.append(doc)
    
    # Batch insert
    _ = collection.insert_many(documents)
    
    # Create indexes
    create_indexes(collection)
    
    client.close()

if __name__ == "__main__":
    insert_faqs(FAQ_FILE_PATH)