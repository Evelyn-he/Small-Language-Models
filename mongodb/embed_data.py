from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

BATCH_SIZE = 128
VECTOR_FIELD = "embedding"
DB_NAME = "slm-capstone-proj"
COLLECTION_NAME = "products"

load_dotenv()
client = MongoClient(os.environ["MONGO_URI"])
collection = client[DB_NAME][COLLECTION_NAME]

# result = collection.update_many(
#     {},                # {} matches all documents
#     {"$unset": {"embedding": ""}}
# )

# print(f"Removed 'embedding' field from {result.modified_count} documents.")

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model Loaded")

def build_text(doc):
    fields = {
        "Title": doc.get("Title"),
        "StockCode": doc.get("StockCode"),
        "UnitPrice": doc.get("UnitPrice"),
        "StockQuantity": doc.get("StockQuantity")
    }

    for field, value in fields.items():
        if value in (None, ""):
            print(f"⚠️ Empty field '{field}' in document {doc.get('_id')}")

    return " ".join(str(value) for value in fields.values() if value not in (None, ""))

cursor = collection.find(
    {VECTOR_FIELD: {"$exists": False}},  # Only documents without embeddings
    {
        "Title": 1,
        "StockCode": 1,
        "UnitPrice": 1,
        "StockQuantity": 1
    }
)

batch_texts = []
batch_ids = []
i = 0

for doc in cursor:
    text = build_text(doc)
    if not text:
        continue

    batch_texts.append(text)
    batch_ids.append(doc["_id"])

    if len(batch_texts) >= BATCH_SIZE:
        i += 1
        print(f"Processing Batch {i}...")
        vectors = model.encode(batch_texts, normalize_embeddings=True, convert_to_numpy=True)

        # --- BULK WRITE ---
        operations = [
            UpdateOne({"_id": _id}, {"$set": {VECTOR_FIELD: vector.tolist()}})
            for _id, vector in zip(batch_ids, vectors)
        ]
        if operations:
            collection.bulk_write(operations)
            print(f"✅ Batch {i}: Updated {len(operations)} documents in bulk.")

        batch_texts.clear()
        batch_ids.clear()

# Handle remaining documents
if batch_texts:
    vectors = model.encode(batch_texts, normalize_embeddings=True, convert_to_numpy=True)
    operations = [
        UpdateOne({"_id": _id}, {"$set": {VECTOR_FIELD: vector.tolist()}})
        for _id, vector in zip(batch_ids, vectors)
    ]
    if operations:
        collection.bulk_write(operations)
        print(f"✅ Final Batch: Updated {len(operations)} documents in bulk.")

print("Embeddings added successfully.")
