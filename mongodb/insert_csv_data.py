"""
This script takes in a .csv file and inserts the contents into mongodb.
"""
import pandas as pd
from pymongo import MongoClient
from dateutil import parser
from dotenv import load_dotenv
import os


"""
CONFIG
"""

CSV_PATH = "../data/business_data.csv"
DB_NAME = "slm-capstone-proj"
COLLECTION_NAME = "products"

"""
CONNECT TO DATABASE
"""
load_dotenv()

client = MongoClient(os.environ["MONGO_URI"])
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

"""
LOAD CSV
"""
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["Title"])


documents = []

for _, row in df.iterrows():
    try:
        doc = {
            "StockCode": str(row["StockCode"]),
            "Title": row["Title"].strip(),
            "UnitPrice": float(row["UnitPrice"]),
            "StockQuantity": int(row["StockQuantity"]),
        }
        documents.append(doc)
    except Exception as e:
        print(f"Skipping row due to error: {e}")

"""
INSERT INTO MONGODB
"""
if documents:
    collection.insert_many(documents)
    print(f"Inserted {len(documents)} documents.")
else:
    print("No documents to insert.")