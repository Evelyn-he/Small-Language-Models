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

CSV_PATH = "../data/updated_customer_data.csv"
DB_NAME = "slm-capstone-proj"
COLLECTION_NAME = "purchases" 

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
df = pd.read_csv(CSV_PATH)  # Use pipe separator
df = df.dropna(subset=["Title"])
df = df.head(75000)  # Take only first 75,000 rows


documents = []

for _, row in df.iterrows():
    try:
        doc = {
            "TrackingNumber": str(row["TrackingNumber"]),
            "StockCode": str(row["StockCode"]),
            "Title": row["Title"].strip(),
            "Quantity": int(row["Quantity"]),
            "DeliveryDate": parser.parse(row["DeliveryDate"]),
            "UnitPrice": float(row["UnitPrice"]),
            "CustomerID": int(row["CustomerID"]) if pd.notna(row["CustomerID"]) else None,
            "Address": row["Address"].strip()
        }
        documents.append(doc)
    except Exception as e:
        print(f"Skipping row due to error: {e}")

"""
INSERT INTO MONGODB
"""
collection.drop()
if documents:
    collection.insert_many(documents)
    print(f"Inserted {len(documents)} documents.")
else:
    print("No documents to insert.")