# import sqlite3
# from pathlib import Path
# import csv
# from datetime import datetime

# CUSTOMER_DATABASE_PATH = Path(__file__).parent / "data" / "customer_large_database.db"
# OUTPUT_FILE_PATH = Path(__file__).parent / "user_data.txt"

# def get_user_with_orders(user_id, redact=True):
#     """Fetch user info and nested orders from the database."""
#     conn = sqlite3.connect(CUSTOMER_DATABASE_PATH)
#     cur = conn.cursor()

#     cur.execute(
#         """
#         SELECT InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, Country
#         FROM orders
#         WHERE CustomerID=?
#         """, (user_id,))
    
#     orders = [
#         {
#             "TrackingNumber": no,
#             "StockCode": s,
#             "PurchasedItemDescription": d,
#             "Quantity": q,
#             "PurchaseDate": date,
#             "UnitPrice": u,
#             "Country": c
#         }
#         for no, s, d, q, date, u, c in cur.fetchall()
#     ]

#     # Sort the orders to be from most to least recent. The most recent orders will be used later for context augmentation
#     orders.sort(
#         key = lambda o: datetime.strptime(o["PurchaseDate"], "%m/%d/%Y %H:%M"),
#         reverse=True
#     )

#     conn.close()

#     if not orders:
#         return None

#     return orders

# def get_user_data(user_id, redact):
#     user_data = get_user_with_orders(user_id, redact)
#     if not user_data:
#         raise RuntimeError(f"User ID {user_id} not found.\n")

#     return user_data

# def write_to_output_txt(user_data):
#     headers = user_data[0].keys()

#     with open(OUTPUT_FILE_PATH, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=headers)
#         writer.writeheader()
#         writer.writerows(user_data)
