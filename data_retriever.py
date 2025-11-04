import sqlite3
from pathlib import Path
import csv

CUSTOMER_DATABASE_PATH = Path(__file__).parent / "data" / "customer_large_database.db"
OUTPUT_FILE_PATH = Path(__file__).parent / "user_context.txt"

def get_user_with_orders(user_id, redact=True):
    """Fetch user info and nested orders from the database."""
    conn = sqlite3.connect(CUSTOMER_DATABASE_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, Country
        FROM orders WHERE CustomerID=?
    """, (user_id,))
    orders = [{"InvoiceNo": no, "StockCode": s, "Description": d, "Quantity": q, "InvoiceDate": date, "UnitPrice": u, "Country": c} for no, s, d, q, date, u, c in cur.fetchall()]

    conn.close()

    if not orders:
        return None

    return orders

def get_user_context(user_id, redact):
    user_data = get_user_with_orders(user_id, redact)
    if not user_data:
        raise RuntimeError(f"User ID {user_id} not found.\n")

    return user_data

def write_to_output_txt(user_context):
    headers = user_context[0].keys()

    with open(OUTPUT_FILE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(user_context)

def main():
    """Main function to take input and print augmented query."""
    try:
        user_id = int(input("Enter user ID: ").strip())
    except ValueError:
        print("Invalid user ID. Please enter a number.")
        return

    query = input("Enter query: ").strip()
    if not query:
        print("Query cannot be empty.")
        return

    redact = False
    user_context = get_user_context(user_id, redact)
    write_to_output_txt(user_context)


if __name__ == "__main__":
    main()