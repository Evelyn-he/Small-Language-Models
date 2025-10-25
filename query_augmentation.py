import sqlite3
import json
from pathlib import Path

CUSTOMER_DATABASE_PATH = Path(__file__).parent / "customers.db"

conn = sqlite3.connect("customers.db")
cur = conn.cursor()

def get_user_with_orders(user_id, redact=True):
    """Fetch user info and nested orders from the database."""
    conn = sqlite3.connect(CUSTOMER_DATABASE_PATH)
    cur = conn.cursor()

    cur.execute("SELECT id, name, email, address FROM users WHERE id=?", (user_id,))
    user = cur.fetchone()
    if not user:
        return None
    
    user_id, user_name, email, address = user
    
    cur.execute("""
        SELECT item_name, price, order_date, status
        FROM orders WHERE user_id=?
    """, (user_id,))
    orders = [{"item": i, "price": p, "date": d, "status": s} for i, p, d, s in cur.fetchall()]

    conn.close()

    if redact:
        user_id = "[REDACTED]"
        user_name = "[REDACTED]"
        email = "[REDACTED]"
        address = "[REDACTED]"
    
    return {
        "id": user_id,
        "name": user_name,
        "email": email,
        "address": address,
        "orders": orders
    }

def augment_query(user_id, query, redact=True):
    """Build augmented query string including user context."""
    user_data = get_user_with_orders(user_id, redact)
    if not user_data:
        return f"User ID {user_id} not found.\n"

    context_json = json.dumps(user_data, ensure_ascii=False)
    return f"{query} (User context: {context_json})\n"

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
    augmented = augment_query(user_id, query, redact)
    print("\n--- Augmented Query ---\n")
    print(augmented)

if __name__ == "__main__":
    main()