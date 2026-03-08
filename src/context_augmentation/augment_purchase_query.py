from typing import List, Dict, Any
from collections import defaultdict
import re
from dateparser.search import search_dates
from datetime import datetime, timedelta

class PurchaseRetriever:
    def __init__(self, collection, user_id):
        self._collection = collection

        # Calculate information about aggregate item data once
        pipeline = [
            {
                "$match": {
                    "CustomerID": user_id  # Filter for specific customer first
                }
            },
            {
                "$sort": {"OrderDate": -1}  # Sort by OrderDate descending within each group
            },
            {
                "$group": {
                    "_id": "$Title",
                    "stock_code": {"$first": "$StockCode"},  # Get one stock code (should be same for same title)
                    "most_recent_order": {"$first": "$OrderDate"},  # Most recent order (first after sort)
                    "most_recent_tracking_number": {"$first": "$TrackingNumber"},  # Tracking number of most recent order
                    "total_spent": {
                        "$sum": {
                            "$multiply": ["$Quantity", "$UnitPrice"]
                        }
                    },
                    "total_quantity_purchased": {"$sum": "$Quantity"},
                    "tracking_numbers": {"$addToSet": "$TrackingNumber"}  # All unique tracking numbers
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "title": "$_id",
                    "stock_code": 1,
                    "most_recent_order": 1,
                    "most_recent_tracking_number": 1,
                    "total_spent": {"$round": ["$total_spent", 2]},
                    "total_quantity_purchased": 1,
                    "tracking_numbers": 1,
                }
            },
            {
                "$sort": {"most_recent_order": -1}  # Most recent first
            }
        ]
        self.item_stats = list(self._collection.aggregate(pipeline))
        self.item_stats = list(self._collection.aggregate(pipeline))
        
    def _extract_invoice_number_from_query(self, query: str, user_id: int):
        """
        A function to try extracting invoice number from a query.

        1. If the invoice number exists directly in the query, extract using regex.    

            The format for Invoices is 6 numbers that can be preceeded by a "C"
            in the case of returns.

            Example of purchase: 536392
            Example of return: C536379

        2. If the query mentions a date, try extracting the invoice number associated with the date.


        Returns: Invoice Number if found, None if not.
        """

        # Try extracting invoice number directly.
        pattern = r"\b([A-Z]?)(\d{6})\b"
        match = re.search(pattern, query.upper())
        if match:
            letter, digits = match.groups()
            return f"{letter}{digits}"
 
        # Try extracting date to get invoice number.
        matches = search_dates(query)
        if not matches:
            return None
        
        dt = matches[0][1]
        start = datetime(dt.year, dt.month, dt.day)
        end = start + timedelta(days=1)

        print("START: ", start)
        
        doc = self._collection.find_one(
            {
                "CustomerID": user_id,
                "OrderDate": {"$gte": start, "$lt": end}
            },
            sort=[("OrderDate", -1)],
            projection={"TrackingNumber": 1}
        )

        if doc:
            return doc["TrackingNumber"]
        
        return None
    
    def _vector_search_purchases(self, user_id: str, embedded_query, top_k: int=15) -> list[dict]:
        """
        Given a query embedding, search the database for documents with highest similarity.
        """
        query_vector = embedded_query.tolist()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embeddings_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "filter": {
                        "CustomerID": user_id
                    },
                    "numCandidates": 50,
                    "limit": top_k
                }
            },
        ]

        return list(self._collection.aggregate(pipeline))
    
    def _get_latest_purchase(self, user_id: str) -> str:
        
        latest_doc = self._collection.find_one(
            {"CustomerID": user_id},
            sort=[("OrderDate", -1)],
            projection={"TrackingNumber": 1}
        )

        if not latest_doc:
            return []

        invoice_no = latest_doc["TrackingNumber"]
        return invoice_no
    
    # def _format_item_based(self, titles: List[str]) -> str:
    def _format_item_based(self, lst: List[Dict[str, Any]]) -> str:

        titles = list(dict.fromkeys(row["Title"] for row in lst))

        # Filter item_stats for only the requested titles
        filtered_items = [item for item in self.item_stats if item['title'] in titles]

        
        if not filtered_items:
            return "No items found for the given titles."
        
        lines = []
        
        for item in filtered_items:
            lines.append(f"Title: {item['title']}")
            lines.append(f"StockCode: {item['stock_code']}")
            lines.append(f"Most Recent Order: {item['most_recent_order']}")
            lines.append(f"Most Recent Order Tracking Number: {item['most_recent_tracking_number']}")
            lines.append(f"Total Spent: ${item['total_spent']}")
            lines.append(f"Total Quantity Purchased: {item['total_quantity_purchased']}")
            
            # Show tracking numbers (optionally limit if too many)
            tracking_nums = item['tracking_numbers']
            if len(tracking_nums) <= 10:
                lines.append(f"Tracking Numbers: {', '.join(tracking_nums)}")
            else:
                lines.append(f"Tracking Numbers: {', '.join(tracking_nums[:10])}... ({len(tracking_nums)} total)")
            
            lines.append("")  # Blank line between items
        
        return "\n".join(lines)

    def _format_order_based(self, lst: List[Dict[str, Any]]) -> str:

        static_cols = ["TrackingNumber", "DeliveryDate", "OrderDate", "Address"]
        dynamic_cols = ["StockCode", "Title", "Quantity"]

        # Aggregate items by StockCode
        aggregated = defaultdict(lambda: {"Title": "", "Quantity": 0})
        for row in lst:
            code = row["StockCode"]
            aggregated[code]["Title"] = row["Title"]
            aggregated[code]["Quantity"] += row["Quantity"]

        # Compute total spending
        total_amount = sum(
            row["UnitPrice"] * row["Quantity"]
            for row in lst
        )

        lines = []
        lines.append("Order Details:")

        if not lst:
            return "\n".join(lines)

        for col in static_cols:
            lines.append(f"{col}: {lst[0][col]}")

        # Display total instead of unit price
        lines.append(
            f"Total Cost: -${abs(total_amount):,.2f}" if total_amount < 0
            else f"Total Cost: ${total_amount:,.2f}"
        )

        lines.append(f"ITEMS: {dynamic_cols}")
        for code, data in aggregated.items():
            lines.append(f" - [{code}, {data['Title']}, {data['Quantity']}]")

        lines.append("")
        return "\n".join(lines)

    def search(self, args, query, embedded_query, user_id, router, top_k=10):

        route = router.route_purchases(args, embedded_query)

        if route == "item_based":

            # relevant_fields = self._select_relevant_fields(query)
            
            # project_stage = {field: 1 for field in relevant_fields}
            # project_stage["_id"] = 0
            # project_stage["score"] = {"$meta": "vectorSearchScore"}

            top_orders = self._vector_search_purchases(
                user_id=user_id,
                embedded_query=embedded_query,
                # project_stage=project_stage,
                # projection={"_id": 0, "embedding": 0, "score": 0},
                top_k=top_k
            )

            return self._format_item_based(top_orders)

        elif route == "order_based":

            # 1. Find an invoice number for order.
            invoice_number = self._extract_invoice_number_from_query(query, user_id) 
            if invoice_number and args.verbose:
                print(f"\t[DEBUG] Extracted invoice number {invoice_number} from query.")
            
            if invoice_number is None:
                invoice_number = self._get_latest_purchase(user_id)
                if args.verbose:
                    print(f"\t[DEBUG] Found invoice number {invoice_number} from puchase history of database.")

            # 2. Extract invoice (based on invoice number) from database.
            invoice = list(
                self._collection.find({
                    "CustomerID": user_id,
                    "TrackingNumber": invoice_number
                },
                projection={"_id": 0, "embedding": 0, "CustomerID": 0}) # Columns to exclude
            )

            # 3. Format invoice to be SLM-parsable 
            return self._format_order_based(invoice)

        else:
            print(f"Warning: Route {route} does not exist. Skipping data augmentation.")
            return ""
        

