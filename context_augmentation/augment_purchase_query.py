from typing import List, Dict, Any
from collections import defaultdict

class PurchaseRetriever:
    def __init__(self, collection, col_embeddings):
        self._collection = collection
        self._col_embeddings = col_embeddings
    
    def _select_relevant_fields(self, query: str, threshold=0.25):
        """ 
        Calculate the semantic similarity bewteen the user input and various columns in the database.
        This is done so we can later remove irrelevant information from the context augmentation.
        """
        # query_emb = embed_text(query)

        fields = []
        # for col_doc in purchase_col_embeddings:
        #     col_name = col_doc["column"]
        #     col_emb = np.array(col_doc["embedding"], dtype="float32")
        #     sim = cosine_similarity(query_emb, col_emb)
        #     if sim >= threshold:
        #         fields.append(col_name)

        #common query mappings:
        alias_map = {
            "price": ["UnitPrice"],
            "cost": ["UnitPrice"],
            "much": ["UnitPrice"],
            "qty": ["Quantity"],
            "quantity": ["Quantity"],
            "number": ["Quantity"],
            "many": ["Quantity"],
            "latest":["InvoiceDate"],
            "last":["InvoiceDate"],
            "when":["InvoiceDate"],
            "tracking":["TrackingNumber"],
            "where":["Country"],
        }
        query_lower = query.lower()

        for keyword, mapped_fields in alias_map.items():
                if keyword in query_lower:
                    for f in mapped_fields:
                        if f not in fields:
                            fields.append(f)

        # Always keep Title for semantic product matching
        if "Title" not in fields:
            fields.append("Title")

        return fields

    def _vector_search_purchases(self, user_id: str, embedded_query, project_stage, top_k: int=15) -> list[dict]:
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
            {
                "$project": project_stage
            }
        ]

        return list(self._collection.aggregate(pipeline))
    
    def _get_latest_purchase(self, user_id: str) -> list[dict]:
        
        latest_doc = self._collection.find_one(
            {"CustomerID": user_id},
            sort=[("InvoiceDate", -1)],
            projection={"InvoiceNo": 1}
        )

        if not latest_doc:
            return []

        invoice_no = latest_doc["InvoiceNo"]

        return list(
            self._collection.find({
                "CustomerID": user_id,
                "InvoiceNo": invoice_no
            },
            projection={"_id": 0, "embedding": 0, "CustomerID": 0}) # Columns to exclude
        )

    
    def _format_specific(self, lst: List[Dict[str, Any]], fields: List[str]) -> str:
        static_cols = ["StockCode", "UnitPrice", "Country"]
        dynamic_cols = ["InvoiceNo", "Quantity", "InvoiceDate"]

        grouped = defaultdict(list)
        for row in lst:
            grouped[row["Title"]].append(row)

        lines = []
        active_dynamic = [c for c in dynamic_cols if c in fields]

        for title, rows in grouped.items():
            lines.append(f"ITEM: {title}")

            # print static fields once (use first row)
            for col in static_cols:
                if col in fields:
                    value = rows[0][col]
                    lines.append(f"{col}: {value}")

            if active_dynamic:
                lines.append(f"EVENTS: {active_dynamic}")
                for row in rows:
                    event = [str(row[c]) for c in active_dynamic]
                    lines.append(" - [" + ", ".join(event) + "]")

            lines.append("")

        return "\n".join(lines)

    def _format_latest(self, lst: List[Dict[str, Any]]) -> str:
        static_cols = ["InvoiceNo", "InvoiceDate", "UnitPrice", "Country"]
        dynamic_cols = ["StockCode", "Title", "Quantity"]

        # Aggregate items by StockCode
        aggregated = defaultdict(lambda: {"Title": "", "Quantity": 0})
        for row in lst:
            code = row["StockCode"]
            aggregated[code]["Title"] = row["Title"]
            aggregated[code]["Quantity"] += row["Quantity"]

        lines = []
        lines.append("Latest Order Details:")
        for col in static_cols:
            value = lst[0][col]
            lines.append(f"{col}: {value}")
        
        lines.append(f"ITEMS: {dynamic_cols}")
        for code, data in aggregated.items():
            lines.append(f" - [{code}, {data['Title']}, {data['Quantity']}]")
        
        lines.append("")
        return "\n".join(lines)

    def search(self, args, query, embedded_query, user_id, router, top_k=10):

        route = router.route_purchases(args, embedded_query)

        if route == "specific_item":

            relevant_fields = self._select_relevant_fields(query)
            
            project_stage = {field: 1 for field in relevant_fields}
            project_stage["_id"] = 0
            project_stage["score"] = {"$meta": "vectorSearchScore"}

            top_orders = self._vector_search_purchases(
                user_id=user_id,
                embedded_query=embedded_query,
                project_stage=project_stage,
                top_k=top_k
            )

            return self._format_specific(top_orders, relevant_fields)

        elif route == "latest_purchase":
            most_recent_order = self._get_latest_purchase(user_id)
            return self._format_latest(most_recent_order)

        else:
            print(f"Warning: Route {route} does not exist. Skipping data augmentation.")
            return ""
        

