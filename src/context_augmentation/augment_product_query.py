from typing import List, Dict, Any

class ProductRetriever:
    def __init__(self, collection):
        self._collection = collection
    
    def _vector_search_products(self, embedded_query, top_k: int=4):
        query_vector = embedded_query.tolist()
        pipeline = [
            {
                "$vectorSearch" : {
                    "index": "embeddings_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 20,
                    "limit": top_k
                }
            },
            {
                "$project" : {
                    "_id": 0,
                    "embedding": 0,
                }
            }
        ]

        return list(self._collection.aggregate(pipeline))
    
    def _format(self, lst: List[Dict[str, Any]]) -> str:
        if not lst:
            return "No relevant products found.\n"

        # Dynamically get all columns from the first document
        all_cols = list(lst[0].keys())

        lines = []
        lines.append("Relevant Products:")

        for row in lst:
            for col in all_cols:
                lines.append(f"{col}: {row[col]}")
            lines.append("")

        return "\n".join(lines)
        
    
    def search(self, embedded_query, top_k=4):
        top_orders = self._vector_search_products(embedded_query=embedded_query)
        return self._format(top_orders)