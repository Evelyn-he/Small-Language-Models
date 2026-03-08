import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Router:

    def __init__(self, collection_routing, purchases_routing):
        self._route_embeddings = {}
        for doc in collection_routing.find():
            self._route_embeddings[doc['route']] = np.array(doc['embedding'])
        
        self._purchases_route_embeddings = {}
        for doc in purchases_routing.find():
            self._purchases_route_embeddings[doc['route']] = np.array(doc['embedding'])

    def _router(self, args, query_embedding, db_embeddings):
        similarities = {}
        for route, embedding in db_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            similarities[route] = similarity

        best_route = max(similarities, key=similarities.get)
        best_score = similarities[best_route]

        if args.verbose:
            print(f"\t[DEBUG] Routed to {best_route} with confidence {best_score}")

        return best_route


    def route_collection(self, args, query_embedding):
        return self._router(args, query_embedding, self._route_embeddings)

    def route_purchases(self, args, query_embedding):
        return self._router(args, query_embedding, self._purchases_route_embeddings) 
