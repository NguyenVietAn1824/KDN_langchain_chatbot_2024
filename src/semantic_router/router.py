import numpy as np
import json
import os
from typing import List
import openai

class SemanticRouter():
    
    def __init__(self, embedding, file_path : str, routes):
        self.routes = routes
        self.embedding = embedding
        self.routesEmbedding = {}
        self.file_path = file_path

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(self.file_path, "r") as f:
                self.routesEmbedding = json.load(f)
        else:
            for route in routes:
                self.routesEmbedding[route.name] = [self.embedding.encode(sample).tolist() for sample in route.samples]
                print([sample for sample in route.samples])

            with open(file_path, "w") as f:
                json.dump(self.routesEmbedding, f, ensure_ascii=False, indent=4)
        
           

    def get_routes(self):
        return self.routes

    def guide(self, query: str):
        queryEmbedding = self.embedding.encode(query)
        queryEmbedding = np.array(queryEmbedding)
        queryEmbedding = queryEmbedding / np.linalg.norm(queryEmbedding)
        scores = []

        # Calculate the cosine similarity of the query embedding with the sample embeddings of the router.

        for route in self.routes:
            routesEmbedding = self.routesEmbedding[route.name] / np.linalg.norm(self.routesEmbedding[route.name])
            score = np.mean(
                np.dot(routesEmbedding, queryEmbedding.T).flatten())
            scores.append((score, route.name))

        scores.sort(reverse=True)
        print(scores)
        return scores[0]
    










