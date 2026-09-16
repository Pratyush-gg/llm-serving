"""
Semantic Router Module for Multi-Adapter Serving System.
Routes incoming queries to specialized LoRA adapters ('sql', 'json', 'code')
or falls back to 'base' for out-of-domain / ambiguous queries.

Powered by fastembed (BAAI/bge-small-en-v1.5) on ONNX Runtime CPU (~2.5ms latency).
"""

import time
from typing import Dict, List, Tuple
import numpy as np
from fastembed import TextEmbedding

# Initialize singleton embedder
ROUTER_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_embedder = None

def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=ROUTER_MODEL_NAME)
    return _embedder

def compute_centroid_anchor(phrases: List[str], embedder: TextEmbedding) -> np.ndarray:
    """Compute normalized centroid embedding vector across representative phrases."""
    vecs = np.array(list(embedder.embed(phrases)))
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    return mean_vec / (norm if norm > 0 else 1.0)

class SemanticRouter:
    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold
        self.embedder = get_embedder()
        self.intent_anchors = self._build_anchors()

    def _build_anchors(self) -> Dict[str, np.ndarray]:
        anchors = {
            "sql": [
                "write a SQL query",
                "select from table where",
                "sql database query schema SELECT",
                "query columns from database table",
                "database schema CREATE TABLE SELECT FROM WHERE",
            ],
            "json": [
                "extract fields and output valid JSON object",
                "convert text into json format schema",
                "parse user, order_id, and amount into json",
                "extract entities into structured json",
                "extract invoice order details user order_id amount as json",
            ],
            "code": [
                "write a python function with code implementation",
                "def solution(args) -> return value",
                "write python code def function",
                "python function implementation def",
                "write a python script with function definition",
            ],
        }
        return {
            intent: compute_centroid_anchor(phrases, self.embedder)
            for intent, phrases in anchors.items()
        }

    def route_detailed(self, prompt: str) -> Dict:
        """Route prompt with confidence scores and latency breakdown."""
        t0 = time.perf_counter()
        
        # Embed single prompt
        vec = list(self.embedder.embed([prompt]))[0]
        norm = np.linalg.norm(vec)
        vec = vec / (norm if norm > 0 else 1.0)

        # Dot product with normalized anchors = cosine similarity
        scores = {
            intent: float(np.dot(vec, anchor))
            for intent, anchor in self.intent_anchors.items()
        }
        
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]
        route = best_intent if confidence >= self.threshold else "base"
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "route": route,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "latency_ms": round(latency_ms, 2),
        }

    def route_query(self, prompt: str) -> str:
        """Simple routing interface returning target adapter name or 'base'."""
        return self.route_detailed(prompt)["route"]

# Global default router instance
_default_router = None

def get_router(threshold: float = 0.60) -> SemanticRouter:
    global _default_router
    if _default_router is None or _default_router.threshold != threshold:
        _default_router = SemanticRouter(threshold=threshold)
    return _default_router

def route_query(prompt: str, threshold: float = 0.60) -> str:
    """Convenience functional API."""
    return get_router(threshold=threshold).route_query(prompt)

if __name__ == "__main__":
    print("Testing SemanticRouter with sample queries...")
    router = get_router()
    
    test_queries = [
        ("SELECT name, age FROM users WHERE age > 21", "sql"),
        ("Extract user and amount: John paid $45.50 for order 123", "json"),
        ("Write a Python function to check if a number is prime", "code"),
        ("What is the capital of France?", "base"),
        ("Write a poem about the sunrise", "base"),
    ]

    for q, expected in test_queries:
        res = router.route_detailed(q)
        print(f"Query: '{q[:50]}...' -> Route: {res['route']:<5} (Expected: {expected:<5}) | Score: {res['confidence']:.3f} | {res['latency_ms']}ms")
