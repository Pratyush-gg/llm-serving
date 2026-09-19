import os
import sys
import time
import pickle
from typing import Dict, List, Tuple, Optional
import numpy as np
from fastembed import TextEmbedding

ROUTER_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_LEARNED_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "learned_router.pkl"
)

# Singleton embedder
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
    """Centroid-based Cosine Similarity Router."""

    def __init__(self, threshold: float = 0.55):
        self.strategy_name = "centroid"
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

        # Include synthetic base score for chart consistency
        scores_with_base = dict(scores)
        scores_with_base["base"] = max(0.0, 1.0 - confidence)

        return {
            "route": route,
            "strategy": "centroid",
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores_with_base.items()},
            "latency_ms": round(latency_ms, 2),
        }

    def route_query(self, prompt: str) -> str:
        """Simple routing interface returning target adapter name or 'base'."""
        return self.route_detailed(prompt)["route"]


class LearnedRouter:
    """Trained Neural MLP Classifier on BGE-Small Embeddings."""

    def __init__(self, model_path: str = DEFAULT_LEARNED_MODEL_PATH, threshold: float = 0.50):
        self.strategy_name = "learned"
        self.model_path = model_path
        self.threshold = threshold
        self.embedder = get_embedder()
        self.model = None
        self.classes = ["sql", "json", "code", "base"]
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Learned router model not found at '{self.model_path}'. "
                f"Run 'python scripts/train_router.py' first to train the classifier."
            )
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.classes = list(data.get("classes", self.classes))
        self.int_to_label = data.get("int_to_label", {i: l for i, l in enumerate(self.classes)})
        self.val_acc = data.get("validation_accuracy", 0.0)

    def route_detailed(self, prompt: str) -> Dict:
        """Route prompt using learned MLP probabilities."""
        t0 = time.perf_counter()

        vec = list(self.embedder.embed([prompt]))[0]
        norm = np.linalg.norm(vec)
        vec = vec / (norm if norm > 0 else 1.0)
        vec = vec.reshape(1, -1)

        probs = self.model.predict_proba(vec)[0]
        scores = {}
        for idx, prob in enumerate(probs):
            cls_key = self.model.classes_[idx]
            label = self.int_to_label.get(cls_key, str(cls_key))
            scores[label] = float(prob)

        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]

        # If highest confidence is below fallback threshold, fallback to base
        if best_intent != "base" and confidence < self.threshold:
            route = "base"
        else:
            route = best_intent

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "route": route,
            "strategy": "learned",
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "latency_ms": round(latency_ms, 2),
        }

    def route_query(self, prompt: str) -> str:
        return self.route_detailed(prompt)["route"]


# Router cache
_routers: Dict[str, object] = {}

def get_router(
    strategy: Optional[str] = None,
    threshold: Optional[float] = None,
    model_path: str = DEFAULT_LEARNED_MODEL_PATH,
) -> object:
    """
    Factory function for routers.
    Strategy can be:
      - 'learned': Uses the trained MLP classifier
      - 'centroid': Uses the cosine-similarity centroid router
      - None / 'auto': Uses 'learned' if model file exists, else 'centroid'
    """
    global _routers

    if strategy is None:
        strategy = os.environ.get("ROUTER_STRATEGY", "auto").lower()

    if strategy == "auto":
        strategy = "learned" if os.path.exists(model_path) else "centroid"

    cache_key = f"{strategy}_{threshold}_{model_path}"
    if cache_key in _routers:
        return _routers[cache_key]

    if strategy == "learned":
        try:
            th = threshold if threshold is not None else 0.50
            router = LearnedRouter(model_path=model_path, threshold=th)
        except Exception as e:
            print(f"[ROUTER WARNING] Could not initialize LearnedRouter ({e}). Falling back to Centroid.")
            th = threshold if threshold is not None else 0.55
            router = SemanticRouter(threshold=th)
    elif strategy == "centroid":
        th = threshold if threshold is not None else 0.55
        router = SemanticRouter(threshold=th)
    else:
        raise ValueError(f"Unknown router strategy: '{strategy}'. Supported: 'learned', 'centroid'")

    _routers[cache_key] = router
    return router

def route_query(prompt: str, strategy: Optional[str] = None, threshold: Optional[float] = None) -> str:
    """Convenience functional API."""
    return get_router(strategy=strategy, threshold=threshold).route_query(prompt)


if __name__ == "__main__":
    print("Testing Router Module...")
    
    test_queries = [
        ("SELECT name, age FROM users WHERE age > 21", "sql"),
        ("Extract user and amount: John paid $45.50 for order 123", "json"),
        ("Write a Python function to check if a number is prime", "code"),
        ("What is the capital of France?", "base"),
        ("Write a poem about the sunrise", "base"),
    ]

    for strat in ["centroid", "learned"]:
        print(f"\n--- Strategy: {strat.upper()} ---")
        try:
            r = get_router(strategy=strat)
            for q, expected in test_queries:
                res = r.route_detailed(q)
                print(f"Query: '{q[:40]}...' -> Route: {res['route']:<5} (Exp: {expected:<5}) | Conf: {res['confidence']:.3f} | {res['latency_ms']}ms | Scores: {res['scores']}")
        except Exception as ex:
            print(f"Strategy {strat} unavailable: {ex}")
