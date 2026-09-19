import re
import ast
import json
import time
import sqlite3
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field

from src.router import get_router

@dataclass
class CandidateEvaluation:
    adapter: str
    router_score: float
    quality_score: float
    combined_score: float
    response_snippet: str
    validation_notes: str

@dataclass
class CascadeResult:
    response: str
    final_adapter: str
    cascade_triggered: bool
    router_confidence: float
    candidates_evaluated: List[Dict[str, Any]]
    selection_reason: str
    routing_latency_ms: float
    total_latency_ms: float


class DomainQualityScorer:
    """Lightweight rule-based and syntax validators for generated completions."""

    @staticmethod
    def score_sql(text: str) -> Tuple[float, str]:
        """Validates SQL generation quality."""
        clean = text.strip()
        # Remove markdown code fences if present
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:sql)?\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
            clean = clean.strip()

        score = 0.0
        notes = []

        # Check for core SQL keywords
        sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", "INSERT", "CREATE"]
        matched_keywords = [kw for kw in sql_keywords if re.search(r"\b" + kw + r"\b", clean, re.IGNORECASE)]
        if matched_keywords:
            score += 0.4
            notes.append(f"Found keywords: {', '.join(matched_keywords[:3])}")
        else:
            return 0.1, "Missing standard SQL statement keywords"

        # Check balanced parentheses
        if clean.count("(") == clean.count(")"):
            score += 0.2
            notes.append("Balanced parentheses")
        else:
            notes.append("Unbalanced parentheses")

        # Check SQLite syntactic completeness
        try:
            # Append semicolon for sqlite completeness check if missing
            test_stmt = clean if clean.endswith(";") else clean + ";"
            if sqlite3.complete_statement(test_stmt):
                score += 0.3
                notes.append("Valid SQLite statement syntax")
            else:
                score += 0.1
                notes.append("Incomplete SQLite statement")
        except Exception:
            notes.append("Syntax error in SQLite statement")

        # Penalize conversational fluff
        if len(clean.splitlines()) > 5 or "here is the sql" in clean.lower():
            score -= 0.1
            notes.append("Contains conversational padding")
        else:
            score += 0.1

        final_score = max(0.0, min(1.0, score))
        return final_score, "; ".join(notes)

    @staticmethod
    def score_json(text: str) -> Tuple[float, str]:
        """Validates JSON extraction structure and schema conformity."""
        clean = text.strip()
        # Strip markdown fences
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
            clean = clean.strip()

        # Find first JSON object candidate
        match = re.search(r"(\{.*\})", clean, re.DOTALL)
        candidate = match.group(1) if match else clean

        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                return 0.3, "Valid JSON but not a JSON object"

            # Check expected entity extraction fields (user, order_id, amount)
            expected_keys = {"user", "order_id", "amount"}
            present_keys = expected_keys.intersection(data.keys())

            if len(present_keys) == 3:
                return 1.0, "Valid JSON perfectly matching target schema (user, order_id, amount)"
            elif len(present_keys) >= 1:
                return 0.7, f"Valid JSON matching partial schema fields: {list(present_keys)}"
            else:
                return 0.5, f"Valid JSON object with generic keys: {list(data.keys())[:3]}"
        except json.JSONDecodeError as err:
            return 0.1, f"Invalid JSON string: {err.msg}"

    @staticmethod
    def score_code(text: str) -> Tuple[float, str]:
        """Validates Python code syntax and function definition."""
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:python|py)?\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
            clean = clean.strip()

        score = 0.0
        notes = []

        # Check for function definition
        if re.search(r"\bdef\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(", clean):
            score += 0.4
            notes.append("Contains Python function definition (def)")
        else:
            notes.append("No function definition found")

        # Check Python AST parsing
        try:
            tree = ast.parse(clean)
            score += 0.4
            notes.append("Parses cleanly into valid Python AST")

            # Check if has return statement
            has_return = any(isinstance(node, ast.Return) for node in ast.walk(tree))
            if has_return:
                score += 0.2
                notes.append("Contains return statement")
        except SyntaxError as e:
            score = max(0.0, score - 0.2)
            notes.append(f"Python syntax error at line {e.lineno}")

        final_score = max(0.0, min(1.0, score))
        return final_score, "; ".join(notes)

    @staticmethod
    def score_base(text: str) -> Tuple[float, str]:
        """Evaluates general text output quality."""
        clean = text.strip()
        if not clean:
            return 0.0, "Empty response"
        if len(clean) < 15:
            return 0.3, "Response is overly terse (<15 characters)"
        if len(clean) > 80:
            return 0.9, "Well-formed informative natural language response"
        return 0.7, "Moderate length response"

    @classmethod
    def score(cls, route: str, text: str) -> Tuple[float, str]:
        """Dispatch validator based on candidate route."""
        if route == "sql":
            return cls.score_sql(text)
        elif route == "json":
            return cls.score_json(text)
        elif route == "code":
            return cls.score_code(text)
        else:
            return cls.score_base(text)


class CascadeRouter:
    """
    Orchestrates confidence-based cascade routing.
    Threshold: queries with router confidence below cascade_threshold trigger evaluation
    across top candidate adapters.
    """

    def __init__(
        self,
        base_router=None,
        cascade_threshold: float = 0.70,
        margin_threshold: float = 0.15,
        router_weight: float = 0.45,
        quality_weight: float = 0.55,
    ):
        self.router = base_router or get_router()
        self.cascade_threshold = cascade_threshold
        self.margin_threshold = margin_threshold
        self.router_weight = router_weight
        self.quality_weight = quality_weight
        self.scorer = DomainQualityScorer()

    def route_and_generate(
        self,
        prompt: str,
        generate_fn: Callable[[str, str], str],
        force_adapter: Optional[str] = None,
    ) -> CascadeResult:
        """
        Executes cascade routing and returns the optimal completion.
        generate_fn: Callable(route: str, prompt: str) -> str
        """
        t_start = time.perf_counter()

        # Direct override
        if force_adapter:
            t_gen_start = time.perf_counter()
            response = generate_fn(force_adapter, prompt)
            tot_ms = (time.perf_counter() - t_start) * 1000
            q_score, q_notes = self.scorer.score(force_adapter, response)
            eval_item = {
                "adapter": force_adapter,
                "router_score": 1.0,
                "quality_score": round(q_score, 3),
                "combined_score": 1.0,
                "response_snippet": response[:80],
                "validation_notes": q_notes,
            }
            return CascadeResult(
                response=response,
                final_adapter=force_adapter,
                cascade_triggered=False,
                router_confidence=1.0,
                candidates_evaluated=[eval_item],
                selection_reason=f"Manual override forced adapter '{force_adapter}'.",
                routing_latency_ms=0.0,
                total_latency_ms=round(tot_ms, 2),
            )

        # 1. Evaluate primary router
        t_route_0 = time.perf_counter()
        route_info = self.router.route_detailed(prompt)
        routing_latency_ms = (time.perf_counter() - t_route_0) * 1000

        primary_route = route_info["route"]
        primary_conf = route_info["confidence"]
        scores = route_info.get("scores", {})

        # Sort candidate routes by score descending
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_route, top_score = sorted_candidates[0]
        second_route, second_score = sorted_candidates[1] if len(sorted_candidates) > 1 else (None, 0.0)
        score_margin = top_score - second_score

        # Determine if cascade is necessary:
        # Trigger if confidence is below cascade_threshold OR if top-2 difference is tight
        should_cascade = (
            (primary_conf < self.cascade_threshold or score_margin < self.margin_threshold)
            and second_route is not None
            and second_route != top_route
        )

        candidates_data = []

        if not should_cascade:
            # High confidence path: Single shot generation
            response = generate_fn(primary_route, prompt)
            q_score, q_notes = self.scorer.score(primary_route, response)
            candidates_data.append({
                "adapter": primary_route,
                "router_score": round(primary_conf, 4),
                "quality_score": round(q_score, 3),
                "combined_score": round(primary_conf, 4),
                "response_snippet": response[:80].replace("\n", " "),
                "validation_notes": q_notes,
            })
            total_latency_ms = (time.perf_counter() - t_start) * 1000
            return CascadeResult(
                response=response,
                final_adapter=primary_route,
                cascade_triggered=False,
                router_confidence=round(primary_conf, 4),
                candidates_evaluated=candidates_data,
                selection_reason=f"High confidence ({primary_conf:.3f} >= {self.cascade_threshold}) — direct route selected.",
                routing_latency_ms=round(routing_latency_ms, 2),
                total_latency_ms=round(total_latency_ms, 2),
            )

        # 2. Cascade execution: evaluate Top-2 candidate adapters
        top_candidates = [top_route, second_route]
        candidate_evaluations: List[CandidateEvaluation] = []
        candidate_responses: Dict[str, str] = {}

        for cand_route in top_candidates:
            resp_cand = generate_fn(cand_route, prompt)
            candidate_responses[cand_route] = resp_cand

            cand_r_score = scores.get(cand_route, 0.0)
            cand_q_score, cand_q_notes = self.scorer.score(cand_route, resp_cand)
            combined = (cand_r_score * self.router_weight) + (cand_q_score * self.quality_weight)

            eval_record = CandidateEvaluation(
                adapter=cand_route,
                router_score=round(cand_r_score, 4),
                quality_score=round(cand_q_score, 3),
                combined_score=round(combined, 4),
                response_snippet=resp_cand[:80].replace("\n", " "),
                validation_notes=cand_q_notes,
            )
            candidate_evaluations.append(eval_record)
            candidates_data.append({
                "adapter": eval_record.adapter,
                "router_score": eval_record.router_score,
                "quality_score": eval_record.quality_score,
                "combined_score": eval_record.combined_score,
                "response_snippet": eval_record.response_snippet,
                "validation_notes": eval_record.validation_notes,
            })

        # Select highest combined score
        best_cand = max(candidate_evaluations, key=lambda c: c.combined_score)
        other_cand = next(c for c in candidate_evaluations if c != best_cand)

        total_latency_ms = (time.perf_counter() - t_start) * 1000

        if best_cand.adapter != top_route:
            reason = (
                f"Cascade rescue! '{best_cand.adapter}' outperformed '{top_route}' on domain validation "
                f"(quality: {best_cand.quality_score} vs {other_cand.quality_score}, "
                f"combined: {best_cand.combined_score} vs {other_cand.combined_score})."
            )
        else:
            reason = (
                f"Cascade confirmed primary candidate '{best_cand.adapter}' with strong domain validation "
                f"(quality: {best_cand.quality_score}, combined: {best_cand.combined_score})."
            )

        return CascadeResult(
            response=candidate_responses[best_cand.adapter],
            final_adapter=best_cand.adapter,
            cascade_triggered=True,
            router_confidence=round(primary_conf, 4),
            candidates_evaluated=candidates_data,
            selection_reason=reason,
            routing_latency_ms=round(routing_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
        )


if __name__ == "__main__":
    print("Testing Cascade Router with mock generator...")
    
    def mock_gen(route: str, prompt: str) -> str:
        if route == "sql":
            return "SELECT id, name FROM users WHERE active = 1;"
        elif route == "json":
            return '{"user": "Alice", "order_id": "ORD-123", "amount": 49.99}'
        elif route == "code":
            return "def calculate_total(prices):\n    return sum(prices)"
        else:
            return "Here is general knowledge information answering your inquiry."

    cascade = CascadeRouter(cascade_threshold=0.75)
    
    test_queries = [
        "SELECT * FROM items WHERE price < 100",  # High confidence
        "Can you extract user and payment details or write a query for it?",  # Ambiguous
    ]

    for q in test_queries:
        res = cascade.route_and_generate(q, mock_gen)
        print("\n" + "=" * 60)
        print(f"Prompt: {q}")
        print(f"Chosen Adapter: {res.final_adapter} | Cascade Triggered: {res.cascade_triggered}")
        print(f"Reason: {res.selection_reason}")
        print(f"Candidates Evaluated: {json.dumps(res.candidates_evaluated, indent=2)}")
