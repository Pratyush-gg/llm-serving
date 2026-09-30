"""Bootstrap confidence intervals for per-example evaluation results."""
import random
from typing import Dict, List, Optional, Sequence


def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000, confidence: float = 0.95,
                 seed: int = 0) -> Optional[Dict[str, float]]:
    """Percentile bootstrap CI of the mean of `values` (e.g. 0/1 correctness per example)."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    lo_idx = int((1 - confidence) / 2 * n_resamples)
    hi_idx = int((1 + confidence) / 2 * n_resamples) - 1
    return {"mean": round(sum(vals) / n, 4), "low": round(means[lo_idx], 4), "high": round(means[hi_idx], 4), "n": n}


def paired_delta_ci(a: Sequence[float], b: Sequence[float], n_resamples: int = 2000,
                    confidence: float = 0.95, seed: int = 0) -> Optional[Dict[str, float]]:
    """CI for mean(b - a) when a[i] and b[i] are results on the same example i (e.g. base vs adapter)."""
    if len(a) != len(b) or not a:
        return None
    return bootstrap_ci([float(y) - float(x) for x, y in zip(a, b)], n_resamples, confidence, seed)


def with_ci(metrics: Dict, key: str, per_example: List[float]) -> Dict:
    """Attach `<key>_ci95` computed from per-example values to a metrics dict."""
    metrics[f"{key}_ci95"] = bootstrap_ci(per_example)
    return metrics
