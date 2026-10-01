# Combined Benchmark Report
**Routed Multi-Adapter LLM Serving System**
*Generated: 2026-10-01 18:48:42. All figures are measured unless marked as an estimate.*

## 1. Task quality: base model vs. LoRA adapters (natural-language prompts)

| Test set | Metric | n | Base model | LoRA adapter | Difference (95% CI, paired) |
| :--- | :--- | ---: | :--- | :--- | :--- |
| SQL (gretel, real data rows) | `execution_accuracy` | 300 | 41.0% [35.3, 47.0] | **56.7%** [51.0, 62.3] | +15.7 pts [+10.0, +21.3] |
| JSON in-scope (order extraction) | `exact_match_rate` | 60 | 53.3% [41.7, 66.7] | **73.3%** [61.7, 85.0] | +20.0 pts [+6.7, +33.3] |
| JSON out-of-scope (paraloq schemas) | `leaf_f1` | 80 | 44.5% [35.5, 54.1] | **34.7%** [26.4, 42.7] | -9.8 pts [-21.2, +0.8] |
| Code: HumanEval | `pass_at_1` | 164 | 44.5% [36.6, 52.4] | **43.9%** [36.0, 51.8] | -0.6 pts [-8.5, +6.7] |
| Code: MBPP (sanitized test) | `pass_at_1` | 257 | 45.1% [39.3, 51.4] | **51.0%** [45.1, 57.2] | +5.8 pts [+0.0, +11.7] |

Brackets are 95% bootstrap confidence intervals. A difference whose interval includes 0 is not
distinguishable from noise at this sample size. Raw model outputs: `results/raw_outputs/`.

**SQL execution accuracy by schema format in the prompt**

| Schema format | n | Base model | LoRA adapter |
| :--- | ---: | :--- | :--- |
| create | 100 | 47.0% [38.0, 58.0] | 61.0% [51.0, 71.0] |
| compact | 100 | 35.0% [26.0, 44.0] | 54.0% [44.0, 64.0] |
| prose | 100 | 41.0% [32.0, 50.0] | 55.0% [45.0, 65.0] |

## 2. GPU memory (measured on NVIDIA GeForce RTX 4050 Laptop GPU, bitsandbytes 4-bit NF4, double quant)

| Quantity | MB | Source |
| :--- | ---: | :--- |
| Base model weights | 1,100 | measured |
| 3 LoRA adapters | 25 | measured |
| Base + 3 adapters resident | 1,125 | measured |
| Peak during generation (128 new tokens) | 1,144 | measured |
| 3 separate fine-tuned models, weights only | 3,300 | **estimate**: 3 x measured base-model weight memory (each separate model = base with one merged adapter, same size in NF4) |

![Memory profile](memory_profile.png)

## 3. Latency (100 sequential requests, max_tokens=128, engine=peft, NVIDIA GeForce RTX 4050 Laptop GPU)

| Stage | mean | p50 | p95 | p99 |
| :--- | ---: | ---: | ---: | ---: |
| Routing | 16.9 ms | 17.7 ms | 26.0 ms | 28.3 ms |
| Adapter switch | 12.4 ms | 8.6 ms | 35.0 ms | 40.9 ms |
| Generation | 4,815.2 ms | 3,416.4 ms | 10,617.2 ms | 11,758.0 ms |
| Other server overhead | 2.8 ms | 2.0 ms | 7.9 ms | 9.4 ms |
| Server total | 4,847.3 ms | 3,448.8 ms | 10,656.8 ms | 11,815.3 ms |
| Client round trip | 4,863.4 ms | 3,476.5 ms | 10,676.3 ms | 11,838.6 ms |

Decode throughput: 9.7 tokens/s (p50); generated tokens per request: 34 (p50).

Percentiles are per stage and do not add up; the chart stacks means.

![Latency breakdown](latency_breakdown.png)

## 4. Router accuracy (split 'test' of `data/router_testset.jsonl`, 166 prompts)

| Router | Overall | Clear-domain | Ambiguous | p50 latency |
| :--- | ---: | ---: | ---: | ---: |
| Centroid (cosine) | 80.7% [74.1, 86.8] | 96.7% [92.2, 100.0] | 61.8% [50.0, 72.4] | 9.38 ms |
| Learned v1 (original, retired) | 35.5% [28.3, 42.8] | 37.8% [27.8, 47.8] | 32.9% [22.4, 43.4] | 7.07 ms |
| Learned v2 (default) | 87.4% [82.5, 92.2] | 100.0% [100.0, 100.0] | 72.4% [61.8, 81.6] | 9.72 ms |

Clear-domain prompts come from public datasets not used for adapter training (gretel synthetic_text_to_sql, MBPP, Dolly) plus hand-written JSON requests; ambiguous prompts are hand-written.

## 5. Cascade (router=learned, split 'test', real generation)

Thresholds calibrated on the calibration half (164 prompts; rule: max accuracy on calibration split; ties -> lowest trigger rate): confidence < 1.0 or top-2 margin < 0.0. Calibration accuracy 88.4% without cascade -> 93.9% with it.

| Subset | n | Direct routing | With cascade | Triggered | Rescues | Harms | p50 latency direct -> cascade |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| all | 166 | 87.4% [82.5, 92.2] | 91.0% [86.8, 95.2] | 99.4% | 8 | 2 | 5,094 -> 19,351 ms |
| ambiguous | 76 | 72.4% [61.8, 81.6] | 81.6% [72.4, 89.5] | 100.0% | 8 | 1 | 8,234 -> 24,969 ms |
| clear | 90 | 100.0% [100.0, 100.0] | 98.9% [96.7, 100.0] | 98.9% | 0 | 1 | 4,473 -> 15,623 ms |
