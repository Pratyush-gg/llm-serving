# Portfolio & Technical Interview Guide
**Routed Multi-Adapter LLM Serving System**

---

## 1. Executive Summary & Framing

### The Problem
When enterprise systems deploy specialized LLMs for diverse tasks (e.g., text-to-SQL, structured entity extraction, and code generation), the standard approach is to host **separate dedicated models** for each task. 
* **The Cost:** Memory consumption scales linearly ($O(N \times V_{\text{base}})$). Serving three 1.5B models concurrently consumes **14.92 GB of VRAM**, exhausting an NVIDIA T4 GPU and preventing batch scalability.
* **The Naive Alternative:** Merging all capabilities into a single model often causes **catastrophic forgetting**, task interference, and bloated prompt engineering.

### The Solution
This project implements a **Routed Multi-Adapter Serving System**:
1. **Shared Frozen Base:** A single instance of `Qwen2.5-1.5B-Instruct` is kept in VRAM.
2. **Specialized LoRA Adapters:** Three rank-16 QLoRA adapters ($\approx 16\text{ MB}$ each) are loaded dynamically by vLLM.
3. **Semantic Router:** A lightweight CPU embedding classifier (`fastembed` with `BAAI/bge-small-en-v1.5`) inspects incoming prompts in **~6 ms** and routes requests to the optimal adapter or falls back to the base model.
4. **Empirical Proof:** Proves both **memory conservation (-66.1% weights reduction)** and **task correctness** against zero-shot baselines on real held-out datasets.

---

## 2. High-Impact Resume Bullet Points

### For Machine Learning Engineer / LLM Systems Roles:
* **Architected and benchmarked a multi-LoRA serving gateway** using vLLM and FastAPI on an NVIDIA T4 GPU, serving 3 specialized adapters (SQL, JSON extraction, Code) on a shared `Qwen2.5-1.5B` base, **reducing model weights VRAM by 66.1% (8.62 GB $\to$ 2.92 GB)** with **$7.98\text{ GB}$ of free headroom** for concurrent KV-cache allocation.
* **Engineered a zero-overhead semantic intent router** with `fastembed` (`BAAI/bge-small-en-v1.5`) achieving **100% routing accuracy (80/80)** on held-out evaluation queries with a median latency of **6.12 ms** and automatic fallback for out-of-domain queries.
* **Designed an execution-based evaluation harness** incorporating an in-memory SQLite sandbox for SQL result verification, Pydantic schema validation for JSON extraction, and an isolated subprocess test runner with 5-second timeouts for Python code generation.
* **Profiled end-to-end serving performance across 100 requests**, recording a P50 latency of **211.69 ms** (6.12 ms routing, 10.84 ms vLLM adapter switch, 198.27 ms generation) and proving superior task correctness over zero-shot baselines.

---

## 3. Interview Technical Q&A

### Q1: Why use vLLM Multi-LoRA instead of merging adapters or running separate model instances?
> **Answer:**
> "Running separate model instances duplicates base model parameters ($O(N)$ VRAM scaling). On an NVIDIA T4 (16GB), three 1.5B models consume ~14.9 GB, reaching the OOM threshold and leaving zero space for KV-cache concurrency.
> 
> Merging adapters into one base model degrades specialized domain accuracy due to parameter interference and catastrophic forgetting.
> 
> vLLM native Multi-LoRA stores the frozen base weights once ($O(1)$ VRAM scaling) and dynamically activates LoRA matrices ($W + B A$) per request. The LoRA activation overhead is minimal (~10-14 ms), while saving 66.1% of weights VRAM and allowing all adapters to share a single high-throughput PagedAttention KV-cache pool."

---

### Q2: Why use an embedding-based semantic router instead of an LLM-based classifier?
> **Answer:**
> "Using an LLM to classify intent adds significant token latency (150–300 ms) and consumes valuable GPU compute just for routing. 
> 
> Instead, our router uses an ONNX-runtime optimized bi-encoder (`BAAI/bge-small-en-v1.5`) running entirely on CPU. It embeds incoming prompts and performs cosine similarity against normalized centroid exemplar vectors in **~6 ms** (P50: 6.12 ms). It achieves 100% routing accuracy on our test set without consuming a single byte of GPU VRAM."

---

### Q3: How did you evaluate correctness rather than just latency and throughput?
> **Answer:**
> "A core failure mode of LLM serving benchmarks is claiming efficiency while ignoring model degradation. We designed domain-specific ground-truth harnesses:
> 1. **SQL:** We spin up an in-memory SQLite database populated with the example's schema DDL, execute the generated query, execute the gold query, and compare the sorted result row sets. This tests functional equivalence rather than brittle string matching.
> 2. **JSON Extraction:** We validate the output against a Pydantic `ExtractionSchema` and verify per-field accuracy (`user`, `order_id`, `amount`) with floating-point tolerance.
> 3. **Python Code:** We assemble the function with paired unit assertions and run it in an isolated subprocess with a 5-second timeout, preventing hanging threads or infinite loops."

---

### Q4: How does this compare to Mixture of Experts (MoE)?
> **Answer:**
> "MoE routes tokens at every layer during the forward pass between internal feed-forward sub-networks, which requires pre-training or specialized post-training at the base model architecture level.
> 
> Dynamic Multi-LoRA is a **modular, post-hoc serving architecture**. You can fine-tune new task adapters independently on cheap GPUs (15 minutes on a free Colab T4), hot-swap or add adapters to a running production cluster without downtime, and route at the request level rather than the token level."

---

### Q5: What are the primary bottlenecks and how would you scale to 50+ adapters?
> **Answer:**
> "In vLLM, adapters registered in memory have near-zero switching cost. When scaling to 50+ adapters:
> 1. **Host-to-Device Memory Swapping:** Inactive adapters can be stored in CPU host RAM and streamed into GPU memory via asynchronous CUDA streams upon router cache prediction.
> 2. **Predictive Pre-loading:** Because the semantic router runs in ~6 ms on CPU, we can trigger the GPU adapter load in parallel with prompt tokenization.
> 3. **Hierarchical Routing:** For large adapter sets, a two-stage hierarchical semantic tree (domain cluster $\to$ specific adapter) prevents linear similarity decay."

---

## 4. Architectural Trade-Off Matrix

| Dimension | Prompt Routing (Few-Shot) | Separate Model Instances | Merged Single Model | **Routed Multi-LoRA (Our Approach)** |
| :--- | :--- | :--- | :--- | :--- |
| **GPU VRAM Cost** | Lowest ($O(1)$) | Highest ($O(N)$ - OOM risk) | Lowest ($O(1)$) | **Low ($O(1) + \epsilon$ adapters)** |
| **Domain Specialization** | Weak (context limit) | Strong | Moderate (interference) | **High (isolated adapter weights)** |
| **Modular Extensibility** | Weak | Hard (new server needed) | Hard (must re-train) | **Seamless (drop in new adapter)** |
| **Routing Latency** | None | High (proxy routing) | None | **Negligible (~6 ms on CPU)** |
| **Task Accuracy** | Baseline | High | Degraded | **High (matched tuned accuracy)** |
