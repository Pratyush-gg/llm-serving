import os
import sys
import json
import time
import random
import pickle
import argparse
from typing import List, Tuple, Dict

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from fastembed import TextEmbedding

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROUTER_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CACHE_FILE = "data/router_embeddings_cache.npz"
LABEL_LIST = ["sql", "json", "code", "base"]
LABEL_TO_INT = {l: i for i, l in enumerate(LABEL_LIST)}
INT_TO_LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

# Templates & topics to generate diverse out-of-domain (base) queries
OOD_TOPIC_TEMPLATES = [
    # Science & Nature
    "What is the theory of general relativity and how was it proven?",
    "Explain how photosynthesis converts light energy into chemical energy in plants.",
    "Why is the sky blue during the day and red during sunset?",
    "How do airplanes generate lift using Bernoulli's principle and Newton's laws?",
    "What is quantum entanglement and why did Einstein call it spooky action at a distance?",
    "How does the human immune system recognize and fight off viral infections?",
    "What are tectonic plates and what causes volcanic eruptions?",
    "Explain the life cycle of a massive star from nebula to black hole.",
    "What is the difference between nuclear fission and nuclear fusion?",
    "How does CRISPR-Cas9 gene editing technology work?",
    "Why do oceans have tides and how does lunar gravity influence them?",
    "What are black holes and what is the event horizon?",
    "How do neurons communicate across synaptic junctions?",
    "Explain the Doppler effect with a real-world acoustic example.",
    "What causes lightning and thunder during a severe thunderstorm?",
    
    # History & Civilization
    "Summarize the primary causes that led to the fall of the Western Roman Empire.",
    "What was the historical significance of the Magna Carta signed in 1215?",
    "How did the Silk Road shape cultural and economic exchange between East and West?",
    "What triggered the Industrial Revolution in Great Britain during the 18th century?",
    "Explain the geopolitical consequences of the Treaty of Versailles in 1919.",
    "Who was Alexander the Great and what was the extent of his empire?",
    "Describe the architectural achievements of the Ancient Egyptian Old Kingdom.",
    "What factors contributed to the outbreak of World War I in Europe?",
    "How did the printing press invented by Gutenberg revolutionize literacy and religion?",
    "Discuss the causes and global repercussions of the 1929 Great Depression.",
    "What role did codebreakers at Bletchley Park play during World War II?",
    "Explain the origins and major milestones of the Space Race during the Cold War.",
    
    # Philosophy & Literature
    "What are the key differences between Stoicism and Epicureanism?",
    "Can you explain the philosophical concept of existentialism according to Sartre?",
    "Summarize the central moral dilemma in Shakespeare's tragedy Hamlet.",
    "What is Plato's Allegory of the Cave and what does it reveal about human perception?",
    "Explain Kant's categorical imperative and deontological ethics.",
    "What are the primary themes explored in George Orwell's dystopian novel 1984?",
    "Describe the concept of the social contract in Hobbes, Locke, and Rousseau.",
    "How does utilitarianism define the greatest happiness principle?",
    "What is the significance of the hero's journey in mythologist Joseph Campbell's work?",
    "Discuss the stream of consciousness literary technique in Virginia Woolf's novels.",
    
    # Arts, Culture & Music
    "Who painted the Mona Lisa and what makes its composition revolutionary?",
    "Describe the evolution of jazz music from New Orleans blues to bebop.",
    "What are the defining characteristics of Renaissance art versus Baroque art?",
    "How did Impressionism challenge conventional academic painting in 19th-century France?",
    "Explain the culinary history and cultural traditions behind authentic Italian pizza.",
    "What is the cultural significance of the Japanese tea ceremony (Chado)?",
    "Describe the architectural philosophy behind Gothic cathedrals and flying buttresses.",
    "How did the Beatles influence modern songwriting and studio recording techniques?",
    
    # Everyday Life, Advice & Creativity
    "Give me 5 creative ideas for a healthy, high-protein breakfast without eggs.",
    "What are the best evidence-based habits for improving sleep quality and duration?",
    "Provide practical tips for managing stage fright and public speaking anxiety.",
    "How can a beginner start training for a 10K running race safely?",
    "Write a short, evocative poem describing autumn leaves falling in an empty park.",
    "What are effective techniques for conflict resolution in a workplace team?",
    "Suggest three compelling science fiction novels for someone new to the genre.",
    "How does compound interest work and why is starting early beneficial for retirement?",
    "What is the difference between aerobic and anaerobic exercise for cardiovascular health?",
    "Give advice on how to cultivate mindful meditation for stress reduction.",
    
    # Geography & General Knowledge
    "What is the difference between climate and weather patterns?",
    "What is the highest mountain peak in North America and where is it located?",
    "Why is the Mariana Trench the deepest point in the world's oceans?",
    "What are the major desert biomes and how do organisms adapt to extreme aridity?",
    "Explain the difference between renewable and non-renewable energy resources.",
    "What is the capital of Australia and why was it chosen over Sydney and Melbourne?",
    "How do ocean currents like the Gulf Stream regulate global temperatures?",
    "What are the seven wonders of the ancient world and which one still stands?",
]

def generate_diverse_ood_prompts(target_count: int = 600) -> List[str]:
    prompts = list(OOD_TOPIC_TEMPLATES)
    prefixes = [
        "Please explain: ",
        "Can you describe ",
        "Provide an overview of ",
        "Summarize the key ideas behind ",
        "In simple terms, what is ",
        "What are the historical origins of ",
        "Write a comprehensive explanation of ",
        "What should someone know about ",
        "Give me a detailed comparison regarding ",
        "Could you elaborate on ",
    ]
    suffixes = [
        " Provide clear reasoning.",
        " Explain in two to three concise paragraphs.",
        " Give a balanced overview with examples.",
        " Highlight the most important takeaways.",
        " Focus on the foundational principles.",
        " Keep the explanation accessible to a beginner.",
        " Outline the main pros and cons.",
    ]
    
    random.seed(42)
    while len(prompts) < target_count:
        base = random.choice(OOD_TOPIC_TEMPLATES)
        p = random.choice(prefixes)
        s = random.choice(suffixes)
        clean_base = base.rstrip("?")
        if random.random() < 0.5:
            new_prompt = f"{p}{clean_base.lower()}?{s}"
        else:
            new_prompt = f"{base}{s}"
        
        if new_prompt not in prompts:
            prompts.append(new_prompt)
            
    return prompts[:target_count]

def load_jsonl_prompts(filepath: str, max_count: int = 600) -> List[str]:
    prompts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "prompt" in item:
                prompts.append(item["prompt"])
            if len(prompts) >= max_count:
                break
    return prompts

def train_learned_router(
    output_dir: str = "models",
    samples_per_domain: int = 600,
    random_state: int = 42,
    use_cache: bool = True,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("\n" + "=" * 65, flush=True)
    print("TRAINING LEARNED ROUTER (2-LAYER MLP CLASSIFIER)", flush=True)
    print("=" * 65, flush=True)
    
    # 1. Check if embeddings cache exists
    X, y_int = None, None
    if use_cache and os.path.exists(CACHE_FILE):
        print(f"Loading cached embeddings from {CACHE_FILE}...", flush=True)
        try:
            cached = np.load(CACHE_FILE, allow_pickle=True)
            X = cached["X"]
            y_int = cached["y_int"]
            print(f"   [+] Loaded cached features: {X.shape}, labels: {y_int.shape}", flush=True)
        except Exception as e:
            print(f"   [!] Cache read failed: {e}. Recomputing...", flush=True)
            X, y_int = None, None

    if X is None or y_int is None:
        print("1. Loading domain prompts from training datasets...", flush=True)
        sql_prompts = load_jsonl_prompts("data/sql_train.jsonl", samples_per_domain)
        json_prompts = load_jsonl_prompts("data/json_train.jsonl", samples_per_domain)
        code_prompts = load_jsonl_prompts("data/code_train.jsonl", samples_per_domain)
        base_prompts = generate_diverse_ood_prompts(samples_per_domain)
        
        print(f"   [+] SQL  samples : {len(sql_prompts)}", flush=True)
        print(f"   [+] JSON samples : {len(json_prompts)}", flush=True)
        print(f"   [+] CODE samples : {len(code_prompts)}", flush=True)
        print(f"   [+] BASE samples : {len(base_prompts)}", flush=True)
        
        prompts = sql_prompts + json_prompts + code_prompts + base_prompts
        y_int = np.array(
            [0] * len(sql_prompts) +
            [1] * len(json_prompts) +
            [2] * len(code_prompts) +
            [3] * len(base_prompts),
            dtype=np.int32
        )
        
        print("\n2. Computing BGE-small embeddings via FastEmbed (ONNX CPU)...", flush=True)
        t0 = time.perf_counter()
        embedder = TextEmbedding(model_name=ROUTER_MODEL_NAME)
        embeddings = list(embedder.embed(prompts, batch_size=128))
        X = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
        
        embed_time = time.perf_counter() - t0
        print(f"   [+] Computed {len(X)} embeddings in {embed_time:.2f}s ({len(X)/embed_time:.1f} samples/sec)", flush=True)
        
        # Save cache
        np.savez_compressed(CACHE_FILE, X=X, y_int=y_int)
        print(f"   [+] Cached embeddings saved to {CACHE_FILE}", flush=True)

    # 3. Stratified Train / Validation Split
    print("\n3. Performing Stratified Train / Validation Split (85% train, 15% val)...", flush=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_int, test_size=0.15, random_state=random_state, stratify=y_int
    )
    print(f"   [+] Train set : {X_train.shape[0]} samples", flush=True)
    print(f"   [+] Val set   : {X_val.shape[0]} samples", flush=True)
    
    # 4. Train MLP Classifier with integer labels
    print("\n4. Fitting MLP Classifier (hidden_layer_sizes=(128, 64))...", flush=True)
    t_train = time.perf_counter()
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        n_iter_no_change=15,
        verbose=False,
    )
    mlp.fit(X_train, y_train)
    train_duration = time.perf_counter() - t_train
    print(f"   [+] Training completed in {train_duration:.2f}s across {mlp.n_iter_} iterations.", flush=True)
    
    # 5. Evaluate on Validation Set
    val_preds = mlp.predict(X_val)
    val_acc = float(np.mean(val_preds == y_val))
    print(f"\n   [+] Validation Accuracy: {val_acc * 100:.2f}%", flush=True)
    print("\nValidation Classification Report:", flush=True)
    print(classification_report(y_val, val_preds, target_names=LABEL_LIST, digits=4), flush=True)
    
    cm = confusion_matrix(y_val, val_preds)
    print("Confusion Matrix:", flush=True)
    print(f"{'':<8}" + "".join([f"{l:>10}" for l in LABEL_LIST]), flush=True)
    for idx, l in enumerate(LABEL_LIST):
        print(f"{l:<8}" + "".join([f"{cm[idx][j]:>10}" for j in range(len(LABEL_LIST))]), flush=True)
        
    # 6. Save Model Artifacts
    model_path = os.path.join(output_dir, "learned_router.pkl")
    meta_path = os.path.join(output_dir, "learned_router_meta.json")
    
    payload = {
        "model": mlp,
        "classes": LABEL_LIST,
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "embedding_model": ROUTER_MODEL_NAME,
        "validation_accuracy": val_acc,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
        
    metadata = {
        "model_file": "learned_router.pkl",
        "embedding_model": ROUTER_MODEL_NAME,
        "architecture": "MLP(384 -> 128 -> 64 -> 4)",
        "classes": LABEL_LIST,
        "num_train_samples": int(X_train.shape[0]),
        "num_val_samples": int(X_val.shape[0]),
        "validation_accuracy": round(val_acc, 4),
        "iterations": int(mlp.n_iter_),
        "training_time_sec": round(train_duration, 2),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\n[OK] Model successfully saved to {model_path}", flush=True)
    print(f"[OK] Metadata successfully saved to {meta_path}", flush=True)
    print("=" * 65 + "\n", flush=True)
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Learned Router")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--samples", type=int, default=600, help="Samples per domain")
    args = parser.parse_args()
    train_learned_router(output_dir=args.output_dir, samples_per_domain=args.samples)
