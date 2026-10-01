"""Keep library caches inside the repo. Import before transformers, datasets, fastembed or matplotlib."""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_CACHE_DIR = os.path.join(REPO_ROOT, ".model_cache")

_DEFAULTS = {
    "FASTEMBED_CACHE_PATH": os.path.join(LOCAL_CACHE_DIR, "fastembed"),
    "HF_MODULES_CACHE": os.path.join(LOCAL_CACHE_DIR, "hf_modules"),
    "HF_DATASETS_CACHE": os.path.join(LOCAL_CACHE_DIR, "datasets"),
    "MPLCONFIGDIR": os.path.join(LOCAL_CACHE_DIR, "matplotlib"),
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_XET": "1",  # xet writes logs outside the repo
    "HF_XET_CACHE": os.path.join(LOCAL_CACHE_DIR, "xet"),
}
for _key, _value in _DEFAULTS.items():
    os.environ.setdefault(_key, _value)


def local_files_only(repo_id: str) -> bool:
    """Load the model offline if it is already cached, otherwise download it (first run)."""
    mode = os.environ.get("HF_LOCAL_FILES_ONLY", "auto")
    if mode != "auto":
        return mode == "1"
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
