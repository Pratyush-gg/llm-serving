"""
Keep library caches inside the repository (project rule: never write outside the repo).

Import this module BEFORE importing transformers, huggingface_hub, datasets, fastembed or
matplotlib, because they read these environment variables at import time.

- Base model weights (Qwen) are read from the existing global Hugging Face cache in offline
  mode (`HF_LOCAL_FILES_ONLY`), so nothing is downloaded or written there.
- Everything that may be downloaded or written (router embedding model, datasets, dynamic
  modules, matplotlib font cache) goes to the gitignored `.model_cache/` folder in the repo.
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_CACHE_DIR = os.path.join(REPO_ROOT, ".model_cache")

# Read-only use of the global HF cache for base model weights (set to "0" to allow downloads).
HF_LOCAL_FILES_ONLY = os.environ.get("HF_LOCAL_FILES_ONLY", "1") == "1"

_DEFAULTS = {
    "FASTEMBED_CACHE_PATH": os.path.join(LOCAL_CACHE_DIR, "fastembed"),
    "HF_MODULES_CACHE": os.path.join(LOCAL_CACHE_DIR, "hf_modules"),
    "HF_DATASETS_CACHE": os.path.join(LOCAL_CACHE_DIR, "datasets"),
    "MPLCONFIGDIR": os.path.join(LOCAL_CACHE_DIR, "matplotlib"),
    "HF_HUB_DISABLE_TELEMETRY": "1",
    # The Xet download backend writes logs to ~/.cache/huggingface/xet/logs even when the
    # download target is elsewhere. Use plain HTTP downloads and keep any xet cache local.
    "HF_HUB_DISABLE_XET": "1",
    "HF_XET_CACHE": os.path.join(LOCAL_CACHE_DIR, "xet"),
}

for _key, _value in _DEFAULTS.items():
    os.environ.setdefault(_key, _value)
