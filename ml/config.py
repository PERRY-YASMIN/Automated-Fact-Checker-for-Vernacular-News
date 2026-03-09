from pathlib import Path

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "ml" / "data"

# Data files
SAMPLE_POSTS_PATH = DATA_DIR / "sample_posts.jsonl"
VERIFIED_FACTS_PATH = DATA_DIR / "verified_facts.jsonl"

# Models
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Retrieval
TOP_K_FACTS = 5
MIN_SIMILARITY = 0.4  # threshold for \"reasonable\" match