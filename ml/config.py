from pathlib import Path
import torch

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "ml" / "data"

# Data files
SAMPLE_POSTS_PATH = DATA_DIR / "sample_posts.jsonl"
VERIFIED_FACTS_PATH = DATA_DIR / "verified_facts.jsonl"

# Model/cache paths
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
CLAIM_DETECTOR_MODEL_DIR = MODELS_DIR / "claim_detector"
VERIFIER_MODEL_DIR = MODELS_DIR / "verifier"
RETRIEVAL_CACHE_DIR = PROJECT_ROOT / "ml" / "cache" / "retrieval"

# Models
# LaBSE: Language-agnostic BERT Sentence Embeddings — trained on 109 languages
# with bilingual sentence pairs. Produces identical (or near-identical) vectors
# for semantically equivalent sentences in different languages, enabling
# Hindi <-> English cross-lingual retrieval without any translation step.
EMBEDDING_MODEL_NAME = "sentence-transformers/LaBSE"

# Verifier (Milestone 2): multilingual NLI for claim vs fact
# Premise = retrieved fact text, Hypothesis = extracted claim text
VERIFIER_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
VERIFIER_BATCH_SIZE = 8

# Retrieval — LaBSE cosine scores for cross-lingual equivalent sentences are
# typically 0.85+; unrelated sentences fall below 0.4.
TOP_K_FACTS = 5
MIN_SIMILARITY = 0.40  # primary threshold (EN input)
MIN_SIMILARITY_HI = 0.35  # threshold for Hindi input (LaBSE is balanced, kept slightly lower)
MIN_SIMILARITY_FALLBACK = 0.20  # last-resort: return best available match

# Verifier decision thresholds
# RETRIEVAL_SUPPORT_THRESHOLD must be above the retrieval noise floor (~0.49 for
# unrelated facts) to prevent high-contradiction NLI scores on irrelevant facts
# from overriding correct entailment scores on the true matching fact.
NLI_DECISION_THRESHOLD = 0.45
NLI_WEAK_SIGNAL_THRESHOLD = 0.38
RETRIEVAL_SUPPORT_THRESHOLD = 0.50
RETRIEVAL_STRONG_THRESHOLD = 0.60


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"