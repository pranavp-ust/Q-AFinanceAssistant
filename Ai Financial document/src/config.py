"""Central configuration, loaded from environment variables / .env."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).absolute().parent.parent
load_dotenv(BASE_DIR / ".env")


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_fallback_model: str = os.getenv("OPENAI_FALLBACK_MODEL", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local").lower()

    retrieval_top_k: int = _i("RETRIEVAL_TOP_K", 6)
    # cosine distance cut-off; the offline hashing embedding produces larger distances
    max_relevant_distance: float = _f("MAX_RELEVANT_DISTANCE",
                                      0.9 if os.getenv("EMBEDDING_PROVIDER", "local").lower() == "hash" else 0.75)
    review_confidence_threshold: float = _f("REVIEW_CONFIDENCE_THRESHOLD", 0.6)
    kpi_discrepancy_pct: float = _f("KPI_DISCREPANCY_PCT", 1.0)
    llm_max_retries: int = _i("LLM_MAX_RETRIES", 3)
    memory_window_turns: int = _i("MEMORY_WINDOW_TURNS", 6)

    docs_dir: Path = BASE_DIR / "data" / "documents"
    structured_dir: Path = BASE_DIR / "data" / "structured"
    financials_csv: Path = BASE_DIR / "data" / "structured" / "financials.csv"
    vectorstore_dir: Path = BASE_DIR / "vectorstore"
    logs_dir: Path = BASE_DIR / "logs"
    collection_name: str = "financial_reports"


settings = Settings()
settings.logs_dir.mkdir(parents=True, exist_ok=True)
