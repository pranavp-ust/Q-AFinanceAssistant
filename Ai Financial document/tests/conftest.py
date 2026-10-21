"""Tests run fully offline: no LLM, hashing embeddings, separate vector-store collection."""
import os
import sys
from pathlib import Path

os.environ["LLM_PROVIDER"] = "none"
os.environ["EMBEDDING_PROVIDER"] = "hash"
ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def knowledge_base():
    from src.config import settings
    if not settings.financials_csv.exists():
        import runpy
        runpy.run_path(str(ROOT / "data" / "generate_data.py"), run_name="__main__")
    from src.vectorstore import get_collection, ingest_directory
    if get_collection().count() == 0:
        ingest_directory()
    yield


@pytest.fixture
def graph():
    from src.graph import build_graph
    return build_graph()
