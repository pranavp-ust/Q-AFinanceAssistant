"""RAG ingestion and retrieval over ChromaDB.

Retrieval is hybrid (vector + BM25 keyword, fused with Reciprocal Rank Fusion).

Ingestion: PDF -> pages (pypdf) -> chunks (LangChain RecursiveCharacterTextSplitter)
-> embeddings -> Chroma, with metadata {company, fiscal_year, doc_type, source, page}
so retrieval can filter by the company / year the user is asking about.
"""
import hashlib
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import settings
from src.monitoring import logger

SLUG_TO_COMPANY = {
    "nimbus_retail": "Nimbus Retail Ltd",
    "orion_software": "Orion Software Inc",
    "vertex_energy": "Vertex Energy plc",
}


class HashingEmbedding(EmbeddingFunction[Documents]):
    """Fully offline embedding (feature hashing of word uni/bi-grams, L2-normalised).

    No model download needed - useful behind corporate proxies that block HuggingFace
    downloads. Lower semantic quality than a neural model, but solid for keyword-heavy
    financial text.
    """
    DIM = 2048

    def __init__(self):
        pass

    STOP = set("a an the of in on for to and or is was were what which who how much many did does do be by "
               "with from as at it its this that these those their our we you your i me million fy".split())

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            vec = np.zeros(self.DIM, dtype=np.float32)
            words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in self.STOP and len(w) > 1]
            words = [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words]  # crude stemming
            grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
            for g in grams:
                vec[int(hashlib.md5(g.encode()).hexdigest(), 16) % self.DIM] += 1.0
            vec = np.log1p(vec)
            n = np.linalg.norm(vec)
            out.append(vec / n if n else vec)
        return out

    @staticmethod
    def name() -> str:
        return "hashing_ngram"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbedding":
        return HashingEmbedding()


def _embedding_function():
    from chromadb.utils import embedding_functions as ef
    p = settings.embedding_provider
    if p == "openai":
        return ef.OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"],
                                          model_name="text-embedding-3-small")
    if p == "gemini":
        return ef.GoogleGenerativeAiEmbeddingFunction(api_key=os.environ["GOOGLE_API_KEY"])
    if p == "hash":
        return HashingEmbedding()
    return ef.DefaultEmbeddingFunction()  # local all-MiniLM-L6-v2 (ONNX, downloaded once)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(settings.vectorstore_dir))
    return client.get_or_create_collection(
        f"{settings.collection_name}_{settings.embedding_provider}",  # one collection per embedding model
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def parse_filename(path: Path) -> dict:
    """`nimbus_retail_FY2024_annual_report.pdf` -> company / year / doc_type."""
    stem = path.stem
    m = re.match(r"(?P<slug>.+?)_FY(?P<year>\d{4})_(?P<doc>.+)", stem)
    if not m:
        return {"company": "", "fiscal_year": 0, "doc_type": stem}
    return {"company": SLUG_TO_COMPANY.get(m["slug"], m["slug"].replace("_", " ").title()),
            "fiscal_year": int(m["year"]), "doc_type": m["doc"]}


RRF_K = 60
_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100,
                                           separators=["\n\n", "\n", ". ", " ", ""])


def ingest_pdf(path: Path, company: str | None = None, fiscal_year: int | None = None) -> int:
    meta = parse_filename(path)
    if company:
        meta["company"] = company
    if fiscal_year:
        meta["fiscal_year"] = int(fiscal_year)
    reader = PdfReader(str(path))
    ids, docs, metas = [], [], []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for i, chunk in enumerate(_splitter.split_text(text)):
            # prefix keeps company/year inside the embedded text, which improves recall
            header = f"[{meta['company'] or 'General'} | FY{meta['fiscal_year'] or 'n/a'} | {meta['doc_type']}]\n"
            ids.append(hashlib.md5(f"{path.name}-{page_no}-{i}".encode()).hexdigest())
            docs.append(header + chunk)
            metas.append({**meta, "source": path.name, "page": page_no})
    if ids:
        get_collection().upsert(ids=ids, documents=docs, metadatas=metas)  # idempotent re-ingest
        _bm25_index.cache_clear()
        indexed_companies.cache_clear()
    logger.info("ingested %s: %d chunks", path.name, len(ids))
    return len(ids)


def ingest_directory(directory: Path | None = None) -> dict:
    directory = directory or settings.docs_dir
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {directory}. Run `python data/generate_data.py` first.")
    return {p.name: ingest_pdf(p) for p in pdfs}


def _tokens(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in HashingEmbedding.STOP and len(w) > 1]
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words]


@lru_cache(maxsize=1)
def indexed_companies() -> tuple[str, ...]:
    """Companies with at least one indexed chunk, including uploaded reports not in the ledger."""
    try:
        col = get_collection()
        if col.count() == 0:
            return ()
        metas = col.get(include=["metadatas"])["metadatas"]
    except Exception as exc:  # store missing / broken -> ledger companies only
        logger.warning("could not list indexed companies: %s", exc)
        return ()
    return tuple(sorted({m["company"] for m in metas if m.get("company")}))


@lru_cache(maxsize=1)
def _bm25_index():
    """In-memory BM25 index over all chunks (rebuilt after each ingestion)."""
    data = get_collection().get(include=["documents", "metadatas"])
    docs = [(i, d, m, Counter(_tokens(d))) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]
    df = Counter(t for *_, tf in docs for t in tf)
    n = len(docs) or 1
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    avgdl = sum(sum(tf.values()) for *_, tf in docs) / n
    return docs, idf, avgdl


def _bm25(query: str, companies, fiscal_year, k: int, k1: float = 1.5, b: float = 0.75) -> list[tuple]:
    docs, idf, avgdl = _bm25_index()
    q = set(_tokens(query))
    scored = []
    for i, d, m, tf in docs:
        if companies and m.get("company") not in companies:
            continue
        if fiscal_year and m.get("fiscal_year") != int(fiscal_year):
            continue
        dl = sum(tf.values())
        s = sum(idf.get(t, 0) * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / avgdl)) for t in q if t in tf)
        if s > 0:
            scored.append((s, i, d, m))
    return sorted(scored, key=lambda x: -x[0])[:k]


def search(query: str, companies: list[str] | None = None, fiscal_year: int | None = None,
           k: int | None = None) -> list[dict]:
    """Hybrid search: vector similarity + BM25 keyword match, fused with Reciprocal Rank Fusion.

    Metadata filters (company / fiscal year) apply to both retrievers.
    Each hit carries `distance` (cosine, vector side) and `bm25` (keyword side) scores.
    """
    col = get_collection()
    if col.count() == 0:
        raise RuntimeError("Vector store is empty. Run `python ingest.py` first.")
    k = k or settings.retrieval_top_k
    filters = []
    if companies:
        filters.append({"company": companies[0]} if len(companies) == 1 else {"company": {"$in": companies}})
    if fiscal_year:
        filters.append({"fiscal_year": int(fiscal_year)})
    where = None if not filters else filters[0] if len(filters) == 1 else {"$and": filters}
    res = col.query(query_texts=[query], n_results=k * 2, where=where)

    fused: dict[str, dict] = {}
    for rank, (cid, doc, meta, dist) in enumerate(zip(res["ids"][0], res["documents"][0],
                                                      res["metadatas"][0], res["distances"][0])):
        fused[cid] = {"text": doc, "distance": round(float(dist), 4), "bm25": 0.0, **meta,
                      "rrf": 1 / (RRF_K + rank)}
    for rank, (score, cid, doc, meta) in enumerate(_bm25(query, companies, fiscal_year, k * 2)):
        hit = fused.setdefault(cid, {"text": doc, "distance": 1.0, "bm25": 0.0, **meta, "rrf": 0.0})
        hit["bm25"] = round(score, 3)
        hit["rrf"] += 1 / (RRF_K + rank)
    return sorted(fused.values(), key=lambda h: -h["rrf"])[:k]
