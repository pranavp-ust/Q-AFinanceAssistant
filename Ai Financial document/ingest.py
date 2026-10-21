"""Index all PDFs in data/documents into the Chroma vector store.

Usage:  python ingest.py
"""
from src.vectorstore import get_collection, ingest_directory

if __name__ == "__main__":
    counts = ingest_directory()
    for name, n in counts.items():
        print(f"  {name:<45} {n:>3} chunks")
    print(f"Indexed {sum(counts.values())} chunks from {len(counts)} files "
          f"(collection now holds {get_collection().count()}).")
