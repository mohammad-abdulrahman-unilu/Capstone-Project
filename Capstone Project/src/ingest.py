import argparse
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import chromadb
import dask.bag as db
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def iter_chunks(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        end = start + chunk_size
        yield text[start:end]


def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    files = list(data_dir.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found under {data_dir}")
    bag = db.from_sequence(files, npartitions=min(32, len(files)))

    def read_file(path_str: str) -> Tuple[str, str]:
        path = Path(path_str)
        return path.name, path.read_text(encoding="utf-8")

    return bag.map(lambda p: read_file(str(p))).compute()


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma with sentence-transformer embeddings.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing .txt documents.")
    parser.add_argument("--persist-dir", type=Path, default=Path("artifacts/chroma"), help="Chroma persistence path.")
    parser.add_argument("--collection", default="financial_reports", help="Chroma collection name.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Character chunk size.")
    parser.add_argument("--overlap", type=int, default=120, help="Character overlap between chunks.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    args = parser.parse_args()

    start_time = time.time()
    docs = load_documents(args.data_dir)
    print(f"Loaded {len(docs)} documents from {args.data_dir}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(args.persist_dir))
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    chunk_records = []
    for doc_id, text in docs:
        for idx, chunk in enumerate(iter_chunks(text, args.chunk_size, args.overlap)):
            if not chunk.strip():
                continue
            chunk_records.append(
                {
                    "id": f"{doc_id}-chunk-{idx}",
                    "text": chunk,
                    "metadata": {"source": doc_id, "chunk_index": idx},
                }
            )
    print(f"Prepared {len(chunk_records)} chunks.")

    # Embed and store in batches to handle large corpora.
    for i in tqdm(range(0, len(chunk_records), args.batch_size), desc="Embedding + writing"):
        batch = chunk_records[i : i + args.batch_size]
        embeddings = model.encode([b["text"] for b in batch], batch_size=args.batch_size, show_progress_bar=False)
        collection.add(
            ids=[b["id"] for b in batch],
            documents=[b["text"] for b in batch],
            metadatas=[b["metadata"] for b in batch],
            embeddings=embeddings.tolist(),
        )

    elapsed = time.time() - start_time
    print(f"Ingestion complete. Persisted to {args.persist_dir} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
