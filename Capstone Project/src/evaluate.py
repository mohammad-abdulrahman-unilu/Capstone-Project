import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import chromadb
import numpy as np
from llama_cpp import Llama
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from rag_qa import answer_question


def load_eval(dataset_path: Path) -> List[dict]:
    rows = []
    with dataset_path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_corpus(collection) -> Tuple[List[str], List[dict]]:
    res = collection.get(include=["documents", "metadatas"])
    docs = res["documents"]
    metas = res["metadatas"]
    flat_docs = [d for sub in docs for d in sub] if isinstance(docs[0], list) else docs
    flat_metas = [m for sub in metas for m in sub] if isinstance(metas[0], list) else metas
    return flat_docs, flat_metas


def bm25_baseline(question: str, corpus: List[str]) -> str:
    tokenized = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.split())
    best_idx = int(np.argmax(scores))
    return corpus[best_idx]


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG answers vs. references using ROUGE-L.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to JSONL with id, question, reference.")
    parser.add_argument("--persist-dir", type=Path, default=Path("artifacts/chroma"), help="Chroma persistence path.")
    parser.add_argument("--collection", default="financial_reports", help="Collection name.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.2)
    args = parser.parse_args()

    if not os.environ.get("LLAMA_CPP_MODEL_PATH"):
        raise EnvironmentError("LLAMA_CPP_MODEL_PATH must be set for evaluation.")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(args.persist_dir))
    collection = client.get_or_create_collection(name=args.collection)
    llm = Llama(model_path=os.environ["LLAMA_CPP_MODEL_PATH"], n_ctx=4096, temperature=0.2)

    corpus, _ = load_corpus(collection)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rows = load_eval(args.dataset)
    rag_scores = []
    bm25_scores = []

    for row in rows:
        q = row["question"]
        ref = row["reference"]
        start = time.time()
        rag_answer, _ = answer_question(
            question=q,
            collection=collection,
            embed_model=embed_model,
            llm=llm,
            top_k=args.top_k,
            min_score=args.min_score,
        )
        rag_time = time.time() - start

        base_answer = bm25_baseline(q, corpus)
        rag_score = scorer.score(ref, rag_answer)["rougeL"].fmeasure
        base_score = scorer.score(ref, base_answer)["rougeL"].fmeasure
        rag_scores.append(rag_score)
        bm25_scores.append(base_score)

        print(f"[{row['id']}] RAG Rouge-L: {rag_score:.3f} | BM25: {base_score:.3f} | Latency: {rag_time:.2f}s")

    print(f"\nAverage Rouge-L — RAG: {np.mean(rag_scores):.3f}, BM25 baseline: {np.mean(bm25_scores):.3f}")


if __name__ == "__main__":
    main()
