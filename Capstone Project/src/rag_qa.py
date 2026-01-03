import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer


def load_llm() -> Llama:
    model_path = os.environ.get("LLAMA_CPP_MODEL_PATH")
    if not model_path:
        raise EnvironmentError("LLAMA_CPP_MODEL_PATH is not set. Point it to a quantized GGUF instruct model.")
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=0,  # auto
        temperature=0.2,
    )


def retrieve(
    collection,
    embed_model: SentenceTransformer,
    question: str,
    top_k: int,
    min_score: float,
) -> List[Dict]:
    q_emb = embed_model.encode([question])[0]
    res = collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k, include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = 1 - dist  # cosine distance to similarity
        if score < min_score:
            continue
        hits.append({"text": doc, "metadata": meta, "score": score})
    return hits


def build_prompt(question: str, contexts: List[Dict]) -> str:
    context_block = "\n\n".join(
        [f"[Source: {c['metadata'].get('source')}#chunk{c['metadata'].get('chunk_index')}] {c['text']}" for c in contexts]
    )
    return f"""You are a financial analyst assistant. Answer the question using ONLY the sources provided.
If the answer is not in the sources, say you cannot find it. Return a concise paragraph and cite sources inline.

Question: {question}

Sources:
{context_block}

Answer with citations like (source#chunk)."""


def answer_question(
    question: str,
    collection,
    embed_model: SentenceTransformer,
    llm: Llama,
    top_k: int,
    min_score: float,
) -> Tuple[str, List[Dict]]:
    retrieved = retrieve(collection, embed_model, question, top_k, min_score)
    if not retrieved:
        return "I could not find supporting information in the indexed documents.", []
    prompt = build_prompt(question, retrieved)
    output = llm(
        prompt,
        max_tokens=512,
        stop=["\n\nSources:", "Sources:"],
    )
    text = output["choices"][0]["text"].strip()
    return text, retrieved


def main():
    parser = argparse.ArgumentParser(description="Query a Chroma-backed RAG index with a local quantized LLM.")
    parser.add_argument("--persist-dir", type=Path, default=Path("artifacts/chroma"), help="Chroma persistence path.")
    parser.add_argument("--collection", default="financial_reports", help="Collection name.")
    parser.add_argument("--question", required=True, help="User question to answer.")
    parser.add_argument("--top-k", type=int, default=4, help="Top-k documents to retrieve.")
    parser.add_argument("--min-score", type=float, default=0.2, help="Minimum cosine similarity to accept.")
    args = parser.parse_args()

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(args.persist_dir))
    collection = client.get_or_create_collection(name=args.collection)
    llm = load_llm()

    start = time.time()
    answer, sources = answer_question(
        question=args.question,
        collection=collection,
        embed_model=embed_model,
        llm=llm,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    elapsed = time.time() - start

    print(f"\nAnswer ({elapsed:.2f}s):\n{answer}\n")
    print("Sources used:")
    for s in sources:
        print(f"- {s['metadata']['source']} chunk {s['metadata']['chunk_index']} (score={s['score']:.3f})")


if __name__ == "__main__":
    main()
