# Big Data Capstone — Financial Report RAG (LLM)

End-to-end Retrieval-Augmented Generation (RAG) pipeline for financial/ESG reports. The project demonstrates ingestion of large text corpora with Dask, vector search with ChromaDB, LLM generation with a quantized GGUF model via `llama-cpp-python`, evaluation, and bias/ethics notes.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set path to a quantized instruct model (e.g., Mistral 7B Instruct GGUF q4_k_m)
export LLAMA_CPP_MODEL_PATH=/path/to/model.gguf

# Ingest documents into Chroma (default uses sample reports under data/)
python src/ingest.py --data-dir data/sample_reports --persist-dir artifacts/chroma

# Run a RAG query
python src/rag_qa.py --persist-dir artifacts/chroma --question "What drove margin expansion?"

# Evaluate generated answers vs. references (toy example)
python src/evaluate.py --persist-dir artifacts/chroma --dataset data/eval/eval.jsonl
```

## Project components
- `src/ingest.py`: Chunk large text files with Dask, embed with `all-MiniLM-L6-v2`, store in ChromaDB (persisted locally).
- `src/rag_qa.py`: Retrieve top-k chunks and answer with `LlamaCpp` (quantized model). Simple prompt chaining: retrieval summary → final answer.
- `src/evaluate.py`: Computes ROUGE-L vs. provided reference answers; baseline BM25 for comparison.
- `proposal.md`: Problem definition, objectives mapping, datasets, models, and plan.
- `notes/bias_ethics.md`: Bias/privacy/ethics analysis and mitigations.

## Requirements
- Python 3.10+
- A local quantized GGUF instruct model (recommended: Mistral-7B-Instruct-v0.2 Q4_K_M) and `LLAMA_CPP_MODEL_PATH` env var.
- For larger corpora, set `--data-dir` to your document root and increase `--chunk-size`/`--persist-dir` accordingly.

## Big Data considerations
- Dask-based chunked loading to avoid memory blowups on multi-GB corpora.
- Vector store persists to disk; replace with managed vector DB (Pinecone/Qdrant) as needed.
- Embedding and retrieval are batchable; adjust `--batch-size` for throughput.

## Evaluation & impact
- Compare RAG answers vs. references with ROUGE-L; baseline BM25 → quantifies uplift.
- Track latency (ingest/query) via printed timings; extend with Prometheus/OpenTelemetry as desired.

## Next steps
1. Swap sample docs with your financial corpus; point `--data-dir` accordingly.
2. Add domain-specific evaluators (e.g., factuality, citation rate).
3. Optional PEFT finetune a small model on Q&A pairs; update `rag_qa.py` to load adapters.
