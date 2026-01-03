# Proposal — Financial Report Analyzer (RAG with Quantized LLM)

## Problem
Analysts sift through large volumes of financial and ESG disclosures to answer questions on performance, guidance, risks, and controls. Manual review is slow and expensive. We will build a Retrieval-Augmented Generation (RAG) assistant that ingests multi-GB corpora of filings/reports and answers analyst queries with grounded citations.

## Objectives (pick ≥3, we cover 5/5)
1) **Technical implementation:** End-to-end RAG (ingest → vector store → LLM).  
2) **Scalability/performance:** Dask ingestion, batch embeddings, disk-persisted Chroma; ready to swap for distributed stores (Qdrant/Pinecone).  
3) **Advanced technique:** Quantized GGUF instruct model via `llama-cpp` + prompt chaining (retrieval summary → final answer).  
4) **Ethical/bias analysis:** See `notes/bias_ethics.md` for privacy, hallucination, and fairness mitigations.  
5) **Quantifiable impact:** ROUGE-L uplift vs. BM25 baseline; latency metrics (ingest/query) for time-to-insight.

## Dataset
- Provided sample docs under `data/sample_reports/` (earnings + ESG). Replace with your corpus (10K/20F, earnings transcripts, ESG reports).
- Files are assumed as UTF-8 text; PDFs can be preconverted with `pypdf` or Apache Tika (add step in ingest).

## Methodology
1. **Ingestion:** Stream text files with Dask, clean, split into overlapping chunks (chunk size, overlap tunable).  
2. **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers` (CPU friendly).  
3. **Vector store:** ChromaDB persisted locally (`artifacts/chroma`), can swap to managed DB.  
4. **Retrieval:** kNN search + score thresholding; prompt builds a cited context block.  
5. **Generation:** Quantized instruct LLM via `llama-cpp-python` (`LLAMA_CPP_MODEL_PATH`).  
6. **Evaluation:** ROUGE-L vs. references in `data/eval/eval.jsonl`; BM25 baseline with `rank-bm25`.  
7. **Deployment (mock):** CLI scripts; ready for API wrapper (FastAPI) and scheduler (Airflow) hooks.

## Expected outcomes
- Faster analyst Q&A; measurable ROUGE uplift vs. BM25 baseline; reduced time-to-insight by retrieval + LLM synthesis.
- Demonstrated path to scale (chunked processing, disk-backed vectors) and to production (swap vector DB, add observability).

## Timeline (suggested)
- Week 1: Data collection + ingestion pipeline.  
- Week 2: RAG wiring + prompt tuning.  
- Week 3: Evaluation, bias/privacy review, documentation.  
- Week 4: Hardening (caching, deployment scaffold).
