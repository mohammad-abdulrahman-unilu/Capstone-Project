# Bias, Ethics, and Privacy Considerations

## Data privacy
- Financial/ESG reports may include sensitive forward-looking statements or PII (rare but possible in appendices). Mitigation: run PII scrubbing before ingestion (regex for emails/phones, spaCy NER), and avoid storing raw data in shared vector DBs.

## Hallucinations & grounding
- Risk: LLM invents numbers or citations. Mitigation: force retrieval-grounded prompts, include source snippets, and reject answers when similarity scores fall below a threshold. Expose a `--min-score` flag.

## Bias & fairness
- Model biases can skew interpretations (e.g., assigning blame to specific regions or demographics). Mitigation: use neutral prompts, add a safety post-check that removes demographic attributions unless present in sources.

## Compliance & auditability
- Keep provenance: store document IDs and chunk offsets; surface top-k sources in answers for audit trails.

## Deployment hygiene
- Run models locally or in a VPC; avoid sending proprietary text to external APIs. Prefer quantized local models (GGUF) with `llama-cpp`.
- Log prompts/responses securely; rotate keys if remote embeddings/LLMs are used.
