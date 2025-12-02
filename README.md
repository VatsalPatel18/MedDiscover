# MedDiscover
MedDiscover is an AI-powered tool designed to assist biomedical researchers using RAG-LLM models fine-tuned on PubMed literature.

## CLI evaluation (headless)
Install the package (or use it in editable mode), set your `OPENAI_API_KEY`, and run the built-in evaluator:

```bash
pip install .
export OPENAI_API_KEY=...
# optional: ALLOW_MEDCPT_CPU=1 to force MedCPT on CPU

meddiscover-eval \
  --pdfs med_discover_ai/eval_samples/sample_pdfs/fmed-11-1345659.pdf med_discover_ai/eval_samples/sample_pdfs/reviewer_comments_aug12.pdf \
  --qa_csv med_discover_ai/eval_samples/sample_qa.csv \
  --embedding_model "MedCPT (GPU Recommended)" \
  --llm_models gpt-4.1-mini \
  --k 3 \
  --max_tokens 64 \
  --out_dir ./eval_outputs_demo
```

- For Ada-based retrieval, switch `--embedding_model` to `OpenAI Ada-002 (CPU/Cloud)`.
- RAGAS metrics are optional; if dependencies are missing or the QA CSV lacks a `reference` column, they fall back to `None`.
- Re-ranking stays disabled on CPU; enable `--rerank` only when a GPU and cross-encoder are available.
