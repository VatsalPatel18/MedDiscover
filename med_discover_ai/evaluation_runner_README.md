# Evaluation Runner

Standalone CLI that reuses the existing MedDiscover code to:
1) ingest and chunk PDFs,
2) build a FAISS index,
3) run retrieval + LLM answer generation for a QA list,
4) compute metrics, and
5) save per-model CSVs plus a combined table.

## Dependency note
This runner expects the MedDiscover library to be available. When using it from the `MedDiscover-Analysis` repo, set `MEDDISCOVER_PATH` to the path of the MedDiscover source (sibling clone), e.g.:
```bash
export MEDDISCOVER_PATH=../MedDiscover
```
Alternatively, run the script from the monorepo root where `MedDiscover/` sits alongside this folder.

## Quick start
```bash
python -m evaluation_runner.run_evaluation \
  --pdfs /path/to/a.pdf /path/to/b.pdf \
  --qa_csv /path/to/questions.csv \
  --embedding_model "MedCPT (GPU Recommended)" \
  --llm_models gpt-4.1-mini,gpt-4.1-nano \
  --k 5 \
  --out_dir ./eval_outputs
```

Expected QA CSV columns: `query,gt`. Additional columns are ignored.

Outputs (under `--out_dir`):
- `per_model/<model>.csv` — one row per question with: `query, gt, ans, retrieved_contexts, rouge1, rouge2, rougeL, bleu, faithfulness, answer_correctness, context_recall, context_precision, answer_relevancy, accuracy, model`
- `all_models_combined.csv` — concatenation of all per-model rows.
- `artifacts/index.faiss` and `artifacts/doc_metadata.json` — built index + metadata for reuse.
