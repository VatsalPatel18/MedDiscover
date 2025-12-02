#!/usr/bin/env python3
"""
End-to-end evaluation runner that reuses the MedDiscover pipeline:
 - ingest + chunk PDFs
 - embed + build FAISS index
 - retrieve + generate answers for a QA list across multiple models
 - compute metrics (ROUGE/BLEU + RAGAS metrics when available)
 - write per-model CSVs and a combined table matching prior schema
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# --- Reuse core library pieces ---
from med_discover_ai.chunking import chunk_text
from med_discover_ai.pdf_utils import extract_text_from_pdf
from med_discover_ai.embeddings import embed_documents
from med_discover_ai.index import build_faiss_index, save_index
from med_discover_ai.retrieval import search_and_rerank
from med_discover_ai.llm_inference import get_llm_answer
from med_discover_ai.evaluation import evaluate_response
from med_discover_ai.config import (
    CHUNK_SIZE,
    OVERLAP,
    AVAILABLE_EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_LLM_MODEL,
    DEFAULT_K,
    DEFAULT_RERANK_ENABLED,
    DEFAULT_MAX_TOKENS,
    get_embedding_model_id,
)

# Optional ragas metrics (used in prior quant_* runs)
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness as ragas_faithfulness,
        answer_correctness as ragas_answer_correctness,
        context_recall as ragas_context_recall,
        context_precision as ragas_context_precision,
        answer_relevancy as ragas_answer_relevancy,
    )
    from datasets import Dataset

    HAS_RAGAS = True
except Exception:
    HAS_RAGAS = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_qa(qa_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(qa_csv)
    if not {"query", "gt"}.issubset(df.columns):
        raise ValueError(f"QA CSV must contain columns ['query','gt']; found {df.columns.tolist()}")
    return df[["query", "gt"]]


def build_index_from_pdfs(pdf_paths: List[Path], embedding_model: str, out_dir: Path):
    """Copy of gradio_app.process_pdfs_interface logic adapted for CLI."""
    all_chunks, metadata_list = [], []
    doc_id_counter = 0
    for pdf in pdf_paths:
        text = extract_text_from_pdf(str(pdf))
        if not text or text.startswith("Error reading"):
            print(f"Skipping {pdf}: no text extracted.")
            continue
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        if not chunks:
            print(f"Skipping {pdf}: no chunks produced.")
            continue
        for chunk_id, chunk_text_content in enumerate(chunks):
            metadata_list.append(
                {
                    "doc_id": doc_id_counter,
                    "filename": pdf.name,
                    "chunk_id": chunk_id,
                    "text": chunk_text_content,
                }
            )
            all_chunks.append(chunk_text_content)
        doc_id_counter += 1

    if not all_chunks:
        raise RuntimeError("No text extracted/chunked from provided PDFs.")

    embeddings = embed_documents(all_chunks, embedding_model)
    if embeddings is None or embeddings.shape[0] == 0:
        raise RuntimeError("Embedding generation failed.")

    # Select metric based on embedding model (same heuristic as gradio_app)
    model_id = get_embedding_model_id(embedding_model)
    use_ip = model_id == AVAILABLE_EMBEDDING_MODELS.get("MedCPT (GPU Recommended)")
    index = build_faiss_index(embeddings, use_ip_metric=use_ip)
    if index is None:
        raise RuntimeError("FAISS index build failed.")

    # Persist artifacts for reuse
    artifacts_dir = out_dir / "artifacts"
    ensure_out_dir(artifacts_dir)
    save_index(index, str(artifacts_dir / "index.faiss"))
    with open(artifacts_dir / "doc_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    return index, metadata_list


def compute_ragas_metrics(query: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
    """Single-sample ragas evaluation to mirror prior quant_* metrics."""
    if not HAS_RAGAS:
        return {
            "faithfulness": None,
            "answer_correctness": None,
            "context_recall": None,
            "context_precision": None,
            "answer_relevancy": None,
        }

    # Skip ragas if there is no context or if the answer is an error string.
    if not contexts or str(answer).startswith("Error"):
        return {
            "faithfulness": None,
            "answer_correctness": None,
            "context_recall": None,
            "context_precision": None,
            "answer_relevancy": None,
        }

    try:
        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truths": [[ground_truth]],
            }
        )
        scores = ragas_evaluate(
            ds,
            metrics=[
                ragas_faithfulness,
                ragas_answer_correctness,
                ragas_context_recall,
                ragas_context_precision,
                ragas_answer_relevancy,
            ],
        )
        return {k: float(v) for k, v in scores.items()}
    except Exception as e:
        print(f"Warning: ragas evaluation failed ({e}). Falling back to None metrics.")
        return {
            "faithfulness": None,
            "answer_correctness": None,
            "context_recall": None,
            "context_precision": None,
            "answer_relevancy": None,
        }


def evaluate_one(query: str, gt: str, contexts: List[Dict[str, Any]], ans: str) -> Dict[str, Any]:
    context_texts = [c.get("text", "") for c in contexts]
    ragas_scores = compute_ragas_metrics(query, ans, context_texts, gt)
    rouge_bleu = evaluate_response(gt, ans)
    accuracy = None
    if ragas_scores.get("answer_correctness") is not None:
        accuracy = float(ragas_scores["answer_correctness"] > 0.5)

    return {
        "rouge1": rouge_bleu.get("rouge1"),
        "rouge2": rouge_bleu.get("rouge2"),
        "rougeL": rouge_bleu.get("rougeL"),
        "bleu": rouge_bleu.get("bleu"),
        **ragas_scores,
        "accuracy": accuracy,
    }


# --------------------------------------------------------------------------- #
# Main run loop
# --------------------------------------------------------------------------- #
def run(args):
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)
    per_model_dir = out_dir / "per_model"
    ensure_out_dir(per_model_dir)

    qa_df = load_qa(Path(args.qa_csv))
    pdf_paths = [Path(p) for p in args.pdfs]

    print(f"Building/using index from {len(pdf_paths)} PDFs with embedding model '{args.embedding_model}'...")
    index, metadata = build_index_from_pdfs(pdf_paths, args.embedding_model, out_dir)

    all_rows = []
    for model in args.llm_models:
        model_rows = []
        print(f"\n=== Evaluating model: {model} ===")
        for _, row in qa_df.iterrows():
            query = str(row["query"])
            gt = str(row["gt"])

            candidates = search_and_rerank(
                query=query,
                index=index,
                doc_metadata=metadata,
                embedding_model_display_name=args.embedding_model,
                k=args.k,
                enable_rerank=args.rerank,
            )

            answer, context_text_block, _usage = get_llm_answer(
                query=query,
                retrieved_candidates=candidates,
                llm_model=model,
                max_tokens=args.max_tokens,
            )

            metrics = evaluate_one(query, gt, candidates, answer)
            row_out = {
                "query": query,
                "gt": gt,
                "ans": answer,
                "retrieved_contexts": json.dumps([c.get("text", "") for c in candidates], ensure_ascii=False),
                **metrics,
                "model": model,
            }
            model_rows.append(row_out)
            all_rows.append(row_out)

        model_df = pd.DataFrame(model_rows)
        model_out_path = per_model_dir / f"{model.replace(':', '_')}.csv"
        model_df.to_csv(model_out_path, index=False)
        print(f"Saved {len(model_df)} rows to {model_out_path}")

    combined_df = pd.DataFrame(all_rows)
    combined_out = out_dir / "all_models_combined.csv"
    combined_df.to_csv(combined_out, index=False)
    print(f"\nCombined results written to {combined_out} ({len(combined_df)} rows).")


def parse_args():
    parser = argparse.ArgumentParser(description="MedDiscover evaluation runner")
    parser.add_argument("--pdfs", nargs="+", required=True, help="PDF file paths.")
    parser.add_argument("--qa_csv", required=True, help="CSV with columns query,gt.")
    parser.add_argument(
        "--embedding_model",
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help=f"Embedding model display name (default: {DEFAULT_EMBEDDING_MODEL_NAME}).",
    )
    parser.add_argument(
        "--llm_models",
        default=DEFAULT_LLM_MODEL,
        help="Comma-separated list of LLM model identifiers (OpenAI or ollama:...).",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k for retrieval.")
    parser.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        default=DEFAULT_RERANK_ENABLED,
        help="Enable cross-encoder rerank (GPU only).",
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable re-ranking even if GPU is available.",
    )
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens for generation.")
    parser.add_argument("--out_dir", default="./eval_outputs", help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Normalize llm_models input to list
    args.llm_models = [m.strip() for m in args.llm_models.split(",") if m.strip()]
    run(args)


def main():
    args = parse_args()
    args.llm_models = [m.strip() for m in args.llm_models.split(",") if m.strip()]
    run(args)
