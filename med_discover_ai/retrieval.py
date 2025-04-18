# med_discover_ai/retrieval.py
import torch
import numpy as np
import json
import os
from med_discover_ai.config import (
    USE_GPU, CROSS_ENCODER_MODEL, MAX_ARTICLE_LENGTH, DOC_META_PATH, DEVICE,
    DEFAULT_K, DEFAULT_RERANK_ENABLED,
    get_embedding_model_id # Import helper
)
# Pass embedding_model_display_name to embed_query
from med_discover_ai.embeddings import embed_query

# --- Global Variables for Re-ranking Model (initialized conditionally) ---
cross_tokenizer = None
cross_model = None

# --- Initialization ---
def initialize_reranker():
    """Initializes the re-ranking model if GPU is available."""
    global cross_tokenizer, cross_model
    # Only load if GPU is available AND a cross-encoder model is configured
    if USE_GPU and CROSS_ENCODER_MODEL:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"Loading MedCPT Cross-Encoder model ({CROSS_ENCODER_MODEL}) for re-ranking (GPU)...")
            cross_tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
            cross_model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL).to(DEVICE)
            cross_model.eval() # Set model to evaluation mode
            print("Cross-Encoder model loaded successfully.")
        except ImportError:
            print("Error: 'transformers' library not found. Cannot use MedCPT Cross-Encoder.")
            cross_model = None # Ensure model is None if loading fails
        except Exception as e:
            print(f"Error loading MedCPT Cross-Encoder model: {e}")
            cross_model = None # Ensure model is None if loading fails
    else:
        if not USE_GPU:
            print("Re-ranking with Cross-Encoder is disabled (CPU mode).")
        elif not CROSS_ENCODER_MODEL:
            print("Re-ranking disabled: No Cross-Encoder model configured.")

# Call initialization when the module is loaded
initialize_reranker()

# --- Metadata Loading ---
def load_metadata(meta_path=DOC_META_PATH):
    """Loads document metadata from a JSON file."""
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file not found at {meta_path}.")
        return None
    try:
        with open(meta_path, "r", encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Metadata loaded successfully from {meta_path}.")
        return metadata
    except Exception as e:
        print(f"Error loading metadata from {meta_path}: {e}")
        return None

# --- Re-ranking Function ---
def rerank_candidates(query, candidates):
    """
    Re-ranks candidate documents using the MedCPT Cross-Encoder.
    Requires GPU and initialized cross-encoder model.
    """
    # Check again if model is loaded, in case initialization failed
    if not USE_GPU or not cross_model or not cross_tokenizer:
        # No need to print here, handled during search_and_rerank call
        return None

    if not candidates:
        print("Warning: No candidates provided for re-ranking.")
        return np.array([])

    print(f"Re-ranking {len(candidates)} candidates using MedCPT Cross-Encoder (GPU)...")
    pairs = [[query, candidate["text"]] for candidate in candidates]

    try:
        with torch.no_grad():
            encoded = cross_tokenizer(
                pairs, truncation=True, padding=True, return_tensors="pt", max_length=MAX_ARTICLE_LENGTH
            )
            encoded = {key: val.to(DEVICE) for key, val in encoded.items()}
            outputs = cross_model(**encoded)
            logits = outputs.logits.squeeze(dim=1)
        print("Re-ranking finished.")
        return logits.cpu().numpy()
    except Exception as e:
        print(f"Error during re-ranking with Cross-Encoder: {e}")
        return None # Indicate failure

# --- Combined Search and Re-ranking ---
def search_and_rerank(query, index, doc_metadata, embedding_model_display_name, k=DEFAULT_K, enable_rerank=DEFAULT_RERANK_ENABLED):
    """
    Performs dense retrieval using FAISS with the selected embedding model,
    optionally re-ranks (GPU only), and returns sorted candidates.

    Parameters:
        query (str): The user query.
        index (faiss.Index): The loaded FAISS index.
        doc_metadata (list): List of document metadata dictionaries.
        embedding_model_display_name (str): Display name of the embedding model selected in UI.
        k (int): Number of top results to retrieve initially.
        enable_rerank (bool): Whether to attempt re-ranking (requires GPU and loaded model).

    Returns:
        list: Sorted candidate dictionaries. Empty list on major failure.
    """
    if not query or query.isspace():
        print("Error: Cannot search with an empty query.")
        return []
    if index is None:
        print("Error: FAISS index is not available.")
        return []
    if doc_metadata is None:
        print("Error: Document metadata is not available.")
        return []

    # Step 1: Embed the query using the *selected* embedding model
    print(f"Embedding query for search (Model: {embedding_model_display_name}, k={k}, re-rank={enable_rerank})...")
    query_embedding = embed_query(query, embedding_model_display_name) # Pass the selected model name
    if query_embedding is None:
        print("Error: Failed to embed query. Aborting search.")
        return []

    # Ensure query embedding is float32 for FAISS
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)

    # Check if index dimension matches query embedding dimension
    if index.d != query_embedding.shape[1]:
        print(f"FATAL ERROR: FAISS index dimension ({index.d}) does not match query embedding dimension ({query_embedding.shape[1]}) for model '{embedding_model_display_name}'.")
        print("This likely means the index was built with a different embedding model.")
        print("Please re-process PDFs with the currently selected embedding model.")
        # Return an empty list or raise a specific error? Returning empty for now.
        return []


    # Step 2: Dense Retrieval using FAISS
    print(f"Performing FAISS search for top {k} candidates...")
    try:
        scores, inds = index.search(query_embedding, k)
        print(f"FAISS search returned {len(inds[0])} results.")
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    # Step 3: Retrieve Candidate Metadata
    candidates = []
    retrieved_indices = inds[0]
    retrieved_scores = scores[0]

    for score, ind in zip(retrieved_scores, retrieved_indices):
        if ind < 0 or ind >= len(doc_metadata):
            print(f"Warning: Invalid index {ind} returned by FAISS search. Skipping.")
            continue
        entry = doc_metadata[ind].copy()
        entry["retrieval_score"] = float(score)
        candidates.append(entry)

    if not candidates:
        print("No valid candidates found after FAISS search.")
        return []

    print(f"Retrieved {len(candidates)} initial candidates.")

    # Step 4: Optional Re-ranking (Only if GPU enabled, checkbox checked, and model loaded)
    rerank_scores = None
    perform_rerank = USE_GPU and enable_rerank and cross_model is not None
    if perform_rerank:
        print("Attempting re-ranking...")
        rerank_scores = rerank_candidates(query, candidates)
        if rerank_scores is not None:
            print("Assigning re-rank scores...")
            if len(rerank_scores) == len(candidates):
                for i, score in enumerate(rerank_scores):
                    candidates[i]["rerank_score"] = float(score)
            else:
                print(f"Warning: Mismatch between candidates ({len(candidates)}) and re-rank scores ({len(rerank_scores)}). Skipping score assignment.")
                rerank_scores = None # Treat as if re-ranking didn't happen for sorting
        else:
            print("Re-ranking failed or produced no scores.")
            rerank_scores = None # Ensure it's None for sorting logic
    else:
         if enable_rerank and not USE_GPU:
             print("Re-ranking skipped: GPU not available.")
         elif enable_rerank and not cross_model:
             print("Re-ranking skipped: Cross-encoder model not loaded.")
         elif not enable_rerank:
             print("Re-ranking skipped: Disabled in UI.")


    # Step 5: Sort Results
    # Determine sort key: Use rerank_score if re-ranking was successful, otherwise retrieval_score
    sort_key = "rerank_score" if perform_rerank and rerank_scores is not None else "retrieval_score"

    # Determine reverse sort order:
    # - True (descending) for rerank_score (higher is better)
    # - True (descending) for retrieval_score if using MedCPT/IP (higher is better)
    # - False (ascending) for retrieval_score if using OpenAI/L2 (lower is better)
    is_medcpt_embedding = get_embedding_model_id(embedding_model_display_name) == "ncbi/MedCPT-Article-Encoder"
    reverse_sort = True if sort_key == "rerank_score" or (sort_key == "retrieval_score" and is_medcpt_embedding) else False

    print(f"Sorting candidates by '{sort_key}' (reverse={reverse_sort})...")
    try:
        candidates_sorted = sorted(
            candidates,
            key=lambda x: x.get(sort_key, -np.inf if reverse_sort else np.inf),
            reverse=reverse_sort
        )
        print("Candidates sorted successfully.")
        return candidates_sorted
    except Exception as e:
        print(f"Error sorting candidates: {e}")
        return candidates # Fallback to unsorted
