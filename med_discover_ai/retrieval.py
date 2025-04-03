# med_discover_ai/retrieval.py
import torch
import numpy as np
import json
import os
from med_discover_ai.config import (
    USE_GPU, CROSS_ENCODER_MODEL, MAX_ARTICLE_LENGTH, DOC_META_PATH, DEVICE,
    DEFAULT_K, DEFAULT_RERANK_ENABLED
)
from med_discover_ai.embeddings import embed_query
# Removed index loading from here, should be passed in

# --- Global Variables for Re-ranking Model (initialized conditionally) ---
cross_tokenizer = None
cross_model = None

# --- Initialization ---
def initialize_reranker():
    """Initializes the re-ranking model if GPU is available."""
    global cross_tokenizer, cross_model
    if USE_GPU:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print("Loading MedCPT Cross-Encoder model for re-ranking (GPU)...")
            cross_tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
            cross_model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL).to(DEVICE)
            cross_model.eval() # Set model to evaluation mode
            print("Cross-Encoder model loaded successfully.")
        except ImportError:
            print("Error: 'transformers' library not found. Cannot use MedCPT Cross-Encoder.")
        except Exception as e:
            print(f"Error loading MedCPT Cross-Encoder model: {e}")
    else:
        print("Re-ranking with Cross-Encoder is disabled (CPU mode).")

# Call initialization when the module is loaded
initialize_reranker()

# --- Metadata Loading ---
def load_metadata(meta_path=DOC_META_PATH):
    """
    Load document metadata from a JSON file.

    Parameters:
        meta_path (str): Path to the metadata JSON file.

    Returns:
        list or None: List of document metadata dictionaries, or None if loading fails.
    """
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file not found at {meta_path}.")
        return None
    try:
        with open(meta_path, "r", encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Metadata loaded successfully from {meta_path}.")
        return metadata
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {meta_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading metadata from {meta_path}: {e}")
        return None

# --- Re-ranking Function ---
def rerank_candidates(query, candidates):
    """
    Re-ranks candidate documents using the MedCPT Cross-Encoder.
    Requires GPU and initialized cross-encoder model.

    Parameters:
        query (str): The user query.
        candidates (list): List of candidate dictionaries, each must have a "text" key.

    Returns:
        np.array or None: An array of relevance scores (logits) for each candidate,
                          or None if re-ranking cannot be performed.
    """
    if not USE_GPU or not cross_model or not cross_tokenizer:
        print("Re-ranking skipped: GPU not available or Cross-Encoder model not loaded.")
        return None # Indicate that re-ranking was not performed

    if not candidates:
        print("Warning: No candidates provided for re-ranking.")
        return np.array([])

    print(f"Re-ranking {len(candidates)} candidates using MedCPT Cross-Encoder (GPU)...")
    # Prepare pairs for the cross-encoder: [query, candidate_text]
    pairs = [[query, candidate["text"]] for candidate in candidates]

    try:
        with torch.no_grad(): # Disable gradient calculations for inference
            # Tokenize the pairs
            encoded = cross_tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=MAX_ARTICLE_LENGTH # Use appropriate max length for cross-encoder
            )
            # Move tensors to the configured device (GPU)
            encoded = {key: val.to(DEVICE) for key, val in encoded.items()}

            # Get model outputs (logits represent relevance score)
            outputs = cross_model(**encoded)
            logits = outputs.logits.squeeze(dim=1) # Remove unnecessary dimension

        print("Re-ranking finished.")
        return logits.cpu().numpy() # Return scores as a NumPy array
    except Exception as e:
        print(f"Error during re-ranking with Cross-Encoder: {e}")
        return None # Indicate failure

# --- Combined Search and Re-ranking ---
def search_and_rerank(query, index, doc_metadata, k=DEFAULT_K, enable_rerank=DEFAULT_RERANK_ENABLED):
    """
    Performs dense retrieval using FAISS, optionally re-ranks using MedCPT Cross-Encoder (GPU only),
    and returns sorted candidate documents.

    Parameters:
        query (str): The user query.
        index (faiss.Index): The loaded FAISS index.
        doc_metadata (list): List of document metadata dictionaries.
        k (int): Number of top results to retrieve initially.
        enable_rerank (bool): Whether to perform re-ranking (only applies if USE_GPU is True).

    Returns:
        list: A list of candidate document dictionaries, sorted by relevance.
              Each dictionary includes 'text', 'filename', 'chunk_id', 'retrieval_score',
              and potentially 'rerank_score'. Returns empty list on major failure.
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

    # Step 1: Embed the query
    print(f"Embedding query for search (k={k}, re-rank={enable_rerank})...")
    query_embedding = embed_query(query)
    if query_embedding is None:
        print("Error: Failed to embed query. Aborting search.")
        return []

    # Ensure query embedding is float32 for FAISS
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)

    # Step 2: Dense Retrieval using FAISS
    print(f"Performing FAISS search for top {k} candidates...")
    try:
        # `index.search` returns distances/scores and indices
        # For IndexFlatL2 (CPU/OpenAI): scores are squared Euclidean distances (lower is better)
        # For IndexFlatIP (GPU/MedCPT): scores are inner products (higher is better)
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
            continue # Skip invalid indices

        # Create a copy to avoid modifying the original metadata list
        entry = doc_metadata[ind].copy()
        entry["retrieval_score"] = float(score) # Store the raw score from FAISS
        candidates.append(entry)

    if not candidates:
        print("No valid candidates found after FAISS search.")
        return []

    print(f"Retrieved {len(candidates)} initial candidates.")

    # Step 4: Optional Re-ranking (GPU only)
    rerank_scores = None
    if USE_GPU and enable_rerank:
        rerank_scores = rerank_candidates(query, candidates) # This returns None if it fails or is skipped
        if rerank_scores is not None:
             print("Assigning re-rank scores...")
             if len(rerank_scores) == len(candidates):
                 for i, score in enumerate(rerank_scores):
                     candidates[i]["rerank_score"] = float(score)
             else:
                 print(f"Warning: Mismatch between number of candidates ({len(candidates)}) and re-rank scores ({len(rerank_scores)}). Skipping score assignment.")
                 rerank_scores = None # Treat as if re-ranking didn't happen for sorting
        else:
            print("Re-ranking was skipped or failed.")

    # Step 5: Sort Results
    # Determine the sorting key based on whether re-ranking was performed successfully
    sort_key = "rerank_score" if (USE_GPU and enable_rerank and rerank_scores is not None) else "retrieval_score"
    # Determine reverse sort order: True for scores where higher is better (IP, rerank), False for L2 distance
    reverse_sort = True if sort_key == "rerank_score" or (sort_key == "retrieval_score" and USE_GPU) else False

    print(f"Sorting candidates by '{sort_key}' (reverse={reverse_sort})...")
    try:
        # Handle potential missing 'rerank_score' key if reranking failed midway
        candidates_sorted = sorted(
            candidates,
            key=lambda x: x.get(sort_key, -np.inf if reverse_sort else np.inf), # Default to worst score if key missing
            reverse=reverse_sort
        )
        print("Candidates sorted successfully.")
        return candidates_sorted
    except Exception as e:
        print(f"Error sorting candidates: {e}")
        return candidates # Return unsorted candidates as fallback

